#!/usr/bin/env python3
"""
BBB-crossing Lipid Module Screener for LNP-mRNA Brain Delivery.

Reference: Wang et al. (2025, Nature Materials) — 72 BLNPs designed by conjugating
BBB-crossing small molecules to ionizable amino lipids.

This module adds a 5th dimension to GateScore-ML: "conjugatability" (S_conjugate).
It systematically screens molecular libraries for candidates that are both:
  A) Predicted BBB-gating molecules (GateScore > threshold), AND
  B) Chemically conjugatable to lipid tails (has -OH, -NH2, -COOH, etc.)

Output: Ranked table of Top-N candidates for BLNP design, suitable for
publication (J Chem Inf Model / J Cheminform / Brief Bioinform level).

GateScore-Lipid = GateScore-ML (4D) + S_conjugate (new 5th dim)
  → Total = 0.35*Gating + 0.25*Chemistry + 0.15*Novelty + 0.10*Safety + 0.15*Conjugate

Usage:
  from de_novo_pipeline.bbb_lipid_filter import LipidModuleScreener

  screener = LipidModuleScreener()
  # Quick test on BBB-GatingDB
  df = screener.screen_from_db("outputs/bbb_gating_dataset_v1.csv")
  # Or screen a custom SMILES list
  df = screener.screen(smiles_list)
  # Export for publication
  screener.export_for_publication(df, "outputs/lipid_module_candidates.csv")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.DataStructs import BulkTanimotoSimilarity, TanimotoSimilarity
except ImportError:
    raise ImportError("RDKit required: pip install rdkit")

try:
    from .scoring import GatingScorer, FilterFunnel
    from .features import Featurizer
except ImportError:
    from scoring import GatingScorer, FilterFunnel
    from features import Featurizer


# ══════════════════════════════════════════════════════════════════════════════
# SMARTS Patterns for Conjugatable Functional Groups
# ══════════════════════════════════════════════════════════════════════════════

CONJUGATABLE_SMARTS = {
    # ── Hydroxyl groups (→ carbonate ester or acetal linkage) ──
    "aliphatic_OH": {
        "smarts": "[C;!$(C=O);!$(C=CS)]-[OH1]",
        "linkage": "carbonate_ester",
        "priority": 1,  # Highest — most widely used in BLNP (CD/MK/TD series)
        "ref_blnp": "CD6, MK16, TD8",
        "notes": "Primary/secondary aliphatic alcohol; preferred for carbonate/acetal"
    },
    "phenol_OH": {
        "smarts": "c-[OH1]",
        "linkage": "carbonate_ester",
        "priority": 2,
        "ref_blnp": "None (phenolic carbonates less stable)",
        "notes": "Phenolic OH; less ideal due to reduced carbonate lability"
    },

    # ── Amine groups (→ amide or carbamate linkage) ──
    "primary_amine": {
        "smarts": "[C;!$(C=O);!$(C=N)]-[NH2]",
        "linkage": "amide",
        "priority": 2,
        "ref_blnp": "LD10, DS11 (amide-linked series)",
        "notes": "Primary aliphatic amine; amide linkage is stable/bioavailable"
    },
    "secondary_amine": {
        "smarts": "[C;!$(C=O)]-[NH1]-[C;!$(C=O)]",
        "linkage": "carbamate",
        "priority": 3,
        "ref_blnp": "TM series (some carbamate linkages)",
        "notes": "Secondary amine; carbamate linkage has tunable stability"
    },

    # ── Carboxyl group (→ ester linkage) ──
    "carboxyl": {
        "smarts": "[C](=[O])-[OH1]",
        "linkage": "ester",
        "priority": 2,
        "ref_blnp": "None explicitly",
        "notes": "Carboxylic acid; ester linkage, less common in BLNP but viable"
    },
}

# Known BBB-crossing modules already used in the BLNP paper (for novelty scoring)
_BLNP_KNOWN_MODULES = {
    "MK-0752": "O=C(N[C@@H](C)C(F)(F)F)C1=CC=C(F)C=C1",
    "D-Serine": "N[C@@H](CO)C(=O)O",
    "Memantine": "C[C@@]12C[C@H]3C[C@@H](C1)C[C@@H](C2)C3",
    "Dopamine": "NCCC1=CC=C(O)C(O)=C1",
    "L-DOPA": "N[C@@H](CC1=CC=C(O)C(O)=C1)C(=O)O",
    "Nicotine": "CN1CCC[C@H]1C1=CC=CN=C1",
    "Atenolol": "CC(C)NCC(O)COC1=CC=C(CC(N)=O)C=C1",
}


# ══════════════════════════════════════════════════════════════════════════════
# pKa Estimation for Ionizable Lipid Compatibility
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_amine_pka(mol) -> Optional[float]:
    """Heuristic pKa estimation for the most basic amine in a molecule.

    Based on atom type + neighbour electronegativity — NOT a full QM calculation.
    For production use, replace with ChemAxon or Epik pKa predictor.

    Returns:
        Estimated pKa of strongest basic center, or None if no ionizable amine.
    """
    pka = None
    best_center_pka = None

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue

        is_amine = any(
            bond.GetBondType() == Chem.BondType.SINGLE
            and bond.GetBeginAtomIdx() != atom.GetIdx()
            for bond in atom.GetBonds()
            if bond.GetEndAtom().GetAtomicNum() == 6
        ) or any(
            bond.GetBondType() == Chem.BondType.SINGLE
            for bond in atom.GetBonds()
            if bond.GetOtherAtom(atom).GetAtomicNum() == 6
        )

        if not is_amine:
            continue

        # Base pKa values by nitrogen type
        hybridization = atom.GetHybridization()
        if hybridization == Chem.HybridizationType.SP3:
            degree = atom.GetDegree()
            if degree == 1:  # primary amine R-CH2-NH2
                base_pka = 10.5
            elif degree == 2:  # secondary amine R2NH
                base_pka = 11.0
            elif degree == 3:  # tertiary amine R3N
                base_pka = 10.0
            else:
                base_pka = 9.0
        elif hybridization == Chem.HybridizationType.SP2:
            base_pka = 6.5
        else:
            base_pka = 9.0

        # Apply rough corrections for electron-withdrawing neighbours
        penalty = 0.0
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 8:  # O proximity
                penalty += 1.5
            elif neighbor.GetAtomicNum() == 9:  # F proximity
                penalty += 2.0
            elif neighbor.GetAtomicNum() == 17:  # Cl proximity
                penalty += 1.0

        # Check for carbonyl beta-position (amide-like)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:
                for n2 in neighbor.GetNeighbors():
                    if n2.GetAtomicNum() == 8 and any(
                        b.GetBondType() == Chem.BondType.DOUBLE
                        for b in n2.GetBonds()
                        if b.GetOtherAtom(n2).GetIdx() == neighbor.GetIdx()
                    ):
                        penalty += 5.0  # Amide: drastically reduced basicity

        adj_pka = base_pka - penalty
        if best_center_pka is None or adj_pka > best_center_pka:
            best_center_pka = adj_pka

    return best_center_pka


# ══════════════════════════════════════════════════════════════════════════════
# Main Screener Class
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LipidModuleScreener:
    """Systematic screener for BBB-crossing lipid modules.

    Combines GateScore-ML (4D scoring) with a 5th dimension: conjugatability.

    Parameters
    ----------
    gatenet_model_path : str or None
        Path to trained GateNet model (.joblib). If None, uses heuristic S_gating.
    min_gatescore : float
        Minimum Total_Score (from 4D scorer) for a molecule to be considered.
    max_mw : float
        Maximum molecular weight of the BBB module (before lipid conjugation).
    max_conjugatable_groups : int
        Maximum number of conjugatable groups allowed (avoid cross-linking).

    Scoring weights (5D):
        w_gating=0.35, w_chemistry=0.25, w_novelty=0.15, w_safety=0.10, w_conjugate=0.15
    """

    gatenet_model_path: Optional[str] = None
    min_gatescore: float = 50.0     # Minimum Total_Score from 4D scorer
    max_mw: float = 500.0           # Max MW for BBB module candidate
    min_mw: float = 80.0            # Min MW (exclude tiny fragments)
    max_conjugatable_groups: int = 3  # Allow up to 3 (prefer 1-2 for mono-conjugation)

    # 5D scoring weights
    w_gating: float = 0.35
    w_chemistry: float = 0.25
    w_novelty: float = 0.15
    w_safety: float = 0.10
    w_conjugate: float = 0.15

    def __post_init__(self):
        self._base_scorer = GatingScorer(
            gatenet_model_path=self.gatenet_model_path
        )
        self._known_module_fps = self._init_known_module_fps()

    def _init_known_module_fps(self) -> list:
        """Initialize fingerprints of BLNP-known modules for novelty scoring."""
        fps = []
        for name, smi in _BLNP_KNOWN_MODULES.items():
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        return fps

    def detect_conjugatable_groups(self, smiles: str) -> dict:
        """Detect all conjugatable functional groups in a molecule.

        Returns:
            dict with keys: n_groups, groups_found, dominant_linkage, priority_score,
                            has_ideal_group, details
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"n_groups": 0, "groups_found": [], "dominant_linkage": None,
                    "priority_score": 0.0, "has_ideal_group": False,
                    "details": {}, "error": "Invalid SMILES"}

        groups_found = {}
        for group_name, info in CONJUGATABLE_SMARTS.items():
            try:
                pat = Chem.MolFromSmarts(info["smarts"])
                if pat is None:
                    continue
                matches = mol.GetSubstructMatches(pat)
                if matches:
                    groups_found[group_name] = {
                        "count": len(matches),
                        "linkage": info["linkage"],
                        "priority": info["priority"],
                        "atom_indices": [m[0] for m in matches],
                    }
            except Exception:
                continue

        n_total = sum(g["count"] for g in groups_found.values())

        # Determine dominant linkage (highest priority group, not most numerous)
        dominant = None
        best_priority = 99
        for name, ginfo in groups_found.items():
            if ginfo["priority"] < best_priority:
                best_priority = ginfo["priority"]
                dominant = ginfo["linkage"]

        # Priority score: 0-100
        # Best = has exactly 1 group of priority 1 (aliphatic_OH)
        has_ideal = False
        priority_score = 0.0

        if n_total == 0:
            priority_score = 0.0
        elif "aliphatic_OH" in groups_found and groups_found["aliphatic_OH"]["count"] == 1 and n_total == 1:
            # Ideal: single aliphatic OH (like borneol, menthol)
            priority_score = 100.0
            has_ideal = True
        elif "aliphatic_OH" in groups_found and groups_found["aliphatic_OH"]["count"] >= 1:
            priority_score = 85.0
            has_ideal = True
        elif "primary_amine" in groups_found and n_total == 1:
            priority_score = 75.0
            has_ideal = True
        elif "carboxyl" in groups_found and n_total == 1:
            priority_score = 65.0
        elif n_total >= 1 and n_total <= self.max_conjugatable_groups:
            priority_score = 50.0
        elif n_total > self.max_conjugatable_groups:
            # Too many reactive groups
            priority_score = max(5.0, 40.0 - 10.0 * (n_total - self.max_conjugatable_groups))

        return {
            "n_groups": n_total,
            "groups_found": list(groups_found.keys()),
            "dominant_linkage": dominant,
            "priority_score": priority_score,
            "has_ideal_group": has_ideal,
            "details": groups_found,
        }

    def _s_conjugate(self, smiles: str, mol) -> dict:
        """Compute S_conjugate: conjugatability score (0-100).

        Incorporates:
          - Group availability (priority_score from detect_conjugatable_groups)
          - Synthesizability adjustment (fewer groups → easier)
          - Position accessibility (terminal/exposed groups preferred)
        """
        conj = self.detect_conjugatable_groups(smiles)
        base = conj["priority_score"]

        # Bonus for exposed/terminal position (not in ring)
        if conj["n_groups"] >= 1:
            try:
                n_ring_groups = 0
                for gname, ginfo in conj["details"].items():
                    for aidx in ginfo.get("atom_indices", []):
                        atom = mol.GetAtomWithIdx(aidx)
                        if atom.IsInRing():
                            n_ring_groups += 1
                # Prefer non-ring (exocyclic) groups for easier conjugation
                n_exocyclic = conj["n_groups"] - n_ring_groups
                if n_exocyclic > 0:
                    base = min(100.0, base + 10.0)
                if n_ring_groups > 0 and n_exocyclic == 0:
                    base = max(5.0, base - 15.0)
            except Exception:
                pass

        # Penalty for too many reactive groups
        if conj["n_groups"] > self.max_conjugatable_groups:
            base = max(5.0, base - 15.0 * (conj["n_groups"] - self.max_conjugatable_groups))

        return {
            "S_conjugate": round(max(0.0, min(100.0, base)), 1),
            "n_conjugatable_groups": conj["n_groups"],
            "conjugatable_types": ",".join(conj["groups_found"]) if conj["groups_found"] else "none",
            "dominant_linkage": conj["dominant_linkage"] or "none",
            "has_ideal_group": conj["has_ideal_group"],
        }

    def _estimate_blnp_properties(self, smiles: str, mol) -> dict:
        """Estimate LNP-relevant molecular properties post-conjugation.

        These are rough estimates of what the FINAL BL (BBB-crossing Lipid)
        would look like after conjugating this module to an amino lipid tail.

        Reference amino lipid tails from BLNP paper (MK series):
          MK tail ≈ ~350-400 Da, with tertiary amine head group (pKa ~6.7-7.0)
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)

        # Estimated properties after conjugation to a typical MK-style lipid tail
        # Lipid tail contribution: ~380 Da, logP ~+5, TPSA ~+25
        est_bl_mw = mw + 380
        est_bl_logp = logp + 5
        est_bl_tpsa = tpsa + 25

        # Estimate final BL pKa (dominated by the amino lipid's tertiary amine)
        est_pka = _estimate_amine_pka(mol)

        # Check if pKa in optimal range for LNP (BLNP paper: best BLNPs pKa 6.7-7.0)
        pka_assessment = "unknown"
        if est_pka is not None:
            if 6.5 <= est_pka <= 7.5:
                pka_assessment = "optimal"
            elif 5.5 <= est_pka < 6.5:
                pka_assessment = "acceptable"
            elif 7.5 < est_pka <= 8.5:
                pka_assessment = "borderline"
            else:
                pka_assessment = "suboptimal"

        return {
            "est_module_pka": round(est_pka, 1) if est_pka is not None else None,
            "pka_assessment": pka_assessment,
            "est_BL_MW": round(est_bl_mw, 1),
            "est_BL_logP": round(est_bl_logp, 1),
            "est_BL_TPSA": round(est_bl_tpsa, 1),
            "module_MW": round(mw, 1),
            "module_logP": round(logp, 1),
            "module_TPSA": round(tpsa, 1),
        }

    def _novelty_vs_blnp_modules(self, smiles: str, mol) -> float:
        """Compute Tanimoto distance from BLNP paper's known BBB modules (0-100).

        Higher = more novel vs known modules. Lower = similar to what's been done.
        """
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            if not self._known_module_fps:
                return 100.0  # No reference → fully novel
            sims = BulkTanimotoSimilarity(fp, self._known_module_fps)
            max_sim = max(sims) if sims else 0.0
            # Convert similarity to novelty score
            if max_sim < 0.3:
                return 100.0  # Highly novel
            elif max_sim < 0.5:
                return 75.0
            elif max_sim < 0.7:
                return 40.0
            else:
                return 10.0  # Very similar to existing module
        except Exception:
            return 50.0

    def score_one(self, smiles: str) -> dict:
        """Full 5D scoring of a single molecule for BLNP suitability.

        Returns dict with all scores + conjugatability details + BLNP property estimates.
        """
        # Base 4D GateScore
        base = self._base_scorer.score_one(smiles)
        if base.get("Total_Score", -999) < -998:
            return {"SMILES": smiles, "Total_Score_5D": -999.0, "error": base.get("error", "Invalid")}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"SMILES": smiles, "Total_Score_5D": -999.0, "error": "Invalid SMILES"}

        # 5th dimension: conjugatability
        conj = self._s_conjugate(smiles, mol)

        # BLNP property estimates
        blnp_props = self._estimate_blnp_properties(smiles, mol)

        # Novelty vs BLNP-known modules
        novelty_blnp = self._novelty_vs_blnp_modules(smiles, mol)

        # 5D Total Score
        total_5d = (
            self.w_gating * base["S_gating"]
            + self.w_chemistry * base["S_chemistry"]
            + self.w_novelty * base["S_novelty"]
            + self.w_safety * base["S_safety"]
            + self.w_conjugate * conj["S_conjugate"]
        )

        return {
            **base,  # Include all 4D fields
            # 5D additions
            "Total_Score_5D": round(total_5d, 1),
            "S_conjugate": conj["S_conjugate"],
            "n_conjugatable_groups": conj["n_conjugatable_groups"],
            "conjugatable_types": conj["conjugatable_types"],
            "dominant_linkage": conj["dominant_linkage"],
            "has_ideal_group": conj["has_ideal_group"],
            # BLNP property estimates
            **blnp_props,
            # Novelty vs BLNP
            "novelty_vs_BLNP_known": round(novelty_blnp, 1),
            # Pass/fail
            "pass_gatescore": base["Total_Score"] >= self.min_gatescore,
            "pass_mw": self.min_mw <= base["MW"] <= self.max_mw,
            "pass_conjugatable": conj["n_conjugatable_groups"] >= 1,
            "pass_overall": (
                base["Total_Score"] >= self.min_gatescore
                and self.min_mw <= base["MW"] <= self.max_mw
                and conj["n_conjugatable_groups"] >= 1
            ),
        }

    def screen(self, smiles_list: list[str], verbose: bool = True) -> pd.DataFrame:
        """Screen a list of SMILES for BLNP module suitability.

        Returns DataFrame sorted by Total_Score_5D descending.
        """
        results = [self.score_one(smi) for smi in smiles_list]
        df = pd.DataFrame(results)
        if "Total_Score_5D" in df.columns:
            df = df[df["Total_Score_5D"] > -999.0]
            df = df.sort_values("Total_Score_5D", ascending=False).reset_index(drop=True)
        if verbose:
            n = len(df)
            n_pass = df["pass_overall"].sum() if "pass_overall" in df.columns else 0
            n_conj = (df["n_conjugatable_groups"] > 0).sum() if "n_conjugatable_groups" in df.columns else 0
            print(f"[LipidModuleScreener] {len(smiles_list)} input → {n} valid")
            print(f"  Conjugatable: {n_conj} | Pass all filters: {n_pass}")
            print(f"  Top Total_Score_5D: {df['Total_Score_5D'].iloc[0]:.1f}" if len(df) > 0 else "  No valid results")
        return df

    def screen_from_db(self, db_path: str, verbose: bool = True) -> pd.DataFrame:
        """Screen molecules from the BBB-GatingDB CSV.

        Parameters
        ----------
        db_path : str
            Path to bbb_gating_dataset_v1.csv
        """
        df_db = pd.read_csv(db_path)
        smiles_col = None
        for col in ["canonical_smiles", "SMILES", "smiles", "CanonicalSMILES"]:
            if col in df_db.columns:
                smiles_col = col
                break
        if smiles_col is None:
            raise ValueError(f"Cannot find SMILES column in {db_path}. Columns: {list(df_db.columns)}")

        smiles_list = df_db[smiles_col].dropna().tolist()
        if verbose:
            print(f"[LipidModuleScreener] Loaded {len(smiles_list)} molecules from {db_path}")
        df = self.screen(smiles_list, verbose=verbose)

        # Merge back database annotations
        merge_cols = [c for c in df_db.columns if c != smiles_col]
        if merge_cols and smiles_col in df_db.columns:
            df = df.merge(df_db[[smiles_col] + merge_cols], left_on="SMILES",
                          right_on=smiles_col, how="left", suffixes=("", "_db"))
        return df

    def get_top_candidates(self, df: pd.DataFrame, top_k: int = 50,
                           only_pass: bool = True) -> pd.DataFrame:
        """Extract top candidates from screening results."""
        if only_pass and "pass_overall" in df.columns:
            df = df[df["pass_overall"]].copy()
        return df.head(top_k).reset_index(drop=True)

    def export_for_publication(self, df: pd.DataFrame, output_path: str,
                                top_k: int = 50) -> pd.DataFrame:
        """Export ranked candidate list in publication-ready format.

        Produces a CSV with columns organized for a journal supplementary table.
        """
        top = self.get_top_candidates(df, top_k=top_k, only_pass=True)

        # Publication-friendly column order and names
        pub_columns = {
            "SMILES": "SMILES",
            "Total_Score_5D": "GateScore_Lipid_Total",
            "S_gating": "GateScore_Gating",
            "S_chemistry": "GateScore_Chemistry",
            "S_novelty": "GateScore_Novelty",
            "S_safety": "GateScore_Safety",
            "S_conjugate": "GateScore_Conjugate",
            "MW": "Module_MW",
            "cLogP": "Module_logP",
            "TPSA": "Module_TPSA",
            "HBD": "HBD",
            "HBA": "HBA",
            "NumHeteroatoms": "NumHeteroatoms",
            "dominant_linkage": "Recommended_Linkage",
            "conjugatable_types": "Conjugatable_Groups",
            "n_conjugatable_groups": "N_Conjugatable_Groups",
            "pka_assessment": "pKa_Assessment",
            "est_BL_MW": "Estimated_BL_MW",
            "est_BL_logP": "Estimated_BL_logP",
            "est_BL_TPSA": "Estimated_BL_TPSA",
            "novelty_vs_BLNP_known": "Novelty_vs_BLNP_Known",
            "QED": "QED",
            "SA_score": "SA_Score",
            "PAINS_alerts": "PAINS_Alerts",
            "Tox_alerts": "Tox_Alerts",
        }

        export_cols = {k: v for k, v in pub_columns.items() if k in top.columns}
        export = top[list(export_cols.keys())].rename(columns=export_cols)
        export.insert(0, "Rank", range(1, len(export) + 1))

        export.to_csv(output_path, index=False)
        print(f"[LipidModuleScreener] Exported Top {len(export)} candidates → {output_path}")
        return export

    def summary_stats(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics for publication methods section."""
        n_total = len(df)
        n_pass = df["pass_overall"].sum() if "pass_overall" in df.columns else 0
        n_conj = (df["n_conjugatable_groups"] > 0).sum() if "n_conjugatable_groups" in df.columns else 0
        n_ideal = df["has_ideal_group"].sum() if "has_ideal_group" in df.columns else 0

        linkage_dist = {}
        if "dominant_linkage" in df.columns:
            linkage_dist = df[df["dominant_linkage"] != "none"]["dominant_linkage"].value_counts().to_dict()

        pka_dist = {}
        if "pka_assessment" in df.columns:
            pka_dist = df["pka_assessment"].value_counts().to_dict()

        return {
            "n_input": n_total,
            "n_pass_all_filters": int(n_pass),
            "n_conjugatable": int(n_conj),
            "n_ideal_single_OH": int(n_ideal),
            "pass_rate": round(n_pass / n_total * 100, 1) if n_total > 0 else 0.0,
            "linkage_distribution": linkage_dist,
            "pka_distribution": pka_dist,
            "mean_gatescore_5d": round(df["Total_Score_5D"].mean(), 1) if "Total_Score_5D" in df.columns else None,
            "max_gatescore_5d": round(df["Total_Score_5D"].max(), 1) if "Total_Score_5D" in df.columns else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Chemical Space Comparison: BLNP Modules vs Our Candidates
# ══════════════════════════════════════════════════════════════════════════════

def compare_with_blnp_chemical_space(
    our_candidates: pd.DataFrame,
    output_plot: Optional[str] = None,
) -> dict:
    """Compare our top candidates with BLNP paper's known modules in chemical space.

    Computes MW/logP/TPSA distributions and Tanimoto overlap statistics.
    Useful for Figure 3/4 of a J Chem Inf Model paper.

    Returns:
        dict with overlap statistics + PCA coordinates (for plotting).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Collect BLNP known module properties
    blnp_data = []
    for name, smi in _BLNP_KNOWN_MODULES.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        blnp_data.append({
            "source": "BLNP_known",
            "name": name,
            "SMILES": smi,
            "MW": Descriptors.MolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
        })

    df_blnp = pd.DataFrame(blnp_data)

    # Collect our candidate properties (only valid ones)
    our_cols = ["SMILES", "MW", "cLogP", "TPSA", "HBD", "HBA"]
    our_avail = [c for c in our_cols if c in our_candidates.columns]
    df_ours = our_candidates[our_avail].copy()
    if "cLogP" in df_ours.columns:
        df_ours = df_ours.rename(columns={"cLogP": "logP"})
    df_ours["source"] = "GateScore_candidate"

    combined = pd.concat([df_blnp, df_ours], ignore_index=True)

    # Compute overlap statistics
    overlap_stats = {
        "n_BLNP_known": len(df_blnp),
        "n_our_candidates": len(df_ours),
        "MW_range_BLNP": (df_blnp["MW"].min(), df_blnp["MW"].max()),
        "MW_range_ours": (df_ours["MW"].min(), df_ours["MW"].max()) if len(df_ours) > 0 else (0, 0),
        "logP_range_BLNP": (df_blnp["logP"].min(), df_blnp["logP"].max()),
        "logP_range_ours": (df_ours["logP"].min(), df_ours["logP"].max()) if len(df_ours) > 0 else (0, 0),
    }

    # Compute mean Tanimoto between our candidates and BLNP known
    if len(df_ours) > 0:
        our_fps = []
        for smi in df_ours["SMILES"]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                our_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        blnp_fps = []
        for smi in df_blnp["SMILES"]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                blnp_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

        if our_fps and blnp_fps:
            pairwise_sims = []
            for fp_o in our_fps:
                for fp_b in blnp_fps:
                    pairwise_sims.append(TanimotoSimilarity(fp_o, fp_b))
            overlap_stats["mean_tanimoto_vs_BLNP"] = round(np.mean(pairwise_sims), 3)
            overlap_stats["max_tanimoto_vs_BLNP"] = round(np.max(pairwise_sims), 3)
            overlap_stats["min_tanimoto_vs_BLNP"] = round(np.min(pairwise_sims), 3)

    return overlap_stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── Test with known molecules ──
    print("=" * 70)
    print("BBB Lipid Module Screener — Quick Test")
    print("=" * 70)

    test_smiles = [
        # Known gating molecules (should score high)
        "C[C@@]1(CC[C@@H]2C1(C)C)CC[C@H]2O",   # (+)-Borneol → ideal aliphatic OH
        "C[C@@H]1CC[C@@H](C(C)C)[C@@H](O)C1",   # Menthol → aliphatic OH
        "COc1cc(CC=C)ccc1O",                      # Eugenol → phenol OH
        "CC1CCCCCCCCCCCC(=O)C1",                  # Muscone → ketone only (no ideal conj)
        "CC12CCC(CC1)C(C)(C)O2",                  # 1,8-Cineole → ether only
        # BLNP paper known modules
        "N[C@@H](CO)C(=O)O",                      # D-Serine → amine + carboxyl + OH
        "NCCC1=CC=C(O)C(O)=C1",                   # Dopamine → primary amine + catechol OH
        # Negative controls (should score low)
        "CCCCCCCCCCCC",                            # Dodecane → no conj groups
        "c1ccccc1",                                # Benzene → no conj groups
        "CN1C(=O)N(C)C(=O)C2=C1C=CC=C2",          # Caffeine → no reactive groups
    ]

    screener = LipidModuleScreener()
    df = screener.screen(test_smiles)

    print("\n── Top Candidates ──")
    cols_show = ["SMILES", "Total_Score_5D", "S_gating", "S_conjugate",
                 "dominant_linkage", "pass_overall", "pka_assessment"]
    avail = [c for c in cols_show if c in df.columns]
    print(df[avail].to_string())

    # ── Summary ──
    stats = screener.summary_stats(df)
    print(f"\n── Summary Stats ──")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── Export ──
    out_path = "outputs/lipid_module_candidates_test.csv"
    screener.export_for_publication(df, out_path, top_k=50)

    # ── Compare with BLNP chemical space ──
    print("\n── BLNP Chemical Space Comparison ──")
    overlap = compare_with_blnp_chemical_space(df)
    for k, v in overlap.items():
        print(f"  {k}: {v}")
