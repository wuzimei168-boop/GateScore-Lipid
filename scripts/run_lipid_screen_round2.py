#!/usr/bin/env python3
"""
Round 2 大规模筛选: TCM 聚焦库 + 原有数据集
==============================================
合并:
  1. TCM 聚焦库 (2,803 natural product SMILES)
  2. B3DB + Augmented GatingDB (6,803 molecules from Round 1)
  → 去重后 ~9,600 molecules

输出: outputs/lipid_module_round2_full_screen.csv, top50_round2.csv, summary_round2.json
"""

import sys, os, json, time, warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["RDKIT_LOG_LEVEL"] = "ERROR"

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_DIR = Path("/Users/zimei/AI_Workspace/01_projects/P1_AroBrain_Platform")
PIPELINE_DIR = PROJECT_DIR / "de_novo_pipeline"
OUTPUT_DIR = PIPELINE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PIPELINE_DIR))

from features import Featurizer
from bbb_lipid_filter import LipidModuleScreener, compare_with_blnp_chemical_space

print("=" * 70)
print("Round 2: TCM-Focused Large-Scale Screening")
print("=" * 70)

# ── 1. Load datasets ──
all_smiles = OrderedDict()

# (a) Round 1 results
df_r1 = pd.read_csv(OUTPUT_DIR / "lipid_module_full_screen.csv")
for smi in df_r1["SMILES"].dropna():
    smi = str(smi).strip()
    if smi and smi != "nan":
        all_smiles[smi] = "round1"

# (b) TCM focused library
tcm_file = OUTPUT_DIR / "tcm_databases" / "tcm_focused_library.smi"
with open(tcm_file) as f:
    for line in f:
        smi = line.strip()
        if smi and smi != "nan":
            if smi not in all_smiles:
                all_smiles[smi] = "tcm_focused"
            # else: already in round1

unique_smiles = list(all_smiles.keys())
print(f"[1] Loaded: {len(df_r1)} (round1) + 2803 (TCM) → {len(unique_smiles)} unique")

# ── 2. MW Filter ──
valid_smiles = []
valid_sources = []
for smi in unique_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol and 80 <= Chem.Descriptors.MolWt(mol) <= 500:
        valid_smiles.append(smi)
        valid_sources.append(all_smiles[smi])

print(f"[2] After MW filter (80-500): {len(valid_smiles)}")

# ── 3. Batch Featurize ──
print(f"\n[3] Batch featurizing {len(valid_smiles)} molecules...")
t0 = time.time()
featurizer = Featurizer(use_3d=False, use_quantum_xtb=False)
X_batch = featurizer.fit_transform(valid_smiles)
print(f"    Featurized: {X_batch.shape} in {time.time()-t0:.1f}s")

# ── 4. Batch Predict S_gating ──
import joblib
model_path = OUTPUT_DIR / "gatenet_v1" / "gatenet_Ensemble.joblib"
model = joblib.load(str(model_path))
t1 = time.time()
gating_probs = model.predict_proba(X_batch)[:, 1]
S_gating_batch = np.clip(gating_probs * 100, 0, 100)
print(f"[4] Predicted in {time.time()-t1:.1f}s")
print(f"    S_gating: [{S_gating_batch.min():.1f}, {S_gating_batch.max():.1f}], mean={S_gating_batch.mean():.1f}")
print(f"    S_gating > 50: {(S_gating_batch > 50).sum()}")

# ── 5. Per-molecule scoring ──
print(f"\n[5] Computing remaining dimensions...")
screener = LipidModuleScreener(gatenet_model_path=None, min_gatescore=0.0, max_mw=500, min_mw=80)

results = []
for i in range(len(valid_smiles)):
    if (i+1) % 3000 == 0:
        print(f"    {i+1}/{len(valid_smiles)}")

    smi = valid_smiles[i]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue

    S_gat = S_gating_batch[i]
    base = screener._base_scorer.score_one(smi)
    if base.get("Total_Score", -999) < -998:
        continue

    conj = screener._s_conjugate(smi, mol)
    blnp_props = screener._estimate_blnp_properties(smi, mol)
    novelty_blnp = screener._novelty_vs_blnp_modules(smi, mol)

    total_5d = (0.35 * S_gat + 0.25 * base["S_chemistry"] + 0.15 * base["S_novelty"]
                 + 0.10 * base["S_safety"] + 0.15 * conj["S_conjugate"])

    results.append({
        "SMILES": smi,
        "source": valid_sources[i],
        "Total_Score_5D": round(total_5d, 1),
        "S_gating": round(S_gat, 1),
        "S_chemistry": round(base["S_chemistry"], 1),
        "S_novelty": round(base["S_novelty"], 1),
        "S_safety": round(base["S_safety"], 1),
        "S_conjugate": conj["S_conjugate"],
        "MW": base["MW"], "cLogP": base["cLogP"], "TPSA": base["TPSA"],
        "HBD": base["HBD"], "HBA": base["HBA"], "NumHeteroatoms": base["NumHeteroatoms"],
        "SA_score": base["SA_score"], "QED": base["QED"],
        "PAINS_alerts": base["PAINS_alerts"], "Tox_alerts": base["Tox_alerts"],
        "n_conjugatable_groups": conj["n_conjugatable_groups"],
        "conjugatable_types": conj["conjugatable_types"],
        "dominant_linkage": conj["dominant_linkage"],
        "has_ideal_group": conj["has_ideal_group"],
        "est_module_pka": blnp_props["est_module_pka"],
        "pka_assessment": blnp_props["pka_assessment"],
        "est_BL_MW": blnp_props["est_BL_MW"],
        "est_BL_logP": blnp_props["est_BL_logP"],
        "novelty_vs_BLNP_known": round(novelty_blnp, 1),
        "pass_overall": (total_5d >= 50 and 80 <= base["MW"] <= 500 and conj["n_conjugatable_groups"] >= 1),
    })

df_all = pd.DataFrame(results).sort_values("Total_Score_5D", ascending=False).reset_index(drop=True)
print(f"\n    Done. Total valid: {len(df_all)}")

# ── 6. Summary ──
print(f"\n{'='*60}")
print(f"Round 2 Summary")
print(f"{'='*60}")
print(f"  Total screened:       {len(df_all)}")
print(f"  Pass all filters:     {df_all['pass_overall'].sum()}")
print(f"  Conjugatable:         {(df_all['n_conjugatable_groups'] > 0).sum()}")
print(f"  S_gating > 50:        {(df_all['S_gating'] > 50).sum()}")
print(f"  Mean 5D:              {df_all['Total_Score_5D'].mean():.1f}")
print(f"  Max 5D:               {df_all['Total_Score_5D'].max():.1f}")

# Source breakdown
src = df_all['source'].value_counts()
print(f"  Source: tcm_focused={src.get('tcm_focused', 0)}, round1={src.get('round1', 0)}")

# Top 10 from TCM
tcm_top = df_all[df_all['source'] == 'tcm_focused'].head(10)
print(f"\n  Top TCM candidates:")
for i, (_, row) in enumerate(tcm_top.iterrows()):
    print(f"    #{i+1} 5D={row['Total_Score_5D']:.1f}  Gate={row['S_gating']:.1f}  "
          f"Conj={row['S_conjugate']:.1f}  MW={row['MW']:.0f}  logP={row['cLogP']:.1f}")

# ── 7. Export ──
df_all.to_csv(OUTPUT_DIR / "lipid_module_round2_full_screen.csv", index=False)
screener.export_for_publication(df_all, str(OUTPUT_DIR / "lipid_module_top50_round2.csv"), top_k=50)

stats = {
    "round": 2,
    "date": "2026-06-15",
    "total_screened": len(df_all),
    "pass_all": int(df_all['pass_overall'].sum()),
    "mean_5d": round(df_all['Total_Score_5D'].mean(), 1),
    "max_5d": round(df_all['Total_Score_5D'].max(), 1),
    "s_gating_gt50": int((df_all['S_gating'] > 50).sum()),
    "source_distribution": {k: int(v) for k, v in df_all['source'].value_counts().items()},
    "tcm_focused_size": len([s for s in valid_sources if s == 'tcm_focused']),
    "round1_size": len([s for s in valid_sources if s == 'round1']),
}
with open(OUTPUT_DIR / "lipid_module_round2_summary.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n{'='*60}")
print(f"Exported: lipid_module_round2_full_screen.csv, top50_round2.csv, summary_round2.json")
print(f"{'='*60}")
