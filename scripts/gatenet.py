#!/usr/bin/env python3
"""
Task 3: GateScore-ML v0.1 — Train traditional ML models to predict BBB gating behavior.

LIMITATIONS (v0.1, be honest in reports):
  - NOT a GNN / graph neural network. Uses sklearn RF/MLP/HistGB + Morgan FP.
  - G3 (bronze) samples are WEAK labels, only used in semi-supervised pretraining.
  - Negative labels from B3DB "BBB+/BBB-" are passive permeability, NOT verified non-gating.
  - Scaffold split is the PRIMARY evaluation; random split is supplementary.
  - Toxic negative set is undersized (n=5); model may not distinguish gating from nonspecific membrane disruption.

Trains 4 model types + ensemble on BBB-GatingDB gating labels:
  - LogisticRegression
  - RandomForest (500 trees)
  - HistGradientBoosting
  - MLP (128-64-32)

Outputs:
  - Trained model artifacts (.joblib)
  - CV metrics (AUC, MCC, BACC, F1)
  - Feature importance ranking
  - Holdout predictions for analysis

Usage:
  python gatenet.py --data outputs/bbb_gating_dataset_v1.csv --output_dir outputs/gatenet_v1
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from .features import Featurizer
except ImportError:
    from features import Featurizer

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 42


def _build_pipeline(model: BaseEstimator, use_scaler: bool = False) -> Pipeline:
    steps = []
    steps.append(("imputer", SimpleImputer(strategy="median")))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(model)))
    return Pipeline(steps)


def _get_models() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": _build_pipeline(
            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
            use_scaler=True,
        ),
        "RandomForest": _build_pipeline(
            RandomForestClassifier(
                n_estimators=500, max_depth=12, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1,
            ),
        ),
        "HistGradientBoosting": _build_pipeline(
            HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                early_stopping=True, random_state=RANDOM_SEED,
            ),
        ),
        "MLP": _build_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                alpha=0.001, learning_rate="adaptive", max_iter=500,
                early_stopping=True, random_state=RANDOM_SEED,
            ),
            use_scaler=True,
        ),
    }


def _murcko_scaffold(smiles: str) -> str:
    """Extract Bemis-Murcko scaffold from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "INVALID"
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return "NO_SCAFFOLD"
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except Exception:
        return "ERROR"


def train_gatenet(
    data_path: str,
    output_dir: str,
    test_size: float = 0.20,
    use_3d: bool = False,
    use_g3_only_for_pretraining: bool = True,
) -> dict:
    """Train GateScore-ML v0.1 baseline models.

    Parameters
    ----------
    data_path : str
        Path to BBB-GatingDB CSV (from gating_db.py).
    output_dir : str
        Directory to save models and metrics.
    test_size : float
        Fraction for holdout test set.
    use_3d : bool
        Whether to compute 3D conformer features.
    use_g3_only_for_pretraining : bool
        If True, G3 (bronze) samples are excluded from training the final classifier.
        They carry weak labels (traditional_kaiqiao) and should only be used for
        semi-supervised pretraining, not as strong positive training data.

    Returns
    -------
    dict with keys: cv_metrics, scaffold_metrics, feature_importance, holdout_metrics, model_paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    df["SMILES"] = df["SMILES"].fillna("")
    df["murcko_scaffold"] = df["SMILES"].apply(lambda s: _murcko_scaffold(str(s)))

    # Remove invalid SMILES (no scaffold = no valid mol)
    invalid_mask = df["murcko_scaffold"].isin(["INVALID", "NO_SCAFFOLD", "ERROR"])
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        print(f"WARNING: Removing {n_invalid} molecules with invalid SMILES")
        print(f"  Invalid: {df[invalid_mask]['name'].tolist()}")
        df = df[~invalid_mask].copy()

    # Separate G3 (bronze) from G1+G2 (gold+silver) if requested
    if use_g3_only_for_pretraining and "evidence_level" in df.columns:
        g3_mask = df["evidence_level"] == "G3"
        n_g3 = g3_mask.sum()
        print(f"Excluding {n_g3} G3 (bronze) samples from training (weak labels).")
        print(f"  G3 samples are reserved for semi-supervised pretraining only.")
        df_trainable = df[~g3_mask].copy()
        df_g3 = df[g3_mask].copy()
    else:
        df_trainable = df.copy()
        df_g3 = pd.DataFrame()
        n_g3 = 0

    # Count labels in training set
    n_gating = (df_trainable["label"] == "gating_positive").sum()
    n_neg = (df_trainable["label"] != "gating_positive").sum()
    print(f"Training set: {len(df_trainable)} total")
    print(f"  gating_positive (G1+G2):   {n_gating}")
    print(f"  passive_permeation (B3DB BBB+): {(df_trainable['evidence_level']=='PP').sum()}")
    print(f"  non_penetrating (B3DB BBB-):   {(df_trainable['evidence_level']=='NI').sum()}")
    print(f"  toxic negative:           {(df_trainable['evidence_level']=='TX').sum()}")
    print(f"  G3 excluded from training:     {n_g3}")
    print(f"  NOTE: B3DB BBB+ molecules are labeled as 'passive_permeation'.")
    print(f"        This means 'known to cross BBB by passive diffusion'.")
    print(f"        It does NOT mean 'verified non-gating'. Interpret with care.")

    # Create binary target: 1 = gating_positive, 0 = not gating
    y = (df_trainable["label"] == "gating_positive").astype(int).values
    smiles_list = df_trainable["SMILES"].fillna("").values.tolist()

    # ── Featurize ────────────────────────────────────────────────────
    print("Featurizing molecules...")
    featurizer = Featurizer(use_3d=use_3d)
    X = featurizer.fit_transform(smiles_list)
    print(f"Feature matrix: {X.shape}")

    # ── PRIMARY: Murcko Scaffold Split ────────────────────────────────
    # Scaffolds not in training → test set = strict evaluation of scaffold generalization
    scaffolds = df_trainable["murcko_scaffold"].values
    unique_scaffolds = list(set(scaffolds))
    n_test_scaffolds = max(1, int(len(unique_scaffolds) * test_size))
    rng = np.random.RandomState(RANDOM_SEED)
    test_scaffolds_set = set(rng.choice(unique_scaffolds, n_test_scaffolds, replace=False))

    test_mask = np.array([s in test_scaffolds_set for s in scaffolds])
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    idx_train = np.where(train_mask)[0]
    idx_test = np.where(test_mask)[0]

    # Verify no scaffold leakage
    train_scaffolds_set = set(scaffolds[train_mask])
    scaffold_overlap = train_scaffolds_set & test_scaffolds_set
    scaffold_novel = test_scaffolds_set - train_scaffolds_set
    print(f"\nMurcko Scaffold Split (PRIMARY evaluation):")
    print(f"  Train: {len(X_train)} molecules, {len(train_scaffolds_set)} scaffolds")
    print(f"  Test:  {len(X_test)} molecules, {len(test_scaffolds_set)} scaffolds")
    print(f"  Scaffold overlap (leakage check): {len(scaffold_overlap)} (should be 0)")
    if len(scaffold_overlap) > 0:
        print(f"  WARNING: Scaffold leakage detected! {scaffold_overlap}")
    print(f"  Novel scaffolds in test: {len(scaffold_novel)}")

    # ── SUPPLEMENTARY: Random stratified split ───────────────────────
    Xr_train, Xr_test, yr_train, yr_test, _, _ = train_test_split(
        X, y, np.arange(len(y)),
        test_size=test_size, stratify=y, random_state=RANDOM_SEED,
    )

    # ── Cross-validation (random stratified, on training set) ─────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    models = _get_models()

    cv_results = {}
    for name, pipe in models.items():
        print(f"CV (random stratified): {name}...")
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "balanced_acc": "balanced_accuracy",
                "f1": "f1",
                "mcc": lambda est, X, y: matthews_corrcoef(y, est.predict(X)),
            },
            n_jobs=-1,
            return_train_score=False,
        )
        cv_results[name] = {
            "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "balanced_acc_mean": float(np.mean(scores["test_balanced_acc"])),
            "balanced_acc_std": float(np.std(scores["test_balanced_acc"])),
            "f1_mean": float(np.mean(scores["test_f1"])),
            "f1_std": float(np.std(scores["test_f1"])),
            "mcc_mean": float(np.mean(scores["test_mcc"])),
            "mcc_std": float(np.std(scores["test_mcc"])),
        }
        print(f"  AUC={cv_results[name]['roc_auc_mean']:.3f}+/-{cv_results[name]['roc_auc_std']:.3f}"
              f"  MCC={cv_results[name]['mcc_mean']:.3f}+/-{cv_results[name]['mcc_std']:.3f}")

    # ── PRIMARY: Scaffold holdout evaluation ──────────────────────────
    scaffold_metrics = {}
    scaffold_preds = pd.DataFrame({"true_label": y_test, "index": idx_test})

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        scaffold_metrics[name] = {
            "auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "mcc": round(float(matthews_corrcoef(y_test, y_pred)), 4),
        }
        print(f"Scaffold Holdout {name}: AUC={scaffold_metrics[name]['auc']:.4f}, "
              f"MCC={scaffold_metrics[name]['mcc']:.4f}")
        scaffold_preds[f"{name}_prob"] = y_prob

    # Ensembles on scaffold split
    estimators = [(name, clone(models[name])) for name in models]
    for _, pipe in estimators:
        pipe.fit(X_train, y_train)
    ensemble = VotingClassifier(estimators, voting="soft")
    ensemble.fit(X_train, y_train)
    y_ens_prob = ensemble.predict_proba(X_test)[:, 1]
    y_ens_pred = ensemble.predict(X_test)
    scaffold_metrics["Ensemble"] = {
        "auc": round(float(roc_auc_score(y_test, y_ens_prob)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_ens_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_ens_pred)), 4),
        "f1": round(float(f1_score(y_test, y_ens_pred)), 4),
        "mcc": round(float(matthews_corrcoef(y_test, y_ens_pred)), 4),
    }
    scaffold_preds["Ensemble_prob"] = y_ens_prob
    scaffold_preds["Ensemble_pred"] = y_ens_pred
    print(f"Scaffold Holdout Ensemble: AUC={scaffold_metrics['Ensemble']['auc']:.4f}, "
          f"MCC={scaffold_metrics['Ensemble']['mcc']:.4f}")

    # ── SUPPLEMENTARY: Random stratified holdout ──────────────────────
    random_metrics = {}
    for name, pipe in models.items():
        pipe.fit(Xr_train, yr_train)
        y_pred = pipe.predict(Xr_test)
        y_prob = pipe.predict_proba(Xr_test)[:, 1]
        random_metrics[name] = {
            "auc": round(float(roc_auc_score(yr_test, y_prob)), 4),
            "mcc": round(float(matthews_corrcoef(yr_test, y_pred)), 4),
        }
    print(f"\nSupplementary Random Holdout (for reference, NOT primary):")
    for name, m in random_metrics.items():
        print(f"  {name}: AUC={m['auc']:.4f}, MCC={m['mcc']:.4f}")

    # ── Save calibrated models (trained on all training data) ────────
    model_paths = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)  # refit on scaffold-train data
        calibrated = CalibratedClassifierCV(clone(pipe), method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        model_path = output_dir / f"gatescore_ml_{name}.joblib"
        joblib.dump(calibrated, model_path)
        model_paths[name] = str(model_path)

    ensemble_path = output_dir / "gatescore_ml_Ensemble.joblib"
    joblib.dump(ensemble, ensemble_path)
    model_paths["Ensemble"] = str(ensemble_path)

    # ── Feature importance ────────────────────────────────────────────
    # From RandomForest (trained on full data)
    rf = models["RandomForest"]
    rf.fit(X_train, y_train)
    rf_model = rf.named_steps["model"]
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        "feature": featurizer.feature_names_,
        "importance": importances,
    })
    feat_df = feat_df.sort_values("importance", ascending=False)
    feat_path = output_dir / "feature_importance.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"Top 10 features:\n{feat_df.head(10).to_string(index=False)}")

    # ── Save all outputs ─────────────────────────────────────────────
    metrics = {
        "model_name": "GateScore-ML v0.1",
        "model_type": "traditional ML ensemble (RF + HistGB + MLP + LogisticRegression)",
        "not_a_GNN": True,
        "primary_evaluation": "murcko_scaffold_split",
        "scaffold_metrics": scaffold_metrics,
        "random_holdout_metrics": random_metrics,
        "cv_metrics_random_stratified": cv_results,
        "dataset_info": {
            "n_total_raw": int(len(df)),
            "n_g3_excluded": int(n_g3),
            "n_trainable": len(df_trainable),
            "n_gating_positive_G1_G2": int(n_gating),
            "n_negative_passive_perm": int((df_trainable["evidence_level"] == "PP").sum()),
            "n_negative_non_penetrating": int((df_trainable["evidence_level"] == "NI").sum()),
            "n_negative_toxic": int((df_trainable["evidence_level"] == "TX").sum()),
            "n_train_scaffold": int(train_mask.sum()),
            "n_test_scaffold": int(test_mask.sum()),
            "scaffold_overlap_leakage_check": int(len(scaffold_overlap)),
            "negative_label_caveat": "B3DB BBB+ molecules labeled as 'passive_permeation'. This means 'known to cross BBB' NOT 'verified non-gating'.",
            "g3_caveat": "G3 (bronze) samples excluded from training because their labels are weak (traditional_kaiqiao, indirect evidence).",
            "toxic_negative_caveat": f"Only {int((df_trainable['evidence_level']=='TX').sum())} toxic negatives. Model cannot reliably distinguish gating from nonspecific membrane disruption.",
        },
        "feature_count": X.shape[1],
    }
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    scaffold_preds.to_csv(output_dir / "scaffold_holdout_predictions.csv", index=False)
    feat_df.to_csv(feat_path, index=False)

    print(f"\nSaved to {output_dir}:")
    print(f"  {metrics_path}")
    print(f"  scaffold_holdout_predictions.csv")
    print(f"  feature_importance.csv")
    for name, path in model_paths.items():
        print(f"  {path}")

    return {
        "cv_metrics": cv_results,
        "scaffold_metrics": scaffold_metrics,
        "random_metrics": random_metrics,
        "feature_importance": feat_df,
        "model_paths": model_paths,
    }


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "de_novo_pipeline/outputs/bbb_gating_dataset_v1.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "de_novo_pipeline/outputs/gatenet_v1"
    train_gatenet(data, out_dir)
