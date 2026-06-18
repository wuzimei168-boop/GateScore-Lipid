# GateScore-Lipid

A conjugatability-aware computational protocol for prioritizing candidate small-molecule modules in brain-targeted lipid nanoparticle (BLNP) design.

## Overview

GateScore-Lipid is a 5-dimensional scoring framework that integrates one ML-derived gating-associated enrichment signal with four deterministic chemical rules to reduce the experimental search space for BLNP module candidates.

**Key features:**
- **5D scoring**: S_gating + S_chemistry + S_novelty + S_safety + S_conjugate
- **Conjugatability-aware**: SMARTS-based detection of chemically reactive functional groups for lipid conjugation
- **Chemistry-first hybrid**: 4/5 dimensions are deterministic rule-based scores
- **Search-space reduction**: 9,599 molecules → 462 elevated-gating conjugatable candidates → Top 50 synthesis-ready shortlist

## Installation

```bash
git clone https://github.com/wuzimei168-boop/GateScore-Lipid.git
cd GateScore-Lipid
conda env create -f environment.yml
conda activate gatescore-lipid
```

## Quick Start

```bash
# Run the tutorial notebook
jupyter notebook notebooks/GateScore_Lipid_Tutorial.ipynb

# Or use the Top 50 pre-computed screening results
import pandas as pd
results = pd.read_csv("data/lipid_module_top50_corrected.csv")
```

## Repository Structure

```
GateScore-Lipid/
├── README.md
├── environment.yml
├── data/
│   ├── lipid_module_round2_corrected_full_screen.csv   # Full screening results (9,599 molecules)
│   ├── lipid_module_top50_corrected.csv                 # Top 50 candidates
│   ├── lipid_module_round2_corrected_summary.json       # Screening statistics
│   └── gatescore_RF_v1.joblib                           # Trained RandomForest model
├── notebooks/
│   └── GateScore_Lipid_Tutorial.ipynb                   # Interactive tutorial
├── scripts/
│   ├── run_lipid_screen_round2.py                       # Main screening pipeline
│   ├── bbb_lipid_filter.py                              # 5D scoring module
│   ├── features.py                                      # Feature engineering (2,138 dims)
│   └── gatenet.py                                       # GateScore-ML model training
└── figures/
    ├── Figure1_Data_and_Model.png
    ├── Figure2_5D_Scoring.png
    ├── Figure3_Diagnostics.png
    └── Figure4_Candidate_Landscape.png
```

The trained model, full 9,599-molecule screening table, and publication
figures will be distributed as downloadable assets in the GitHub v1.0
release to keep the source repository lightweight.

## Data

The screening was performed on 9,599 molecules from three sources:
- **BBB-GatingDB augmented** (1,231 entries): Curated BBB gating-associated database
- **B3DB** (7,807 entries): Public BBB permeability benchmark
- **TCM-focused natural product library** (2,796 entries): Systematically enumerated monoterpenoid, sesquiterpenoid, and phenylpropanoid scaffolds

## Model

Primary model: RandomForest classifier (500 trees, max_depth=12), calibrated with isotonic regression.

| Metric | Value |
|--------|-------|
| Scaffold AUC | 0.992 |
| Scaffold MCC | 0.662 |
| 5-fold CV AUC | 0.976 ± 0.017 |
| 5-fold CV MCC | 0.775 ± 0.145 |

Training: Murcko scaffold split (80/20, 0% leakage). G3 molecules excluded. 35 feature-valid G1+G2 positives against B3DB permeability-reference background.

## Citation

Wu Z. GateScore-Lipid: A Conjugatability-Aware Computational Protocol for Prioritizing Candidate Small-Molecule Modules in Brain-Targeted Lipid Nanoparticle Design. Manuscript in preparation (2026).

## License

CC-BY 4.0
