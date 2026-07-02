# MenoSCA-FBTS

Reference implementation of **MenoSCA-FBTS** (Menopause-Specific Sleep Classification
Algorithm with Filter-Bank Tangent Space) for automated sleep staging in
menopausal women, as reported in:

> "Menopause-Specific EEG Markers for Automated Sleep Staging: A Cross-Database
> Polysomnographic Study", *Sensors*, 2026.

## Overview

MenoSCA-FBTS couples Riemannian-geometric covariance features with
menopause-specific EEG markers (alpha intrusion index, spindle density proxy,
NREM alpha/delta ratio, slow-wave stability) and is evaluated under
leave-one-subject-out (LOSO) cross-validation on three independent
polysomnographic databases: Sleep-EDF Expanded, ISRUC-Sleep, and DREAMS.

## Repository Layout

```
.
├── sca_fbts_woman.py              # Core MenoSCA-FBTS algorithm
├── experiment_three_groups_paper.py  # Three-group (MW/YW/YM) analysis
├── experiment_sota_comparison.py    # Benchmark against 5 SOTA methods
├── experiment_ablation_study.py     # Ablation study
├── generate_representative_hypnograms.py
├── analysis_comprehensive.py        # Comprehensive metrics analysis
├── run_all_experiments.sh           # One-click runner (Linux / macOS)
├── run_all_experiments.bat          # One-click runner (Windows)
├── json/                            # Subject metadata
│   ├── menopausal_women_subjects.json
│   ├── young_women_control_group.json
│   └── young_men_control_group.json
└── README.md
```

## Requirements

- Python 3.9+
- numpy, scipy, scikit-learn, mne, pyedflib
- torch (for deep-learning baselines: TinySleepNet, EEGNet)
- braindecode (for TinySleepNet / EEGNet baselines)
- yasa, tslearn, matplotlib
- xgboost (ensemble classifier)

A typical install:

```bash
pip install numpy scipy scikit-learn mne pyedflib torch braindecode \
            yasa tslearn matplotlib xgboost
```

## Datasets

The experiments use three publicly available polysomnographic databases. Each
must be downloaded separately and stored at the path indicated by the
corresponding environment variable:

| Database         | Environment variable | Source                                                                              |
| ---------------- | -------------------- | ----------------------------------------------------------------------------------- |
| Sleep-EDF (SC)   | `SLEEP_EDF_DIR`      | <https://physionet.org/content/sleep-edfx/1.0.0/>                                   |
| ISRUC-Sleep      | `ISRUC_DIR`          | <https://sleeptight.isr.uc.pt/ISRUC_Sleep/>                                         |
| DREAMS           | `DREAMS_DIR`         | <https://zenodo.org/record/2650142>                                                 |

Set the variables before running the experiments, e.g.:

```bash
export SLEEP_EDF_DIR=/path/to/sleep-edfx
export ISRUC_DIR=/path/to/ISRUC-SLEEP
export DREAMS_DIR=/path/to/DREAMS
```

## Quick Start

Run the full pipeline (SOTA comparison → ablation → three-group analysis on
each database → representative hypnograms → comprehensive analysis):

```bash
bash run_all_experiments.sh
```

Or step-by-step:

```bash
python experiment_sota_comparison.py
python experiment_ablation_study.py
python experiment_three_groups_paper.py --dataset sleep-edf
python experiment_three_groups_paper.py --dataset isruc
python experiment_three_groups_paper.py --dataset dreams
python generate_representative_hypnograms.py
python analysis_comprehensive.py
```

Results are written to `experiment_results/` as JSON files. Figures are saved
as PNG to the same directory.

## Reproducing the Paper

- **Table 1 (MenoSCA-FBTS performance by group):** `experiment_three_groups_paper.py --dataset sleep-edf`
- **Table 2 (SOTA comparison):** `experiment_sota_comparison.py`
- **Table 3 (Cross-dataset performance):** `experiment_three_groups_paper.py` for each database
- **Table 4 (Sleep architecture statistics):** `analysis_comprehensive.py`
- **Figures 1–2 (Representative hypnograms / hypnodensity):** `generate_representative_hypnograms.py`

## License

This code is released for academic, non-commercial use. Please cite the
associated paper if you use this software.
