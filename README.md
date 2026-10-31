# Modeling-Driver-Evasive-Behavior-in-Response-to-Mobile-Maintenance-Vehicles-on-Expressways

This repository contains the trajectory data, selected source code, summary results, and manuscript figures for the analysis of driver evasive behavior and HGB-ECR trajectory prediction in mobile maintenance work-zone scenarios on expressways.

The results correspond to the current manuscript: five date-held-out folds over 4,890 trajectories, comparing HGB-ECR with five baselines (CNN, GBR, IDM, Kalman-CV, Helly).

## Repository Structure

```text
.
├── Data/      Five trajectory workbooks split into 24 MiB parts, plus manifest.json and reassemble.py
├── Code/
│   ├── model/        HGB-ECR inference, head training, parameter selection, numerical correction
│   ├── core/         Shared data loading, features, models and rollout
│   └── evaluation/   Feature statistics, result verification, component ablation
├── Result/    Summary CSV files used in the manuscript tables
└── Figure/    Figures 1-12 extracted from the manuscript
```

Each directory has its own `README.md` and a `manifest.json` listing file sources and SHA-256 values.

## Data

| Workbook | Size | Parts |
| --- | --- | --- |
| `data_0403.xlsx` | 118.5 MiB | 5 |
| `data_0407.xlsx` | 624.4 MiB | 27 |
| `data_0409.xlsx` | 531.0 MiB | 23 |
| `data_0410.xlsx` | 520.1 MiB | 22 |
| `data_0416.xlsx` | 83.9 MiB | 4 |

The `.partNNN` files are byte slices of the original workbooks, not individual spreadsheets. To restore them, download the whole `Data/` directory and run:

```bash
cd Data
python reassemble.py
```

The script verifies every part and every reconstructed workbook against the SHA-256 values in `manifest.json` and writes the files to `Data/restored_workbooks/`.

The five workbooks comprise 81 parts and preserve all three worksheets (`Sheet1`, `Main_Completion_Stats`, and `Surrounding_Vehicles`). Their checksums were verified on 2026-10-31.

Raw source records (`raw_data_*.xlsx`) are not included. Requests for raw data should be directed to the repository maintainers; any sharing is subject to the data owners' consent, privacy and data-use requirements.

## Code

`Code/` is a source snapshot of the files behind the current results, not a standalone executable package. Trained model weights, caches and the private path registry are not included, and the original imports and paths are kept so the SHA-256 provenance in `Code/manifest.json` can be checked.

## Results

Overall prediction errors across the five held-out dates (manuscript Table 8b; `Result/overall_prediction_errors.csv`):

| Model | ADE (m) | FDE (m) | RMSE (m) | Successful / attempted |
| --- | --- | --- | --- | --- |
| HGB-ECR | 1.214 | 2.720 | 3.129 | 4890 / 4890 |
| CNN | 1.474 | 3.502 | 4.084 | 4890 / 4890 |
| GBR | 1.523 | 3.567 | 4.385 | 4890 / 4890 |
| Kalman-CV | 2.455 | 5.007 | 4.857 | 4890 / 4890 |
| IDM | 3.040 | 4.099 | 6.622 | 4888 / 4890 |
| Helly | 7.060 | 14.704 | 10.230 | 4600 / 4890 |

Other files in `Result/` give per-date errors, horizon errors, endpoint axis errors, confidence intervals, component ablation, endpoint feature SHAP values and scoring coverage. The manuscript reports rounded values.

The per-date errors, component ablations, endpoint axis errors, and grouped SHAP results correspond to manuscript Tables 8a, 9, 10, and 11, respectively.

## Figures

`Figure/` contains Figures 1-12. All 12 PNG files were verified byte-for-byte against the current revised manuscript (`Figures_Revised.docx`) on 2026-10-31. See [Figure/README.md](Figure/README.md) for the figure documentation and [Figure/manifest.json](Figure/manifest.json) for SHA-256 values.
