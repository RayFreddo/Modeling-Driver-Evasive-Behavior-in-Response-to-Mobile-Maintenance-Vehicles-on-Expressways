# Modeling-Driver-Evasive-Behavior-in-Response-to-Mobile-Maintenance-Vehicles-on-Expressways

This repository contains the key data, code, model outputs, and manuscript figures for driver evasive behavior analysis and multi-model trajectory prediction in mobile maintenance work-zone scenarios.

## Repository Structure

```text
.
├── Data/
│   ├── data_0403.xlsx
│   ├── data_0407.xlsx
│   ├── data_0409.xlsx
│   ├── data_0410.xlsx
│   └── data_0416.xlsx
├── Code/
│   ├── prediction_main.py
│   ├── prediction_shared_functions.py
│   ├── feature_analysis_statistics.py
│   └── feature_indicator_schematic.py
├── Result/
│   ├── final_eight_model_results.xlsx
│   └── final_eight_model_results.json
└── Figure/
    └── JTEA_submission_figures/
        ├── figure_01_field_collection.png
        ├── figure_02_sensing_workflow.png
        ├── figure_03_event_extraction.png
	        ├── ...
	        └── figure_12_endpoint_errors.png

```

## Data Availability

The `Data/` directory contains the analysis-ready Excel datasets used by the
code and reported results in this repository:

| File | Availability | Notes |
| --- | --- | --- |
| `data_0403.xlsx` | Included | Analysis-ready dataset. |
| `data_0407.xlsx` | Pending Git LFS upload | Analysis-ready dataset; it is not available in this commit. |
| `data_0409.xlsx` | Included | Analysis-ready dataset. |
| `data_0410.xlsx` | Included | Analysis-ready dataset. |
| `data_0416.xlsx` | Included | Analysis-ready dataset. |

The raw source data are **not** deposited in this public repository. In
particular, files named `raw_data_0403.xlsx`, `raw_data_0407.xlsx`,
`raw_data_0409.xlsx`, `raw_data_0410.xlsx`, and `raw_data_0416.xlsx` are not
available here and should not be expected in a clone of the project.

The public datasets are the repository inputs for the supplied analysis and
prediction code. They support reproduction of the processing, model fitting,
and result-generation steps implemented in `Code/`, but they do not by
themselves reproduce the upstream collection, cleaning, or event-extraction
steps that produced the raw source records.

Requests for access to raw source data should be directed to the repository
maintainers. Any sharing decision must be made by the data owners and is
subject to the applicable consent, privacy, and data-use requirements.

### Cloning the repository

Because `data_0407.xlsx` exceeds GitHub's ordinary file-size limit, it must be
stored with Git LFS. Once it has been uploaded, install Git LFS before cloning,
or run the following in an existing clone before using that file:

```bash
git lfs install
git lfs pull
```
