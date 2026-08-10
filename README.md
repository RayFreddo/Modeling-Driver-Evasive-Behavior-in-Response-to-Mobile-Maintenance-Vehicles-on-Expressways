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
│   ├── data_0416.xlsx
│   ├── raw_data_0403.xlsx
│   ├── raw_data_0407.xlsx
│   ├── raw_data_0409.xlsx
│   ├── raw_data_0410.xlsx
│   └── raw_data_0416.xlsx
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
