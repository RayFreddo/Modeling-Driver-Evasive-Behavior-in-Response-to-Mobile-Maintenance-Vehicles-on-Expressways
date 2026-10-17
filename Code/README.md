# Manuscript code selection

Selected source files for the current five-fold, 4,890-trajectory HGB-ECR manuscript results. This is a source snapshot, not a standalone executable package: the frozen model weights, Excel input files, joblib caches, and historical path registry remain in the private workspace and are not included. The code retains original imports and paths so SHA-256 provenance can be checked. Do not present this snapshot as a one-command reproduction.

The `model` directory contains the current HGB-ECR inference, head-training, parameter-selection and numerical correction sources. `core` contains the shared data, features, model and rollout implementation. `evaluation` contains the result verification and component ablation scripts. Earlier run1-run12 experiments, virtual environments, caches, model weights, per-object predictions, and document-edit utilities were omitted. Source paths and hashes are in `manifest.json`.
