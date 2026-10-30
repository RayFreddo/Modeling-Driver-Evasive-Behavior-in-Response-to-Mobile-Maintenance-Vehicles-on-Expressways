# Manuscript code selection

Selected source files for the current five-fold, 4,890-trajectory HGB-ECR manuscript results. This is a source snapshot, not a standalone executable package: the frozen model weights, Excel input files, joblib caches, and historical path registry remain in the private workspace and are not included. The code retains original imports and paths so SHA-256 provenance can be checked. Do not present this snapshot as a one-command reproduction.

The `model` directory contains the current HGB-ECR inference, head-training, parameter-selection and numerical correction sources. `core` contains the shared data, features, model and rollout implementation. `evaluation` contains the result verification and component ablation scripts. Earlier run1-run12 experiments, virtual environments, caches, model weights, per-object predictions, and document-edit utilities were omitted. Source paths and hashes are in `manifest.json`.

## Risk input update 2026-10-30

The published feature extractor now appends eight inputs to the existing 310
motion/image features: TTC, DRAC, and the front gap, rear gap and available gap
for each adjacent lane. Training and history prediction pass these values into
the model transformer. TTC and DRAC use positive spacing and closing speed
greater than 0.5 m/s. Neighbor inputs use the time-matched snapshot available at
the prediction origin, or explicitly supplied gap columns in that same history.
Missing neighbors remain NaN; available gap requires both neighbors. No future
action-side label or future observation is used to select these inputs.

The schema is `motion310-risk8-v1` (318 columns). Feature caches and both the
base and correction models must be rebuilt/refitted for this schema. The
prediction entry rejects the old frozen model rather than silently dropping
the new inputs. This remains a selected source snapshot with private runtime
dependencies, not a standalone training package. No training or evaluation was
run for this update. The existing `Data`, `Result`, and figure files are unchanged
and describe the previously evaluated model, not new risk-input results.
