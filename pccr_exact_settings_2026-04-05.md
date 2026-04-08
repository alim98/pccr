# PCCR Exact Settings Snapshot (2026-04-05)

This file stores the exact settings used for the strongest overfit10 results collected in the latest sweep.

## Best Run (Current Top In This Sweep)
- Experiment: `pccr_vnext_fix_corr_ov10_20260404-184738`
- Base config: `src/pccr/configs/pairwise_oasis_vnext.yaml`
- Launch script: `run_pccr_vnext_overfit_local.sh`
- Command payload:
```bash
MAX_EPOCHS=120 \
EXPERIMENT_NAME=pccr_vnext_fix_corr_ov10_20260404-184738 \
./run_pccr_vnext_overfit_local.sh \
  --config_override correspondence_weight=0.2 \
  --config_override decoder_fitting_weight=0.1 \
  --config_override decoder_fitting_confidence_percentile=0.5 \
  --config_override decoder_fitting_entropy_threshold=0.9
```
- Best metric point:
  - Dice FG: `0.2468` @ epoch `80`
  - HD95 FG: `1.786`
  - SDlogJ: `0.351`
  - Non-positive Jacobian fraction: `2.98e-05`

## Strong Replicate
- Experiment: `pccr_vnext_corr_rep1_ov10_20260404-193001`
- Base config: `src/pccr/configs/pairwise_oasis_vnext.yaml`
- Command payload:
```bash
MAX_EPOCHS=120 \
EXPERIMENT_NAME=pccr_vnext_corr_rep1_ov10_20260404-193001 \
./run_pccr_vnext_overfit_local.sh \
  --config_override correspondence_weight=0.2 \
  --config_override decoder_fitting_weight=0.1 \
  --config_override decoder_fitting_confidence_percentile=0.5 \
  --config_override decoder_fitting_entropy_threshold=0.9
```
- Best metric point:
  - Dice FG: `0.2463` @ epoch `80`
  - HD95 FG: `1.732`
  - SDlogJ: `0.330`
  - Non-positive Jacobian fraction: `1.59e-05`

## Key Reference
- Historical overfit10 reference:
  - Experiment: `pccr_v4b_overfit10_test`
  - Dice FG: `0.2482` @ epoch `200`
  - HD95 FG: `1.725`
  - SDlogJ: `0.408`

## Applied Next-Step Run (Margin-Weighted Decoder Fitting)
- Experiment: `pccr_vnext_margin_handoff_ov10_20260405-132837`
- Base config: `src/pccr/configs/pairwise_oasis_vnext_margin_handoff.yaml`
- Slurm job: `26075002`
- Command payload:
```bash
MAX_EPOCHS=120 \
CONFIG_PATH=src/pccr/configs/pairwise_oasis_vnext_margin_handoff.yaml \
EXPERIMENT_NAME=pccr_vnext_margin_handoff_ov10_20260405-132837 \
./run_pccr_vnext_overfit_local.sh \
  --config_override correspondence_weight=0.2 \
  --config_override decoder_fitting_weight=0.1 \
  --config_override decoder_fitting_entropy_threshold=-1.0 \
  --config_override decoder_fitting_confidence_percentile=0.0 \
  --config_override decoder_fitting_margin_power=1.0 \
  --config_override decoder_fitting_margin_min=0.0
```

## Current Full-Training Best Setup (V5)
- Config: `src/pccr/configs/pairwise_oasis_v5.yaml`
- Full pipeline launcher: `run_pccr_oasis_v5.sh`
- Command payload:
```bash
RUN_TAG=v5 \
SYNTH_EXPERIMENT_NAME=pccr_v5_synth_v5 \
REAL_EXPERIMENT_NAME=pccr_v5_real_v5 \
./run_pccr_oasis_v5.sh
```
- Notes:
  - Synthetic stage: `max_epochs=50`, `lr=1e-4`
  - Real stage: `max_epochs=200`, `train_num_steps=600`, `lr=2e-5`
  - Correspondence/handoff settings follow best observed overfit setup:
    - `correspondence_weight=0.2`
    - `decoder_fitting_weight=0.1`
    - `decoder_fitting_confidence_percentile=0.5`
    - `decoder_fitting_entropy_threshold=0.9`
