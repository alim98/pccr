# Final Run

## PCCR on TransMorph PKL

Default full-capacity attempt with streamed final cost volume:
```bash
EXPERIMENT_NAME=pccr_transmorph_pkl_v3 sbatch train_pccr_transmorph_pkl.sh
```

Fallback to original legacy final cost volume:
```bash
FINAL_COST_VOLUME_MODE=legacy EXPERIMENT_NAME=pccr_transmorph_pkl_legacy sbatch train_pccr_transmorph_pkl.sh
```

Safe fallback with final cost volume disabled:
```bash
FINAL_COST_VOLUME_MODE=off EXPERIMENT_NAME=pccr_transmorph_pkl_safe sbatch train_pccr_transmorph_pkl.sh
```

If streamed mode still needs less memory:
```bash
FINAL_COST_VOLUME_MODE=streamed FINAL_COST_VOLUME_CHUNK_SIZE=4 EXPERIMENT_NAME=pccr_transmorph_pkl_chunk4 sbatch train_pccr_transmorph_pkl.sh
```

## TransMorph Official Checkpoint

Reproduce official validation metrics:
```bash
/nexus/posix0/MBR-neuralsystems/alim/envs/hvit/bin/python scripts/eval_transmorph_official.py \
  --accelerator gpu \
  --precision fp32 \
  --submission_dir results/transmorph_official/submission_large
```

Expected reference from our successful run:
- direct Dice: `0.8617`
- challenge Dice: `0.8623`
- challenge SDlogJ: `0.1276`

## Notes

- `streamed` keeps final local cost volume on, but computes it memory-efficiently.
- `legacy` restores the old behavior.
- `off` is the safety fallback if training is unstable or still OOMs.

## What Changed

- The old low Dice came from evaluating/training different setups at once:
  - old low-res checkpoints around `[40, 48, 56]`
  - raw Learn2Reg loader
  - new full-res configs
- That made the comparison unfair and the numbers looked artificially bad.
- We fixed this by moving to the official TransMorph-preprocessed OASIS `.pkl` data and by evaluating TransMorph's own official checkpoint first.
- Official TransMorph checkpoint was successfully regenerated at the expected scale:
  - direct Dice: `0.8617`
  - challenge Dice: `0.8623`
  - challenge SDlogJ: `0.1276`
- This confirmed that the data, split, and evaluation scale are now in the correct comparable regime.

## Data Setup

- Official TransMorph-preprocessed OASIS data:
  - train: `/nexus/posix0/MBR-neuralsystems/alim/regdata/oasistransmorph/OASIS_L2R_2021_task03/All`
  - val/test: `/nexus/posix0/MBR-neuralsystems/alim/regdata/oasistransmorph/OASIS_L2R_2021_task03/Test`
- Shape used by this setup: `160 x 192 x 224`
- Label space: `0..35`
- Evaluation labels: `1..35`

## Loader Paths

- Main dataset loader logic:
  - [src/data/datasets.py](/u/almik/others/hvit/src/data/datasets.py)
- PCCR data plumbing:
  - [src/pccr/data.py](/u/almik/others/hvit/src/pccr/data.py)
- PCCR TransMorph-pkl config:
  - [config/pairwise_oasis_transmorph_pkl.yaml](/u/almik/others/hvit/config/pairwise_oasis_transmorph_pkl.yaml)
- PCCR training script:
  - [train_pccr_transmorph_pkl.sh](/u/almik/others/hvit/train_pccr_transmorph_pkl.sh)

## Memory Fix

- The main OOM came from the full-resolution final local cost volume in PCCR.
- We added a reversible memory-efficient `streamed` mode instead of deleting the feature.
- It keeps final local cost volume enabled, but avoids materializing the giant dense tensor at once.
- Toggle modes:
  - `streamed`: new default
  - `legacy`: original behavior
  - `off`: safe fallback
