4node v5: 
 run_pccr_v5.sh — Warm-starts from epoch027.ckpt (val_dice≈0.77) via --checkpoint_path (strict=False, fresh LR schedule). No synthetic training needed.

How to launch:

JOB_SCRIPT="${PWD}/jobs/active/train_pccr_transmorph_pkl_4node.sh" bash run_pccr_v5.sh
To use a later epoch checkpoint if the current run trains further before you launch v5:


WARMSTART_CKPT="/nexus/posix0/MBR-neuralsystems/alim/experiments_pccr/checkpoints/pccr/pccr_full_final_20260418_210108/epoch035.ckpt" \
  JOB_SCRIPT="${PWD}/jobs/active/train_pccr_transmorph_pkl_4node.sh" bash run_pccr_v5.sh


final 4 node: JOB_SCRIPT="${PWD}/jobs/active/train_pccr_transmorph_pkl_4node.sh" bash run_pccr_full_final.sh
# PCCR Experiments

Updated: 2026-04-12

Each tmux here is one experiment. No separate tmux list.

## 1) Half-Fast Full Pipeline

- `tmux`: `halfpccrfinal`
- `status`: `running`
- `current job`: `26265816` (`pccr_tm_pkl`)
- `experiment`: `pccr_transmorph_pkl_half_fast_full_v2_synthetic`
- `type`: `half-size`, full pipeline launcher, currently in `synthetic`
- `shape`: `80 x 96 x 112`
- `start`: started fresh; current job resumed from same experiment `last.ckpt`
- `current progress`: `epoch 255`, `step 25301`
- `dice`: none yet (`synthetic`)
- `tmux state`: launcher alive, monitoring `26265816`
- `run command`:

```bash
EXPERIMENT_PREFIX=pccr_transmorph_pkl_half_fast_full_v2 ./run_pccr_transmorph_half_fast_full_synth_then_real.sh
```

## 2) Half-Fast Immediate Real From Latest Synth

- `tmux`: `pccr_imid_half_real`
- `status`: `running`
- `current job`: `26280440` (`pccr_tm_pkl_real_now`)
- `experiment`: `pccr_transmorph_pkl_half_fast_real_now_20260411`
- `type`: `half-size`, immediate `real` run from latest synth checkpoint
- `shape`: `80 x 96 x 112`
- `start`: first started from latest checkpoint of `pccr_transmorph_pkl_half_fast_full_v2_synthetic`; current job resumed from real `last.ckpt`
- `current progress`: `epoch 45`, `step 4494`
- `dice`: latest `val_dice = 0.7458` at `epoch 39`, `step 3959`
- `tmux state`: launcher alive, monitoring replacement job after timeout
- `run command`:

```bash
SYNTH_EXPERIMENT_NAME=pccr_transmorph_pkl_half_fast_full_v2_synthetic REAL_EXPERIMENT_NAME=pccr_transmorph_pkl_half_fast_real_now_20260411 JOB_NAME=pccr_tm_pkl_real_now ./run_pccr_transmorph_half_fast_real_now_from_latest_synth.sh
```

## 3) Full-Size Full Pipeline

- `tmux`: `finalpccr`
- `status`: `running`
- `current job`: `26281028` (`pccr_tm_pkl`)
- `experiment`: `pccr_transmorph_pkl_v3_vec_full_20260410_125634_synthetic`
- `type`: `full-size`, full pipeline launcher, currently in `synthetic`
- `shape`: `160 x 192 x 224`
- `start`: started fresh; current job resumed from same experiment `last.ckpt`
- `current progress`: `epoch 21`, `step 531`
- `dice`: none yet (`synthetic`)
- `tmux state`: launcher alive, previous job `26252670` timed out and this tmux submitted replacement `26281028`
- `run command`:

```bash
./run_pccr_transmorph_full_synth_then_real.sh
```

## 4) Full-Size Immediate Real From Latest Synth

- `tmux`: `final_pccr_imidiate_real_start`
- `status`: `running`
- `current job`: `26281902` (`pccr_tm_pkl_real_now_full`)
- `experiment`: `pccr_transmorph_pkl_v3_vec_full_real_now_20260412_124135`
- `type`: `full-size`, immediate `real` run from latest full-size synth checkpoint
- `shape`: `160 x 192 x 224`
- `start`: started from `/nexus/posix0/MBR-neuralsystems/alim/experiments_pccr/checkpointspccr/pccr_transmorph_pkl_v3_vec_full_20260410_125634_synthetic/last.ckpt`
- `current progress`: no logged metrics yet
- `dice`: none yet
- `checkpoint`: none yet
- `tmux state`: launcher alive, monitoring `26281902`
- `run command`:

```bash
SYNTH_EXPERIMENT_NAME=pccr_transmorph_pkl_v3_vec_full_20260410_125634_synthetic REAL_EXPERIMENT_NAME=pccr_transmorph_pkl_v3_vec_full_real_now_$(date +%Y%m%d_%H%M%S) JOB_NAME=pccr_tm_pkl_real_now_full ./run_pccr_transmorph_full_real_now_from_latest_synth.sh
```

## 5) Full-Size Optimized Immediate Real From Latest Synth

- `tmux`: `imid_full_real_opt`
- `status`: `running`
- `current job`: `26282781` (`pccr_tm_pkl_real_full_optimized`)
- `experiment`: `pccr_transmorph_pkl_v3_vec_full_real_optimized_20260412_134921`
- `type`: `full-size`, optimized immediate `real` run from latest full-size synth checkpoint
- `shape`: `160 x 192 x 224`
- `start`: started from `/nexus/posix0/MBR-neuralsystems/alim/experiments_pccr/checkpointspccr/pccr_transmorph_pkl_v3_vec_full_20260410_125634_synthetic/last.ckpt`
- `settings`: `final CV radius=6`, `chunk_size=16`, `checkpoint every 25 steps`
- `current progress`: no logged metrics yet
- `dice`: none yet
- `checkpoint`: none yet
- `tmux state`: launcher alive, monitoring `26282781`
- `run command`:

```bash
SYNTH_EXPERIMENT_NAME=pccr_transmorph_pkl_v3_vec_full_20260410_125634_synthetic REAL_EXPERIMENT_NAME=pccr_transmorph_pkl_v3_vec_full_real_optimized_$(date +%Y%m%d_%H%M%S) JOB_NAME=pccr_tm_pkl_real_full_optimized ./run_pccr_transmorph_full_real_optimized_from_latest_synth.sh
```
