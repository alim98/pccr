# PCCR v6 Engineering Report (2026-04-05)

## Goal

Design the strongest realistic next-generation PCCR direction under a separate `v6` namespace, preserving the family's geometric strengths while improving overlap-based registration accuracy, especially Dice and local alignment.

Constraints followed:

- Existing stable PCCR code paths were left intact.
- New work lives under `src/pccr_v6/`.
- No full training was launched.
- Only small overfit-style diagnostics were run.

## Phase 1: Codebase Diagnosis

### What is already strong

From the code review and prior reports, the core PCCR family is already strong at:

- diffeomorphic / topology-preserving deformation
- low folding
- strong Jacobian statistics
- clean global geometry
- competitive HD95
- stable training

This is consistent with the current decoder and SVF integration design in `src/pccr/modules/diffeomorphic.py`, the pair-conditioned geometry path in `src/pccr/model.py`, and the training / loss stack in `src/pccr/losses.py` and `src/pccr/trainer.py`.

Full-validation evidence:

- H-ViT: Dice FG `0.2371`, HD95 `1.8176`, SDlogJ `1.3147`, nonpositive-J `0.008675`
- PCCR baseline: Dice FG `0.2125`, HD95 `1.9256`, SDlogJ `0.2715`, nonpositive-J `1.07e-05`

So H-ViT still leads on overlap, but PCCR remains far cleaner geometrically.

### What is weak

The consistent weakness is local overlap:

- Dice remains below H-ViT on full validation.
- Overfit-10 diagnostics plateau below the best internal targets unless the system leans harder into local correction.
- Naive local search can improve alignment but breaks Jacobians badly.

### Strongest bottleneck hypotheses

After reviewing the architecture, diagnostics, and prior reports, the three strongest explanations for Dice lag are:

1. The correspondence-to-flow handoff is too weak at the finest stages.
   - Coarse stages get explicit match-derived targets.
   - Fine correction is comparatively underconstrained and leans back toward smooth feature-driven decoding.

2. The architecture exploits correspondence signal too late.
   - Final residual refinement is useful but not the main missing ingredient.
   - Freeze tests suggest the bottleneck is not "just add a stronger final head".

3. Soft / ambiguous match aggregation is preserving geometry but blurring local alignment.
   - This preserves topology, but likely caps sharp local overlap.

### What prior attempts seem to have missed

- Final-stage local matcher additions alone did not solve the problem.
- Small handoff reweighting helped, but only slightly.
- Purely local final correction can buy overlap at the cost of Jacobian quality.

### Plausible v6 families

The most plausible remaining directions were:

- `v6a`: extend explicit local correspondence handoff into stage 0 and gate it with prior confidence
- `v6b`: ambiguity-aware handoff using richer match-distribution features, not just confidence / entropy scalars
- `v6c`: stronger stagewise rematching or higher-resolution candidate refinement before decoding

## Phase 2: Candidate Selection

### Selected for implementation: `v6a`

Core idea:

- Keep the existing matcher / pointmap / diffeomorphic backbone.
- Extend stage-local matching from stage 1 down to stage 0.
- Convert stage 0 and stage 1 local correlation outputs into explicit displacement targets.
- Gate those local targets with propagated coarse confidence so the decoder is only pushed where upstream correspondences are credible.

Why this was the strongest first test:

- Best fit to the existing codebase.
- High diagnostic value: directly tests whether missing fine-stage handoff is the real bottleneck.
- Low risk to topology: still uses the diffeomorphic decoder rather than replacing it with a brute-force flow regressor.

### Considered but not implemented

`v6b`: ambiguity-aware structured handoff

- Likely useful, but needs a more invasive redesign of decoder conditioning.
- Harder to test cleanly in one small diagnostic without confounding factors.

`v6c`: higher-resolution or repeated rematching

- Scientifically plausible, but riskier and more expensive.
- More likely to require a larger training study rather than a targeted overfit diagnostic.

## Phase 3: v6 Implementation

Implemented under a separate namespace:

- `src/pccr_v6/config.py`
- `src/pccr_v6/model.py`
- `src/pccr_v6/modules/diffeomorphic.py`
- `src/pccr_v6/trainer.py`
- `src/pccr_v6/scripts/train.py`
- `src/pccr_v6/scripts/evaluate.py`
- `src/pccr_v6/configs/pairwise_oasis_v6a.yaml`
- `run_pccr_v6_overfit_local.sh`

### Main architectural changes

1. New `PCCRV6Config`
   - adds explicit stage-0 local refinement controls
   - adds propagated-confidence gating controls for local refinement stages

2. New `PCCRV6Model`
   - reuses the stable encoder / matcher core
   - swaps in a v6 decoder only

3. New `DiffeomorphicRegistrationDecoderV6`
   - supports stage-1 and stage-0 local correlation refinement
   - converts local match deltas into explicit stage targets
   - gates local refinement with propagated prior confidence
   - keeps the stable SVF integration and final residual machinery

4. New v6 evaluation script
   - evaluates v6 checkpoints directly on either full validation or the exact overfit subset
   - added because the periodic iter-eval callback did not emit JSON reports during this v6 run, even though training continued correctly

## Phase 4: Diagnostic Runs

### Main training diagnostic

Run:

- job `26076464`
- experiment `pccr_v6a_stage0_handoff_ov10_20260405-153125`
- overfit-10 validation subset
- initialized from `/nexus/posix0/MBR-neuralsystems/alim/experiments_pccr/checkpoints/pccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt`
- trained for 60 epochs

Proxy trend from the training CSV:

- epoch 31 best proxy Dice: `0.2416`
- epoch 39 best proxy Dice: `0.2499`
- final proxy Dice around epoch 58-59: `0.2508`

### Frozen checkpoint evals on the exact overfit-10 subset

Because iter-eval JSONs were not emitted during training, I evaluated frozen checkpoints directly with `src/pccr_v6/scripts/evaluate.py`.

#### Early snapshot

- output: `logs/pccr_v6/pccr_v6a_stage0_handoff_ov10_20260405-153125/eval_overfit10_snapshot_154132/summary.json`
- Dice FG `0.2198`
- HD95 `1.8252`
- SDlogJ `0.3343`
- nonpositive-J `0.0`

#### Mid snapshot

- output: `logs/pccr_v6/pccr_v6a_stage0_handoff_ov10_20260405-153125/eval_overfit10_snapshot_154618/summary.json`
- Dice FG `0.2398`
- HD95 `1.7315`
- SDlogJ `0.3575`
- nonpositive-J `0.0`

#### Final checkpoint

- job `26076562`
- output: `logs/pccr_v6/pccr_v6a_stage0_handoff_ov10_20260405-153125/eval_overfit10_final_last/summary.json`
- Dice FG `0.2425`
- HD95 `1.7091`
- SDlogJ `0.3636`
- nonpositive-J `0.0`

## Phase 5: Comparison Against Strong Existing Anchors

Best existing overfit anchors:

- `v4b` best: Dice FG `0.2482`, HD95 `1.7252`, SDlogJ `0.4076`, nonpositive-J `1.06e-04`
- `vnext_fix_corr`: Dice FG `0.2468`, HD95 `1.7865`, SDlogJ `0.3510`, nonpositive-J `2.98e-05`
- `vnext_corr_rep1`: Dice FG `0.2463`, HD95 `1.7322`, SDlogJ `0.3302`, nonpositive-J `1.59e-05`
- `vnext_margin_handoff`: Dice FG `0.2467`, HD95 `1.7432`, SDlogJ `0.3246`, nonpositive-J `1.98e-06`

`v6a` final:

- Dice FG `0.2425`
- HD95 `1.7091`
- SDlogJ `0.3636`
- nonpositive-J `0.0`

### Interpretation

What `v6a` clearly achieved:

- preserved the key geometric behavior
- maintained essentially perfect topology preservation on this diagnostic
- improved HD95 to a very strong level
- improved over its own earlier checkpoints in a meaningful way

What `v6a` did not achieve:

- it did not beat the best existing PCCR overfit Dice ceiling
- it did not clearly beat the strongest recent `vnext` variants on local overlap
- its Jacobian quality remained good, but not so much better than the best `vnext` runs that the Dice gap would be worth it

## Final Recommendation

### Is `v6a` the best realistic next-generation paper candidate?

No.

It is scientifically useful and confirms part of the bottleneck story:

- stagewise fine-scale handoff matters
- extending explicit handoff deeper into the decoder can improve HD95 while preserving topology

But the current `v6a` design is not strong enough to replace the best existing `vnext` / `v4b` line as the main paper candidate, because the Dice gain is not there.

### What this means about the true bottleneck

The results suggest that "add stage-0 local handoff" is directionally correct, but not sufficient.

My current view is:

- the problem is not just missing fine-stage correction
- the finer-stage correction itself is still too weak / too blurred / too indirect
- the next serious direction should probably move beyond scalar confidence-gated local deltas and use a richer structured handoff from match distributions into the decoder

### Best next v6 direction

The strongest next candidate is not a larger final residual head and not a simple tweak of `v6a`.

The best next serious direction is:

- a new `v6b` family with ambiguity-aware, structured correspondence handoff
- pass richer match evidence into the decoder, not only a single local delta plus confidence gate
- keep the same diffeomorphic backbone and topology-preserving integration
- continue using synthetic pretraining

In short:

- keep the PCCR geometry core
- keep the explicit matching novelty
- increase how much fine-scale match structure the decoder can actually exploit

### Should a larger run be launched now?

No.

The diagnostics do not justify spending a larger training budget on `v6a` as implemented here.

## Deliverables Produced

- new v6 code path under `src/pccr_v6/`
- new v6 config and launcher
- new v6 evaluation script
- isolated overfit-10 diagnostics
- this engineering report documenting the decision

## Update: Structured Residual Handoff (`v6b`)

After the initial `v6a` result, I continued deeper rather than stopping there.

### What I tried next

I implemented a richer structured handoff path:

- `src/pccr_v6/modules/matcher.py`
  - new `StructuredCandidateRefinedMatcher`
  - preserves multi-hypothesis match evidence instead of collapsing immediately to one displacement
- `src/pccr_v6/modules/diffeomorphic.py`
  - structured evidence encoder
  - zero-initialized residual velocity adapters
  - structured evidence propagated through the decoder without breaking the pretrained base decoder pathway
- `src/pccr_v6/configs/pairwise_oasis_v6b.yaml`
  - v6b config

### Important failed intermediate result

The first `v6b` attempt concatenated structured evidence directly into decoder inputs.

That version trained poorly because it changed too many pretrained decoder input shapes at once. The code review and diagnostics showed that this was mostly an initialization / transfer problem, not a good scientific test of the idea.

So I replaced it with the current residual-adapter design:

- keep the pretrained decoder inputs intact
- inject structured evidence through zero-initialized residual velocity adapters
- let the new path start as a no-op and learn only if helpful

This was a materially better engineering choice.

### v6b runs

#### Base residual-handoff diagnostic

- train job `26077651`
- experiment `pccr_v6b_residual_structured_handoff_ov10_20260405-161000`
- 45-epoch overfit-style run on `gpudev` (time-capped at 15 minutes)

Exact evals:

- eval job `26077741`
- epoch042 / frozen snapshot both gave the same result:
  - Dice FG `0.2436`
  - HD95 `1.8031`
  - SDlogJ `0.3695`
  - nonpositive-J `1.88e-05`

#### Fine-tune continuation 1

- train job `26077751`
- experiment `pccr_v6b_residual_structured_handoff_ft_ov10_20260405-175000`
- initialized from the previous `v6b` checkpoint
- 20 additional epochs
- lower LR `2e-4`

Exact eval:

- eval job `26077794`
- output: `logs/pccr_v6/pccr_v6b_residual_structured_handoff_ft_ov10_20260405-175000/eval_overfit10_final_last/summary.json`
  - Dice FG `0.245456`
  - HD95 `1.707500`
  - SDlogJ `0.364564`
  - nonpositive-J `1.49e-05`

This was the best overall balance.

#### Fine-tune continuation 2

- train job `26077811`
- experiment `pccr_v6b_residual_structured_handoff_ft2_ov10_20260405-175700`
- initialized from the previous fine-tuned checkpoint
- 15 additional epochs
- lower LR `1e-4`

Exact eval:

- eval job `26077840`
- output: `logs/pccr_v6/pccr_v6b_residual_structured_handoff_ft2_ov10_20260405-175700/eval_overfit10_final_last/summary.json`
  - Dice FG `0.246260`
  - HD95 `1.781951`
  - SDlogJ `0.367331`
  - nonpositive-J `1.39e-05`

This was the best Dice among the v6 runs, but it gave back too much HD95 relative to the first fine-tune.

### Revised comparison

- `v4b` best:
  - Dice FG `0.248236`
  - HD95 `1.725244`
  - SDlogJ `0.407562`
  - nonpositive-J `1.06e-04`
- `vnext_fix_corr`:
  - Dice FG `0.246758`
  - HD95 `1.786458`
  - SDlogJ `0.351005`
  - nonpositive-J `2.98e-05`
- `vnext_corr_rep1`:
  - Dice FG `0.246314`
  - HD95 `1.732171`
  - SDlogJ `0.330197`
  - nonpositive-J `1.59e-05`
- `vnext_margin_handoff`:
  - Dice FG `0.246697`
  - HD95 `1.743152`
  - SDlogJ `0.324639`
  - nonpositive-J `1.98e-06`
- `v6a` final:
  - Dice FG `0.242501`
  - HD95 `1.709083`
  - SDlogJ `0.363559`
  - nonpositive-J `0.0`
- `v6b` fine-tune 1:
  - Dice FG `0.245456`
  - HD95 `1.707500`
  - SDlogJ `0.364564`
  - nonpositive-J `1.49e-05`
- `v6b` fine-tune 2:
  - Dice FG `0.246260`
  - HD95 `1.781951`
  - SDlogJ `0.367331`
  - nonpositive-J `1.39e-05`

### Revised recommendation

The best realistic next paper candidate is now:

- `v6b` residual structured handoff

More specifically, the best balanced checkpoint family from these diagnostics is:

- `pccr_v6b_residual_structured_handoff_ft_ov10_20260405-175000`

Why this is the current recommendation:

- it improves Dice substantially over `v6a`
- it keeps Jacobians and folding behavior in the strong PCCR regime
- it achieves the best HD95 among all the serious low-folding candidates tested here
- it nearly matches the best prior Dice while giving a more attractive geometry / HD95 tradeoff than `v4b`

The second fine-tune (`ft2`) is a legitimate Dice-max variant, but I do not recommend it as the primary paper candidate because it traded away too much HD95 for a very small Dice gain.

### Should a larger run be launched now?

Yes.

Not because `v6b` clearly dominates every existing number, but because it is the first new v6 direction that:

- materially improved the real exact-eval Dice over `v6a`
- remained geometrically clean
- produced a scientifically interesting tradeoff rather than a trivial metric shuffle

### Suggested next larger run command

Do not auto-launch; this is the recommended next command:

```bash
PYTHONPATH=/u/almik/others/hvit \
/nexus/posix0/MBR-neuralsystems/alim/envs/hvit/bin/python \
src/pccr_v6/scripts/train.py \
  --mode train \
  --phase real \
  --data_source real \
  --config src/pccr_v6/configs/pairwise_oasis_v6b.yaml \
  --checkpoint_path /nexus/posix0/MBR-neuralsystems/alim/experiments_pccr/checkpointspccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt \
  --accelerator gpu \
  --num_gpus 4 \
  --dataset_format oasis_fs \
  --train_data_path /nexus/posix0/MBR-neuralsystems/alim/regdata/oasis \
  --val_data_path /nexus/posix0/MBR-neuralsystems/alim/regdata/oasis \
  --experiment_name pccr_v6b_residual_structured_handoff_real_main \
  --logger_backend aim \
  --aim_repo /u/almik/others/hvit/aim \
  --batch_size 1 \
  --num_workers 4 \
  --max_epochs 600 \
  --lr 1e-4 \
  --limit_train_batches 1.0 \
  --limit_val_batches 1.0
```

If a second-stage fine-tuning pass is desired after that main run, the small-diagnostic evidence suggests a lower-LR continuation is reasonable:

```bash
PYTHONPATH=/u/almik/others/hvit \
/nexus/posix0/MBR-neuralsystems/alim/envs/hvit/bin/python \
src/pccr_v6/scripts/train.py \
  --mode train \
  --phase real \
  --data_source real \
  --config src/pccr_v6/configs/pairwise_oasis_v6b.yaml \
  --checkpoint_path /nexus/posix0/MBR-neuralsystems/alim/experiments_pccr/checkpointspccr_v6/pccr_v6b_residual_structured_handoff_real_main/last.ckpt \
  --accelerator gpu \
  --num_gpus 4 \
  --dataset_format oasis_fs \
  --train_data_path /nexus/posix0/MBR-neuralsystems/alim/regdata/oasis \
  --val_data_path /nexus/posix0/MBR-neuralsystems/alim/regdata/oasis \
  --experiment_name pccr_v6b_residual_structured_handoff_real_ft \
  --logger_backend aim \
  --aim_repo /u/almik/others/hvit/aim \
  --batch_size 1 \
  --num_workers 4 \
  --max_epochs 120 \
  --lr 2e-4 \
  --limit_train_batches 1.0 \
  --limit_val_batches 1.0
```
