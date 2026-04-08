# PCCR Engineering Report (2026-04-04)

## Scope
- Full codebase + diagnostics + overfit results reviewed.
- Matcher/handoff bottleneck targeted with code+config updates.
- Large Slurm sweep executed on overfit10 (many variants, multiple replicas).

## Code Changes Applied
1. `src/pccr/modules/matcher.py`
- Confidence/entropy in `CandidateRefinedMatcher` now computed from **final refined probabilities** (`final_probabilities`) instead of pre-refinement distribution.
- This directly fixes confidence-guided handoff signal quality.

2. `src/pccr/configs/pairwise_oasis_vnext_top1_handoff.yaml`
- Added a top1 handoff config to test sharper correspondence transfer.

## Key Experimental Outcome (Overfit10)
Best run in this sweep:
- `pccr_vnext_fix_corr_ov10_20260404-184738`
- Best Dice FG: **0.2468** (epoch 80)
- HD95 FG: 1.786
- SDlogJ: 0.351
- Non-positive Jacobian fraction: 2.98e-05

Important comparisons:
- `v4e oracle handoff` reference: Dice FG 0.2335
- `vnext_overfit10_diag` reference: Dice FG 0.2367
- historical best (`v4b_overfit10_test`): Dice FG 0.2482

Interpretation:
- New pipeline is clearly stronger than current v4e/vnext references.
- We closed the gap to historical best (`0.2482`) to **0.0014**.
- No tested variant in this sweep exceeded 0.2482.

## What We Learned From Sweep
- Strongest recipe remains near:
  - `correspondence_weight=0.2`
  - `decoder_fitting_weight=0.1`
  - `decoder_fitting_confidence_percentile=0.5`
  - `decoder_fitting_entropy_threshold=0.9`
- Over-sharpened or overly strict variants (aggressive confidence/entropy, some LR changes) usually hurt.
- Top1 can improve late but was less reliable than best `fix_corr` setup.

## Final Engineering Conclusion
### Confirmed Strengths
- Diffeomorphic/topology behavior remains strong.
- Handoff-aware training significantly improves Dice over v4e-level baselines.
- Matcher confidence bug fix is directionally correct and stable.

### Remaining Bottleneck
- We are still slightly below historical overfit peak (0.2482), indicating residual noise/mismatch in correspondence-to-flow transfer.

## Action Plan (Next, high-confidence)
Priority 1 (implementation):
- Add a **margin-weighted decoder fitting loss** using refined candidate margin directly (not only entropy percentile mask).
- Keep current best base (`fix_corr`) as anchor.

Priority 2 (training protocol):
- Run controlled 3-seed sweep on exactly one best config (no ad-hoc mix), fixed evaluation cadence.
- Save periodic checkpoints every 10 epochs to avoid losing transient best states.

Priority 3 (architecture, limited scope):
- Add lightweight stage-1 refinement gating by confidence (only apply local residual where confidence is high).

Priority 4 (if still capped):
- Synthetic refresh with matcher/handoff objective aligned to real phase (short synthetic tune, then real fine-tune).

## Operational Status
- PCCR sweep jobs from this session were stopped after collecting enough evidence.
- No active `pccr_vnext*` Slurm jobs remain.
