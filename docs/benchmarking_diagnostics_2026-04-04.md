# PCCR v4 Local Diagnostics

Date: 2026-04-04

This note summarizes the local-GPU diagnostic runs that have actually produced usable outputs so far.

## Sources

- `v4a overfit`: `/u/almik/others/hvit/logs/pccr_v4a_overfit10_test/version_0/metrics.csv`
- `v4b overfit`: `/u/almik/others/hvit/logs/pccr_v4b_overfit10_test/version_0/metrics.csv`
- `v4c overfit`: `/u/almik/others/hvit/logs/pccr_v4c_overfit10_test/version_0/metrics.csv`
- `v4b freeze final refinement`: `/u/almik/others/hvit/logs/pccr_v4b_freeze_final_refinement/version_0/metrics.csv`
- `v4b freeze coarse decoder`: `/u/almik/others/hvit/logs/pccr_v4b_freeze_coarse_decoder/version_0/metrics.csv`
- `v4b freeze matcher`: `/u/almik/others/hvit/logs/pccr_v4b_freeze_matcher/version_0/metrics.csv`
- `v4a oracle decoder`: `/u/almik/others/hvit/logs/pccr_v4a_oracle_decoder/version_0/metrics.csv`
- `v4c residual only`: `/u/almik/others/hvit/logs/pccr_v4c_residual_only/version_1/metrics.csv`
- `v4b proxy20`: `/u/almik/others/hvit/logs/pccr_v4b_proxy20/version_0/metrics.csv`
- `v4b jac=0.0`: `/u/almik/others/hvit/logs/pccr/pccr_v4b_jac_0p0_jac_v4b/iter_eval/epoch_0010.json`
- `v4b jac=0.001`: `/u/almik/others/hvit/logs/pccr_v4b_jac_0p001_jac_v4b/version_0/metrics.csv`
- `v4a real-only`: `/u/almik/others/hvit/logs/pccr_v4a_real_only_sr_v4a/version_0/metrics.csv`

## Results

### 1. Overfit-on-10

Metric source:
- `/u/almik/others/hvit/logs/pccr/pccr_v4a_overfit10_test/iter_eval/epoch_0300.json`
- `/u/almik/others/hvit/logs/pccr/pccr_v4b_overfit10_test/iter_eval/epoch_0200.json`
- `/u/almik/others/hvit/logs/pccr/pccr_v4c_overfit10_test/iter_eval/epoch_0200.json`

| Variant | Best epoch | Dice FG | HD95 | SDlogJ | Nonpositive J |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4a | 300 | 0.2464 | 1.7696 | 0.5111 | 4.91e-4 |
| v4b | 200 | 0.2482 | 1.7252 | 0.4076 | 1.06e-4 |
| v4c | 200 | 0.2455 | 1.7284 | 0.4614 | 2.03e-4 |

Interpretation:
- None of the variants comes close to the target overfit regime (`Dice > 0.6`).
- `v4b` is the strongest of the three, but only by a very small margin.
- This points to a core bottleneck rather than a pure generalization issue.

### 2. Freeze test

Metric source:
- `/u/almik/others/hvit/logs/pccr/pccr_v4b_freeze_final_refinement/iter_eval/epoch_0030.json`
- `/u/almik/others/hvit/logs/pccr/pccr_v4b_freeze_coarse_decoder/iter_eval/epoch_0020.json`

| Trainable block | Best epoch | Dice FG | HD95 | SDlogJ | Nonpositive J |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final refinement only | 30 | 0.2275 | 1.8037 | 0.3407 | 1.28e-4 |
| Coarse decoder only | 20 | 0.2285 | 1.7659 | 0.3622 | 1.28e-4 |
| Matcher branch only | 10 | 0.2317 | 1.8021 | 0.3265 | 1.06e-4 |

Interpretation:
- Training only the coarse decoder works at least as well as training only the final refinement head.
- Training only the matcher branch is actually the strongest of the freeze runs so far.
- The refinement head alone is not the dominant missing piece.
- This shifts the bottleneck diagnosis a bit upstream: correspondence learning still matters at least as much as decoder fitting.

### 3. Oracle correspondence test

Metric source:
- `/u/almik/others/hvit/logs/pccr_v4a_oracle_decoder/version_0/metrics.csv`

Observed validation metrics:
- Best `val_dice`: `0.0651`
- Last `val_dice`: `0.0492`
- Last `val_avg_loss`: `2.3943`

Interpretation:
- As currently wired, the oracle test does not produce a strong decoder result.
- That suggests one of two things:
  1. the decoder/SVF fitting path is still a bottleneck, or
  2. this oracle setup is not yet a strong enough proxy for “perfect correspondence.”
- This result should be treated as informative but not final proof.

### 4. Residual-only test

Metric source:
- `/u/almik/others/hvit/logs/pccr/pccr_v4c_residual_only/iter_eval/epoch_0020.json`

Best observed:
- `Dice FG = 0.0800`
- `HD95 = 3.9009`
- `SDlogJ = 0.0823`
- `Nonpositive J = 0.0`

Interpretation:
- The residual head by itself does not carry registration.
- It stays extremely regular, but overlap is almost at identity level plus a tiny gain.
- This strongly suggests the refinement head is only a correction module, not a replacement registration path.

### 5. Jacobian-Dice sweep

Metric source:
- `/u/almik/others/hvit/logs/pccr/pccr_v4b_jac_0p0_jac_v4b/iter_eval/epoch_0010.json`

Two sweep points are available so far:

| Jacobian weight | Epoch | Dice FG | HD95 | SDlogJ | Nonpositive J |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 10 | 0.0832 | 1.7011 | 0.3353 | 1.20e-4 |
| 0.001 | 1 | val_dice=0.0940 | n/a | n/a | n/a |

Interpretation:
- The sweep is still incomplete, so no real trade-off curve can be claimed yet.
- But the first two points suggest that a tiny Jacobian term does not obviously improve the early proxy result over `jac=0.0`.
- The `jac=0.001` run does not yet have `iter_eval` JSON, so it is only a weak proxy point for now.

### 6. 20-epoch proxy

Metric source:
- `/u/almik/others/hvit/logs/pccr/pccr_v4b_proxy20/iter_eval/epoch_0010.json`

Best observed:
- `Dice FG = 0.0824`
- `HD95 = 1.7332`
- `SDlogJ = 0.3399`
- `Nonpositive J = 1.28e-4`

Interpretation:
- Early proxy quality is still low.
- This is useful mostly as a ranking tool, not as evidence of final quality.

### 7. Synthetic-only vs Real-only

Observed validation metrics from CSV:

| Setting | Best val_dice | Last val_dice | Best val_avg_loss |
| --- | ---: | ---: | ---: |
| v4a real-only | 0.0189 | 0.0079 | 3.4075 |
| v4a synthetic-only | not applicable | not applicable | not applicable |

Interpretation:
- The `real-only` run is currently very weak.
- Even with the limitation that this is CSV validation rather than `iter_eval`, the result is bad enough to be informative.
- This supports the idea that synthetic pretraining is not optional for this architecture family.

## Overall analysis

### Main takeaways

1. `v4b` is the best of the `v4a/v4b/v4c` family so far.
   - It wins the overfit test by a small margin.
   - It also has the cleanest regularity among the overfit trio.
   - It remains the strongest practical base even after the new runs were added.

2. The system still fails the most important diagnostic:
   - Overfitting 10 fixed pairs does not go beyond about `0.25 Dice FG`.
   - That is far below the target regime and means the bottleneck is fundamental.

3. The final refinement head is not the main missing ingredient by itself.
   - `coarse decoder only` is slightly better than `final refinement only`.
   - `residual-only` is dramatically too weak to carry registration.

4. The likely bottleneck is still in the global correspondence-to-flow pipeline.
   - Either the correspondence representation is not precise enough,
   - or the decoder is not exploiting it strongly enough,
   - or the interaction between the two remains too conservative.
   - The new `freeze_matcher` result strengthens the case that the correspondence branch still has meaningful headroom.

### Recommended next move

If only one direction should be pushed next, it should be:

- keep `v4b` as the base,
- improve the correspondence / decoder interaction,
- and do not expect the current residual branch to solve the overlap problem on its own.

Concretely, the current diagnostics argue more for:
- better oracle correspondence wiring,
- better decoder fitting under stronger correspondence signals,
- or a stronger local correction mechanism tied to explicit local matching,

than for simply tuning regularization again.

## Not yet ready for strong conclusions

- `synthetic-only vs real-only vs mixed` is not complete enough yet for a reliable ablation conclusion.
- `synthetic-only vs real-only vs mixed` is still incomplete as a full ablation, but the available `real-only` result is already clearly weak.
- `oracle correspondence` needs one cleaner pass before it can be treated as decisive evidence.
- the Jacobian sweep has started, but it is still too sparse for a real curve.
