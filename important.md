1. The project, end to end
The repo carries the original H-ViT code untouched in src/model/ (used as a baseline), and your own work is a from-scratch model called PCCR (Pair-Conditioned Canonical Registration). Two generations live side by side: src/pccr/ (v5 line) and src/pccr_v6/ (v6 line).

Pipeline (per-pair forward, src/pccr/model.py):


source, target ─► SharedPyramidEncoder (TransMorph backbone)
                  └─ stage features {0:full, 1:1/2, 2:1/4, 3:1/8}
                       │
              ┌────────┴────────────────────────────┐
              ▼                                     ▼
   PairConditionedPointmapHead              StageVelocityDecoders (coarse→fine, stages 3→0)
   (canonical coords at stage 3)             • stage 3: uses pointmap match (raw_disp + conf + entropy)
              │                              • stage 2: StageLocalCorrelationRefiner radius=6
              ▼                              • stage 1: StageLocalCorrelationRefiner radius=6
   CanonicalCorrelationMatcher               • stage 0: zeros for conf/entropy → final residual head
   → MatchOutputs(raw_disp,                       │
                  confidence,                     ▼
                  margin,                  Final residual refinement (full-res)
                  entropy)                  • LocalCostVolumeEncoder radius=8 (channels 16 in v6, 8 in v5)
                                            • image-error inputs (raw, no edges)
                                            • SVF integrator (scaling-and-squaring, 7 steps)
                                                  │
                                                  ▼
                                          final displacement φ_s2t
Forward is run in both directions (S→T and T→S) so all symmetric losses are available — see model.py:229-277.

Loss (src/pccr/losses.py + criterion):
multiscale LNCC (window=9 at scales 1, 2) + Grad3D smoothness + neg-Jacobian penalty + InverseConsistencyLoss + CorrespondenceConsistencyLoss (forward+backward matcher displacements should cancel) + DecoderFittingLoss (decoder velocity must agree with confidence-gated matcher target) + DiceLoss on warped seg.

Training: Lightning DDP, 4 nodes × 4 A100 (16 GPUs), bf16-mixed, batch 1, grad-accum 4 (eff. 64), 394 steps/epoch, cosine LR with 5-epoch warmup, full-res 160×192×224 in v5/v6.

v5 vs v6 in one paragraph: v5 inherits pairwise_oasis_transmorph_pkl_full_final.yaml and only bumps segmentation_supervision_weight 0.05→0.20 and smoothness_weight 0.01→0.015. v6 (config/pairwise_oasis_transmorph_pkl_v6.yaml) widens cost-volume channels 8→16 (force-reinit on warm load), drops pointmap stage 2 (only stage 3), softens inverse_consistency 0.1→0.05, raises smoothness to 0.02, doubles residual_refinement_lr_scale to 2.0.

2. The two running jobs
26694773 (v6)	26694995 (v5)
Run	pccr_v6_v6_from_v5best_20260425_115233	pccr_v5_20260425_115302
Script	src/pccr_v6/scripts/train.py on v6.yaml	src/pccr/scripts/train.py on v5.yaml
Init	--resume_from_checkpoint last.ckpt (3rd requeue)	--checkpoint_path .../pccr_v5_20260420_130721/best-dice-epoch033-dice0.8067.ckpt (fresh schedule, strict=False)
Latest epoch in CSV	30 (started from epoch 24 of resume)	14
Best val_dice	0.7610 (epoch 24 iter-eval)	0.8115 (epoch 14)
Trend	Flat: 0.7510 → 0.7521 → 0.7559 → 0.7565 → 0.7537 → 0.7508 → 0.7529	Climbing: 0.8067 (init) → 0.8076 → 0.8084 → 0.8076 → 0.8114 → 0.8115
val_image (LNCC)	2.046–2.048 (saturated)	2.034–2.040
val_sdlogj	0.60–0.62	0.67–0.70
val_segmentation (1-Dice)	0.71–0.74 (worsening)	0.19 (good)
Errors	none — just torch.load weights_only=False deprecation warnings, expected SLURM warnings	same
Reading the flow: Both jobs started 2026-04-27 10:11. The slurm .out files are tiny because the model summary prints once and Lightning's progress bar isn't being captured — that's normal, not a bug. The CSV metrics under logs/pccr_v6_v6_from_v5best_20260425_115233/version_2/metrics.csv and logs/pccr_v5_20260425_115302/version_2/metrics.csv are the truth.

What is wrong:

v6 (26694773) regressed and is stuck. It started from your v5-best checkpoint at 0.8067, but the architectural reshape (cost-volume 8→16, pointmap stage 2 removed, refinement-LR doubled) effectively re-initialized those weights. After ~30 epochs and 13+ hours it has not climbed past 0.756 — it is converging to a lower local optimum than the v5 it was supposed to surpass. val_image is saturated (~2.046) while val_segmentation is increasing (0.713→0.746), meaning image similarity is plateaued and segmentation alignment is getting worse. That's a net-negative configuration.

v5 (26694995) is healthy. Improving 0.8067 → 0.8115 in 14 epochs, val_segmentation is steady ~0.19, no folding (val_nonpositive_jacobian_fraction ≈ 9e-4 ≈ 0.09%). With the cosine schedule still in early phase, this run will continue to gain.

No actual errors. The "Force-releasing locks for Run …" line in v6 is Aim recovering from the prior crash on requeue — benign but a sign the run has been requeued multiple times (version_2 confirms this is the third resume).

Both runs have higher SDlogJ than ideal. 0.6–0.7 is actually worse than H-ViT's 0.539 (lower=better for SDlogJ). The folding fraction is fine, but the deformation field roughness is high. This is regularization-related.

3. Does the model have potential? Yes — but you're spending compute on the wrong fork
The architecture is conceptually richer than H-ViT. H-ViT is a black-box pairwise flow regressor with self+cross attention. You have:

explicit canonical pointmaps with confidence/margin/entropy,
bidirectional inverse-consistency,
diffeomorphic SVF integration (scaling-and-squaring),
multi-scale local cost-volume refinement at stages 1, 2, and a residual head at stage 0,
a decoder-fitting loss that ties the displacement decoder to the matcher.
That gives you levers H-ViT doesn't have. The math case for parity-or-better is real.

Where you are: v5-best is 0.8115 dice vs H-ViT's 0.876. Gap = ~6.5pp. From the trajectories, v5 will likely reach 0.83–0.84 at convergence with no further changes. To close to ≥0.876, you need additional structural changes — see below.

4. How to beat H-ViT (concrete, ranked)
Operationally, today
Kill the v6 job (26694773) now. It's burning 16 GPUs to converge to a worse local optimum than v5. Your v5 (26694995) is the run that's actually improving. If you want to keep a v6 line alive, restart it from the current v5 best (0.8115), not the old 0.8067, and keep pointmap stage 2 (don't drop it). The "fewer pointmap stages = better" hypothesis is empirically losing.

Loss/training (1–3 weeks of work, +3–5pp dice each)
Push segmentation_supervision_weight further (0.20 → 0.35–0.50), and apply Dice on per-stage warped labels (stage 1 + stage 2 + stage 0), not only the final field. Right now the matcher branch sees no semantic signal — only the final-resolution Dice does. Direct gradient on the matcher from anatomy is the largest unused lever. See src/pccr/trainer.py — the warped seg is already computed for val, just plumb it into per-stage train loss.
Replace Grad3D smoothness with a hyperelastic / bending-energy regularizer of the form α‖∇v‖² + β(‖J‖² + ‖J⁻¹‖²). H-ViT trades SDlogJ for Dice; the right answer is a regularizer that allows large local deformation while penalizing inverse-Jacobian explosion. This is the standard lever LapIRN uses to keep SDlogJ at 0.072 instead of 0.539.
Test-time refinement (TTR). After the network predicts a velocity field, run 30–50 steps of LBFGS on LNCC + λ·smoothness over the velocity at full-res. This is what TransMorph-Bspl-style work uses for the final 1–2pp; it costs ~10s/pair at inference and is worth it on the leaderboard.
Multi-window LNCC. Currently just window=9 at scales 1 and 2. Try windows=[5, 9, 13] averaged. Small structures (hippocampus, amygdala) drive the mean Dice; window=5 captures them, window=13 stabilizes the cortex.
Focal Dice / per-region weighting. Mean over 35 structures hides the bottleneck. Weight smaller structures (look at per-label dice during val) ~2× — H-ViT's win is concentrated there.
Architecture (1–2 months, the real game)
Use the TransMorph-Large checkpoint already in transmorph_large_checkpoint/ as the encoder backbone. The current encoder stage_channels are modest (~13.4M total params); TransMorph-L is much larger and trained on similar data.
Symmetric inference and Lie-mean averaging. At test time, average ½(φ_s2t − φ_t2s∘φ_s2t) to enforce inverse-consistency at evaluation, not just training. Your bidirectional architecture is built for this — exploit it.
The PEARL-Reg direction in idea/plan.md (permutation-equivariant set encoder + co-visibility/matchability head + dense matching head) is genuinely paper-worthy — but that is not a "beat H-ViT on OASIS by Friday" effort. It's a 3–6 month research bet. The one piece I'd lift from it now is the explicit matchability output (a 3-class softmax: matchable / ambiguous / absent) as an extra head. It plugs into your existing confidence-weighting and is roughly two PRs.
Honest assessment
0.85 is achievable with v5 + cosine to convergence + (1) and (4) above. ~2 weeks of compute.
0.86–0.87 is achievable with (1)+(2)+(3) and a TransMorph-L backbone. ~6 weeks.
>0.876 with SDlogJ better than H-ViT's 0.539: that is the angle worth publishing — match H-ViT's Dice while having dramatically better topology. Your diffeomorphic SVF gives you a structural advantage there that H-ViT can't easily match. With (2) hyperelastic regularization + (3) TTR + a moderate-strength backbone, this is a real paper.
The model has potential. The current v6 fork doesn't, but the v5 line plus targeted loss changes does.

