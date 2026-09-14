Memory350 policy precision continuation, 2026-09-13.

The tracker-action experiment now supports a checkpointed policy precision setting. `--policy-precision fp32` selects IEEE FP32 matrix multiplication for the frozen tracker, trainable residual actor, and critic during collection, PPO updates, and evaluation. Evaluation and resume inherit the saved setting unless explicitly overridden. Legacy checkpoints retain TF32 by default so their original evaluations remain reproducible. The frozen context encoder retains its existing explicit BF16 inference and supplies stored FP32 latent tensors to PPO.

The numerical audit found that recomputing the same policy at rollout batch size 8192 and minibatch size 49152 introduced TF32 differences dominated by the frozen tracker's action mean. On actual observations tiled to those shapes, the 99th-percentile absolute probability-ratio error was 4.89%, with a maximum of 11.68%; the same shape comparison had zero error in IEEE FP32. This motivates a numerical consistency correction, without claiming a demonstrated learning benefit.

The user authorized unified policy precision. Both policies were stopped at complete update boundaries, with model, optimizer, normalization, four-rank parameter agreement, and adaptive sampler state verified. Baseline resumes from update 1589 on GPUs 0-3; latent resumes from update 1300 on GPUs 4-7. Both continue without an update limit. Their existing output directories and W&B run IDs are retained. Simulator episodes and context memory restart; learned weights and optimizer progress are restored.

The transition journal is `runs/limb_context_20260913_memory350_compressed_grid256_tracker_action_adaptive_scratch/precision_fp32_resume/`. Its `FP32_READY.json` records immutable resume checkpoints and the validated source snapshot. The original `PPO_READY.json` remains the historical scratch-run preflight. The current supervisor is `scripts/resume_memory350_policy_precision.py`. Periodic evaluation remains every 100 updates, with latent-use interventions every 1000. Comparisons omit updates where the two policies were evaluated under different numerical precisions.

Validation includes policy/evaluation regression tests, three precision-specific tests, and an explicit-FP32 simulator evaluation of the existing latent checkpoint (8 motions, 60 steps, finite metrics and full coverage). Startup restoration and resumed progress are recorded in the transition journal.

No MoE architecture has been implemented. The user requested discussion of actor and critic MoE designs before any network change.
