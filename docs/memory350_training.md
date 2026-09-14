# Memory350: separate context-encoder training version

2026-09-11：用户要求后续训练恢复 50% nominal A。修正版入口和新运行见
[Memory350 nominal50 训练说明](memory350_nominal50_training.md)。本页以下保留原 all-DR 实验记录。

2026-09-09. This implements the approved 50-step short memory plus 30 independent
10-step long-memory chunks, with the frozen tracker's original DR and independent
four-limb U(0,4 kg) payloads. All new files and outputs are inside this project.

2026-09-10: [paired memory intervention analysis at update 7500](memory350_effect_analysis_20260910.md)
finds that removing long memory raises held-out five-step prediction error by 79.1%;
the effect is stronger after a short-history reset. Historical context200 is not a
matched training control because DR, environment count and loss settings also differ.

[Inference cost benchmark](memory350_inference_cost_20260910.md): the current online
entry point takes a median 4.57 ms for one environment and 125 ms for 8192 on a
shared H100 during ongoing training. Frozen chunk caching is a concrete optimization
opportunity; the current raw-memory training implementation remains unchanged.

## Implementation

- `memory350_bank.py` keeps completed raw `(state71, applied_target29, next_state71)`
  transitions. New transitions enter short history first. Evictions accumulate
  in a pending 10-step chunk. Short and long memory do not overlap.
- Reset/motion discontinuities flush complete old-trial chunks into long memory,
  discard incomplete tails, and clear short history. No chunk crosses a reset.
  A physical-parameter session change invalidates only the affected world's memory.
- `memory350_replay.py` saves query-time committed-chunk counters and uses a separate
  raw chunk archive. Subsequent collection cannot leak future chunks into an old
  replay query. Overwritten queries are excluded. Every optimizer forward encodes
  raw history with current weights, avoiding stale learned features in replay.
- `memory350_model.py`: shared 10-interaction encoder (width 128, one layer), then
  attention over 30 summaries (two layers) produces **one** 128-D memory token.
  The final two-layer context encoder receives that token, 50 short tokens and
  CLS (52 tokens), and produces the existing 64-D latent. Predictor capacity is
  unchanged: width 512, six layers, history 10, horizon 5.
- `memory350_objective.py` allows representation learning with padded short history
  whenever short or long memory is usable. Same-world positives remain exact
  +/-5-step views within one episode/motion. The loss is
  `L_prediction + .01 L_positive + .02 L_relation`, with target `2D/(D+.75)`.
- `memory350_inference.py` strictly loads only the new encoder and normalization
  and implements the same causal memory rules for later frozen inference.

The separate entry point is `python -m intact_tracking.cli.forward_memory_train`.
The original encoder, replay, objective, trainer, rollout and frozen loader were
not modified for this implementation. Their pre-implementation SHA256 manifest
is `.runtime/memory350_v1/baseline_source_manifest.json`; launch checks every hash.
New checkpoints have architecture version
`nominal_counterfactual_short50_chunk10_long30_context_v1` and cannot silently be
loaded through the old architecture's loader.

## Training contract

Run root: `runs/limb_context_20260909_memory350`.
Launcher: `scripts/run_memory350_stage1.py`.

- GPU 0,1,2,3; each rank first uses 8192 A environments: 8064 training worlds and
  128 disjoint validation worlds. Counterfactual B uses the same number of state-
  and action-matched nominal simulator slots. A confirmed OOM is the only automatic
  reason to retry at 4096 A environments per rank (3968 training + 128 validation).
- Full dataset `/data_zcy/wxy/motion_data_correct/motion_data_full`, sorted disjoint
  rank shards, no exclusions. Actual file counts, frame counts and list hashes are
  gathered from all four initialized runtimes.
- All A worlds retain checkpoint DR, observation corruption and initial-state
  perturbations, plus hand and mid-shin loads. B remains the clean nominal model
  with identical A starts and physical PD targets. A 0-kg load is not a no-DR A world.
- The live environment and wrapper both use a 1000-control-step episode cap.
  Stage1 retains the original termination conditions, including `ee_body_pos`.
- No training-update limit and no automatic plateau stop. 8000 is only the cosine
  LR time scale; LR then stays at 1e-5. The launcher contains no stage2 jobs.
- Original optimizer settings: global batch 4096 (1024/rank), microbatch 256,
  four optimizer steps/update, BF16, AdamW 3e-4, warmup collection 500 control steps,
  motion-balanced replay capacity 262144/rank, fixed independent validation.
- Full validation every 100 updates; rank-0 training-loss progress every 10. Save
  every 250 updates and whenever validation improves. Each saved checkpoint checks
  exact model-parameter SHA256 agreement across all four ranks.
- Native W&B logging uses account `2486344338@qq.com`, entity
  `2486344338-zhejiang-university`, project `intact-forward-predictor`.
  Formal group is `limb_context_20260909_memory350-stage1`; short implementation
  checks use the separate `...-stage1-smoke` group.

## Verification and current state

CPU checks cover short-first nonoverlap, pending tails, reset flushing, short trials,
per-world parameter invalidation, historical-query causality, chunk/archive wrap,
padding masks, representation gradients under incomplete short history, strict
checkpoint loading and exact agreement between replay and frozen online inference.
See `.runtime/memory350_v1/core_tests.log`.

The four-GPU, 8192-env/rank BF16 smoke completed two updates/eight optimizer steps
without OOM. All three encoder attention stages had finite nonzero gradients, and
the checkpoint parameter-agreement audit passed. It used the full 20,095,126-parameter
model on one motion, so it is an implementation check, not a dataset-performance result.
Evidence: `smoke_8192/completion.json`, `metrics.jsonl`, `run_config.json` and `last.pt`.

The nominal simulator's repeated-rollout diagnostic flags rare contact-sensitive
outliers: the smoke's largest pose-component difference is 0.010657 rad in joint
position; the formal startup's largest is 0.020987 rad in joint position. The logged
pose p99 values (0.000141 / 0.000337) aggregate mixed state components and are not
meter-valued errors. The initial state restore error stays at a few 1e-6. These existing numerical
diagnostics are retained in logs and checkpoints; no exact repeatability claim is made.

Formal `stage1_8192` is running: all 129827 motions / 48085337 frames are loaded,
with rank motion counts `[32457, 32457, 32457, 32456]`. Startup verification observed
89 completed updates / 356 optimizer steps and read back update 80 from W&B.
The actual four workers, GPU assignment, 8192 environments/rank, 1000-step cap,
normalization/validation isolation, finite checkpoint weights and four-rank exact
parameter agreement were checked. Original baseline source hashes still match.
Evidence: `runs/limb_context_20260909_memory350/startup_verification.json`.
Subsequent observation reached 156 updates / 624 optimizer steps. The first
scheduled validation at update 100 completed with finite DR NMSE `0.2698445171`;
this is an early predictor metric, not a downstream policy-performance result.
All three saved context200 checkpoints were also rehashed and are byte-identical
to their stop-time records: `.runtime/memory350_v1/old_checkpoints_unchanged.json`.

[Formal W&B run](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/m350-07a735338e82).
The run is unbounded and continues independently of this chat turn. The launcher
PID and actual torchrun handle are stored in `launcher_process.json` and `state.json`;
`stage1/progress.json` reports current completed updates. At startup, GPU 1–3 also
had unrelated tasks using about 6.8 GiB each, and the 8192 configuration passed
without OOM. On September 10, the user explicitly requested their removal; PIDs
28067 / 28175 / 28335 exited after SIGTERM. The final check at update 7868 found
only the four formal memory workers on GPU 0–3. The formal process remains confined
to visible GPUs 0–3.
