# Short/long context experiment: agreed design

2026-09-09. The user approved the memory organization and asked to increase both
the response-relation loss weight and the target distance. They subsequently
specified **the frozen tracker's original DR plus independent four-limb payloads**.
This replaces the earlier load-only requirement for this new experiment.

## Memory contract

- Short memory contains the latest 50 completed interactions of the current
  episode/motion. New interactions enter short memory first. Reset/motion
  boundaries clear short memory; missing entries are masked.
- Long memory contains the latest 30 disjoint valid chunks of 10 consecutive
  interactions, without overlapping short memory. Evicted short interactions
  accumulate into complete chunks before entering long memory.
- A reset flushes remaining complete chunks from the old trial into long memory,
  drops its incomplete tail and any invalid reset transition, and starts a new
  short history. No chunk may cross a reset/teleport.
- Long memory remains associated with one world and physical parameter session;
  a real parameter change invalidates affected memory. Empty long-memory slots
  require their own validity mask.
- A shared chunk encoder maps each 10-step chunk to one token. Attention over
  the 30 chunk tokens produces **one** long-memory latent. Suggested width is 128,
  matching the existing context transformer.
- The final context encoder combines 50 short tokens, one long-memory token and
  its CLS, then outputs the existing 64-D policy/predictor context latent.
- At most 350 distinct interactions are represented. A staging tail of fewer
  than 10 evicted interactions is not yet visible to the model. Memory fullness
  and prediction-window validity are separate: a padded short history must not
  automatically disqualify a query with usable long memory.
- The new episode cap is 1000 control steps for stage 1 and both stage-2 arms.
  Stage-2 baseline and latent policies must use matching environments.

The hierarchical architecture, replay and frozen inference are now implemented
in separate `memory350_*.py` files and the `forward_memory_train` entry point.
The original baseline sources are unchanged. Four-GPU training at 8192 environments
per rank is running on the full dataset without an update cap. See
[implementation and verified training status](memory350_training.md).

## Representation parameters

The loss/DR values are recorded in
[`configs/experiments/limb_context_memory350_representation.json`](../configs/experiments/limb_context_memory350_representation.json).
The hierarchical version uses the separate `scripts/run_memory350_stage1.py`
launcher and `forward_memory_train` entry point. Its smoke and formal workers
resolve these same values and save them in `run_config.json` and every checkpoint:

```text
--representation-weight 0.01
--representation-relation-weight 2.0
--response-distance-scale 0.75
```

The unchanged default representation weight need not be repeated in the generated
command. The effective loss is
`L_prediction + 0.01 * L_positive + 0.02 * L_relation`.
The response-to-target mapping becomes `2D / (D + 0.75)` instead of `2D / (D + 1)`.
At `D=0.5`, the target grows from `0.6667` to `0.8`; zero response difference still
has zero target distance. The latent distance uses normalized embeddings.

CLI, checkpoint, run configuration and W&B metadata retain the relation multiplier
and distance scale. Probe diagnostics include separate positive/relation losses,
valid-pair counts, actual/target distance means and per-batch p10/p50/p90. Distributed
logs average rank-local quantiles; they are not pooled global quantiles. Quantiles
are excluded from optimizer forwards when `compute_metrics=False`.

Legacy defaults remain `(0.01, 1.0, 1.0)`, preserving prior runs and command identity.
Older checkpoint loss configurations without the new multiplier normalize to 1.0
before strict resume comparison; a changed objective still requires a new run.

## Corrected DR source

Use the checkpoint named by `limb_context_protocol.TRACKER`, SHA256
`fd7bd90d5552e573bbbce1417e9b415c64bb487a76b683ba3c20503b5ec77635`, as the source
of the original DR configuration. Readback of its saved configuration confirms:

- Torso COM offsets: each axis in `[-0.075, 0.075]` m.
- Torso mass offset: `[-1, 1]` kg.
- Encoder bias: `[-0.01, 0.01]`.
- Foot friction: `[0.3, 2.0]`, using the checkpoint's shared-random setting.
- Joint armature multiplier: `[0.8, 1.2]`.
- Torso force pulses: interval `[3, 6]` s, duration `[0.3, 0.5]` s,
  force magnitude setting `[0, 10]` N.

Add independent `U(0,4)` kg loads on each hand and at each mid-shin, retaining the
audited attachment offsets and composite inertias. Preserve the checkpoint's
observation-noise, action and initial-state perturbation settings. Actor corruption
is enabled; critic corruption is disabled, as saved in the checkpoint. The original checkpoint uses the action
type `joint_position_with_mean_history`, with history length 8.

The new `tracker_dr_plus_limb_payload` profile in `limb_context_dr.py` bypasses both
`configure_load_only` and `_keep_startup_events`, preserving original events and
reset callbacks. It is wired through stage 1, PPO, resume and evaluation. A context
checkpoint must match PPO's DR profile; evaluation inherits the residual checkpoint
profile and rejects conflicting overrides. Final frozen-tracker evaluation also
receives this profile explicitly. Old layouts default to `load_only`, and their
existing job commands remain identical.

An endpoint of **0 kg still includes the original tracker DR**. The nominal
counterfactual simulator B remains the unperturbed model: exact A start, identical
physical PD targets, no payload/DR/pushes. It is supervision, not the 0-kg endpoint.

The GPU 0 simulator audit used 128 worlds and 500 control steps, without training:
12656 valid five-step paired windows, 3408 world-steps with active force pulses,
144 reset boundaries and 116 fixed-physics invariance checks. Encoder bias also
remained unchanged across resets. Nominal restore maximum absolute error was
`1.9073486328125e-06`. That earlier environment audit preceded the separate hierarchical
implementation and does not establish learned performance. The first audit exposed a checker issue:
MJLab expands friction geometry regexes into concrete names. The checker now resolves
the saved selectors independently before comparing them, retaining all range checks.
Evidence: `.runtime/limb_context_memory350_task2/tracker_dr_gpu_audit.json`.
All 71 relevant CPU tests passed, including selector expansion, frozen-tracker
evaluation routing, DR mismatch rejection and the representation-gradient tests.
The final source hashes and legacy command compatibility check are recorded in
`.runtime/limb_context_memory350_task2/dr_verification.json`.

## Previous context200 stop

The user authorized stopping the old GPU 0–3 training. All four workers handled
SIGTERM and exited normally after saving at update **24988** (99952 optimizer
steps). `last.pt`, the numbered checkpoint and `best.pt` are retained in
`runs/limb_context_20260909_context200/stage1/`. The completion record reports
`stopped: true`, `stopping_mode: until_user_stop` and best validation DR NMSE
`0.014418506529182196`.

The old pending `baseline_123` is held by its `legacy_queue_release.json`, reserving
GPUs 0–3 for the new experiment. GPU 4–7 processes belong to the other repository
and were not signalled. The old no-EE queue reports `film_123` as failed; its partial
run is not a completed seed-123 result.
