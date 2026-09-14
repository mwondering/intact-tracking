# Five-step preview experiment v2 — user-controlled training

For the current **full dataset + hands/shins 2–4 kg** protocol and launch
commands, see [preview_full_dataset.md](preview_full_dataset.md). The single-motion
commands below remain available as the historical small-DR setting; their defaults
are intentionally not silently changed for existing runs.

This replaces the v1 512-world, state/physics-privileged, warm-started/additive
critic experiment. Old checkpoints remain evaluable, but cannot initialize or
resume v2. The old `adaptation_train --simulator-preview` entry point rejects
new training and points to the entry point below.

## Locked comparison

| | Preview arm | Original-input residual baseline |
|---|---|---|
| Real training worlds (default) | 4096 | 4096 |
| Extra shadow worlds | 4096 | 0 |
| Frozen tracker | Original checkpoint, unchanged | Same checkpoint, unchanged |
| Actor residual input | Original tracker features + 1065 preview features | Original tracker features only |
| Actual residual input width | 1645 + 1065 = 2710 | 1645 |
| Critic input groups | Original `policy` + `priv` + preview | Original `policy` + `priv` only |
| Actual critic input width | 6330 + 1065 = 7395 | 6330 |
| Critic hidden widths | 1024 → 512 → 512 → 256 | 1024 → 512 → 512 → 256 |
| Critic initialization | Random weights and fresh normalization | Random weights and fresh normalization |

Both critics are one feed-forward scalar-value network, with LayerNorm/Mish
hidden blocks. No value addition, old-value branch, critic checkpoint weights,
or inherited normalization statistics. Actor/critic parameters and normalizers
are independent. Both consume the same raw extra observation in the preview arm.

The original actor's estimator/reference frontend is frozen and unchanged.
Residual hidden widths remain 512 → 256 → 128, output 29. The extra observation
is normalized and concatenated directly; there is no 64-D preview bottleneck.
Action mean is `frozen_tracker_action + 1.0 * tanh(residual_output)`; the
residual's output layer starts at zero. Gaussian exploration starts at std 0.1.

### Exactly what is extra

At each real control step, an independent simulator copies the actual current
state/history/reference time/DR physics, then executes the frozen tracker
closed-loop for five steps. It recomputes that tracker's action at every step.
The real residual's future actions are not used. The real world is not advanced
or rewound by the query.

The 1065-D vector is:

- 5 × 71 future target-state coordinates (355).
- 5 × (71 simulated state coordinates + 71 signed state-error coordinates) (710).

State coordinates are root position/quaternion/linear and angular velocity,
29 joint positions, and 29 joint velocities. Horizontal positions share the
current root origin; quaternion signs are aligned before subtraction.

The two scalar body/joint distance features have been removed from v2. They
remain only in the legacy decoder needed for historical checkpoints. Evaluation
can still report body/joint metrics; that does not make them policy inputs.

There is **no** extra current height/contact/physics group in either v2 arm.
The baseline has no preview group at all—not a zero-padded 1065-D input. Any
state/contact information already in the original checkpoint's critic remains
there, as requested. Both arms keep the original checkpoint observation schema.

PPO stores the preview with its behavior-time rollout. PPO minibatches do not
rerun the simulator. This is a privileged stage-one experiment, not deployable
inference, a context-encoder student, or a world-model training objective.

## Launch yourself

Run from `/data_zcy/wxy/intact-tracking`, in two terminals. The `100000` below is
an example budget; choose `--iterations` yourself. It is required, with no hidden
300-update limit or automatic model-selection/experiment queue.

Preview arm, physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m intact_tracking.cli.simulator_preview_train \
  --variant preview --physics dr --num-envs 4096 \
  --iterations 100000 --save-interval 100 \
  --output-dir runs/preview_v2/dr_preview_seed121
```

Original-input residual baseline, physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 .venv/bin/python -m intact_tracking.cli.simulator_preview_train \
  --variant baseline --physics dr --num-envs 4096 \
  --iterations 100000 --save-interval 100 \
  --output-dir runs/preview_v2/dr_baseline_seed121
```

Use a new/empty output directory for each fresh run. Each process uses one GPU;
4096 means real environments per arm, not the sum of real and shadow worlds.
This entry point deliberately rejects distributed `torchrun` to avoid silently
starting multiple uncoordinated copies. Other authorized GPUs can be selected
with `CUDA_VISIBLE_DEVICES`.

The default motion is
`/data_zcy/wxy/motion_data_correct/motion_data_full/AMASS_LAFAN_Qingtong/lafan_qingtong/dance1_subject2.motion.npz`.
Use `--motion-file` to choose another single motion. The default frozen tracker
is the original SPV5-2A `checkpoint_72000.pt`, not a previously trained residual.

### Stop and resume

Press Ctrl+C (or send SIGTERM) **after training initialization**. The handler
finishes the current PPO update, then atomically saves `checkpoint_interrupted.pt`
and `checkpoint_final.pt`. Wait for the process to exit; `kill -9` cannot save.
Regular `checkpoint_N.pt` files are also saved at the chosen interval.

Resume example, adding 5000 updates:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m intact_tracking.cli.simulator_preview_train \
  --variant preview --physics dr --num-envs 4096 \
  --iterations 5000 --save-interval 100 \
  --output-dir runs/preview_v2/dr_preview_seed121 \
  --resume runs/preview_v2/dr_preview_seed121/checkpoint_final.pt
```

Resume restores actor, critic, optimizer, normalizers, exploration std, and
completed-update/warmup progress. It starts at the next update rather than
repeating the last index. Simulator episodes restart; this is not bitwise
continuation of an old simulator snapshot. Keep the original model/PPO/seed/
environment-count flags when resuming; incompatible changes are rejected.
`--iterations` always means **additional updates for this invocation**.

`checkpoint_initial.pt` records zero completed updates. For legacy-style numbered
filenames, `checkpoint_100.pt` represents 101 completed updates. Read the explicit
`completed_updates` field when matching budgets; do not infer it from the name.

## Unchanged training settings

- Original rewards, termination rules, control action mapping and PD settings.
  No reward shaping, distance-based extra reward, BC, or prediction loss.
- Same DR: torso COM ±0.075 m/axis; torso mass additive ±1 kg; encoder bias
  ±0.01 rad; foot friction 0.3–2; armature multiplier 0.8–1.2; right-hand payload
  1–3 kg. No interval pushes or within-episode dynamics jumps.
- Uniform starts on this one motion; initial pose/velocity/joint perturbations
  disabled, observation corruption retained. Episodes last at most 10 s.
- 50 Hz control, five-step preview = 0.1 s.
- 24 rollout steps/world/update: **98,304 real transitions per update**.
- Fixed actor LR 1e-4, critic LR 5e-4; 5 PPO epochs, 4 minibatches; clip 0.2,
  gamma 0.99, GAE lambda 0.95, entropy coefficient 0.0002.
- Default first 20 updates train only the randomly initialized critic. Both
  arms use the same warmup; `--critic-warmup-updates 0` starts joint PPO immediately.
- Seed 121 by default; change `--seed` together across the comparison pair.

The experiment is an input/architecture comparison, not equal parameter count:
the preview arm naturally has a larger first layer. The baseline also avoids the
extra simulator cost. Compare equal real interaction/update budgets, not runtime.

## Evaluate either checkpoint

The existing evaluator now reads v1/v2 metadata and builds the correct wrapper.
For a v2 baseline it builds no preview simulator. Example:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 .venv/bin/python -m intact_tracking.cli.adaptation_eval \
  --checkpoint runs/preview_v2/dr_preview_seed121/checkpoint_final.pt \
  --motion-file /data_zcy/wxy/motion_data_correct/motion_data_full/AMASS_LAFAN_Qingtong/lafan_qingtong/dance1_subject2.motion.npz \
  --physics dr --seed 13001 --repeats 128 --steps 500 \
  --output runs/preview_v2/dr_preview_eval_13001.json
```

Replace checkpoint/output paths to evaluate the baseline with the same settings.
Both models can also be tested under `--physics nominal`. A nominal-trained
residual control can be launched explicitly with `--variant baseline --physics nominal`;
none is started automatically. The shared frozen tracker itself was pretrained
with DR, so these compare subsequent residual training, not two trackers trained
from scratch.

Training logs are TensorBoard event files in the output directory. Actual input
groups/dimensions and reward signature are stored in `run_config.json` and the
checkpoint metadata. `config.yaml` retains the source task definition together
with the effective agent configuration; runtime DR/start overrides are recorded
in `run_config.json`.

## Implementation verification (not a learning result)

- 186 related unit/regression tests passed; Ruff and `git diff --check` passed.
- Each arm completed one actual PPO update at4096 real worlds. Preview also
  allocated4096 shadows. No nonfinite model values; all53 frozen tracker tensors
  remained unchanged; original reward SHA remained unchanged.
- Preview checkpoint resumed successfully: completed updates1→2 and optimizer
  steps20→40. Both actor classes loaded in the evaluator and ran8 worlds×16 steps.
- SIGTERM at a4096-world update boundary saved a valid interrupted checkpoint;
  resuming it advanced completed updates and critic-warmup count1→2.
- Artifacts: `runs/preview_v2_validation_IumuuB/{preview_4096_fixed,baseline_4096_fixed,stop_4096_fixed}`.
  Unsuffixed earlier smoke artifacts exposed a serialization bug and are **not**
  valid handoff checkpoints; they were retained for diagnosis, not repaired in place.

No full training or automatic evaluation queue is running on the assistant's
behalf after this handoff. These short tests establish engineering integration,
not a tracking improvement or completion of the research goal.

## Live W&B mirror (added at the user's request)

Private project: https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2

`scripts/sync_preview_wandb.py` independently tails TensorBoard scalar events;
the running trainers were not restarted or modified. It backfills history and
checks for new events/runs under `runs/preview_v2` every10seconds. Separate log
directories become separate W&B runs, including user restarts with `_1` suffixes.
Model checkpoints, source code and motion data are not uploaded.

Current pair:

- Preview `_1`: https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/pv2-b07d9008d73b
- Baseline `_1`: https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/pv2-f7119ec7d41f

Earlier preview/baseline histories are separate runs in the same project. These
are training-rollout metrics, not fixed-start evaluation results. Charts use
`iteration`; the two TensorBoard `/time` curves instead use `elapsed_seconds`.
W&B's internal `_step` includes both row types, so use `iteration` for comparison.

Sidecar log: `runs/preview_v2/wandb_sync.log`.
Run IDs and per-tag sync cursors: `runs/preview_v2/.wandb_sync/state.json`.
To restart the sidecar if it exits, run in a separate terminal (a lock prevents
duplicate instances):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -u scripts/sync_preview_wandb.py \
  --root runs/preview_v2 --project intact-preview-v2 \
  --entity 2486344338-zhejiang-university --poll-seconds 10
```

Stopping this sidecar does not stop training; local TensorBoard logging continues.
Cloud summary and body/joint history were read back successfully to verify upload.
