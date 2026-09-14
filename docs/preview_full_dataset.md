# Full-dataset, four-limb DR preview comparison

This is the current experiment. Test preview first; do not launch latent variants
until the preview comparison has an efficacy result. Long training is still
user-controlled. The assistant's one-update runs are engineering checks only.

## Locked protocol

- Dataset: every recursive NPZ under
  `/data_zcy/wxy/motion_data_correct/motion_data_full`, without exclusions.
  Inventory on 2026-09-07: **129,827 motions**, 86,401,833,542 archive bytes.
  Runtime full load: **48,085,337 frames**.
  `AMASS_LAFAN_Qingtong`: 42; `sonic_filtered`: 129,785.
- Every file is loaded and eligible for uniform motion sampling; reference start
  time is sampled within the selected file. No small active subset or artificial
  equal weighting of the two top-level datasets. This means nearly all sampled
  motions come from `sonic_filtered`.
- Keep the checkpoint's `qpos_only_actor_fk` representation. Real and shadow
  commands share immutable motion arrays, but not reference cursors, histories,
  state, simulation data, or FK caches. The original action frontend is frozen.
- 4096 real environments per arm; preview additionally has 4096 shadows.
- Five future frozen-tracker steps at 50 Hz. The exact 1065-D v2 preview goes to
  **both actor and critic**. No additional scalar body/joint distances, no new
  current privileged height/contact features, no preview bottleneck.
- Baseline: original checkpoint actor/critic input groups only.
- Both critics start from scratch: 1024–512–512–256–1, no additive value branch.
- Original 20 reward terms unchanged. Reward SHA-256:
  `abee73632d079f08c160584b04ace15532ecf6bd9dfa8139979af518d8a1553c`.
- PPO: 24 steps/environment/update, 5 epochs, 4 minibatches; actor/critic LR
  1e-4/5e-4; default first 20 updates critic-only; actor std 0.1.
- Keep torso COM/mass, friction, encoder bias and armature DR as before. No
  reward edits, stronger armature range, pushes, or within-episode load switches.

### Payload profile `hands-shins-2-4kg`

Each of the four body payloads is sampled **independently, uniformly in 2–4 kg**
per world at startup, and stays fixed during that run. Total added mass is
8–16 kg, not 2–4 kg shared across limbs. It replaces the old right-hand 1–3 kg
profile; no old payload is stacked on top.

| Payload | Simulator body | Position in that body's local frame (m) | Cuboid size (m) |
|---|---|---|---|
| Left hand | `left_wrist_yaw_link` | (0.12, 0, 0) | (0.10, 0.08, 0.08) |
| Right hand | `right_wrist_yaw_link` | (0.12, 0, 0) | (0.10, 0.08, 0.08) |
| Left shin | `left_knee_link` | (0, 0, -0.15) | (0.10, 0.08, 0.10) |
| Right shin | `right_knee_link` | (0, 0, -0.15) | (0.10, 0.08, 0.10) |

The shin positions are a documented implementation choice near mid-shin, not
on the thigh or ankle. Payload mass, COM, principal inertia and inertial frame
are composed with the original body; no extra collision geometry is added.
Runtime audit checks actual added mass on all four bodies and records sampled
mass correlations and a per-world mass fingerprint. Preview copies the actual
expanded model fields, not merely the nominal payload parameters.

## Launch commands

Use free/appropriate authorized GPUs; the commands do **not** stop existing jobs.
GPU 0/1 currently have the older user-launched single-motion runs. Choose the
GPU and timing yourself. Use fresh output directories and choose iterations.

Preview:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m intact_tracking.cli.simulator_preview_train \
  --variant preview --num-envs 4096 --seed 121 \
  --motion-path /data_zcy/wxy/motion_data_correct/motion_data_full \
  --physics dr --dr-profile hands-shins-2-4kg \
  --iterations 100000 --save-interval 100 \
  --output-dir runs/preview_full/dr_preview_seed121
```

Original-input residual baseline:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 .venv/bin/python -m intact_tracking.cli.simulator_preview_train \
  --variant baseline --num-envs 4096 --seed 121 \
  --motion-path /data_zcy/wxy/motion_data_correct/motion_data_full \
  --physics dr --dr-profile hands-shins-2-4kg \
  --iterations 100000 --save-interval 100 \
  --output-dir runs/preview_full/dr_baseline_seed121
```

Repeat the pair with additional training seeds (e.g. 122, 123) before making a
robust learning-effect claim. Compare equal **completed PPO updates**, not equal
wall time. First 20 default updates do not train the actor. Checkpoint names are
zero-based: `checkpoint_100.pt` contains 101 completed updates.

Initial full-catalog NPZ loading takes several minutes (about 10 minutes in the
engineering check). Console `stage=motion_load` lines report progress. This is
startup loading, not PPO iteration time; shadow construction reuses the loaded
arrays rather than reading the corpus a second time.

To resume, repeat the same arguments with `--resume <own-checkpoint>`.
`--iterations` then means additional updates. DR profile, payload configuration,
dataset identity and learning configuration are locked. Do not resume an old
single-motion/right-hand run as the new experiment. SIGINT/SIGTERM saves after
the current complete PPO update; SIGKILL cannot save.

Dataset identity is an ordered **path/size/mtime manifest**, explicitly not a
content SHA of all 86 GB. Single-file checkpoints retain a content SHA. A changed
file list or ordinary file edit changes the manifest hash and blocks resumption.

## W&B

The existing project is
[intact-preview-v2](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2).
The full-dataset runs have group `full-data-preview-v2`, separate from the old
single-motion group. A CPU-only sidecar for `runs/preview_full` discovers new runs
and uploads scalar logs plus protocol metadata. It does not upload checkpoints,
code or motion data. State is `runs/preview_full/.wandb_sync/state.json`; log is
`runs/preview_full_wandb_sync.log`. The old `runs/preview_v2` sidecar is untouched.

If the full-data sidecar is not running, start it in a separate terminal:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -u scripts/sync_preview_wandb.py \
  --root runs/preview_full --project intact-preview-v2 \
  --entity 2486344338-zhejiang-university --poll-seconds 10
```

Use `iteration` as the x-axis. Training curves are not a fixed-start evaluation.

## Full-catalog paired evaluation

The evaluator batches files so each process has at most 4096 worlds including
repeats. With four repeats it evaluates at most 1024 motions per batch. All files
are covered; no random test subset is silently substituted. It evaluates the
residual baseline, preview, and frozen tracker under the **same strong DR**.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python scripts/evaluate_preview_dataset.py \
  --baseline-checkpoint runs/preview_full/dr_baseline_seed121/checkpoint_1000.pt \
  --preview-checkpoint runs/preview_full/dr_preview_seed121/checkpoint_1000.pt \
  --num-envs 4096 --repeats 4 --steps 500 --seed 20001 \
  --output-dir runs/preview_full_eval/seed121_update1001_eval20001
```

Equal completed training updates and matching training hyperparameters, original
tracker, dataset, DR and reward contracts are required. Motion IDs, starts,
physics fingerprints and checkpoint hashes are verified.
Completed batches can be reused by rerunning the exact command. Do not overwrite
the selected checkpoint during evaluation. `comparison.json` reports body/joint
ratios, paired motion-bootstrap 95% intervals, failure rates, newly failed and
rescued episodes, including preview versus frozen DR tracker. New failures are
not hidden by a better aggregate failure rate. The bootstrap describes variation
over motions, not variation over independent training seeds.

This tests in-distribution performance on the full training catalog, not unseen
motion generalization. It answers the present preview-versus-no-preview question;
it does not by itself establish matching a nominal-trained policy's performance.

## Engineering evidence (not efficacy)

- 42-motion, 64-world, four-limb DR consistency audit, 3 trials:
  `runs/preview_v2_validation_limb/multimotion_audit.json`.
  Real state/RNG untouched, reference alignment and first action exact, shared
  catalog verified. Max five-step body deviation 0.1335 mm; max joint L2 deviation
  0.001043 rad. Passed the existing bounded gate, not bitwise/strict replay.
- Copying all allocated simulation data also did not produce exact replay;
  that diagnostic is retained in `multimotion_full_state_audit.json`. It is not
  used to alter the training protocol or justify an efficacy claim.
- Full-catalog **4096-world** one-update checks completed in
  `runs/preview_v2_validation_limb/full_preview_4096` and `full_baseline_4096`.
  Each loaded all 129,827 motions / 48,085,337 frames, completed one full PPO
  update with actor learning enabled, and saved initial/final checkpoints.
- `checkpoint_pair_audit.json`: actual four-limb payload samples identical;
  all 53 original frozen-tracker tensors bitwise unchanged in both checkpoints;
  learned actor/critic state changed, all weights finite, reward SHA unchanged.
- Preview final checkpoint successfully reloaded and evaluated on 42 motions,
  16 steps each (`preview_reload_eval.json`). This is an execution check,
  not a meaningful tracking-performance benchmark.
- `batched_eval_smoke`: two bounded batches, 42 motions, two repeats, 16 steps,
  with baseline, preview, and frozen tracker. Pairing, aggregation and new-failure
  accounting completed successfully. Rerunning reused all six completed results;
  equal-update and reward-lineage checks passed. Short-run metrics are not
  efficacy evidence.
- 90 relevant regression tests passed; Ruff and `git diff --check` passed.

No preview learning improvement has yet been established under this new protocol.
