# LaFAN A/B/C tracker fine-tuning

## Question and scope

Starting from the same pretrained tracker, does a single load-randomized policy
lose endpoint tracking accuracy relative to a policy fine-tuned for that endpoint?
This is an in-distribution **fine-tuning** comparison. The source checkpoint was
itself pretrained with DR; A is not a tracker pretrained from scratch in nominal.
No preview, latent, residual action branch, new observations, or reward changes
are introduced in this experiment.

## Fixed protocol

| Arm | Additional mass on each hand and each shin |
| --- | --- |
| A | All four loads are 0 kg |
| B | All four loads are 4 kg |
| C | 25% of worlds all 0 kg; 25% all 4 kg; 50% independently U(0, 4) kg per limb |

Loads are assigned at startup and remain fixed within each world. Payloads alter
mass, center of mass, and inertia consistently; all other dynamics randomization
is removed. Original observation noise remains enabled in all arms.

- Data: 40 original-named LaFAN files under
  `/data_zcy/wxy/motion_data_correct/motion_data_full/AMASS_LAFAN_Qingtong/lafan_qingtong`.
  Two additional cropped files are excluded to avoid duplicate weighting.
  Each rank sees all 40 files through the same explicit manifest.
- Initialization: same source `checkpoint_72000.pt`, SHA-256
  `fd7bd90d5552e573bbbce1417e9b415c64bb487a76b683ba3c20503b5ec77635`.
- Actor: original control MLP and exploration std are fine-tuned. Original
  height/contact estimator, reference encoder, and all actor normalizers stay frozen.
- Critic: **all original weights and DecayVecNorm statistics restored**;
  original 6330 → 1024 → 512 → 512 → 1 architecture, all parameters trainable.
  Normalization statistics are globally pooled across ranks. No residual critic.
- Actor and critic are checked tensor-by-tensor against the source before learning.
- No critic warmup. Actor and critic both update from the first PPO iteration.
- Same seed 121, 2 GPUs × 2048 environments = 4096 worlds per arm.
- 1000 PPO iterations, 24 steps per world per iteration = 98,304,000 real
  transitions per arm. Five epochs, four minibatches, fixed actor LR 1e-5,
  critic LR 5e-4, entropy coefficient 0.0002; fresh optimizers in all arms.
- Uniform motion/time sampling, reference-state starts, rewind disabled.
- Original 20-term reward contract SHA-256:
  `abee73632d079f08c160584b04ace15532ecf6bd9dfa8139979af518d8a1553c`.

## Training and evaluation

```sh
OMP_NUM_THREADS=1 .venv/bin/python -u scripts/run_lafan_abc.py \
  --output-root runs/abc_lafan_20260907_r2 --iterations 1000 \
  --a-gpus 4,5 --b-gpus 6,7 --c-gpus 2,3
```

The orchestrator trains all three arms and then evaluates A, B, C and the frozen
source tracker in both nominal and hardest environments. Every checkpoint uses
the final 1000-update policy, without selecting on endpoint results.

Each policy/endpoint has three evaluation seeds (20001–20003), 40 motions,
8 starting points per motion, up to 500 control steps (10 seconds) per episode:
960 paired episodes per policy/endpoint. Motion IDs, reference starting frames,
per-step random seeds and actual physics fingerprints must match between policies.

Primary comparisons are C/A in nominal and C/B in hardest. Report body position
error, joint position L2 error, failures, new failures and rescued failures.
Errors average steps within each episode, then weight episodes/motions equally.
Failure-truncated trajectories are included, so error reductions must always be
interpreted together with failures and coverage. Paired bootstrap intervals
resample motions after averaging evaluation seeds. **One training seed only**:
intervals describe evaluation variability, not training-seed robustness.
The mixture also gives each exact endpoint fewer training transitions than its
specialist at the same total budget. A C deficit would establish headroom under
this budget/protocol, not prove that a single policy cannot represent both endpoints.

Results: `runs/abc_lafan_20260907_r2/comparison.json`; raw paired episodes:
`runs/abc_lafan_20260907_r2/eval/`. Each arm records runtime load audits, original
checkpoint restoration, unchanged rewards, unchanged frozen actor components,
and exact equality of final actor/critic states across the two ranks.

Training curves are mirrored to W&B project `intact-preview-v2`, group
`lafan-abc-finetune`, via a separate CPU scalar-only uploader. Training metrics
are not a substitute for the paired endpoint evaluation.

The earlier `runs/abc_lafan_20260907` attempt is incomplete and explicitly marked
invalid. A separate 100-update engineering run with actor LR 1e-4 degraded
nominal evaluation to 153/320 failures (frozen tracker: 1/320), motivating the
common conservative 1e-5 actor LR before any arm's formal 1000-update run.
