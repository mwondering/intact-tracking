# DR tracking adaptation research

## 2026-09-07 01:23 UTC user override — original reward is fixed

All reward-shaped teacher/student results below are historical and do NOT
establish the newly required fixed-reward goal. Both stages need revalidation.
No reward additions, weight/shape changes or failure penalties are permitted.
The old +1 percentage-point failure guard is retired; observed failure increase
does not pass. Fixed-reward teachers restart from the original common tracker,
and stage two replaces privileges via context/deployable observations and
supervised distillation. See adaptation_live.md for the active experiments.

## Strongest stage-one confirmation: clean teacher800xx,19:31 UTC

Frozen`oracle_clean_payload_seed58/checkpoint_500.pt`, SHA256
`5a91d70ef4e8756796d074a7fbff75d8c5f9d5b325dc5121b8daf2267af31daf`.
Freeze19:26:31UTC precedes all80001/80002/80003 evaluations. Manifest
`eval_v2/teacher_clean_confirmation_freeze.json`, report
`eval_v2/teacher_clean_confirmation.json`. Same42motions,1008segments,
one shared DR policy, physical/noise distributions and nominalreference unchanged.
Teacher uses separate noise-free proprioception/FK histories as privileges;
the original noisy deployment groups remain unmodified.

AllTEN meanerrors are below fixed nominal:
bodyposition0.919807,jointL2 0.987757,anchorposition0.936881,
anchorrotation0.866448,bodyrotation0.980678,jointvelocity0.949513,
anchorlinearvelocity0.966284,anchorangularvelocity0.893585,
bodylinearvelocity0.959418,bodyangularvelocity0.931486.
AllTEN paired95%upper tracking-error ratios are below1.05 (largest1.030720).
This is strong evidence for stage-one average-tracking success, NOT equal
failure probability: observedfailures5/1008 vs1/1008,+0.397pp,
paired95%CI[0,+1.091]pp narrowly crosses the+1pp failureguard.
The combined all-metrics-and-failure interval flag is consequentlyfalse.

Stage TWO remains incomplete. Do not infer its success from the teacher.
Bestbroad studentdevelopment now has maxratio~1.086, still above1.02.
Currentexperiments target deployable reference preprocessing and stateestimation;
cohort600xx remains unused. See`adaptation_live.md` for currentjobs.

Earlier sections below are historical milestones.

## Latest valid milestone: broad teacher700xx confirmation,18:52 UTC

`oracle_payload_arm_seed57/checkpoint_250.pt` was frozen before all70001/2/3
evaluations (manifest`eval_v2/teacher_payload_confirmation_freeze.json`). It is
one shared DR policy, with unchanged physics/noise/force limits and all42motions.
Report`eval_v2/teacher_payload_confirmation.json`,1008matchedsegments:

| Error | Teacher / fixed nominal |
| --- | ---: |
| Body position |0.95613|
| Joint position L2 |0.99568|
| Anchor position |0.99289|
| Anchor rotation |0.93652|
| Body rotation |1.00820|
| Joint velocity |0.99811|
| Anchor linear velocity |0.99934|
| Anchor angular velocity |1.00306|
| Body linear velocity |1.00415|
| Body angular velocity |1.02902|

All ten MEAN ratios pass the predeclared1.05 engineering margin, as do the two
primary-error paired95%upper bounds (body0.98489,joint1.03334). This supports
the stage-one average-tracking objective. It does **not** establish strict
all-metric statistical noninferiority or equal failure risk: anchor andbody
angular95%upper ratios are1.07423 and1.07021. Observed failures6/1008 versus
nominal2/1008; difference+0.397pp, paired95%CI[-0.198,+1.190]pp crosses+1pp.

Stage two is still incomplete. Two matched-seed deployable students are being
distilled from this frozen teacher, with/without trainable policy MLP core,
same causal50-frame context inputs and0.05 loss-only velocity/payloadlabels.
No privileged actor inputs survive. Cohort60001/2/3 is still unused and reserved
for a frozen broad student. See`adaptation_live.md` for active run identities.

Earlier results below are chronological history, not the current best result.

> **Evaluation correction (2026-09-06 ~17:17 UTC): ALL v1 tracking evaluations
> below are exploratory, not final acceptance evidence.** A partial call to
> `env.reset(env_ids=...)` also advanced surviving worlds' reference cursors and
> pushed their observation histories. The v2 evaluator isolates finished-world
> resets and asserts survivor cursor/state/history invariance. Baselines and
> candidates are being re-evaluated under `runs/adaptation_goal/eval_v2/`.
> Previously reported primary-metric milestones require v2 reconfirmation.
> Training (normal auto-reset path), the grouped-world physics probe and the
> pointwise standalone export audit are not affected by this evaluator bug.

The first v2 GPU evaluations passed every cursor/state/history invariant,
including147 partial-reset batches for the original DR tracker. Development
body/joint errors: nominal0.030883m/0.484430rad; frozen DR0.044352m/0.626819rad;
teacher2500.029677m/0.474304rad; student2500.029727m/0.474660rad. Their failure
rates are0.298%,2.083%,0.595%,0.595%, respectively. Before seeing new confirmation
data, seeds **40001,40002,40003** were reserved for v2 re-confirmation of the
already frozen teacher250/student250 pair. These evaluations are now complete:

| Frozen model | Body error / nominal (95% CI) | Joint error / nominal (95% CI) | Failure difference (95% CI), percentage points |
| --- | --- | --- | --- |
| Compact teacher250 | 0.9793 [0.9527,1.0118] | 0.9708 [0.9406,1.0017] | +0.298 [-0.298,+0.992] |
| Context student250 | 0.9829 [0.9548,1.0176] | 0.9721 [0.9416,1.0034] | +0.298 [-0.397,+1.290] |

These establish a primary-error milestone, **not whole tracking equivalence**:
root-position error remains 1.428x/1.435x nominal, and velocity/orientation errors
also remain elevated. Student failure uncertainty crosses the +1pp guard. Further
training is now focused on the best broad oracle-feature teacher and complete
privilege removal from that teacher. Current development oracle-feature1250 has
worst ten-metric ratio1.103, substantially better than the compact teacher on
the non-primary metrics. See `adaptation_live.md` for active run identities.

V2 student250 development ablations and software export check:
- Normal: body0.0297269m,joint0.474660rad,failure0.595%.
- Zero latent: body0.0299348m,joint0.476808rad,failure1.190%.
- Cross-world rolled latent: body0.0297413m,joint0.475255rad,failure0.893%.
- Independent CPU TorchScript: body0.0296949m,joint0.474453rad,failure0.595%.
  All500-step reset/timeline invariants passed. Full-loop floating-point
  trajectories are not bit-identical (anchor mean differs by about8mm), while
  pointwise artifact equivalence was separately audited before export.
The latent has measurable but small tracking effects in these ablations; do not
claim it explains most improvement. The prior held-out-world physics probe is
valid and shows payload information is readable from the latent, but that is
not itself proof of a large causal control benefit.

## Single-network core interpolation experiment (~17:50 UTC)

`scripts/merge_adaptation_base.py` interpolates only the aligned `tracker.mlp`
weights. Receiver: oracle-feature1500; donor: compact metric-teacher500. Frozen
preprocessing normalizers must match bit-for-bit. Receiver privileged encoder,
residual head, input mode and critic are preserved. Inference remains one core,
one encoder and one correction; there is no ensemble or environment-based choice.

Development results for donor fractions0.25/0.50/0.75:
- 0.25: body0.028352m,joint0.473266rad,anchor0.248616m,fail0.595%,
  worst ten-metric ratio1.060(body angular velocity), currently best broad result.
- 0.50: body0.027049m,joint0.441140rad,anchor0.277290m,fail0.298%,
  worst ratio1.124(anchor position).
- 0.75: body0.026767m,joint0.426065rad,anchor0.298833m,fail0.595%;
  root regression prevents broad equivalence.

The moderate-reward continuation of oracle1250 reached body0.03066/joint0.5111
at250 but worst velocity ratio1.098. It was stopped, retaining250. GPU0 now
fine-tunes the merged0.25 single network: `oracle_merged_smooth_seed52`,
4096envs2500updates, tracking x4, action-rate restored to x1, primaryweight2,
auxiliaryweight8, actorLR2e-5. Hypothesis: recover velocity smoothness while
preserving the merged policy's primary-error headroom and nominal-like root path.
No new confirmation seeds have been consumed for these candidates.

At18:05UTC, before any new broad-confirmation data: reserve **50001/50002/50003**
for the next frozen broad teacher candidate, and **60001/60002/60003** for the
next frozen broad student candidate. Candidates will be chosen on development
only and identified by checkpoint SHA256 before their confirmation runs start.
These seeds are not used for training, interpolation weights, or checkpoint
selection. The earlier400xx compact-primary confirmation remains separate.

### Frozen broad teacher confirmation500xx (~18:19 UTC)

Frozen `oracle1750_base_merge020/checkpoint_0.pt`, SHA256
`34cd8a4e8fc3a8d17c892335d3a14c6c42ddd2c67f18b6b000b0f5234ef520c7`.
The freeze manifest predates all500xx outputs. In1008 matched segments:
- Body position ratio0.9320 [0.9109,0.9537], joint position0.9680 [0.9357,1.0020].
- Anchor position0.9869 [0.9146,1.0691], anchor rotation0.9422 [0.9081,0.9753],
  body rotation0.9929 [0.9617,1.0260].
- Joint velocity1.0242, anchor linear velocity1.0055, anchor angular1.0155,
  body linear1.0166. Body angular velocity remains1.05433 [1.02768,1.08329].
- Observed failures: teacher0/1008, nominal1/1008. Zero observed failures is
  not a claim of zero population failure probability.

Both primary criteria and failure guard pass. Nine mean metrics are<=1.025x,
but the strict all-ten<=1.05x guard still narrowly fails, and all-metric interval
equivalence is not established. **Do not change the threshold to call this a pass.**
Next candidates must not be tuned to500xx; use development and a fresh cohort
for any later broad confirmation. Full result`eval_v2/teacher_broad_confirmation.json`.

Authorized 2026-09-06: use physical GPUs 0–3; change code only inside this repository.
Dataset: `/data_zcy/wxy/motion_data_correct/motion_data_full/AMASS_LAFAN_Qingtong`.
The dataset contains 42 motion NPZ files. Existing runs used the larger parent dataset;
their training metrics are motivation, not the acceptance benchmark.

## Objective

1. Train one shared policy with unrestricted privileged observations so that average
   tracking errors under the existing DR distribution approach the nominal policy
   evaluated under nominal physics.
2. Replace the teacher's privileged observations with a context encoder and deployable
   observations. The context encoder must itself consume only deployable history.
   Privileged training targets and a privileged critic are allowed; privileged actor
   inputs at student inference are not.
3. If a candidate fails, diagnose and change the approach; launching training or
   improving training reward is not completion.

## Fixed comparison and acceptance (before new results)

- Retain checkpoint startup DR, including base COM/mass, friction, encoder bias,
  armature, and the prior experiment's fixed 1–3 kg right-hand payload. No random pushes.
  Never shrink DR to meet the objective. Nominal disables persistent DR and payload.
- Evaluate all 42 motions with equal numbers of deterministic-policy episodes per
  motion, matched start frames, at most 500 control steps, and independent physics seeds.
  No adaptive sampling or failure-based motion exclusions during evaluation.
- Development seed: 10001. Confirmation seeds: 20001, 20002, 20003, unseen in training
  and not used for selecting a candidate. Reference motions may have been seen in
  tracker pretraining; this is not a claim of generalization to unseen motions.
- Primary metrics: episode/motion-balanced mean body-position error (m) and joint
  position L2 error (rad, the repository convention). Also record anchor position and
  orientation, body orientation, joint/body/root velocities, failures and coverage.
- Stage 1 provisional equivalence margin: both primary errors at most 1.05 times the
  nominal baseline, with failure probability no more than 1 percentage point higher.
  Stage 2 target: at most 1.02 times nominal for both, same failure guard. Report exact
  ratios, per-seed variation and paired bootstrap intervals, even if margins pass.
- Freeze a valid nominal checkpoint before candidate selection. Include the existing
  nominal residual checkpoint and an equal-training-budget nominal control to avoid
  declaring success against a deliberately weak baseline.

## Initial evidence and approach

Existing residual PPO freezes the large tracker and clips its correction with
`0.25*tanh`; observed corrections reach this bound. The current latent actor consumes
raw 71-D simulator state, so it does not satisfy the student input contract.

First candidates: shared privileged residual policies with larger correction capacity,
explicit physical parameters and true state/tracking errors. Warm-start from the
existing tracker; compare with nominal and nonprivileged controls. If correction
heads remain limiting, unfreeze the policy and/or change the optimization objective.

Teacher-to-history adaptation follows the general RMA/A-RMA idea, with further
student PPO or action distillation if latent regression alone loses tracking accuracy:
https://arxiv.org/abs/2107.04034 and https://ashish-kmr.github.io/a-rma/ .

## State

- GPU/PyTorch/Warp environment checks passed; GPUs 0–3 were cleared as authorized.
- User approved the objective and clarified complete replacement of privileged inputs.
- Research implementation and fixed evaluation are in progress.

## Initial fixed evaluation, development seed 10001

336 matched segments (8 per motion), 500-step maximum; all start frames verified equal.

| Policy / physics | Body position (m) | Joint position L2 (rad) | Failure rate |
| --- | ---: | ---: | ---: |
| Original tracker / nominal | 0.031559 | 0.498774 | 0.595% |
| Existing nominal residual, update 1000 / nominal | 0.030918 | 0.484586 | 0.298% |
| Original tracker / DR + payload | 0.045566 | 0.643843 | 2.679% |

The existing nominal checkpoint is
`runs/residual_policy_no_latent_nominal_v13_run2/checkpoint_1000.pt`.
The initial body/joint DR-to-nominal ratios are 1.474 and 1.329, respectively.
Full metrics and paired motion-bootstrap intervals: `runs/adaptation_goal/initial_comparison.json`.

## First training batch

Each run starts from the same original tracker, uses all 42 motions, seed 42,
1024 environments, 24 rollout steps, 2000 PPO iterations, 5 PPO epochs, identical
reward/termination, and the original adaptive motion sampler. Outputs are under
`runs/adaptation_goal/`. New experiments use local TensorBoard; no external logging.

| Run | GPU | Privileged actor branch | Residual scale |
| --- | ---: | --- | ---: |
| nominal_scale1_seed42 | 0 | no | 1.0 |
| dr_control_scale1_seed42 | 1 | no | 1.0 |
| oracle_scale2_seed42 | 2 | yes, 64-D learned bottleneck | 2.0 |
| oracle_scale1_seed42 | 3 | yes, 64-D learned bottleneck | 1.0 |

Teacher privilege consists of the original critic's 4602-D `priv` group plus a
495-D vector of actual model-relative mass/COM/inertia/armature/friction and encoder
bias. It is normalized and encoded into 64 values; only this branch supplies privileged
actor information. Base tracker features remain frozen and deployable.

`scripts/watch_adaptation_evals.py` monitors the explicitly identified training PIDs
and evaluates each stable 250-iteration checkpoint on the development seed. It records
subprocess completion/failures in `runs/adaptation_goal/evaluation_watch.log`.
Do not mistake development evaluations or successful training for completion.

Student history-encoder structure and input-isolation tests are prepared; student
training has not started, pending the first-stage result.

## Development decisions, 15:32 UTC

At update 250, oracle scale 1 improved body error to 0.04000 m (vs DR control
0.04255 m). At update 500 it remained 0.039955 m, with joint error 0.644269 rad;
both still substantially exceed the frozen nominal acceptance baseline. The
scale-2 candidate was no better (0.04077 m / 0.65288 rad) and was stopped; its
checkpoint 500 and all measurements remain available. Oracle scale 1 and both
controls continue, so this decision does not assume the first method cannot
improve with more training.

New GPU-2 run: `compact_unfrozen_tracking2_seed42`, seed 42, 2048 environments,
2000 iterations, initial actor LR 5e-5. Its privileged branch uses current true
robot state excluding absolute x/y, contact state and the same physics vector
(566 dimensions total), without privileged future reference windows. The actor
MLP can be fine-tuned; its sensor and reference preprocessors stay frozen.
Tracking reward weights are multiplied by 2 and action-rate weights by 0.5.
Evaluation physics, motion distribution, error metrics and thresholds are unchanged.
Training rewards from this run are therefore not directly comparable to the first batch.

Three independent-seed results for the existing nominal checkpoint:

| Seed | Body position (m) | Joint position L2 (rad) | Failure rate |
| --- | ---: | ---: | ---: |
| 20001 | 0.03003 | 0.47501 | 0.000% |
| 20002 | 0.03077 | 0.47901 | 0.000% |
| 20003 | 0.03053 | 0.47525 | 0.298% |

Preparation for stage two now includes an on-policy teacher/student action
distillation CLI. Its optional latent regression weight defaults to zero, since
a teacher latent may mix information that cannot be uniquely recovered from
sensor history; task actions, conditioned also on deployable reference observations,
are the primary supervision. This CLI has not yet been run or validated in an
environment and is not a claimed experimental result.

## Further development, 15:57 UTC

- First-generation oracle scale 1 plateaued: body error 0.03986 m and joint
  error 0.68861 rad at update 1250. It was stopped; checkpoints through 1250
  and completed evaluations remain. Increasing reward in this run had not been
  attempted; it retains the original training objective.
- Compact, unfrozen, tracking-weight-2 teacher at update 500: 0.03860 m body,
  0.61902 rad joint, 0.893% failure. It continues. This is improvement, not success.
- An analytic gravity/Coriolis torque-difference compensation probe was worse:
  0.05773 m body, 0.97618 rad joint, 8.631% failures. It is not a learned-policy
  result and is not used for acceptance. The separate nominal-twin zero-correction
  identity audit passes after handling invariant singleton model-field rows.
- Substituting true height/contact and clean reference into the original tracker,
  without training, gives 0.04377 m / 0.62169 rad / 1.488% failure in DR. This small
  improvement does not explain the whole nominal/DR gap.
- The next GPU-3 candidate uses this oracle tracking-feature route, full policy-MLP
  fine-tuning, tracking weights times 4, action-rate weights times 0.25, and a larger
  4096-environment batch. Its feature path is privileged and must also be replaced
  for stage two. The distillation implementation now targets complete teacher
  actions relative to the student's own deployable base features to support this.

All tracked repository code from before the research remains unmodified. Research
code is in newly added modules/scripts/tests; existing untracked `wandb/` and the
space-named directory are preserved. No source outside this repository is edited.

## Metric-aligned training, 16:13 UTC

Compact unfrozen teacher update 750 improves body error to 0.03744 m, but
joint L2 remains 0.61746 rad. Source inspection found that the original joint
tracking reward uses mean absolute error over joints, unlike evaluation's joint
L2. The keypoint training reward also uses different semantic points/alignment
from evaluation. New optional rewards optimize exponential transforms of the
actual joint L2 and yaw-aligned equal-body position error (scales 0.4 rad and
0.03 m). Reward timing retains the environment contract; evaluation is unchanged.

`metric_aligned_compact_seed45` starts from compact checkpoint 750 with a fresh
optimizer, 4096 environments, 4000 updates, two new reward weights of 10,
original tracking weights, action-rate weights times 0.25, and uniform motion
sampling without rewinds. It shares GPU 1 while the original DR control finishes.
The first short runtime check exposed a multi-world quaternion broadcasting
error; this was fixed and covered by a batch-size-two unit test. The revised GPU
smoke completed two PPO updates successfully. Failed smoke logs remain recorded.

Optional evaluation diagnostics now report per-joint RMSE and the fraction of
final-substep actuator forces above 95% of the force limit. Compact update 750
has its largest errors in the right wrist pitch/yaw (0.227/0.225 rad RMSE), with
6.94%/4.33% saturation fractions, respectively. This is not yet evidence that
the requested average performance is physically impossible. The diagnostic
rerun differs slightly from the first evaluation (0.03738 m, 0.61652 rad,
2 instead of 3 failures), so seeded GPU simulation should not be described as
bitwise reproducible; independent confirmation seeds remain essential.

At 16:17 UTC the nonprivileged DR control was stopped after its saved update
1750. Its development joint error had worsened from 0.644 at update 250 to
0.703 at update 1500, while body error stayed around 0.0405 m. All checkpoints
and evaluations are retained; nominal/control comparisons can use matched
updates through 1750. Concurrent trainers on GPU 1 tripled collection latency,
so this releases that GPU to the metric-aligned candidate. This is a recorded
resource/early-stop decision, not a claim that the original 2000-update plan
was completed. Nominal control continues to its planned endpoint.

## First primary-metric confirmation, 16:31 UTC

`metric_aligned_compact_seed45/checkpoint_250.pt` is frozen as the first
teacher passing the preregistered **two-primary-metric + failure** criterion.
Three confirmation seeds (1008 balanced segments total), compared with the
original frozen nominal acceptance checkpoint:

| Measurement | Nominal | DR teacher | Ratio / difference |
| --- | ---: | ---: | ---: |
| Body position | 0.030444 m | 0.029834 m | 0.97998, CI [0.95455, 1.01472] |
| Joint position L2 | 0.476425 rad | 0.465322 rad | 0.97669, CI [0.94534, 1.00944] |
| Failure probability | 0.0992% | 0.3968% | +0.2976 pp, CI [-0.2976, +0.8929] pp |

Evidence: `runs/adaptation_goal/teacher_250_confirmation.json`. This is **not**
equivalence of all tracking metrics: global anchor position remains 46.6% higher,
and velocity/orientation metrics are 9–19% higher. This limitation was explicitly
reported to the user. Stage two can now test complete privilege replacement for
the confirmed primary-metric teacher, while the other teacher candidates continue
addressing broader tracking quality. The entire user goal is not marked complete.

The compact tracking2 trainer was stopped after saved/evaluated update 1250
(0.03479 m / 0.58539 rad) to release GPU 2 for context adaptation. All artifacts
remain. The nominal control completed 2000 updates normally; its final checkpoint
is weaker than the original acceptance baseline, which is **not** changed.

Stage-two final confirmation will use fresh seeds **30001, 30002, 30003**, chosen
before any student training results. Development remains seed 10001. Teacher
confirmation results will not be reused as the student's final held-out evidence.
Distillation learns teacher actions from deployable history and observations;
latent regression defaults to zero because the privileged latent also contains
state information (e.g. global heading) not identifiable from history alone.

## Student primary-metric result, 16:41 UTC

`context_action_distill_seed43/checkpoint_250.pt`, evaluated after actually
removing privileged observation keys on every action, has development body
0.029737 m and joint L2 0.475436 rad, failure 0.893%. New confirmation seeds
30001/2/3 give body 0.030366 m vs nominal 0.030770 (ratio 0.98687, CI
[0.95821,1.01847]) and joint L2 0.475377 vs 0.480585 (ratio 0.98916, CI
[0.95507,1.02525]). Failure difference +0.5952 pp, CI [-0.0992,+1.4881] pp.

Thus the original **point-estimate** primary criterion passes, but the more
conservative 95%-interval evidence does not fully fit the 2% margin, and other
tracking metrics still lag. This is a measured milestone, **not whole-goal
completion**. Detailed report: `runs/adaptation_goal/student_250_confirmation.json`.
Training continues. Zero/shuffled-context rollouts are running to measure actual
context dependence. Do not treat a small action-imitation loss as sufficient
evidence that the latent is an informative environment encoding.

The new optional `--auxiliary-tracking-weight` adds a reward averaging eight
normalized error exponentials: anchor position/orientation, body orientation,
joint velocity, anchor linear/angular velocity and body linear/angular velocity.
Scales (0.25,0.075,0.125,4.4,0.21,0.45,0.267,1.07) were selected from nominal
development magnitudes. These use the evaluation conventions and the normal
environment reward timing. A GPU smoke with weight16 passes; unit tests cover
both qpos-only and full-reference velocity conventions. Total relevant tests:
20 passed. No physics/observation scope or evaluation metrics were changed.

## Latent identifiability and inference artifact

Context250 latent probe uses an independent physics seed14643 and holds out
physical worlds, keeping all temporal observations of each world together
(384 training worlds,128 heldout,18 samples/world). A ridge readout of current
latent predicts payload mass with heldout R²0.718 and RMSE0.318kg, versus
R²0.045 for a current-sensor ridge baseline. Averaging latent within each episode
gives R²0.866. Other parameter readouts are weak; this supports payload encoding,
not identification of every DR parameter. Zero/shuffle rollout ablations have
only small performance effects, so most measured improvement cannot be attributed
to context adaptation alone. See `context_250_physics_probe.json` and ablation
JSON files within the context-distillation run.

`context_export.py` and its CLI create a self-contained CPU TorchScript action
module with an8199-dimensional flat deployable input and29 normalized outputs.
It contains learned preprocessing, the context encoder, base MLP and residual,
not the privileged critic. Input ordering is quaternion4, term-major50-frame
sensor history6100, known reference input1900, noisy-sensor-FK key bodies195.
History sensor dimensions are(29,29,3,3,29,29), oldest to newest within each
term. The caller owns history buffers and the original action scale/offset/order.

Prototype artifact: `runs/adaptation_goal/context_export_smoke/policy_cpu.ts`.
96 real-observation action checks across multiple batch sizes pass, max absolute
difference1.75e-5. A fresh isolated Python process importing only torch (no
intact_tracking modules) loaded it and passed the same checks. Full336-segment
closed-loop CPU-artifact evaluation: body0.029842m, joint0.476317rad, failure
1.190%; small differences from GPU inference are expected from floating-point
simulation divergence. This validates the software inference path, not hardware
safety or completion of the broader tracking goal.
