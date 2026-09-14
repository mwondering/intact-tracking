# Single-motion, five-step simulator oracle (2026-09-07)

**Superseded for new training:** the user now controls training. Use
[the v2 setup and commands](simulator_preview_v2.md): 4096 real worlds, strict
original-input baseline, no scalar distance inputs, scratch single-path critics.
The rest of this file records the historical v1 experiment, not current defaults.

## Locked question

Can one DR-trained residual policy, above the unchanged frozen tracker, learn
better when it sees the five-step closed-loop future of that tracker from its
own current state, and the signed deviation of that future from the reference?
This is a privileged, nondeployable stage-one upper-bound experiment. No world
model and no context student in this experiment.

- Motion selected before new results: `lafan_qingtong/dance1_subject2.motion.npz`
  under `/data_zcy/wxy/motion_data_correct/motion_data_full/AMASS_LAFAN_Qingtong`.
- Horizon **5**, per user instruction (0.1 s at the original control frequency).
- Recompute tracker actions closed-loop at every shadow step, including its
  observed history. Do not hold the first action or use a stale motion-index tape.
- Shadow starts from the real residual-visited state, history, reference time,
  action memory and actual randomized physics; never modifies the real world.
- Original reward, physics distribution, PD/action mapping, limits and frozen
  tracker are unchanged. Original reward signature:
  `abee73632d079f08c160584b04ace15532ecf6bd9dfa8139979af518d8a1553c`.

## Controls

1. Nominal training, masked simulator future.
2. DR training, masked simulator future.
3. DR training, true simulator future.

All three use the same architecture, current state/physics privileges, future
reference information, initial residual, training budget and original critic.
Both DR arms compute shadow rollouts; only the future-outcome slots differ.
The critic initially receives no extra future group, isolating the actor prior.
**User update09:59:** future experiments must expose preview/latent information
to the critic too. The initial three runs remain the actor-only ablation. New
true/zero pairs use `PreviewAwareHeftCritic`: original6330-D value plus a
zero-output7971->256->128->1 branch receiving the SAME1641-D behavior-time
privileged tensor as the actor. Independent weights/normalization, no gradient
into actor. Mask outcomes once before both networks. Preview-critic defaults
on for newly launched preview runs; disable explicitly for the old ablation.
Evaluate all arms in identical nominal and DR test worlds; the primary goal is
DR-trained/oracle in DR versus nominal-trained/control in nominal. Also retain
the original tracker and the earlier fixed nominal policy as reference points.
Use matched starts and physical seeds; show errors, failures and new failures.
Positive evidence needs a matched training comparison, not only a post-hoc
latent shuffle or sensitivity. A single motion does not establish full-dataset
success. Within-episode physics jumps are a separate later stress test.

## Required implementation audits before performance claims

- Source and shadow model fields agree, including derived randomized fields.
- Branching leaves real simulation state, histories, commands and RNG intact.
- Replayed tracker rollouts agree with a real five-step tracker continuation;
  report errors, including contact-sensitive differences (no silent tolerance
  relaxation). Check reference offsets and action ordering explicitly.
- Future is behavior-time observation stored with the PPO rollout; minibatches
  cannot recompute it from another state or access the residual's own future.
- Frozen tracker weights and initial residual action identity are checked.
- Test true, zero and mismatched previews after training as secondary evidence
  of use; do not confuse those interventions with training-level benefit.

## Status

Training launch: seed121,512env,24steps,300updates,actorLR1e-4fixed,
criticLR5e-4,warmup20,std0.1,entropy0.0002,uniform/reference starts.
Three initial checkpoints have exactly identical72 actor and17 critic tensors.
Original reward SHA and empty reward_changes confirmed in all three artifacts.
Primary endpoint is `checkpoint_final.pt` after300updates (stored iter299).
Legacy checkpoint100/200 filenames correspond to101/201 completed updates;
these are diagnostic curves only, with equal counts across arms.

Implementation and two-update PPO smoke completed; 28 targeted unit/regression
tests passed. The reference rollout and initial action agree exactly; branch
queries leave real-state/history/RNG tensors unchanged. All expanded physical
model fields were compared exactly, including derived DR fields.

### Explicit numerical qualification (before efficacy training)

The original strict pose `1e-5` / full-state `1e-4` repeatability gate **failed**.
Copying all simulator arrays, including contacts, did not eliminate this. Same
shadow/same state repeats also differ around contacts, consistent with the
atomic parallel contact/constraint assembly in the GPU solver. We do not change
the physical solver to hide this, and do not claim bitwise future truth.

A separate, explicitly approximate-oracle acceptance rule was added and tested
on a fresh seed 12002, 64 worlds x 12 queries x 5 steps, in each physics mode.
This requires mean body/joint prediction error <1% of actual tracking error,
worst body point error <1 mm and worst joint-vector error <0.005 rad, plus exact
source-state/RNG preservation, physics equality and first-action/reference
agreement. Both modes passed. Across batches, maximum mean-error/tracking-error
ratios were DR body 0.00232%, joint 0.00418%; nominal body 0.00158%, joint 0.00187%.
Worst body point errors: DR 0.378 mm; nominal 0.402 mm. Worst joint-vector errors:
DR 0.00439 rad; nominal 0.00310 rad. Velocity outliers are larger (DR state max
0.166), so this is NOT a precision velocity-prediction claim.

Results: `runs/simulator_preview_single_motion/audit_*bounded_12002.json`.
Strict failures remain in `audit_dr_12001.json` / `audit_dr_full_12001.json`.
Efficacy training uses the bounded-numerical-preview interpretation, not a
misreported strict-replay pass.

### 512-world audit and user's numerical-accuracy clarification

At512worlds x8queries on seed12003 the former max-error caps did NOT pass:
worst body point1.614mm, joint-vector0.00708rad. Mean-error/tracking ratios
remained small: maximum batch body0.00121%,joint0.00165%. This is recorded as
`passed:false` in `audit_dr_bounded_512_12003.json`; do not overwrite that result.
User explicitly clarified that repeat error need not be zero, only sufficiently
small. Continue the efficacy experiment using a numerical simulator preview,
reporting these mean and tail errors, not claiming exact/strict oracle replay.
No simulator/reward modification was made to improve this audit. Future
conclusions concern learning with this measured noisy preview; body/joint/failures
are still evaluated in the real, unchanged environment.

GPU 0–3 were verified free after terminating the four occupying training jobs
under the user's explicit authorization. GPU 4–7 were not touched.
