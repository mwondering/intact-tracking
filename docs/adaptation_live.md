# Live research handoff

## Current override — full dataset / four-limb DR; preview first

See `docs/preview_full_dataset.md`. User confirmed independent 2–4 kg on each
hand and each SHIN, replacing the old right-hand-only payload. Entire requested
root contains 129,827 NPZs (42 AMASS/LAFAN/Qingtong + 129,785 sonic), 48,085,337
loaded frames. Do not change rewards, frozen tracker, or other DR ranges. First
test raw five-step preview to both actor/critic against the original-input
residual baseline; no latent experiment yet. User reiterated efficacy is the
focus, not chasing exact simulator replay. Formal training remains user-controlled.

Implemented `preview_protocol.py`, scoped immutable motion sharing, multi-motion
train/audit CLI, explicit DR profile with resume lock, bounded full-catalog paired
evaluation including frozen DR tracker, W&B full-data group. Only repo files
modified. Existing GPU 0/1 user training and old W&B mirror are untouched;
foreign jobs on GPU 2/3 were not killed. One-update engineering jobs only.

New full-data W&B mirror PID42885/session72074 watches `runs/preview_full`;
log `runs/preview_full_wandb_sync.log`, state `.wandb_sync/state.json` under root.
Same private W&B project. Old mirror PID61200/session48062 remains running.

Engineering checks: 90 relevant regression tests passed; 42-motion/64-world
strong-DR simulator audit passed existing bounded gate (not strict bitwise).
Full-state copy diagnostic did not eliminate numerical deviation; no asserted
root cause. Full preview and baseline 4096 one-update runs completed with original
rewards and all 129,827 motions; preview had 4096 real + 4096 shadow environments.
`checkpoint_pair_audit.json` verifies identical actual payload samples, all
53 frozen tracker tensors bitwise unchanged, finite updated actor/critic state.
Preview reload/evaluation succeeded on 42 motions x 16 steps. Batched evaluator
smoke test completed two batches with baseline, preview and frozen DR tracker.
Exact rerun reused all six results; matched completed-update/hyperparameter and
reward-lineage checks passed. Full train/eval/mirror launch commands are ready in
the new protocol document. No long full-data training was launched.
All artifacts under `runs/preview_v2_validation_limb`, not formal W&B training.

No full-data preview efficacy conclusion yet. Compare equal completed updates,
paired fixed starts/DR, body/joint CIs and new failures; do not confuse short
engineering updates with evidence of learning benefit or nominal matching.

## Current override — W&B live upload; performance investigation paused

User explicitly asked to upload logs to W&B and set speed work aside. No profiling
program or production speed change was launched in the interrupted investigation.
An independent CPU TensorBoard-to-W&B sidecar is now running, PID61200, exec
session48062; user training processes were not stopped/restarted by the assistant.
Project `2486344338-zhejiang-university/intact-preview-v2` was created PRIVATE.
Script `scripts/sync_preview_wandb.py`, poll10s, root `runs/preview_v2`.
Live state/run links: `runs/preview_v2/.wandb_sync/state.json`; log `wandb_sync.log`.
Four directories uploaded separately; newly created baseline `_1` auto-discovered:
preview `_1` id `pv2-b07d9008d73b`, baseline `_1` id `pv2-f7119ec7d41f`, historical
preview id `pv2-1a755e0220f3`, historical baseline id `pv2-a8738f008d76`.
Cloud summaries/history confirmed (preview `_1` atiteration62, baseline `_1`24
at verification; continuing to advance). Original scalar tags retained, charts
use `iteration`, `/time` duplicate curves use `elapsed_seconds`. Do not compare
W&B internal `_step`, which also counts time-axis rows. Only scalar logs and
selected experiment config uploaded; no checkpoints/code/dataset/credentials.
Two sidecar regression tests and Ruff passed. Preserve the sidecar unless the
user asks to stop it; do not resume autonomous training or speed optimization.

## Current — Sep 7: user takes over training; v2 implementation handoff

The user explicitly requested code/launch commands, not autonomous experiments.
Do NOT restart training/evaluation queues or resume the old research goal on your
own. The historical v1 trainers finished; remaining v1 evaluation queue2711 and
its scoped child30775 were stopped for the handoff (queue58458 had already exited).
No checkpoints or completed results were deleted; GPUs4–7 were not touched.

New entry point: `python -m intact_tracking.cli.simulator_preview_train`.
Commands/specification: `docs/simulator_preview_v2.md`.
Default4096 real envs per arm; preview adds4096 shadows. Exactly two requested
arms: original tracker actor/critic input residual baseline versus additional
five-step simulator target/state/signed-error information for BOTH networks.
Extra dimension1065, no old566 current privilege group or scalar body/joint
distance features. Actor residual inputs1645 versus2710; no64-D preview bottleneck.
Both critics are scratch single-path HEFT MLPs6330/7395->1024->512->512->256->1,
fresh normalization, no checkpoint value weights or additive value branches.
Original rewards/frozen tracker unchanged. User chooses `--iterations` explicitly;
Ctrl+C/SIGTERM saves at a complete PPO update boundary. Resume restores optimizer,
normalizers, action std and update/warmup count; physics episodes restart.

Only bounded engineering validation was launched by the assistant, under
`runs/preview_v2_validation_IumuuB/`. The initial4096 smoke exposed RSL's in-place
configuration mutation (missing class_name in saved artifacts). Fixed by copying
the config in ResidualOnPolicyRunner; do not use the unsuffixed first smoke
checkpoints. Corrected runs use `preview_4096_fixed` / `baseline_4096_fixed`.
Both completed4096-world PPO/save tests; preview resumed for one further update;
both checkpoints evaluated on8worlds x16steps (load/integration tests ONLY).
The actor tracker's53 tensors remain bitwise unchanged. Scratch critic input
shapes and original reward SHA checked. No learning-efficacy claims from these
short tests. The v1 preliminary tracking results below remain historical and
do not describe v2 performance. Old v1 checkpoints are evaluation-only for v2.
Final verification:186 related tests passed, Ruff/diff checks passed. Actual
SIGTERM/save/warmup-resume roundtrip passed at4096worlds in `stop_4096_fixed`;
completed count1->2, warmup count1->2. All bounded tests completed; no assistant
training/evaluation queue should remain after this handoff.

## Latest —10:42 UTC Sep7: PRELIMINARY conclusion ready; final300 jobs remain active

Latestuserwantedworkuntilinitialconclusion; nowmatchedAC101updatecurvecomplete.
DeliverChineseinitialresult, do NOTclaimstage1complete. Report:
`docs/simulator_preview_preliminary.md` (complete), paired10metrics/conditionalCIs
`runs/simulator_preview_single_motion/eval/seed13001/summary.json`.

Singlemotiondance1; trainingseed121;128matchedstarts/DRworlds,500steps:
frozenDR body.0524574874,joint.8684639960,fail1;
AC DRzero .0486533744/.8538578476,fail1;
AC DRtrue .0488273180/.8511456563,fail0;
AC nominalzero testednominal .0299706478/.6070405625,fail0.
True/zero body1.0035752 CI[.9953905,1.0112309],joint.9968236
CI[.9929596,1.0005190]: no clearmeantrackinggain. Failure1→0 is recoveryof
sameworld111/start5259 thatfrozenandbothzerocontrolsfail; no newfailures,but
oneeventnotrobustnessproof. TrueDR/newmatchednominal gap62.92%/40.21%.
Oldfixednominaltargetgap55.66%/35.24%. Mainconclusion: functionalinputuse
verified(criticRMSresponse.211), averagebenefitofordinarypreviewconcatenation
NOTestablished; notnominallevel. No rewardchanges/frozentensorchanges.

Actoronly201updatepairalsozeroish: true .0461398420/.8494948468,fail0;
zero .0462459685/.8472646949,fail0. Ratios.99771/1.00263. Earlier101update
testseeds13001/13002 bothshowbodytinygain,jointtinyregression. One training
seed; no generalizationortraining-repeatclaim. Fixed300endpointstillpredeclared.

Backgroundjobs allverifiedlive10:41: GPU0 old41055/newAC2689; GPU1 old41058/
newAC2690; GPU2 old41059/newAC2691. GPU3queuesold58458/AC2711 handle200/final
ownphysicscurves,finalcrossmatrix andzero/shuffleablation. Originalqueueexec48168;
ACqueue36946. Logs names documentedbelow. No needterminateanything forreport.
Proposednextdirection(notimplemented): structurepreviewerror→actioncorrection,
or supplylocalaction/errorresponse ratherthan onlybaseline-future observation;
do NOTpresentthisasestablishedcause. Contextstage2notstartedforthismotionroute.

## Latest —10:25 UTC Sep7: second test seed runs while AC reaches~90

StillworkuntilinitialACmatchedconclusion. Extra actoronlycheckpoint100 tests
seed13002runconcurrentlyGPU3: true exec46770,zero exec95191, outputs
eval/seed13002/dr_{true,zero}_seed121_100_test_dr.json,
logs runs/simulator_preview_eval_{true,zero}_100_seed13002.log.
Theyshouldfinish~10:27, nearAC100readiness. No nominal orfrozen13002baselineyet.
Thisisadditionaltestseed,notindependenttraining. All512trainingstillactive.

## Latest —10:22 UTC Sep7: waiting for AC101-update matched efficacy results

Do continue until preliminary conclusion (explicit latestuserinstruction).
Actor+critic runs~74iterations,oldactoronly~167. Both300updateendpointscontinue.
AC100expected~10:28;128world500stepevals~3min/condition,GPU3queuescanoverlap.
Userlasttoldexpect~ten-plusminutesforfullmatchedresults,notthatgoalissolved.
AC50functionalprobe completed: criticzero-preview RMSdifference.21139,
shuffle.19161 (valueSTD1.6839),actorzero.001699/shuffle.002086;
nonzerooutcomegradientsactor.001355,critic.02555. Thisonlyprovesfunctionaluse,
notvalueaccuracy/policygain. Savedprobe/dr_true_ac_seed121_50.json.
ProbeCLI nowadditionallyreportsactor_residual_rms on FUTURE runs.
Singlemotionnominaltrainingcheckpoint100result.03017298/.60776631,fail0.
ActoronlytrueDRrelativegap61.2%/40.65%; noinitialgoalsuccess.

Newtests/test_simulator_preview_comparison.py validatesratios,newfailures,
protocol/world/rewardmismatchrejection. Combined10newpreviewtests passed;
156broaderadaptationtests passed inlastsubset; ruffclean.
Numericalsimulationerrorsremainasdocumented, useracceptsnonzero sufficiently
smallerror; do not spendremainingtimechasingbitwisezeroinstead ofefficacy.

## Latest —10:17 UTC Sep7: actor-only101-update curve is a weak/null preview result; AC pending

User asks to continue until a preliminary conclusion, numerical repeat error
need only be small. Continue until matched actor+critic curve is evaluated;
do not end with only “jobs launched.” Sixtrainersstillrunning, sameGPU0–2pairs.
Oldactoronly at~150iterations, newAC~55; evaluatorGPU3queues wait100/200/final.

Own-physics firstcurve complete atcheckpoint100 =101updates,128starts500steps:
- frozenDR .0524574874/.8684639960,fail1;
- DRtrue actoronly .0486401508/.8548226456,fail0;
- DRzero actoronly .0488385527/.8533199296,fail1;
- nominalzero actoronly nominaltest .0301729817/.6077663128,fail0.
True/zero ratios body.9959376 CI[.9900232,1.0017841],joint1.0017610
CI[.9991402,1.0043292]. Conditionalworld/start CIs, one motion/checkpointpair.
Thus body~0.4%better,joint~0.18%worse: no clear preview-specific learning gain;
zero failures is onlyone observedrecovery, not robustnessproof. True/frozenDR
body7.28%/joint1.57%better; cannotattributethatgain topreview. TrueDR/nominal
trained nominal gap61.2%/40.65%. User toldtheseactoronlynegativecomparisons.
Detailed10metrics andpairedfailurechecks in eval/seed13001/summary.json,
generatedby scripts/summarize_simulator_preview.py; test comparisons10passed.

NewCLI simulator_preview_probe.py: separate nonacceptance input interventions,
zero/shuffleONLYlast720outcomes,keep921commonstatephysicsreference fixed.
Measuresactor/criticoutputchangesandactualgradientswrtoutcomes. Smoke12004passed
nonzeroactor/criticgradient; notefficacyproof. FormalACcheckpoint50probe currently
runningGPU3 exec8656, log runs/simulator_preview_probe_ac121_50.log,
output probe/dr_true_ac_seed121_50.json. AC100notyetavailable.
All3ACinitialactors/criticsbitwiseequal; originalvalue0differenceauditpassed.
Alloldtrackerweightsbitwiseunchangedinitial→100. Originalrewardsignatureunchanged.

## Latest —10:06 UTC Sep7: user requires shared preview to critic; continue until preliminary conclusion

New instructions: give preview/latent to critic too; continue verification until
a preliminary conclusion. User explicitly says simulator repetition need NOT
be zero, only sufficiently small. Clarified512-world audit numerical tails in
commentary: maxbody1.614mm,maxjointL2.00708rad; mean/bodytracking<.00121%,
mean/jointtracking<.00165%. Former hardmax gate failed and remains failed inJSON;
we continue a measured numerical-preview experiment, no solver/reward change.

Implemented `PreviewAwareHeftCritic` in simulator_preview.py. It reads exactly
actor's same1641-D storedPRIVILEGE (masking is shared before bothnetworks).
Original6330 value plus7971->256->128->1 zero-output branch. Independentnormalizer/
weights, no sharing actor parameters/gradients. Freshpreview CLI defaultscriticON;
explicit--no-simulator-preview-critic isactoronly ablation. Running oldjobs stay
oldclass andmetadata. New3-updateACsmoke12004passed; initialvalue difference0,
frozen tracker tensorsunchanged. Unit pairedstates variedpreview testpasses.

NEW3trainers launched10:03approx onGPU0/1/2 alongsideoldtrainers:
- dr_true_ac_seed121 exec42680; dr_zero_ac_seed121 exec78315;
- nominal_zero_ac_seed121 exec80042.
Same512env300updates,seed/RNG121,24steps,20criticwarmup,LR1e-4/5e-4,std.1,
entropy.0002,uniform/reference starts,original reward. EvalACqueue exec36946,
log runs/simulator_preview_evaluations_ac_13001.log, GPU3.
OldGPU3queue54915 was stopped (child57528finishednormally), replacedbyPID58458,
exec48168, log runs/simulator_preview_evaluations_13001_resumed.log.
This corrected primary endpoint file: FINAL after300updates hasiter299; there
is no checkpoint_300.pt. Legacycheckpoint100/200 means101/201updates.
Bothqueues nowusecheckpoint_final.pt forlabel300. Do notwaitnonexistent300.

Newcomparison scripts/summarize_simulator_preview.py validates matchedworlds,
starts/protocol/reset/reward; reports10metrics,newfailures,conditionalworld/start
bootstrapCIs explicitlyNOT acrossmotion/traininguncertainty. Noefficacy resultyet.
Asrequested, give preliminary diagnostic conclusion when firstmatchedACcurve
hasbeen tested; finalpredeclared300endpointremainsunchanged, no bestckptselection.

Baseline seed13001,128repeats500steps,onone dance1motion:
frozen nominal body.0305581522,joint.6128975578,fail0;
frozenDR .0524574874/.8684639960,fail1;
oldfixednominal evaluatednominal .0313676503/.6293764801,fail0.
Allbaseline4crossconditionscompletedunder eval/seed13001/.


## Latest —09:53 UTC Sep7: NEW single-motion FIVE-step simulator-preview experiment

Current user scope supersedes the older active routes below. Return to stage1:
frozen tracker + residual, privileged five-step closed-loop tracker future from
the residual's CURRENT state/history/actual physics, plus target deviation.
User explicitly rejected starting with one step: horizon=5. No world model,
no reward modifications, no subagents, no code writes outside this repo.

Plan and numerical qualification: `docs/simulator_preview_plan.md`.
New module `simulator_preview.py`; integrated optional `--motion-file` and
`--simulator-preview true|zero` into adaptation_train/eval. Original architecture
PrivilegedAdaptationActor, state_physics566 + common future reference355 +
true/masked future outcome720 =1641 inputs ->256/128 encoder->64 latent ->
residual512/256/128, zero-output initial residual, original deployable frozen
tracker frontend. Critic gets no preview. All rewards untouched (SHA abee736...).

Smoke2 PPO updates +100-step full evaluator passed. 159 tests passed.
Strict simulator repeatability FAILED even copying all arrays; separate bounded
preview gate passed on fresh seed12002,64worlds x12queries x5 in BOTH nominal/DR.
Source state/history/RNG and actual expanded physics exact; first action and
reference exact. Mean pose prediction errors <0.005% of actual tracking error,
worst body point<0.403mm, joint vector<0.00440rad. Velocity outliers up to0.166:
do NOT claim bitwise future truth or precision velocity prediction. Full strict
failures and later bounded audit files retained under runs/simulator_preview_single_motion/.

Selected one motion BEFORE results: lafan_qingtong/dance1_subject2.motion.npz,
6574frames. No full-dataset generalization claim. Primary checkpoint fixed at300;
100/200 only learning-curve diagnostics. 512env,24steps,300updates, seed121,
training RNG121, actorLR1e-4fixed, critic5e-4,warmup20,std.1,entropy.0002,
uniform sampling/reference starts, original DR distribution, scale1.

ACTIVE launched09:50; physical GPUs only0–3:
- GPU0 `runs/simulator_preview_single_motion/dr_true_seed121`, exec4653.
- GPU1 `.../dr_zero_seed121`, exec8168.
- GPU2 `.../nominal_zero_seed121`, exec42262.
- GPU3 evaluation queue `scripts/run_simulator_preview_evals.py`, exec85046.
  Logs `runs/simulator_preview_evaluations_13001.log`, eval/seed13001/.
  Baselines: frozen tracker and fixed old nominal, each tested nominal andDR;
  candidate curves own physics100/200/300, then full cross matrix at300;
  post-hoc true teacher zero/shuffle ablations last.128repeats,500steps.
  Queue waits for stable checkpoints and reports failures, never overwrites evals.

No efficacy result yet. GPU0–3 other user-authorized occupying jobs1731,23667,
38119,3631 were terminated09:31; no files deleted; GPU4–7 untouched.
Old goal-tool objective remains stale blocked audit-only and cannot be edited
with provided tools; user authorization after that clearly permits this work.


## Latest —06:30 UTC Sep7: independent nominal diagnosis complete; gated route active

User's question has been answered with new measurements in Chinese commentary;
research jobs continue. SAME NOMINAL seeds98001/2/3,1008segments, all9 evals done
under eval_v2/teacher_nominal_diagnosis_980xx/. World fingerprints/starts/motion
IDs and protocol identical across the3 models at eachseed. One trainedteacher,
not3 independent training runs. Generic compare reports correctly reject these
as ultimate-goal acceptance (both sides nominal), not as diagnostic evidence.

| Policy | Body | Joint | Failures/1008 |
| --- | ---: | ---: | ---: |
| original frozen tracker | .03079445 | .48399478 | 1 |
| oracle-wide initial, zero residual | .02788254 | .45737074 | 0 |
| oracle-wide400 | .03223682 | .51179765 | 0 |

teacher400/frozen body1.04683856 CI[1.01775105,1.08156294],
joint1.05744458 CI[1.04335545,1.07323450].
initial/frozen body.90544039 CI[.88995872,.91838989],
joint.94499107 CI[.93876067,.95073179].
teacher400/initial body1.15616508 CI[1.12125268,1.19985559],
joint1.11899955 CI[1.10170616,1.13907022]. All3 seeds same directional result.
Reports teacher400_vs_frozen.json, initial_vs_frozen.json, teacher400_vs_initial.json.
Both oracle policies0fail does NOT prove equal failure probability.

ACTIVE training now:
- GPU2 gated teacher fixed_reward_separated_gated_oracle_seed77 PID44909,
  session25145; watcherPID49429/session2634.4096env1000/save100,compact7 lowrank
  rank2/shared0, exact nominal prior with separated deployable action base,
  oracle/clean residual inputs; scale1,LR2e-5fixed,critic5e-4,warmup50,std.1,
  entropy.0002,seed77/RNG770001,uniform/reference, original reward signature.
  Initial source-action audit0; no100eval yet. No student quality claim.
- GPU1 refinement108 PID20796, watch24086. Latest200 body.03876558,
  joint.61626534,fail4; regressing, check300 before keeping this trainer longer.

Stopped exact old wideMLP8687 after joint regression500→600;700 also retained.
The watcher completed all existing700evals, no failures; best400 preserved.
Fine-tune104frozen/unfrozen BOTH completed501, watchers complete. Neither route
improved oldoracle300 overall. Sharedwide59054 andseparatedstate64992 already stopped.
No checkpoint deleted, unrelated GPUs4–7 untouched. No agents.

New gated teacher smoke109 complete4updates: only16up/down tensors plusstd learn;
trained nominal full336 result.03080050/.48345410,fail1. Pointwise source identity
is exact; do not claim bitwise closed-loop identity (numeric trajectory variance).
Stage2 smoke109 now COMPLETE4updates+full336v2+CPUexport:
encoder-only source94_1500 (12encoder tensors,4normalizer tensors match),
109 teacher controller/prior/preprocessing tensors bitwise preserved and frozen.
7physical latent + feature adapter, IMU/rightaligned, no rootquat/FK extras toencoder,
latentMSE1,featureMSE.1,64env, no teacher mixing, original reward unchanged.
Actual eval strips privileged actor keys eachstep;fail6 is smoke, NOTquality.
Artifact fixed_reward_separated_gated_context_smoke109_export/policy_cpu.ts
SHA300057652852d516993aa12b4854ca5cb02602976d5c8dddf4952980261c89ab;
checkpointSHA14a8e0e67e98402524daa66e92574e654fd2e09a9619b4058ff0dd7c00582bcb.
8 fresh CPU processes max5.90085983e-6, required CPUthreads1. No privileged inputs.
New combined architecture test + existing modulation tests12passed; ruffclean.
Only local source edit in this continuation is added regression test, new route
uses existing flags. Final student600xx remains unused. Final goal NOT achieved.

## Latest —06:23 UTC Sep7: nominal regression localized to learned residual

USER asks why teacher400 is worse than frozen tracker in the same nominal table.
Answered in Chinese commentary: yes, body/joint are worse; fewer failures do not
erase that regression. New zero-residual INITIAL checkpoint evaluation disproves
the tentative explanation that the oracle frontend itself caused this regression.
All same development seed10001,336segments/500steps, exact matching world
fingerprints and starts, files eval_v2/new_teacher_nominal_10001/:

| Policy | Body | Joint | Failures |
| --- | ---: | ---: | ---: |
| frozen.json | .0308283769 | .4900344991 | 2 |
| fixed_nominal.json | .0309239219 | .4843087843 | 1 |
| teacher_initial.json | .0282723299 | .4659700127 | 2 |
| teacher400.json | .0323747070 | .5189603941 | 1 |

Initial residual final weight/bias exactly zero; all53 tracker/preprocessing
tensors bitwise unchanged from initial to400. Initial oracle frontend improves
nominal body/joint8.29%/4.91%, but DR learning degrades them14.51%/11.37% relative
to that initial policy. Full teacher is5.02%/5.90% worse than original tracker.
DR payload training support is1–3kg, nominal payload0; this is a concrete support
difference, NOT proof of the precise learning mechanism. Same reward unchanged.
Do not keep presenting oracle-input distribution shift as the established cause.

New initial DR diagnostic completed: teacher_initial_dr_diagnostic.json body
.0424668013,joint.6049777689,fail5. Thus400 improves DR body11.77%/joint1.159%
against its own oracle zero-residual initialization; do not attribute the entire
15.6%/4.7% gain vs raw frozen tracker to learned physical conditioning.

Stopped exact regressing trainers59054(sharedwide,700 retained) and64992
(separated full-state prior,600 retained). All checkpoints preserved. Shared700
.03803398/.62522747,fail1; split600 .04093288/.64578268,fail6.
Context106 joint full2000 COMPLETE: .04196855/.62169888,fail2; no clear gain.
Refinement108 first100 .03833507/.60091325,fail4, no gain yet; trainer20796 continues.

NEW existing-flags combination smoke109: exact nominal prior/deployable base,
oracle/clean features ONLY residual, compact7 direct physical low-rank rank2,
shared rank0. True nominal code guarantees zero residual after learning; this is
NOT a DR failure or student's estimated-code guarantee. Fresh original-reward
training only. Smoke4updates/64env complete, source action difference0 initially,
only16 low-rank up/down tensors plus explorationstd changed. Reward SHA unchanged.
New combined architecture test passed with existing tests12passed.
Full nominal smoke evaluation currently session34919 onGPU2, file
separated_gated_smoke109_nominal.json in same new_teacher_nominal_10001 folder.
NO full gated training yet at this note. No subagents, no skill, no reward edits.

Predeclare independent SAME NOMINAL diagnostic seeds98001/98002/98003 for exactly
frozen tracker, oracle-wide initial, oracle-wide400. Selection fixed now, before
new-seed evaluation. Purpose is regression/frontend diagnosis, NOT ultimate goal
acceptance and NOT a DR-on/off training-only causal comparison. One trainedteacher.
Final student600xx remains unused. Task continues despite user progress questions.

## Latest —06:07 UTC Sep7: USER asks teacher architecture and nominal-test interpretation

NEW USER: "任务仍然继续...这个新方案的结构是什么?如果这个指标是在nominal环境下
测出来的,我认为已经能说明有效果了." ExplainedinChinesecommentary: cited15.6%/4.7%
is SAME DR testing improvement; fixednominaltarget was NOMINAL testing. Newteacher
is NOTLoRA:statephysics566→256/128→64latent,ordinary512/256/128→29residual,
oraclecleanfeatures,frozenoriginalcore,no nominalprior. Relativeolderoracle changed
residualscale.25→1 andLR.001adaptive→2e-5fixed; freshheadfromcommontracker,notwarmstart.
Reward/realactuator/actionscalesunchanged. ExplicitlyrecognizedDRimprovementasvalid
intermediategain,notdismissedbecausefinalgoalunmet; notinput-onlycausality.

Three NEW same-NOMINAL evalsdev10001 currentlyrunning toanswerdirectly:
eval_v2/new_teacher_nominal_10001/{teacher400,frozen,fixed_nominal}.json.
Teacherfixed_reward_oracle_wide_smalllr_seed77/checkpoint400, othersoriginaltracker/
exactfixednominal1000. GPUs1/2/3;PIDs20597/20599/20601,sessions91101/77443/37142.
All336segments500steps. At06:07aboutstep300. NEEDreportresultsasavailable.
Thisisarchitecture/inputvariant comparison,NOTpureDRtrainingon/off(the3paired
trainingseedcontrolalreadyreported13.5%/9.2%nominaltestgap).

NEWfullteacherrefinement108 GPU1PID20796/session56430,watchjustlaunched
fixed_reward_teacher_refinement_seed108_watch.log. InitialactiondifferenceEXACT0,
signedoracle_large300source;1024env1000/save100,refinementscale1,LR1e-4fixed,
critic1e-4,warmup0,std.05,entropy.0002,seed108/RNG1080001,uniform/reference.
Sourceencoder/normalizer/base/headfrozen; onlynewrefinement+explorationstdtrain.
CAUTION currentlogger residual_saturation_fraction stillcomparestotalresidualwith
originalscale.25,notcombined1.25bound. Telemetryonly; don'tinterpretitascapacity
proof. Actualactionbound/originalrewardunchanged,fixfutureloggingifneeded.

Refinement107teacher/student4updates+full336v2DONE (studentfail4,notquality).
69tracker/originalhead/newheadstudenttensorsbitwiseequalteacher,context/adapterlearn.
CPU8freshprocessesmax4.827976e-6,artifact
e3c4ad320a946d6df10f1731c2273b29c42a2b363000c13e5c0d6477cb8dab31.
Fulltests288passed55warnings47.35s. ExistingGPU1jointdistill106continues,500
body.042653/joint.628931/fail4,notbetterthaninitializedstudent1011000.

## Latest —05:58 UTC Sep7: USER progress report delivered, explicitly continue experiments

NEW USER asked: "你可以不终止任务,给我报告一下你已经尝试的技术路线，以及目前效果."
Delivered detailedChineseprogressreportincommentary(~05:57), coveringmatchednominal,
causal7physicsinput,negative64,teacherarchitecture,labels,stage2,CPUfailures.
UserexplicitlydoesNOTwanttasksstopped; continuework, doNOTstopmerelybecauseprogress
wasasked. Noagents. Goaltoolgetat05:56confirmedstaleBLOCKEDoldobjectiveinitialauditonly;
cannotreactivatewithavailableAPI. Userlaterauthorizationisclear. Donotmarkcomplete.

LATESTteacherwide_smalllr77checkpoint400: body.03746828,joint.59796604,
vsfixednominal1.220405/1.238340,worst10ratio1.318831,fail2/new1vsfrozenDR.
Newlowestbodybutnotdominatesoldoracle300joint. Oldoracle300.0382017/.5914685.
Sharedwide500body.0380853/joint.606235/fail3,regressingjoint. Separatedprior300
.0413765/.626470/fail6(new5),notyetgood. SmallLRunfreezing104at200:
unfrozen.0383028/.603397/fail2 vsfrozen.0385279/.595889/fail4; tradeoff,notwin.

Stage2oracle1013000COMPLETE(final.0419051/.620131/fail2), best1000/1500~.0417/.618/fail1.
Action-only1051500COMPLETE(final.0416095/.619297/fail3); no clearoverallgain.
NEWGPU1jointstudent106 PID58352/session27699,watchsession39199 (recheckPID):
teacheroracle_large300,studentinit101checkpoint1000,1024env2000/save250,
4rollout/4grad,batch2048/replay32768,teacherfade0,latentMSE.1/featureMSE.1,
TRAINbase+residual+context+featureadapter,LR3e-5,seed106/RNG1060001,
IMU/rootquat/rightaligned/FK. No new privilegedactorroute, all labels training-only.

Payloadgravityfull336devprobesCOMPLETE,nobenefit; DO NOTbuildfixedskipbasedonthis:
gain0(control)body.038296,joint.592683,fail2;
.25 .038827/.594243,fail5; .5 .040279/.639894,fail3;1 .046978/.846680,fail6.
Controlnodignosticsagainfail2vsbodydiagfail4; causeofrepeatdifferenceunresolved.
CodekeptDIAGNOSTIConlywithacceptancerejection. Fullsuite286passed55warnings47.64s
BEFOREnextrefinementcode. Nominalprobeexactzeropayloadcorrectionpassed.

NEW genericteacherrefinementarchitecture (NOTgravityskip): refinement_scale(default0)
adds zero-output512/256/128MLPboundedactioncorrection; sourceoriginalteacherhead,
encoder, normalization allfrozen viafreeze_teacher_trunk; originalrewardunchanged.
CLI--initialize-from signedteacher --add-teacher-refinement --refinement-scale1;
addsONLYrefinement_mlp statekeys, assertsinitialsourceactionEXACT0.
Teacher/studentclassesretainlegacyinferencewhenrefinementNone(noextraadd0kernel).
Studentcopiesrefinementhead,trainabilityfollowsfreeze_student_residual; context
meanunsupportedwithrefinement. Distilloptimizer/export includeoptionalhead.
TeacherprivnormalizerupdateDISABLEDwhenfreezetrunk: preservationincludesstats.
New3?actual2unit tests plusfixedseparatedexportfixture; fullsuitecurrentlylog
fixed_reward_teacher_refinement_full_tests.log (readlatestcount). Firstoldsubset
failedonlynewexportfixtureomittedrefinement_mlp attr,fixturefixedexplicitNone.

GPU1 teacherrefinement smoke1074updatesCOMPLETE,initialmean difference0;
checkinitial/finalonlyrefinement_mlp/distributionchanged,allsourcebuffersfrozen.
Fullv2teacherlaunchedsession(recheck), studentrefinement_context_smoke1074updates
JUSTLAUNCHED. Needstudentfullv2/CPUexport/frozenheadcopyaudit BEFOREfullrefinementrun.
No fullrefinementtrainingyet. Suggestedafterpassing: warmstartoldoracle300,scale1,
LR1e-4fixed,std.05,originalcritic1e-4,1024env/save100,500-1000updates.
Watchersforexisting6fulltrainersstillrunning, NONEstoppedinresponsetouserreport.

## Latest —05:43 UTC Sep7: runtime confirmation, right-arm gap, gravity feasibility probe

STILLoriginalrewardgoal, no userprogressrequest, noagents. NEW running jobs:
GPU2separated_prior_oracle77 PID64992/session30020, initialauditednominalaction
EXACT0 bothcached/uncached;100worst1.53793/fail4,notbetterthanbestteacher.
GPU0oracle300_unfrozen104 PID998/session89330;GPU3oracle300_frozen104 PID1139/
session28752. BothinitializeEXACTsignedoracle_large300,1024env501planned/save100,
LR2e-6fixed,critic1e-4,no warmup,std.05,entropy.0002,seed104/RNG1040001,
uniform/reference. ONLYtrain_base_policyflagdiff/outputdir;initial72actor/17critic
tensorsbitwiseequal. Thisisconservativewarmstartedbase-unfreezingcontrol,NOTold
catastrophicLR.001unfreeze. Sharedwatchsession96313 coversboth+splitprior (PIDrecheck).
GPU1NEWaction-onlylatentdistill105 PID909/session71011,watchsession66084:
teacheroracle_large300, studentinitcontext101checkpoint1000,1024env1500updates/save250,
teacherfade0,latentMSE0,featureMSE.1,LR3e-4,freezecontroller,IMU/rootquat/rightaligned/FK,
seed105/RNG1050001. TestactionequivalentlatentwithoutforcingteacherlatentMSE.
101stillrunning3000,2000worst1.45664/fail2;1055001.46314/fail3. Neitherclosetonominal.
Oldnewteacherjobs0shared_wide59054/3MLPwide8687running: shared400worst1.35127/fail4,
MLPwide2001.35416/fail2. Rechecklatest.

Splitprior smoke103fullstudent336v2passedinput/reset,fail6(notrainedquality).
CPU8freshprocessesauditmax4.664063e-6,artifact
67f9295599f4f283a71f2eb8476db019df7f1f70588dd9a8a405ff6f9bef5d1e.

LOWRANKcontext94_1500 CPUexportFULLCLOSEDLOOP950xx COMPLETE3seeds:
exported/Pythonall10pointmax1.002120, body1.001195,joint1.000581,
fail12vs10 (2/5/5vs1/4/5); notobservedreliability-equivalent.
Exported/frozenDRbody.918872,joint.979462,fail12vs15;
exported/fixednominal1.314894/1.257575,fail12vs2. Reports
context_replacement_950xx/exported_vs_{candidate,frozen_dr,fixed_nominal}.json.
Artifactstillauditedd104b5...,checkpoint461017... unchanged. No finalgoalpass.

Fulljoint/bodydiagnosticsofbestoracle_large300 vsfixednominaldev10001:
rightarmaccountsfor73.96%bodyposnetgap/88.47%bodyrotnetgap. LargestjointRMSEgap
rightwristpitch(.23955vs.09791rad), thenrightshoulderpitch/elbow/wristyaw.
IMPORTANTjoint_names andactuator_names areDIFFERENTorder; joinBYNAMEforsaturation,
neverzipjointRMSEwithactuatorsaturation. SourceJSONseparatenamelistsarecorrect.
Diagnosticsrerunoracle300fail4vsoriginal2, body.03846865vs.0382017;
samecheckpointSHA/worldfingerprints/motionstarts. AdditionalfailuresfallAndGetUp1_subject4
start3778 andfallAndGetUp2_subject3 start3477. DoNOTclaimstable2failures.
Userinformed, keepbothresults; cause(numeric nondeterminism vsread-onlydiagnosticcache)
notyetisolated. NewNONdiagnosticcontrolrerunlaunchedbelow.

NEW analyticalPAYLOADONLYgravityprobe inoracle_compensation.py:
currentworldcenter+hingeaxes/anchors, ancestor-mask torque=-Jpos.T*m*g,
mapunsaturatedPDtorquetoaction/clamp±3. Nominalmass0=>EXACTzerocorrection;
no realmodel/statewrites,no nominaltwin, nocontact/inertialcorrection.
DIAGNOSTICONLY,nolearnedpolicysuccess. Evalflag--payload-gravity-compensation,
studentforbidden; compare_adaptation_evalsEXPLICITLYrejectsnonzeroasacceptance.
Sign/mask/zero/acceptanceunit14passed; fullsuitecurrentlysession(recheck).
NominalandDR42env8stepssmokesCOMPLETE,nominalidentityauditedzero.
NOWfourfull336/500dev10001evalsonBESToracle300:
oracle300_payload_gravity_{control,025,050,100}_seed10001.json,
gains0/.25/.5/1, GPUs1/0/3/2. Allno body/jointdiagnostic flags. No new policy.
Ifhelpful, canconsiderLEARNEDstructurewithsameoriginalreward; NOTimplementedyet.
Oldfullbiascompensationnegativeprobe notresurrectedassuccess.

## Latest —05:29 UTC Sep7: separated prior smoke passed, full run launched

Fullsuite284passed55warnings47.31s; firstsuiteattemptinheritedCUDA_VISIBLE_DEVICES=2,
causingonlymocklauncher envexpectationfailure. Retriedwithenv-uCUDA_VISIBLE_DEVICES;
no test weakened. Logfixed_reward_split_prior_full_tests_retry.log.
Splitprior teacher smoke102failedbeforetrainingbecauseoldnominalidentityauditstripped
oraclefieldsneededbynewresidualpath. Fixedaudit'suncachedgroups toretainrequiredoracle
fields; no assertionrelaxed. Smoke103complete4updates: initialnominalprioraction
differenceEXACT0 cached/uncached,zeroresididentity0;53tracker+8prior tensorsunchanged,
8residual+6encoder tensorslearned. Audit incheckpoint_initial.pt['infos'].
Student1034updatescompleted; teacher53tracker+8prior+8residual tensorsallbitwise
preserved. Fullv2student/CPUexport launchedsessions31331/61736,recheckresults.
GPU2fullfixed_reward_separated_prior_oracle_seed77 JUSTLAUNCHED (recheckPID/session).
4096env2000/save100,scale1,LR2e-5fixed,critic5e-4/warmup50,std.1,entropy.0002,
seed77/RNG770001,uniform/reference,statephysics64latent,oraclefeatures/clean,
exactnominalprior+separateddeployablebase. Needwatcher100andinitialidentityaudit.

950xxadditionalfrozenDR/nominalreferences COMPLETE. Student94_1500/frozenDR
body.917775,joint.978894,fail10vs15,new3/recovered8; worldfingerprintsequal.
Student/fixednominalbody1.313325,joint1.256845,fail10vs2,new10/recovered2.
Reportscontext_replacement_950xx/student_vs_{frozen_dr,fixed_nominal}.json.
OriginalstudentselectionSHAunchanged. Final600xxUNUSED;fullgoalNOTachieved.

## Latest —05:24 UTC Sep7: negative64 control, oracle300 distillation, separated prior implementation

User's original-reward goal continues; NO progress/stop request. No subagents.
970xx actuator64 shared16 input-only500 comparison COMPLETE: normal/zero body
1.026536,joint1.009772,fail18vs17/1008,new8/recovered7. All10meansworse.
Initial118actor/17criticbitwiseequal, argsinputmode/outputonly. One trainseed77.
Stopped exact trainers25236/25361 after500retained, no filesdeleted.

Oracle_large300 all-path posthoc height/contact dev10001 finished:
removeheight body/joint1.00290/1.00175,fail2; contact1.00478/1.00234,fail2;
both1.00497/1.00322,fail3; unmodifiedfail2. Pathsoraclefrontend ANDstatephysics
latent replaced. NOTretrainingcausality. Filesfixed_reward/oracle_large300_remove*.
Stoppedoracle_lowrank62481 after800retained (700worse1.36564/fail5,8001.37768/fail6).
Best remainsoldoracle_large300. Newwideupdates havenot passednominal.

RUNNING (recheck): GPU0 oracle_shared_wide77 PID59054/session64901,
watch6324/session50900.4096env,compact7lowrankrank2+shared16,oraclefeatures/clean,
scale2,LR2e-5fixed,warmup50,std.1,entropy.0002,seed77/RNG770001;100worst1.40019/fail6.
GPU3 oracle_wide_smalllr77 PID8687/session41628,watchsession81583;4096env,
statephysics64latent/plainMLP,oracle/clean,scale1,LR2e-5fixed,otherwisematchedabove.
GPU1 oracle300_context101 PID8615/session56927,watchsession2790;
teacheroracle_large300,1024env,3000updates/save250,4rollout/4grad,batch2048/replay32768,
teacherfade750,latentMSE1,featureMSE.1,frozenbase/residual,IMU/rootquat/rightaligned/FK,
LR3e-4,seed101/RNG1010001. No oldshapedorstudentinit.600running,eval250/500pending.

GPU2 sixadditionalreferenceevals95001/2/3 launched simultaneously:
context_replacement_950xx/frozen_dr_seed*.json ANDfixed_nominal_seed*.json.
Sessions14427/16421/15319/91278/40573/90789. Recheckcompletion.
Thesecontextualizeexistingteacher500/student1500results,notnewstudentselection.

Oracle_shared_context_smoke99 COMPLETE4updates+full336v2input/reset(fail6,noquality).
CPUexport succeededretryafterinitialCUDA_VISIBLE_DEVICESemptyerror (exportpreparation
needsGPU;artifactitselfCPUonly).8freshprocessesmax4.559755e-6,artifact
e3390149c9f8a7d93fdea6d6fe644f8c89e195df65f584100e297dd36014e54c.
Sameoutputwasemptyafterearlyfailure,nothingoverwrittenexceptnewretrylog.

NEWCODEexplicitfrozen_nominal_prior_deployable_base(defaultFalse)teacher/student:
auditednominalactionbranch alwaysoriginaldeployablefeatures;oracle/contextadapter
featuresonlynewresidual. Requiresprior,frozencore/ref;legacyguardsdefaultunchanged.
Helperbase_action_from_features andstudentbase_action_from_adapted_features unify
inference/distillrollout/replaygradient/diagnostics; CPUexportpreservesrawbasefeatures.
CLIflagfreshprivileged+prioronly; loadedactorsretainsavedflag. No rewardschanged.
Initialoldtests118passed;new3unit tests pendinglogfixed_reward_split_prior_unit_tests.
Fullsuitepending. GPU2splitprior teacher64env4updates smoke102JUSTLAUNCHED;
needzeroresidnominalsourceaudit/frozenweightaudit,student4updates+fullv2+CPUexport
BEFOREfullsplitpriortraining. No fullsplitpriorrunlaunchedyet.

Do notconfuse raw64negative withpositivecompact7twoindependenttrainingpairs;
compact76/77 results andcontext950results in05:00/04:36docs remainvalid.
Final600xxstillunused. Goaltoolstale blockedcannotreactivate;donotmarkcomplete.

## Latest —04:40 UTC Sep7: causal input evidence, context replacement, new physical architecture

IMPORTANT latestUSER intent remains: beating DR-TRAINED residual is an independently
meaningful intermediate milestone; don't dismiss progress merely because finalnominal
targetunmet. Need separatecombinedarchitecture+inputgain fromcausalinputgain. No new
progress/stop request has arrived. Continueoriginalrewardgoal. Noagentsallowed.

COMPLETED full495→64FiLM INPUT-ONLYpair onnew94001/2/3, fixed500,102? actual86actor/
27criticinitialtensorsbitwiseequal; exactargsdiffonlyinputnormal/zero+outputdir.
Bothcriticstruephysics. Normal/zero body.993424 CI[.985204,1.002243],joint.992182
CI[.986856,.998555],fail14vs16/1008;new5/recovered7,perseed2/2,4/7,8/7;
failureCI[-.0128968,.0119048]. Directsmalljointbenefit fromphysicsinput, body/risk
uncertain, ONEtrainingseedonly. Report`eval_v2/input_control_940xx/full77/comparison.json`.
Zero53132STOPPEDafterretained500. Normal24775previouslySTOPPEDafter500. No filesdeleted.

COMPLETED contextreplacement95001/2/3: lowrank77teacher500 vscontext94student1500.
All102nonencodercontroller/preprocessingtensorsBITWISEequal;studentdeployable-only.
Student/teacherbody.997038 CI[.990752,1.002053],joint.996130 CI[.990192,.999758];
all10pointmax1.002450. Fail10vs14/1008,new2/recovered6,perseed1/4,4/5,5/5;
failureCI[-.0138889,.00396825]. Goodintermediatereplacementevidence, NOTfinalgoal:
teacherstillfarfromfixednominal; nozeroriskguarantee. Report
`eval_v2/context_replacement_950xx/comparison.json`. TeacherSHA
132ec7f57bece83012ba4565002a77d4c6277dd6da59280dd2ee7c9b07679219,
studentSHA461017433cd4696066f69f622eb87a86d3eb1ec105d37a1de8b46eab6544306a.
StudentCPUexport8freshprocessesmax5.72205e-6,artifact
d104b5bc9f4b371922c50ca9307e700a0a9b58a887ab819abfa7837499bd5559;
`fixed_reward_lowrank_context94_1500_export/`. No full exported-policy closedloop
confirmation yet; numeric+Pythonclosedloopaudits only. Final600xxUNUSED.

Context94encoder-only2000COMPLETE(PID22381/session64023,watch70316completed).
Jointdistill96controller+encoder1500COMPLETE(PID4354/session63984,watch10015):
startscontext94_1000,sameteacher500,LR1e-4,teacherfade250.1250worst1.44469/fail3,
final1.45527/fail4. Didnotdemonstrateclearimprovement; nofreshconfirmation.
Initialcontextfailureextra isfallAndGetUp2_subject2,start6896,length226(4.52s),
notstartup. Causalmean50/250oncontext94INITIAL onlytinymeanchanges,fail4both;
files`lowrank_context94_initial_mean{50,250}_seed10001.json`. Mean250body.0410284,
joint.614891 vsinstant.0411076/.615903,notreliabilityfix. Theseconversions used
oldercodehashbutonlylaterguard/autogradchanges, inferenceunchanged; noacceptance.

CURRENT FULL JOBS(recheckPID/cmdline):
GPU0 compact7 true/constant seed76 PID25154/session60561 &25381/session38424,
watch12284. Planned501updates (fixedcheckpoint500),save100,1024env,LR2e-5fixed.
GPU3 compact7constant77PID40906/session36883/watch9480,500notyetretainedat04:38
(400evaluated), STOPafter500asbudgetedinputcomparison. True77source500retained.
Bothinputcomparisons predeclared94001/2/3, waitingcontrollers:
compact77PID1388/session9380, gpus1/3;
compact76PID1467/session64284,gpus0/0. Logs`input_control_940xx_{compact77,compact76}.log`.
Output`eval_v2/input_control_940xx/{compact77,compact76}`. Script nowwaitsstablefiles,
freezesSHA, reauditsinitialmodels/args, neverselectsearlierevaluation. At200seed76
truebody.041174/joint.611929/fail1 vsconstant.042085/.622337/fail7; promisingONLY
interim, keep500selection. Seed77bodyat200slightlyworsetruebutjointbetter; needfull.

GPU2 NEW fixed_reward_oracle_lowrank_seed77 PID62481/session12276/watch74269:
compact7directcode,rank2perphysics, oracleheight/contact/reference+cleanproprio,
NOprior, originalcritic,4096env,scale.25,LR.001adaptive,2000planned/save100,
seed77/RNG770001,warmup50/std.1/entropy.0002,uniform/reference,originalreward.
100worst1.40793/fail7,2001.38082/fail6,3001.35685/fail5. Coreweightsfrozen.
STOPPEDoldoracle_large34701after600 (body.038623/joint.625472,regressing); keepbest
oracle300body.038202/joint.591468/fail2. Nofull-goalpass.

JUSTSTARTED GPU3 fixed_reward_actuator_shared_normal_seed77 session6581;
GPU1 fixed_reward_actuator_shared_zero_seed77 session37031; PIDsrecheck, NOwatchers
yet. Both1024env/2000planned/save100,LR2e-5fixed,scale1,seed77/RNG770001,
std.1/entropy.0002/warmup50,uniform/reference,originalcritic,NOprior/NOoracles.
NEWactuator_physics64 rawcode =compact[:6](payload,torso,COMxyz,meanfootfriction)
+29 individualarmatureratios/.2 +29 encoderbias/.01,canonicalrobotjointorder.
Nolearnedphysicalcompression, alltrueq/rootheight/contactabsent; footgeometry
frictionsSTILLaveraged, don'tclaimfullylosslessDRrepresentation.
LOWRANKshared16 +conditional64(rank1each) at8frozencorelayers. Sharedpathlearns
commonDRcorrection evenwhenactorφZERO; conditionalpathlearnsparameter-dependent
corrections. Onepolicy, notmodelensemble. Afterlearningφ0doesNOTguaranteezero
residualbecauseglobalpathisactive. Initialzero-outputidentitystillassertedexactly.
Needactualfullinitialpairaudit +watchers +predeclare500 new970xxcomparison.

CODEchanges: PhysicalLowRankResidual optionalshared_rank(default0preservesallold
states/inference); teacher/student shared_low_rank option; teacherzero-input
nowalsoallowedphysical-lowrankwithpositivelearnablesharedpath. Teacher supports
compact7oractuator64,studentdirectcodes7/64. Meanonlyconversionstill7only;
encoder-onlytransferstillSTRICT7only, notsilently reusedfor64. Newteacheroracles
areallowedwithlowrankfrozencore; studentcanhaveexplicitdeployablefeatureadapter.
Fixedreferencebranchinputgradient: removedno_gradaroundreference(features),
parametersremainfrozen, correctbothbranchderivativesfortrainablefeatureadapter.
Inferenceunchanged; unitensureszero-outputequalbranchesalsozeroinputgradient.
Fullsuite279passed55warnings47.83s (`fixed_reward_shared_lowrank_full_tests.log`).

SMOKES: oraclelowrank95teacher/featureadapterstudent shorttrain +full336DR input/
reset audit +CPU8process exportcompleted (studenttrackingbadfail8, notgoal),
artifact0cccd65060b48a5f5b32f21649308c4af54f18ea3b7b8d5b36899bb535a024ca,
max6.31809e-6. Actuator64purelowrank97teacher4updatesandfull336completed(fail6),
contextsmoke974updatescompleted; contextfull336session61338/CPUexport91580were
runningat04:36—checklogs. Shared64+16normal/zero98smokesbothcomplete4updates,
initialactor/criticbitwiseequal. Normalconditional+sharedlearn;zeroONLYsharedlearn,
core/preprocessfrozen. Audit`fixed_reward_actuator_shared_smoke98_audit.log`.
Sharedcontext98smokejustlaunchedsession95664, needsfullDRv2/CPUexport/audit.

Nextpriority: completecompact76/77causalinput500confirmations; inspectnewshared
architectureandoraclelowrank; deployablelatentreplacementalreadycloseforweak
teacher, so mainbottleneckremainsfirst-stagefixednominalquality. Keeporiginalreward
guard(shaabee73632d079f08c160584b04ace15532ecf6bd9dfa8139979af518d8a1553c),
nooldshapedinitialization/rewards, GPUs0–3only. Goaltoolstale blockednotreactivatable;
doNOTmarkcomplete. No newuserprogressrequest; continueuntilgoaloruserasks.

## Latest —04:08 UTC Sep7: 930xx complete, second-stage encoder-only transfer

Three freshsameDR evaluations COMPLETE; report
`eval_v2/dr_residual_confirmation_930xx/comparison.json`. Lowrankphysical500 /
ordinaryDRresidual500: body.989410 CI[.977325,1.000155],joint.964964
CI[.957483,.972731]; all3 pointimproved. Bodyangularvel1.002027. Failures9vs9,
perseed2/1,2/3,5/5, NEW5/recovered5; failureCI±.00892857. OnlyoneTRAINseed,
architecturediff; NOTcausalinputproof orfullnominalgoal. Nominalfinal600xxUNUSED.
Legacyreference`matched_dr_seed77` predatesruntimerewardsignature; initialsummary
failedclosed. Added EVALUATION-ONLY audit of equal savedtask.reward/decimation/sim,
originalargs, emptyrewardchanges, sameadaptation_rewards.py sourceSHA, samebase
tracker and noinitialization/teacher. No checkpointmodified, no retrospective
signature, stillINELIGIBLEnewtraininginit. Reportretainslimitation. Scriptresume
alsofixedtuple/listplancomparison; frozenSHA/seedsunchanged,sixevalsnotrerun.

CURRENT FULL JOBS:
GPU0 NEW fixed_reward_lowrank_seed76 PID25154/session60561 and
fixed_reward_constant_lowrank_seed76 PID25381/session38424; sharedwatchseeprocess
`fixed_reward_lowrank_input_seed76_watch.log`. Both1024env/fixedLR2e-5/scale1,
warmup50/std.1/entropy.0002/originalreward, seed76/RNG760001,501plannedupdates,
fixedcheckpoint500 (iteration0counts) predeclared,save100. True102actor/17critic
initialtensorsbitwiseequal, argsdiffonlyinputmode/outputdir. Audit
`fixed_reward_compact_input_seed76_initial_audit.log`.
GPU1 NEW fixed_reward_lowrank_context_transfer_seed94 PID22381/session64023,
watch70316. TeacherNOprior compact7lowrank77checkpoint500, sourceENCODERONLY
`fixed_reward_compact_context_pretrain_seed88/checkpoint_final.pt`,1024env,
2000distillupdates,save250,teacherfractionfade500,4rollout/4gradient,batch2048,
latentMSEweight1,LR3e-4,seed94/RNG940001,frozencontroller,IMU+rightaligned50history.
GPU2 oracle_large34701/session82603/watch77421 plusphysics_critic_prior24775/
session69961/watch43356; theirfull500statscontinue, nofull-goalpass.
GPU3 zero_actor_physics53132/session95479/watch40345 (sameFiLMarchitecture
input-onlypairwith24775), plusconstant_lowrank77PID40906/session36883/watch9480.
Constant77sameoriginal1024settingsaslowrank77,plan2000butfixed500comparison;
initial102actor/17criticbitwiseequal, modeconstantonesinsteadtrue7. Allranks
active/learnable, no worldinfo. Audit`fixed_reward_compact_input_pair_initial_audit.log`.

STOPPED46273oraclecriticafterretained600(worst1.43240/fail3);5001.41093/fail4,
notimprovementoverplaincritic. STOPPED13337nonlinearlowrankafter500(worst1.58363/
fail9), no filesdeleted. Oldwide12669/narrow60228/compactlarge15241stopsremain.

NEWdistill`--context-encoder-initialization`: transferONLY7physicalcodeencoder,
validatehistoryflags/inputnormalizationbitwise, sourcesignedreward andactual
exact-SHA teachertruthsemantics (legacycontextmetadata resolvedusingitsrecorded
teacher). Rejectlearnedlatent/constantcode/changedinputsemantics/memory/fullinit
mixes. Othercontroller/prior/preprocessing tensorsbitwiseunchanged. Metadata
`latent_target_contract`nowexplicit. Fullsuite276passed54warnings47.04s.

Smoke93fullDR INITIAL actor (NOnewdistillupdates) body.0411076372,joint.6159031433,
root.322632022,fail4 vs teacher3. Ratiosvs7physicalteacher500body1.003753,
joint1.001689,all10max1.003753. Promisingcode-replacementprecision butextra
failure, notacceptance. RelativeordinaryDRresidualbody.98899,joint.96709approx.
Fullseed94INITIALactor114tensorsbitwiseequal tosmoke93INITIAL; canreusethediagnostic
conceptually butrespectcheckpointSHAforsubsequentpairedstats. File
`eval_v2/fixed_reward/lowrank_context_transfer_smoke93_initial_seed10001.json`.
Comparison`lowrank500_vs_transferred_encoder_initial_development.json`.
Smoke4updatesshowONLYencoderlearned, currentteachercontroller102tensors exact,
oldpriornottransferred. Exportsmoke93FINAL8freshCPUprocessesmax4.58956e-6,
artifactSHA4452229668b946f7df9ceb8eba060a0495f7eb36e2412a70dcbc898efbf41add.
DoNOTconfusefinalsmokeexportwithinitialevaluatedactor orfulltrainedstudent.

Next: waitinputcontrols500(predeclared), normal/zeroFiLM500, student94checkpoints.
Inputablationresultcouldshowarchitecturealoneaccountsformostgain; don'tassume
privilegecausality. Contextinitialmeanreplacementisclose butfailuresneedattention.
Keeporiginalreward/fixednominaltarget,continueuntilstage2oruserprogressrequest.

## Latest —03:52 UTC Sep7: intermediate DR-residual comparison and input control

USER added: beating DR-TRAINED residual is meaningful intermediate progress,
separate from the unchanged fixed-nominal ultimate target. Not a progress/stop
request. Latest body/joint positive result vs ordinary DR residual: fixed500,
train77/eval10001, identical1024env/fixedLR2e-5/scale1/originalreward/no prior,
compact7lowrank ratios.985294/.965462, failures3vs4/336; bodyangularvel1.000349
slightlyworse. Architecture/input jointlydiffer; one train/eval seed only.
Report`eval_v2/fixed_reward/dr_residual_vs_lowrank77_500_development.json`.
New`confirm_dr_residual_comparison.py` freezes checkpointSHA/seeds93001/2/3 and
runs2arms onGPU0/1, session86751, log`dr_residual_confirmation_930xx.log`.
Directory`eval_v2/dr_residual_confirmation_930xx`, nofinal600xx used.

CURRENT FULL TRAINERS:
GPU0 oracle_physics_critic PID46273/session20677/watch28879;
GPU1 nonlinear_lowrank PID13337/session11014, NEWwatch49407;
GPU2 oracle_large PID34701/session82603/watch77421, physics_critic_prior
PID24775/session69961/watch43356;
GPU3 NEW zero_actor_physics PID53132/session95479, watch40345.
All runnames havefixed_reward_ prefix and_seed77 suffix. Latestoraclelarge300
worst1.35018/fail2;4001.36296/fail3. Oraclecritic3001.38055/fail3.
Physicscriticprior3001.52140/fail6; nonlinear3001.53800/fail5 (200fail8).
No full-goal pass. GPU4–7 untouched.

STOPPED fullphysicsprior narrow60228 after600(1.55242/fail5), compactpriorlarge
15241 after500(1.52154/fail6), wide12669 after600(1.59118/fail9). Allartifacts
retained; nofilesdeleted. Older03:23PIDtable below is historical.

NEW actor_physics_input normal/zero: zero permitted onlyfullphysics noncentered
residuals. Inputzeroing BEFOREencoder/normalizer; critictruth unchanged. Global
encoderbias andstate-conditionedresidualstilllearn. TestcoversNaNtruthmasking.
Fullzero run is paired EXACTLYwith physicscriticprior24775: all86actor/27critic
initialtensorsbitwiseequal, argsdiffonlyinputflag/outputdir (newdefaultsnormalized).
Audit`fixed_reward_actor_physics_input_pair_initial_audit.log`.
Smoke91frozenweights/learning/zero-normalizermean checks passed. Firstauditattempt
usedwrongresidual prefix andfailed; reruncorrectresidual_mlp prefix passed.
Fullsuite273passed54warnings45.36s,`fixed_reward_zero_actor_physics_full_tests.log`.

NEW nonlinear learnedlatentlowrank: full495physics→64code centeredexactlyat
E(phi)-E(0),rank64over8frozenMLPlayers, nominalprior/physicscritic, scale1.
Truezeroφ givesexactzeroresidualpointwise; no claimclosedloopbitwiseidentity.
Teacher smoke90 nominal336 body.0309031/joint.484577/fail1 versusfreshnominal
.0307015/.482877/fail1 (smallGPUclosedloopnumericaldifferences). Contextsmoke90
fullDR336 body.0453259/joint.630215/fail7, input/resetchecks pass NOTtracking.
CPU export8freshprocesses max5.48363e-6, SHA
58dfc9c05eac752cb687c6ced2bc04c65a7b6be1fc076a2aa68b5e0c6fc4c3e8.
Exportandcontextsmokeevals complete(session69548/77228).

Next: monitor930xx fixedcomparison; addsamearchitecture7code inputablation if
needed (zero centeredcode woulddisablelowrankcapacity, unfair). Finishmatched
FiLMzero/normalatpredeclared500 checkpoint. Continueteacher/studentwithoutreward
changes. Goaltoolstale blocked cannotreactivate viaAPI; donotmarkcomplete.

## Latest —03:23 UTC Sep7: matched comparison complete; physics-aware critic

ALL matched76/77/78 training and36evaluations COMPLETE. Final aggregate
`eval_v2/matched_controls_920xx/three_training_pairs_summary.json`; readable
`docs/adaptation_audit_results.md` top. Same nominal DR/nominal body1.134927
CI[1.104178,1.172023], joint1.091959 CI[1.075521,1.109281], all10 worse in each
of3 training pairs. Failures14vs7/3024; failure-differenceCI crosseszero.
SameDR body.899120, joint1.002319, failures39vs65; failureCI also crosseszero.
No longer run/monitor matched controller/evaluator. Final600xx UNUSED.

Current ORIGINAL-REWARD trainers (each4096env,2000planned,save100,seed77/RNG770001,
actorLR.001adaptive,critic5e-4,warmup50,std.1,entropy.0002,uniform/reference):
GPU0 fullphysics-prior scale.25 PID60228/session44168 (watch26792).
GPU1 compact-prior-lowrank scale.25 PID15241/session60572 (watch55309).
GPU2 oracle-large scale.25 PID34701/session82603 (watch77421), plus NEW
`fixed_reward_physics_critic_prior_seed77` PID24775/session69961 (watch43356).
GPU3 `fixed_reward_fullphysics_prior_wide_seed77` PID12669/session2760 (watch21726).
Latestnewwide andphysicscritic are same full495physics→64FiLM + frozen audited
nominalprior, scale1. Wide vs GPU0 differ ONLYresidualbound.25/1 andoutputdir:
all86actor/17critic initial tensors bitwise equal. Saved
`fixed_reward_fullphysics_bound_pair_initial_audit.log`. About.3% ofGPU0residual
samples nearbound (95% criterion), not wholesale saturation; hypothesis only.
Simulatoractionscale, torque/PDlimits,physics,noise,rewards all unchanged.

NEW `adaptation_critic.py`: actual495physics in separate CRITIC_PHYSICS group;
original warmstarted critic + zero-output state/physics value MLP256,128.
Original critic `policy`/`priv` includes states/references/torques but no explicit
DRparameters (inspected actual observation functions). This may reduce value
aliasing; NOT established as cause. PPO return/reward targets unchanged.
Newphysicscritic run differs fromwide ONLYthiscritic flag/outputdir, all86actor
and17commoncritic tensors bitwise equal, initial value maxdifferenceEXACT0.0.
Audit`fixed_reward_physics_critic_pair_initial_audit.log`. Newbranchnormalizer/
weights actuallychanged in64env4update smoke89; tracker/prior weights stayed
frozen. Teacher/context complete336v2 input/reset smokes passed, NOTtracking.
Latestwhole suite271passed54warnings45.35s,`fixed_reward_physics_critic_full_tests.log`.
Freshcriticaddition to an old initialization is intentionally rejected; resume/
freshoptimizer init from an existing physicscritic config is supported bycode.

STOPPED old1024env prior-lowrank PID21725 afterretained500:200–400worst~1.50;
5001.52026/fail7 (nominal1/frozenDR6). Artifacts retained. Other oldstops below.
Current bestoracle-large200 worst1.36271/fail2. Body.0388404, joint.589017,
root.296035: vsnominal1.26510/1.21981/1.19508; vsfrozenDR.87514/.93891/.85948.
OnthisONEdevseed its2failures bothfrozenDRfailures:0new,4recovered. Theyare
fightAndSports1_subject4 start2524/length130 andfallAndGetUp2_subject3
start4221/length25. No broadfailureguarantee; nofirst/secondstageacceptance.

Context PRETRAINseed88 completed2000/session46035, SHA
d88d54815239855a218791af0ba903c7b0bf66fa3cdedadfb8851169322077c2.
Teacher remains UNQUALIFIED prior200; controller frozen,only7Dencoder trained.
Finalworst1.50738/fail5;1000worst1.49980/fail4. Stillfarfromgoal.
Direct latent diagnostic newseed14644/512worlds: finalphysicalRMSE
[payload.34779kg,torso.53233kg,COMxyz.01869/.01432/.03236m,friction.4000,
armatureratio.022595], R²[.6586,.1841,.7983,.8988,.3812,.3525,-.0834].
Files`fixed_reward_compact_context_pretrain_seed88_probe500.json`and`_probe_final.json`.
No fitted decoder inthese direct scores; offlinewhole-episodeaverage ridgeprobe
is onlydiagnostic, never deployment evidence. No teacher/control transfer yet.

500latentablation, allsameworld fingerprints: normalbody.0424256/joint.624579/
root.342096/fail4;zero.0452096/.628877/.393450/fail6;shuffle.0426731/.622733/
.350253/fail4. Smallshuffleimpact: doNOT claimstrongenvironment-specificcontrol.

Implemented CAUSALSTATICMEANONLY (not action smoothing): ContextActor flag
context_mean_only with context_latent_mean=True,frozen7Dphysicalencoder only.
Prefixmean thenEMA1/horizon,resetper-world. No latentconcatenation, code stays7D.
`scripts/add_causal_physics_memory.py` converts a signed student to a NEWdir,
keepsALLactor tensorsbitwise, removesstaleoptimizer, records architecture-only
change/sourceSHA/rewardcontract. Currentdistill full-init rejects thisnewmemory
contract; don't accidentally useits oldconcat training path. Eval/PPO wrappers
already size memoryfromlatentdim. Exportexplicitmean/count/reset, manifest
mean_only checkedagainbyeval.269tests aftermemory,271aftercritic.

`fixed_reward_compact_context88_500_mean50`: full336v2/reset checks pass,body
.0421204/joint.622847/root.347467/fail4 (mixed tinyeffects, notgoal). Eightfresh
CPUprocessaudits passed,max5.72205e-6,artifactSHA
2a5681db8aaaf77d3738359e7c1c417018e76a922f2c0515c8a2dacacd8a97e4.
`fixed_reward_compact_context88_final_mean50` fullv2:body.0424749/joint.623662/
root.342089/fail5,notbetter. Itincludesfullpostprocessing sourcehashaudit;500
conversion predatedthatmetadataaddition buthash/weights/rewardchecks passed.
No final600xx used. Keepworkinguntilstage2achieved oruserasksprogress.

## Latest —02:56 UTC Sep7: new larger-batch teachers and context preparation

STOPPED lowrank30503 and bilinear56977 after retaining700. Bilinear500best
worst1.44603;600/7001.45850/1.45862, failures2/3. Lowrank400best1.44636;
6001.45935/fail7;7001.47403/fail6. Neither passes. No artifacts deleted.

Current GPU0 `fixed_reward_fullphysics_prior_seed77`, PID60228/session44168,
watchsession26792. Frozen audited nominal prior + full495 physical-only
parameters → learned64 code → FiLM residual. No privileged current state,
height/contact/reference; full per-joint armature and encoder bias included.
Input wrapper freshly reads physics every attach. New physics schema has a
no-state-read regression test. Full4096env initial identities all EXACT0.0.

GPU1 `fixed_reward_compact_prior_large_seed77`, PID15241/session60572,
watchsession55309: same compact7 lowrank architecture/prior as GPU2 old branch,
but4096env, residualbound.25, actorLR.001 adaptive, other settings unchanged.
GPU0 fullphysics uses these same settings. Both2000planned, save100, seed77,
modelRNG770001, warmup50, std.1, entropy.0002, original reward hash unchanged.
These optimize new input/architectures; not an isolated one-factor ablation of
batch/LR/bound. Nominal baseline was4096globalenv/.25bound/LR.001adaptive;
we did NOT copy its entropy.005 or alter reward. GPU2 prior lowrank1024/
scale1/LR2e-5fixed PID21725 continues:100worst1.53011/fail6;
2001.50188/fail4. Waiting for later checkpoints.

GPU3 `fixed_reward_compact_context_pretrain_seed88`, PID15311/session46035,
watchsession34389(interval500): teacher GPU2 prior200, same frozen controller,
only causal50frame+noisyIMU right-aligned encoder learns7 physical coordinates.
1024env,2000distillupdates, actionMSE+latentMSE(weight1), teacher mixture fades
tozero500. This is system-identification PREPARATION using an UNQUALIFIED teacher,
not a completed first/second stage. If transferring to a later teacher, transfer
only compatible encoder weights; existing full-student init would also overwrite
the controller and MUST NOT be mistaken for encoder-only transfer.
New read-only direct7-code probe reports actual latent physical RMSE/bias without
fitting a decoder. Not yet run on a substantial trained checkpoint.

Fullphysics teacher/context smoke87 complete336v2 reset/input checks passed, not
tracking. Context CPU export eightfreshprocesses max5.66244e-6,
artifactSHAff940fdd6da50f282c9b449923442690c2432486097c78b3bc73d2edcb8f3834.
All-tests run `pytest tests -o addopts='' -q`:266passed54warnings42.78s;
afterward added watcher-race and direct-code tests (targeted checks pass/pending).

Matched all3 training pairs COMPLETE, exactfinal1000 saved. Evaluator PID818
exited spuriously when final78file was younger than15seconds but controller
already exited normally. Fixed final_pair_ready to wait for existing fresh files
even after controller exits, still reject missing files. Three aggregation/watcher
tests passed. Restarted evaluator PID13503/session31627, log
`v2_matched_controls_920xx_watch_resume.log`, resumes existing results and freezes
78finalSHA. GPU1/3 run remaining12 evaluations concurrently with new trainers.
No retraining/reselection. Aggregate all3 only after78complete. Final600xx unused.

## Latest —02:44 UTC Sep7: audited nominal prior, fixed reward

Current trainers: GPU0 lowrank PID30503/session38488; GPU2 bilinear
PID56977/session32185 plus nominal-prior lowrank PID21725/session61746;
GPU1/3 matched seed78 PIDs26336/26337. Prior watcher session99450,
`fixed_reward_nominal_prior_lowrank_seed77_v2_watch.log`. Other watcher/controller
identities below unchanged. Modulated4834 and oracle-frozen30884 STOPPED at500
after development plateau; artifacts retained. No shaped teacher restarted.

New `fixed_reward_nominal_prior_lowrank_seed77_v2` freezes the exact original-
reward nominal baseline as an action prior, then adds compact-physics low-rank
adapters. Provenance: `docs/adaptation_nominal_prior_audit.md`; source SHA
800da8c40016bba3263e685e9694e51e82b83a5e1c3df3e2e80543bd47ad8f1f,
saved clean training commit692f90bf9dd9900229f8a5d201fb20e1db2c314d,
all53 tracker/preprocessing tensors bitwise original. Generic historical
checkpoint rejection remains enabled; this is not permission for shaped init.
Checkpoint_initial has matching original reward signature, empty reward_changes,
explicit prior lineage, and all three initial action identity checks EXACT0.0.
Source actor construction populates its TensorDict cache: the first full attempt
failed audit before training because the audit shared that mutable observation.
Fixed audit using independent recursive clones, without relaxing tolerance.
1024env identity77 smoke passed, then fresh v2 full run launched02:35:43.

Prior teacher/context smoke86 and 8-process CPU export passed interface/reset/
no-GT checks, NOT tracking acceptance. Export artifact SHA
24d42fc195c6c13dd97326cf956f1d57f3f9a663b833324f9624dac0796f7f87,
max independent difference5.1856e-6. Latest full regression185passed/48warnings.

Matched independent training seed77 complete: same nominal test DR/nominal
body1.119198, joint1.092528; all10 metrics worse, failures5vs3/1008.
Together with seed76, hierarchical interim aggregate body1.134618
CI[1.09964,1.17476], joint1.096193 CI[1.07990,1.11205]. Seed78 pending;
do not treat evaluation seeds as independent training runs.

Current development: oracle-frozen400 worst1.36704/fail6 (stopped);
bilinear500 worst1.44603/fail4;600 worsens1.45850/fail2;
lowrank400 worst1.44636/fail4;5001.45160/fail3. Reference nominal fail1,
same-physics frozen DR fail6. NO candidate meets fixed-reward tracking and
failure criteria. Final student600xx seeds remain untouched. Continue work.

## Latest —02:17 UTC Sep7: fixed reward, physical-code architectures

STOPPED original-reward full-base fine-tune PID4677 atretainedcheckpoint200:
eval100worst1.96812/fail11;eval200worst1.98381(bodyrot)/fail24vsnominal1/frozenDR6.
No more full-base unrestricted updates. Allartifactsretained.
STOPPED plainlatent PID30793 atretainedcheckpoint500:200–500plateauworst~1.47,
latestfail5; allartifactsretained. Originalwatcher43870 stillwatchesoracle.

CURRENTtrainers (all ORIGINALreward, nohistoricalshapedinit):
GPU0 modulated4834/session12444 + NEW lowrank30503/session38488;
GPU2 oracle-frozen30884/session88089 + NEW bilinear56977/session32185;
GPU1/3 matchednominal/drseed78 PIDs26336/26337. Controlsseed77completed;920xx
evaluationinprogress. Seed78actualinitialactor/criticbitwiseauditPASS, bothreward
signaturesabee73632d079f08c160584b04ace15532ecf6bd9dfa8139979af518d8a1553c.
Controller59123/session48845 finishesafter78; evalwatcher818/session91984.
Newbilinearwatchersession40272/logfixed_reward_bilinear_seed77_watch.log;
newlowrankwatchersession32524/logfixed_reward_lowrank_seed77_watch.log.

Bothnewphysicalteachers use compact_physics7 DIRECT(nolearnedprivilegeencoder):
[payloadkg/3, torsoΔmasskg, torsoCOMxyz/.075, meanfootfrictionΔ/2,
 meanarmatureratioΔ/.2]. AllcenteredatTRUE NOMINAL, freshlyreadactualfields,
noheight/contact/truecurrentstate/truereference. Originalnoisyfrontendunchanged.
FullDRdistribution/payload/sensorynoise/terminationconstraintsunchanged; selected
actorinputs omitencoderbias andper-jointarmaturevariation, notenvironmentDR.

Bilinearresidual: MLP(features)→29×7 coefficients, multiplycodeand sum,
thenoriginalboundedscale1*tanh. Lowrank: sevenphysicalcoordinatesgate rank14
updates(2directionspercoordinate) throughoutall8originalMLPlinearlayers;
originalandreferencecopyweightsallfrozen, onlydown/upadapters+stdlearned.
ReturnsconditionalMLP−frozenMLP insidestandardresidualbound.
TeacherzeroTRUEphysicalcode givesEXACTzerocorrection foranylearnedweights;
NOT aDRfailureguarantee, NOT guaranteedifstudentpredictsimperfectcode.
Full1024envinitialaudits forbothactualjobs maxnominalcodeactiondiff=0.0 and
initialDRactiondiff=0.0 incheckpoint_initial.infos.bilinear_function_audit.
Bothsamebudget/settingsasseed77comparators:2000planned,1024env,save100,
LR2e-5fixed,warmup50,std.1,entropy.0002,uniform/reference,noinitcheckpoint.

Student ofeither is ContextAdaptationActor with7D causalhistoryencoder output;
samefrozenteachercorrectionnetwork, actionMSE+codeMSE labels only, noGTactioninput.
Actual64env4updatebilinearsmoke84 andlowranksmoke85_v2 PASSED; onlyintended
weightschanged. Correspondingcontextsmokespassed, complete336v2teacher/student
structuralruns passed (nottrackingpasses); CPUexports8freshprocessespassed:
bilinearartifactSHA1122f4fed23cfaf459f27b9df224691e1cfcf8461eca4c90abee3d73ee18051b,
maxdiff5.1260e-6; lowrankartifactSHAb2b17d39e52d2e02c4353e580366b79b713417ddbe48bf8dc16e90f7461a1ada,
independentmaxdiff5.9456e-6,8349deployableinputs. Paths
fixed_reward_bilinear_context_smoke84_export andfixed_reward_lowrank_context_smoke85_export.

Firstlowranksmoke85FAILED beforetraining: Sequential.children()deduplicates
sharedELUmodules inRSL-RLMLP. FixedusingSequentialiterationlist(base_mlp), kept
failedfolder/log, addedrealRSLMLP regression. Contexttrainabilityalsoexplicitly
re-freezeslowrankreference/basecopies afterglobalresidual.requires_grad toggle.
LatestFULLregressions **183passed**,48warnings,11.63s;
fixed_reward_lowrank_full_tests.log. Reward/signaturelock remainsenabled.

Latestqualifying-candidateNONE: oracle-frozen400worst1.36704/fail6,
oracle3001.36900/fail2, oracle5001.37573/fail5;modulated3001.44716/fail3;
bilinear1001.49207/fail8. AllagainstfreshSAMEfixednominalcheckpointfail1 and
samephysicsfrozenDRfail6. No finalstudent600xx used. Keepworking untilstage2done
oruserasksprogress. DoNOT claimsmallsmokepassesaretrackingimprovement.

Unimplemented possiblefuture cleanwarmstart: frozenprior from the EXISTING
original-reward nominal checkpoint800da8... (notanyshapedmodel), plusphysical
adapters; wouldneedexact-SHA provenanceaudit andunchangedtrackerfrontendweights,
sincecurrenthardlock correctlyrejectsALLlegacyunauditedinit. DoNOT silently
whitelist ormistake thisunimplementedideaforanexperiment. Currentjobsallstart
fromoriginaltracker, sohaveunambiguouscleanrewardlineage.

## Latest —01:53 UTC Sep7

Fixed-reward teacher architecture runs added, same common original tracker and
training settings as other seed77teachers:
GPU0 fixed_reward_modulated_seed77 PID4834/session12444 (latent-conditioned
hidden layers, tracker frozen); GPU2 fixed_reward_oracle_unfrozen_seed77 PID4677/
session68012 (same clean oracle frontend but base MLP trainable). Both2000planned,
save100. Additional watcher PID8445/session73441,
fixed_reward_architecture_seed77_watch.log. GPU0/2 each host2trainers; original
paired controls GPU1/3 unchanged. All six jobs use ORIGINAL rewards.

Actual modulated64env4update smoke83 passed: action output initiallyzero,
trackerallweightsunchanged, modulation heads receive gradients and update.
Teacher→context student smoke passed; full336v2teacher/student structuralchecks
passed, NOTtrackingpass. CPUexport8independentprocesses passed withmaxdiff
6.7949295e-6, SHA c02cd72961cc558a32b0bf0a6f904b3538e4c2aa8eed3a45959d35f92ef181ec,
fixed_reward_modulated_context_smoke83_export/policy_cpu.ts,8349deployableinputs.
Actual forbidden--failure-penalty1 CLI invocation was rejected BEFORE env/output
creation. Original task/sim/action/termination code remains tracked-unchanged.
Full latest regressions **180passed**,45warnings,12.98s;
fixed_reward_full_regression.log. Ruffchangedfilespassed.

Strict development scoreboard scripts/summarize_fixed_reward.py writes
fixed_reward_development_scoreboard.json, only clean-lineage runs; all10errors,
failure≤fixednominal andsamephysicsfrozenDR, fingerprint requirement, neverfinal.
Fresh SAME checkpoint references eval_v2/fixed_reward/reference_*.json include
physicalfingerprints. Currentnominalbody.0307015/joint.482877/fail1; currentfrozenDR
body.0443820/joint.627344/fail6. Earlier v2singledevreferences hadslightlydifferent
closedloopnumbers (nominalbody.0308825, frozenDRfail7); seeds/starts/SHAidentical,
do NOTclaim bitwise repeatable GPU closedloop. Nominal target checkpoint unchanged
andnewmeanbodybaseline is slightlySTRONGER. Allnewteacher physicalworldfingerprints
matchfreshfrozenDR. Allcandidate screens fail. Oracle200worstbodyrot1.37602,fail2;
latent200worst1.46952,fail4;latent300worst1.48066,fail7;mod100worst1.48764,fail8.

First new strict trainingpair seed76 final920xxreport COMPLETE. Same nominaltest
DR-trained/nominal-trained ratiosbody1.149977(CI1.11788–1.18871),joint1.099792
(1.08534–1.11629),root1.209815,bodyrot1.254622,all10worse. Fail6vs2/1008,
4new/0recovered. SameDRtestbodyratio.904769,joint1.001562,fail16vs22;
report eval_v2/matched_controls_920xx/training_seed76_comparison.json.
Thus direction replicates oldseed42 butnotidenticalmagnitude/budget; remaining
seed77at~500, then78pending. New aggregate_matched_controls.py (2testsPASS)
will poolthreeINDEPENDENTtrainingpairs withtraining/eval/motionbootstrap after
completion; refusesmissingdata/mismatchedphysics. DoNOTtreatevalseedsastrainingreplicates.

Unimplemented possible next architecture ifcurrentonesplateau: physics-only
teacher code centeredatnominal with differential residual f(features,z)-f(features,0)
andfrozenoriginalbase; guaranteeszero learnedcorrection fortrue nominalphysics,
NOTnoregressioninDR andNOTautomaticnominalidentityforcontextstudent. Merelyanidea,
notimplementedorclaimed. DoNOTchange rewards, initializations toshapedlineage,
orphysical DR tosolve remaining tracking/failure gaps. Goalstillincomplete.

## Latest —01:39 UTC Sep7: clean fixed-reward restart running

GPU0 fixed_reward_latent_seed77 PID30793/session72630; GPU2
fixed_reward_oracle_seed77 PID30884/session88089. Both from ORIGINAL tracker,
NO checkpoint initialization; same seed77/modelRNG770001,1024env,2000updates,
save100,LR2e-5fixed,critic5e-4,warmup50,std.1,entropy.0002,uniform/reference.
All tracker modules FROZEN. Latent-only uses state_physics566→64; oracle adds
the prior separate clean frontend, nothing else differs. Actual pre-update
actor AND critic tensors bitwise equal between these two. Reward20terms/dt
signature abee73632d079f08c160584b04ace15532ecf6bd9dfa8139979af518d8a1553c.
Watcher session54996, logfixed_reward_seed77_watch.log, eval_v2/fixed_reward.

Hardlock implementation adaptation_reward_contract.py: CLI rejects all shaping
flags; reward mutation code REMOVED from adaptation_train; config/function/
weight/params/dt equality before env construction, signature saved to checkpoint.
Init/resume/BC teacher/DAgger teacher and optional student init require matching
audited fixed-reward lineage, so old shaped or unaudited models are eval-only.
Eval now records training_reward_audit; comparison rejects missing lineage and
ANY observed failure increase (old1pp rule removed). Separate same-physics
frozen failure comparison still needed. 174 regressions passed, then31 targeted
reward/comparison tests passed after acceptance update.

Smokes fixed_reward_latent_smoke82/oracle_smoke82 completed3updates64env;
alltracker tensors unchanged, encoders/residuals updated, identical reward hash.
fixed_reward_context_smoke82 clean teacher distillation completed3updates;
full336 v2 student check session19329 running. Teacher oracle initial fullv2
body.042393/joint.604281/fail5/336, still far from goal. No training success yet.

Prepared optional latent_modulation architecture: same shared residual layers,
latent drives bounded per-hidden-feature gain/bias, zero action output atinit,
defaultFalse leaves currentjobs unchanged. Unit/TorchScript checks passed;
actual GPU teacher→student/export smoke still required before full use.
Controls seed76complete with920xxreport; seed77 PIDs25544/25545 continue,
then78. Seed76/77 trainers began before hardlock source patch but have unchanged
original rewards; seed78 will automatically get the lock. No retroactive claims
that their saved configs already contain signatures. Final student600xx untouched.

## Latest —01:23 UTC Sep7: USER REQUIRES ORIGINAL REWARD, HARD OVERRIDE

No reward additions, penalties, reweighting, shaping, alignment changes or
adaptive reward objectives are permitted. Improve inputs/architecture and use
distillation; failures are an evaluation gate, not a new reward term.
Verified trainers GPU0 PID52134 oracle_failure_guard_seed80 and GPU2 PID2685
context_static_correction_seed81_v2 were SIGTERM-stopped at checkpoint300.
All artifacts retained. Original-reward paired controls GPU1/3 continue.
Prior shaped teacher/student checkpoints are historical, NOT evidence of the
fixed-reward goal, and NOT clean initialization for new causal experiments.
Both stages need revalidation under this constraint. Fresh teachers start from
the original common tracker; stage two may distill a clean-lineage teacher.
Adding fail-closed reward configuration/function/weight/dt signature checks.
No final student600xx seeds have been used. Goal remains incomplete.

## Latest —01:10 UTC Sep7

Static teacher actualstudentconversion PASS: allactorweights unchanged across
onecriticwarmupdate, correction_use_privilegeFalse andmetadata privilegedFalse.
Full336studentv2 passed61624, fail0 (nottrackingpass). CPUexport79389 passed8fresh
processes, maxactiondiff8.7023e-6, artifactSHA
c4af408f9c44289c04c9bddf0fa86a9a98d6aa4bae529e71019b3c65bee797b5,
`context_static_student_smoke81_export/policy_cpu.ts`. Inference8349deployableonly.

Firstfullstaticlaunch `context_static_correction_seed81` FAILEDbeforetraining:
initialaudit compared uncachedsource to cachedtarget, maxdiff.001566>2e-4.
Allfileskept (no trainingcheckpoint). Fixed like-for-like comparison forBOTH
cachedtraining anduncacheddeployment; source residual slicealsoexplicitcontiguous.
Actual1024env2step `context_static_identity_audit81` nowPASSED withoutloosening
tolerance; check checkpoint_initial.infos for exactmaxdiff (74202 result).

FULLrestart `context_static_correction_seed81_v2` GPU2 PID2685/session2662,
logv2_context_static_correction81_v2.log. Source linear1050, extraONLYtrue3static
physics, allsource+decoderfrozen; .25boundedcorrectionMLP only(+std).
1024env1500updates/save50, LR1e-5fixed,warmup50,std.1,uniform/reference,
tracking4/primary2linear/aux8linear/rightarmangular8/offset1, failurecost1000.
Actualinitialactionequalityassert runsBEFORE firstupdate. Needwatcheronceconfigready.
Originalsafe teacher52134 GPU0 continues; dev50/100maxratio.9863/.9905,
failure3/336 both, no currentfailureimprovement. Controls44382/44383 GPU1/3continue.

NEWautomatic finalpairedcontrol evaluator `watch_matched_control_evals.py`,
PID818/session91984 (restartedafterclosure-binding lintfix; noevalhadbegun), log
v2_matched_controls_920xx_watch.log. Waitscontroller59123 forfinal1000 eachseed76/77/78;
freezesSHA BEFORE92001/2/3 evaluations; bothnominal/DR test, GPU1/3 co-located
withnexttrainingpair. Outputs eval_v2/matched_controls_920xx. Samephysics
fingerprints+initialbitwiseaudits enforced; pertrainingseedreport, no pretending
threeevalseedsarethreetrainingseeds. 600xx untouched. Goalstillincomplete.

## Latest —01:03 UTC Sep7

ALL54 audit910xx evals complete, matrix11825/30655 exit0. Finalsummary
`audit_910xx/summary.json`, readable `docs/adaptation_audit_results.md`.
TeacherDR8fails vsfrozenDR9, BUT vsnominaltarget1; failureCI[+.099,+1.488]pp.
Teacher nominal8vsfrozennominal2. Studentrotation350DR4vsnominal1, maxratio1.086.
Height+contact allpathreplacement all10DRmeanerrorchanges<1%, NOTmajorbenefit.
Oldmatchedsame-nominal bodyratio1.28257/joint1.34924, all10worseDRtraining.

ControlsREALseed76 launchedGPU1PID44382/GPU3PID44383; controller59123/session48845
thenruns77/78 automatically. All62actor+17critic initialstates BITWISEEQUAL,
sameagentcfg. 1000updates1024env ~35minperpair. No failurecost inthesecontrols.

FAILED identityfeatureDAgger24710 stoppedstable800. Eval100/200/300maxratio
1.207/1.229/1.251, failures2/3/4; laterwatcher87226 stillfinishingthrough800.
Allcheckpointskept. SourceexactidentityandCPUexport8freshprocessespassed;
thisisalgorithmicfailure, not deployabilitysuccess. ExportmanifestSHA370ff5...49bc,
maxactiondiff1.0669e-5.

GPU0 NEW oracle_failure_guard_seed80 PID52134/session86412, watcher12799/log
v2_oracle_failure_guard80_watch.log. Sourceclean500, freezeswholetracker,
trainsprivilegeencoder/residual/std. 4096env1500updates/save50, LR3e-6fixed,
warmup50,std.1, originalsampling/starts, originalteachertracking4/primary2exp/
aux8exp/rightarmangular8/offset1; adds truefailurecost weight−1000 (dt .02,
cost20/event), no timeoutpenalty/termination/physicschange. Actual64env3step
smoke andfull336v2 passed; source tracker allweights/buffersunchanged.

GPU2 NEWstaticcorrection actual64env3step smoke81 PASSED, onlycorrectionMLP/std
updated; decoder/context/source unchanged. Full336GTmodev2 passed43510,
fail0; thisisstructural, nottrackingpass. No trueheight/contact/state/reference,
only centeredpayload/COMx/y truths. Studentconversion1criticwarmupdate15159
running in context_static_student_smoke81; next verify noactorchanges, fullv2
studentmode, CPUexport, then launchGTtraining. Staticfullrun NOTlaunchedyet.
TrainCLI nowaudits actualinitialactions onadd-static-correction and storesmaxdiff
in checkpoint_initial.infos; priorGPUstatic3stepsmoke predatesthatnewassertion.
Need check newfullinitialactionaudit, cannot claim it alreadypassed in GPU.

## Latest —00:53 UTC Sep7

New stage-two identity-frontend DAgger LAUNCHED GPU2 PID24710/session51814,
`context_identity_features_dagger_seed79`, logv2_context_identity_features_dagger79.log.
Teacher clean500, source studentlinear1050; sourcecontext/base/residual frozen,
only zero-output additivefeatureadapter trained (256/128). Actions initially
EXACTLY source (0.0 maxdiff). 1024env3000updates/save100, LR3e-5, featureloss.01,
student-onlyrollins, seed79/RNG790001, existingIMU/rightalignment unchanged.
Opt-in warmadapterupgrade inDistillCLI; allothersemanticchanges stillrejected.
CPUtestspassed,64env3updates onlyadaptertensorschanged; full336v2smokepassed
allreset/privilegeguards (nottrackingpass). CPUexportaudit15406 running.
Plaincontext31555 stopped atstable1450, allcheckpointsretained; sharedwatcherfinishing.

Newpairedcontrols controllerPID59123/session48845 waits audit30655, then GPUs1/3
seeds76/77/78, each1000updates1024env, bothno-privilegedresidualsameoriginaltracker.
SameLR2e-5fixed/std.1/warmup50/uniform/reference/rewardsoriginal. Actual64env2step
nominal/DRsmokes have all62actor +17critic initialtensors BITWISE IDENTICAL and
sameagentcfg. Realinitialcheckpoint saved pre-update. Thesecontrols notstarted yet.

54-evaluationaudit nearingthirdseedDR. Partial91001 summary at
`audit_910xx/summary_partial.json` andv2_audit_partial91001.txt. Allsame-physics
fingerprintchecks passed. Height+contact removal DR body+0.465%,joint+0.239%,
root−0.076%; no largefirstseeddependence. TeacherDR2fails vsfrozenDR5/studentDR0,
butteacher2vsnominaltarget0, stillmustfix. Seed91002 teacherDR3 vsfrozenDR1;
do NOT generalize firstseedcount. TeacherfailuresoftenfallAndGetUp; preserveall.
OldmatchedDRvsnominaltraining onnominal: firstseed10/10errorsworse (body+28.3%,
joint+34.9%). Full3seedreportpending; additionaltrainingseeds areessential.

Added opt-in trainingfailurecost (truefailure only, no timeout, normaldt scaling),
default0; notusedbyold/current jobs ormatchedcontrols. StaticcorrectionCPUtests
passed initialidentity/sourcefreeze/GTpoisoning/freshphysics/privateexportreject,
but actualGPUstaticcorrection remainsunlaunched. Prefer safety-focusedteacher
refinement onceGPU0auditdone, pendingcompletefailureaudit, overuncertainnewbranch.

## User-directed audit —00:40 UTC Sep7

User explicitly requests same-nominal-test DR/no-DR training control, privileged
teacher validation, all-path height/contact ablations, continued distillation,
and no failure-rate degradation. Continue until stage two succeeds OR user asks
for progress. Details: `docs/adaptation_audit_plan.md`.
Matrix script `scripts/audit_adaptation_matrix.py`, session11825, log
`v2_audit_matrix_910xx.log`, physicalGPUs0/1/3, 54 v2 evaluations frozen before
91001/2/3 audit/development seeds (NOT600xx). Includes same-physics frozen tracker.
Rotation2/4 trainers61085/61167 SIGTERM-stopped at checkpoint1000, allfileskept.
Plaincontext31555 GPU2 remainsrunning. Tests150 passed; all-path privilege
poisoning tests and actual42env3stepGPUsmoke51253 passed. No acceptance claim.
Snapshot actualphysics fingerprints +failure termination terms added audit-only.
Training CLI adds opt-in construction/rollout RNG reseed and initialcheckpoint
save for future paired controls; default paths unchanged. Newpair notlaunchedyet.
Staticcorrection integrations exist but notvalidated/launched; paused for audit.

## IN PROGRESS —00:16 UTC Sep7

New static correction module written, NOT integrated/tested/launched yet:
`adaptation_correction.py`, ContextPhysicsCorrectionActor subclassesContextActor.
Frozen originalsource +zero-output correctionMLP256/128, boundedscale.25.
Truecode3=(payloadkg−2,COMx/.075,COMy/.075); NO truecurrentstate/cleanreference.
Studentmode usescopiedcontextphysicshead indices3/5/6 fromcurrent64Dlatent,
clipped[-1,1]. Originalcontext/base/residual/referenceallfrozen. Staticwrapper
freshlyreadsactualmodelmass/COM, no stalelabels. Initializationhelpercopiesall
sourceweights, duplicatephysicshead, newdelta weightszerooutput. Needstrict
GPUinitialactionequality, frozen-source updateaudit, GTpoisoning, eval/export
classification guard: subclassisContextActor but TEACHER MODE MUST NOT be
classifiedasstudent. Centralconfig_requires_privilege helperinnewmodule.
TODO integrateCLI/load_actor/wrapper/studentclassification/probe/export BEFORE
anyrun. Noexistingjobusesnewmodule. No newphysicsactuatorlimit/rangechange.

Lowexplorationrotation4 PID17411 stoppedafterstable350; at300 max1.1724,
body1.0529,joint1.1092,rot1.1211. Allfileskept. GPU0nowfree fornewteacher
oncevalidated; other3existingPPOscontinue. Currentbestdevstillrotation4_350
max1.0562; later550max1.1059. Userinformedofnewteacher-correctionroute.

## Latest update —23:59 UTC

NEWbestdevelopment: rightarmrotation4 checkpoint350 seed10001max1.0562,
body.9978,joint1.0351,root1.0562,bodyrot1.0431,fail1/336. Notstage2pass;
singledevpointonly. Keeptraining, do notprematurelyconsume600xx.
At350rotation2max1.0784/bodyrot1.0496; lownoise100max1.0886/bodyrot1.0611.
Plaincontext600max1.0849/bodyrot1.0675, stillrunningbutnooverallrecord.
Current4activejobsPIDs unchanged23:55. No newcodeimplemented sincepose reward.
All new hypotheses beyondcurrenttrials remainunimplemented; donotconfuse
brainstorming(referencefrontends, readoutfeatures,gravitycompensation)withwork.

## Latest update —23:55 UTC

GPU0lownoise rotation4 PID17411/session38906, watcher67500/log
v2_context_right_arm_rotation4_low_noise_ppo_watch.log. Otheractive unchanged:
GPU1rotation2 61085, GPU2plaincontext31555, GPU3rotation4 61167.
Allfulltests2351 session92718 exit0; ruffpassed. No codechanges sincepose reward.

CLOSED diagnostichypothesis: currentresidual range saturation is NOT supported
by existing logs. ReadTensorBoard withoutchangingmodels/physics: oldlinearall
1497–1499 residualmax.78–.81 andsaturationfraction0; rotation4steps317–319
max.70–.77/fraction0; plaincontext591–593max.80–.84/fraction0. RMSabout.09.
Thusdonotexpandresidualrangeonassumptionitishitting±1. Actualforcesdoapproach
actuatorlimitinsomewriststeps (previousdiagnostic), a differentcondition.
KnownnominalG1 action-scale formula.25*effort/stiffness wasread-onlyinspected,
butno action scale, gain, torque limit or residual bound changed.
ContextshuffleactionRMS~.031 inrotation4,~.026inplain; notperformanceablationproof.

Earlylatestpointsat23:49: plain400joint1.0271/bodyrot1.0493/root1.081;
aux500joint1.03845/bodyrot1.06045/root1.094(stopped). Rotation4at200
bodyrot1.04292 butroot1.09931, stillFAIL. Preserveall10metrics/failureguard.
900xxbesttested remainslinear1050 all10improvedvsolderPPO butworstbodyrot
1.07378; nofinal600xxuse. Continuegoal, donotclaimsuccessfromongoingjobs.

## Latest update —23:51 UTC

PPOauxiliarysupervision31480 nowSIGTERM-stopped afterstable550, allfileskept.
At500 max1.09429/bodyrot1.06045/joint1.03845, noadvantageoverplaincontextPPO.
Plain31555 continues onGPU2: at400 body.9957/joint1.0271/bodyrot1.0493,
root1.081; stillfailswholetracking. No claimofauxiliaryidentificationbenefit.

GPU0 NEW `context_right_arm_rotation4_low_noise_ppo_seed75`, session38906,
PIDfromlatestps/tool. ExactmatchtoGPU3rotation4 setup/source/seed except
initialactionexplorationstd .03 vs.1. This DOESNOT reduce sensor/reference
noise, physicalDR, payload or evaluationnoise. Actorfrozenbase/context,
trainresidual, LR3e-6,warmup50,4096env1500updates/save50. Source linear1050
sameSHA2b21aa...ccc3f. Logv2_context_right_arm_rotation4_low_noise_ppo.log.
Needstartwatcher afterrunconfigready. Otheractive:GPU1rotation2 PID61085,
GPU2plain31555,GPU3rotation4 PID61167. At200rotation4bodyrot1.04292,
joint1.03464,root1.09931/fail0; rotation2bodyrot1.0524,root1.09364/fail2.
Earlyposegainbutnooverallpass. 600xx remainsunused.

Read-onlyverified actualspv1 sensor cache: history androbot_key_body shareONE
noisysample withsamejointq/qd/gyro keys, so noindependentsamplefusionloophole.
No frontenddenoising,retrieval/datasettemplating,decoder-inputaugmentation,
analyticgravitycompensation orfeatureadapterwarmstart implemented. Those were
hypotheses only; evaluatecurrenttrials first. Goalremainsincomplete.

## Latest update —23:36 UTC

Right-armPOSE trials nowLAUNCHED, allsmoke/fulltests passed:
- GPU1 `context_right_arm_rotation2_ppo_seed75`, PID61085/session88517.
- GPU3 `context_right_arm_rotation4_ppo_seed75`, PID61167/session54432.
- Sharedwatcherlogv2_context_right_arm_rotation_pair_ppo_watch.log (latesttool).
Both4096env1500updates/save50, same source linear1050, seed75, fixedLR3e-6,
warmup50,std.1, uniform/reference, base+contextfrozen, originaltracking4,
primary2/aux8 bothlinear, rightarmangularVELOCITY8, offset1. OnlyaddedPOSE
termweights2vs4 differ. Nootherrewardweights, physics,sensor,evalchanges.
Unitrotationtests passed27615, fullsuite65126 exit0, ruffpassed; actual64env
2updates changesonlyresidual/std. Fullv2smoke76036 exit0/allasserts/fail0,
body.0308914,joint.502625,root.267011—nottrackingpass.

GPU3oldmemoryPPO52483 SIGTERM-stopped afterstable650; checkpointsretained,
watcherfinishing. Eval550/600max1.08436/1.08438, no broadadvantageoverlinear1050.
GPU1oldlinear18150 completed1500normally, session24519 exit0. Sharedwatcher63328
may still be finishingfinaleval; do notkill newrotationtrainerwhencheckingGPU1.
GPU0aux31480 andGPU2plain31555 continue, around300updates. All4 authorized
GPUsassignedtoactivePPO. GPUs4–7untouched.

Jointdiagcomplete72179 exit0: linear1050rightwristpitch/yawRMSE .20592/.19598rad.
Atleast95%-force-limit final-substep sample fractions .078743/.039952.
This isNOT whole-substep dutycycle and doesNOT provefundamentallimit, but
physicalcapacitymaycontribute; limitsremainunchanged. File
eval_v2/linear1050_joint_diagnostics_seed10001.json.
Teacher'srightarmerrorisonly2–7%belowstudent, despitebetterglobalaverage;
reportthiscaveat, donotattributeallloadedarmerrorsolelytocontextinformation.

Read-onlyobsconfigcheck: estimatorhistory jointqnoise.01,qd.5,gyro.2;
robot_key_body uses SAME spv1 sensor functions/noiseparameters, biasedTrue.
Not a noisier secondobservation source to magically replace. Rootquatnoise.1.
No FKreplacement/analyticalgravitycompensation/warmfeatureadapter implemented.
600xxstillunused; stage2incomplete. Needcontinue ratherthanfinalmerelylaunching.

## Latest update —23:32 UTC

Per-bodyv2diagnostics complete, seed10001, allsameprotocol andall22bodies:
`nominal_body_diagnostics_seed10001.json`, `linear1050_body_diagnostics_seed10001.json`,
`teacher_clean500_body_diagnostics_seed10001.json` (77979/35720/92350 exit0).
Studentrightshoulderyaw1.178×nominal,wristroll1.343,wristyaw1.605,
shoulderroll1.126,elbow1.354,wristpitch1.501. Mostlegs/leftarmBETTERthannominal.
BUTcleanprivteacher alsohaslargearmerror: wristyaw.24042rad vsstudent.24620,
nominal.15342; wristpitch.18294 vsstudent.19670,nominal.13106. Teacher'slow
aggregate comesfromotherbodyimprovements. Thusnotsolelycontextinformationgap,
mayberewardallocation/loadedactuatorlimit. Thisisdiagnostic,notproof oflimit.
Nativefailnom1/student0/teacher4 intheseevals; includeallterminalerrors.
Joint/torquesaturationdiagnostic1050 underway GPU3 (latesttoolsession).

IN PROGRESS, fullrunNOTlaunched: targetedright-armPOSE reward, rather than
globalbodyrotationboost. Existing right_arm_angular_tracking penalizes angular
VELOCITY, notorientation. Addedright_arm_rotation_tracking: sameyawalignment
asbodyrotmetric, meansixrightarmbodies, linear1-min(error/.15,5).
OptionalCLI--right-arm-rotation-weight default0; allotherrewardsuntouched.
Sharedright-armindexhelper preservesoldangularreward behavior. Unit tests
left-armexclusion andglobalyawalignment passed (27615); ruffpassed.
64env2updateGPU1smoke completed15341 exit0, base+contextfrozen, onlyresidual/
stdchanged. v2fullsmokeeval76036 running; fulltests65126 running.
Proposedfulltrials weights2and4 fromlinear1050, nootherchanges, GPU1onceold
linearfinishes andGPU3ifmemoryPPOplateauevidencewarrantsstop. Notlaunchedyet.
No analyticgravityfeedforward, reference-conditionedcontext, orwarmfeature
adapteraddition implemented; these were hypotheses only.
600xxstilluntouched; goalnotcomplete. Chinese statusupdated23:22.

## Latest update —23:21 UTC

Allthreeadditionaldev900xxlinear1050 evaluations completed (40486 exit0).
`development_validation_900xx/linear1050_comparison.json`: ratios body1.013865,
joint1.038401,root1.051636,anchorrot.998912,bodyrot1.073777,jointvel1.018863,
anchorlin1.050690,anchorang1.047886,bodylin1.046450,bodyang1.051971.
ALL10 improved from physicsPPO750 on this cohort, but stage2stillFAIL.
Failure5/1008 vsnominal0,CI[0,+1.1905]pp. RootCI[.97930,1.12468],
bodyrotCI[1.04760,1.10418]. ConfirmationflagsFalse/600xxuntouched.
Prefreeze23:13:13 and checkpointSHA recorded23:17above. Userinformed.

CurrentpairPPOwatcherPID33913/session48628;50evalsunderway. GPU0aux31480,
GPU2plain31555; GPU1oldlinear18150~1300; GPU3memory52483~400.
Next possible low-risk trial (NOT launched yet): boost bodyrotation-only linear
reward using existing --auxiliary-metric-weights. Mainremaining900xxgapbodyrot
7.38%, whileanchorrotationmeanmatches. For factor4 useauxweight11 andweights
1,1,4,1,1,1,1,1, preserving everyotherauxtermcoefficient1. Factor2 usesweight9
and1,1,2,1,1,1,1,1. Samephysical/eval/inputcontracts. First inspectper-body
diagnostics to checkwhethererrorconcentrates onrightarm or broaderchain.
No referenceconditionedcontext or warmfeatureadapter implemented.

## Latest update —23:17 UTC

Old slowphysics PPO47254 completed1500 normally (session67456 exit0), watcher
53494 exited0/noerrors/finalpresent. Bestdev remains1200max1.07444. Allremaining
causalmeanDAgger1600 evaluations finished: best1200max1.077, last1600max1.148.
Late1200smallgain was acknowledged to user; it does not justify stage2success.

Bias-supervised probe1000 seed19649 complete (session1974): trained39head mean
biasRMSE.00577738rad vszeroprior.00577825, essentiallyno gain; instantaneous
latent meanbiasR²−.03914. Reportcontext_bias003_1000_probe_seed19649.json.
BothbiasDAggertrainers now SIGTERM-stopped:3221 afterstable1500,3368 after1250;
allfileskept, watcher6083/session37146 finishingremainingcheckpoints.
Evalbias1250max1.10235; control1000max1.13274, neitherbeatsbestexistingstudent.

NEW paired PPO trials, same frozeninitial linearglobal1050 checkpoint SHA
2b21aa60913373acb8a64872d7f9c5891e09318eb756617280080284e58ccc3f:
- GPU0 `context_trainable_aux_ppo_seed74`, PID31480/session68960.
- GPU2 `context_trainable_plain_ppo_seed74`, PID31555/session79493.
Sharedwatcherlogv2_context_trainable_pair_ppo_watch.log (sessionlatesttool).
Both4096env1500updates/save50, fixedLR3e-6,warmup50,std.1,seed74,
uniform/referencetrainingstarts, tracking4/primary2/aux8/rightarm8/offset1,
primary+auxlinear, basefrozen, context+controllertrainable, no memory.
Auxiliaryarmadds2 supervisedsteps/update on SAME PPOrollout, batch2048,
bodyvelocity/payloadloss.05, sixphysicstargetloss.01. SamePPOoptimizer used;
onlyencoder+headreceiveauxgradients, controller/criticreceiveNONEinthesesteps.
All auxiliary updates disabledduringcriticwarmup. Privilegedtargetsaredisjoint
labelkey `adaptation_context_auxiliary_target`; actorandencoderinputsstripit.
New ContextSupervisionWrapper and ContextAuxiliaryPPO inadaptation_ppo.py;
CLI--context-aux-steps default0 preservesolderruns. No extraoptimizerstate.
Conflictingmemory/teacher/oracleconfigs rejected; weights validatedfinite.
Focused tests+fullsuite passed (27890/63949 exit0); ruffpassed. Actual64env
3update smoke: warmupactoridenticaltosourceexceptstdintentionaloverride,
thenonlycontext/head/residual/stdchanged, baseunchanged. Full336segmentv2
smoke completed/allasserts/fail0, body.0308422,joint.502554,root.268321;
not a tracking pass. Logs v2_context_auxiliary_ppo_smoke{,_eval}.log.

GPU1 oldlinearall18150 stilltraining~1200; GPU3causalmeanPPO52483~300 remains.
Additionaldev linear1050 vsnominal900xx currentlyrunning(session40486,GPU1);
90001/90002JSONdone,90003pending. Prefrozen23:13:13 manifest
eval_v2/development_validation_linear1050_900xx_freeze.json. This isDEVELOPMENT,
notfinalconfirmation; 600xxuntouched. Compareall3onlyafterallcomplete.
No reference-conditionedcontext, warmfeatureadapteraddition or joint-trajectory
biasproxy implemented; those were hypotheses only. Currentbiaslabelsstayloss-only.

## Latest update —23:03 UTC

Stopped explicitly verified teacher PID28275 after stable1200, and causalmean
DAgger41878 after stable1600. ALL files retained, watchers finish remainingevals.
Teacher1000/1100/1200 max1.1367/1.1462/1.1533 after completefeaturetransition;
memoryDAgger700/800/900 max1.1091/1.1014/1.1119, no broadimprovement. Memory
PPO52483 remains running, not stopped with DAgger. Oldslowphysics47254 nearing
final1500 still may be running alongside it onGPU3; recheckbeforeallocation.

NEW paired trials, source same slowphysicsPPO750, neitherusescausalmean:
- GPU0 `context_bias003_dagger_seed73`, PID3221/session21109.
- GPU2 `context_bias_control_dagger_seed73`, PID3368/session98001.
- Shared watcher session37146/logv2_context_bias_pair_dagger_watch.log, interval250.
4096env3000updates/save250, LR1e-4, uniform/original trainingstarts, teacher
rollins0, IMU/rightaligned, aux.05/physics.01, traincontext+residual, basefrozen.
Both seed73 and optionaltraining-RNGreseed730001 AFTERmodelconstruction avoid
discarded head-initialization RNG confounds. Bias armadds29loss-only targets
encoder_bias/.01 withweight.03; head10->39, first10oldoutputsexactlypreserved,
new29weights/biaszeros. Controlkeepshead10/weight0. Physics draws happenbefore
modelconstruction, initialrolloutmetrics nearlyidentical (GPUtiny numerics):
body.037579130/.037579115, joint.533838153/.533836782. Noassertionofbitwise
trajectories. Existingbias event isstartup inactualsourceconfig; no resampling
or rangechanges. Moreworldsthanold1024runs controls potentialphysics-IDmemorization.
Heldoutprobe isrequired—lowertrainingbiaslossaloneisnotidentificationevidence.

New CLI --context-encoder-bias-weight defaults0, requiresphysicsloss>0 and
trainablecontext. Auxiliary slices arefirst4bodyvel/payload,next6physics,last29
bias. Probe39headnowreportsheldoutbiasRMSE inrad vszeroprior. Policy/export
actionarchitecture unchanged; auxiliaryhead neverusedinactions.
Newwarmstarthelperallowlistonly10->39lossoutputexpansion; allaction/encoder
weights copiedunchanged. Smoke64env2updatesinitialactualactionmaxdiff0.0,
updatesonlyencoder12/head4/residual8tensors, baseunchanged; no privilegekeys
incheckpointactor. Fullv2smoke336segmentspassedallasserts/fail0, nottrackingpass.
Biashead/semantic tests andfullsuite passed (sessions94589/55463 exit0);
ruffimportformatfixed. Logs v2_context_bias_supervision_smoke{,_eval}.log,
v2_tests_bias_full.log. RNGflag added aftersmoke, read-onlyseed resetonly.
Final600xx remainsunused. Goalnotcomplete.

## Latest update —22:54 UTC

Causal memory implementation COMPLETE and structurally tested. DefaultsFalse
leave older actors unchanged. 5 focused tests plus full adaptation/comparison/
residual-policy/residual-training suite passed (session3261 exit0, log
v2_tests_memory_export_full.log); ruff passed. Wrapper selects only deployable
keys for producer, same-step idempotence, per-world resets, replay snapshots.
Unit tests cover prefix/EMA/reset, frozen-producer guard, input poisoning,
initial zero-padded function equality, new-column gradients, mean ablation.
PPO memory storage and DAggersmokes64env2updates both completed; only residual
weights changed (plus exploration std for PPO). Source PPO750 vs initialized
memory actor actual GPU maximum action difference EXACT0.0.

GPU2 auxiliary-linear trainer17408 SIGTERM-stopped after stable750; files kept.
GPU2 NEW `context_causal_mean_dagger_seed72`, PID41878/session68711, watcher
PID44135/session5178, logs v2_context_causal_mean_dagger{,_watch}.log.
1024env3000updates/save100, LR1e-4, teacherrollins0, seed72, frozencontext/base,
teacherpayload250, source slowphysicsPPO750, IMU/rightaligned/aux10, causalmean
horizon250. Mean branch64 new inputs zero-initialized, totalcontext128.
Earlyeval100/200/300/400 root1.114/1.114/1.140/1.126—NOT improved. Stilltraining.

GPU3 NEW `context_causal_mean_linear_ppo_seed72`, PID52483/session58121,
watcher session84675/logv2_context_causal_mean_linear_ppo_watch.log.
4096env1500updates/save50, source SAME exact-initial memory actor (smoke
checkpoint_initial.pt, not its updatedfinal), fixedLR3e-6,warmup50,std.1,
uniform/reference trainingstarts, tracking4/primary2/aux8/rightarm8/offset1,
primary+auxlinear. Frozenbase/context; nophysical/eval/inputnoisereduction.
Coexists briefly with oldslowphysics47254 approaching1500, fitsGPU3memory.
GPU1 linearall18150 continues (best700max1.06711); GPU0teacher28275 at1100
stillworst1.146(fullmix1), no endpointsuccess. No final600xx use.

StatefulCPUexport NOW supported by StatefulContextInferenceModule and CLI/eval:
raw8349D + previousmean[B,64]+count[B,1]+reset[B] -> action,newmean,count.
Statecallerowned, initializedzeros, exactlyoncepercontrolstep; no extra sensors,
teacher or simulator library in the TorchScript artifact. Flatstatelessclass
still REJECTS memory actor. Eight fresh python-I torch-only processes passed,
batch1/2/5/104, sequential12steps plus syntheticprefix/EMA/resetboundaries.
Artifact `context_causal_mean_smoke_export_fp32/policy_cpu.ts`, SHA
a55b56e0ba12b14babd0d92ec038fe47fe42faf3485f9effb44ebe7490a58a81,
maxactiondiff7.882714e-6. Strict2e-4tol retained. Firstexportattemptfailed
2/6656 latent components (max.000257) with GPU cuDNN TF32; explicitlyFP32
GPUconvolution reference passed. Training/eval GPUdefaults not changed. CPU
stillrequires1thread. This does NOT claim GPU/CPU trajectories bitidentical.
Full336segment statefulCPUv2smoke completed (session82419, log
v2_context_causal_mean_export_eval.log): allisolation/timelineassertionspassed,
fail1,body.0313707,joint.507964,root.269159. NativeGPU sameupdatedsmoke fail0,
root.269643, so numerical path differences do alter closedloopoutcomes.
Neither smoke is a tracking-pass or hardwarevalidation.

New diagnostic `context_physics750_bias_probe_seed19648.json` complete:
512world384train/128heldout,18times. 29trueencoderbiaslabelsONLYinridge probe,
neveraction. MeanR²instant−.03606,offlineepisodeavg−.05924,currentrawsensor
−.05330; meanRMSE.005818/.005882/.005865rad vsconstant.005750rad.
NojointpositiveheldoutR². Thus existinglatent does NOT linearly identifybias,
even though it identifiespayload/COM; not a proof of fundamentalunobservability.
Probe CLI optional--encoder-bias-diagnostic; oldphysical7label API unchanged.
Readable Chinese status updated22:53 with full900xxstudenttable.

## Latest update —22:43 UTC

IN PROGRESS, not yet trained: causal context memory. Current64D context plus
past-and-current64D mean; prefix mean up to250 samples, then EMA gain1/250.
Frozen encoder; zero-pad new64 residual input columns to preserve source action
at initialization. Reuses existing noisy measurements, no future samples or
physical labels. Motivation is physics1500 held-out diagnostic payload R²
.874 instantaneous vs .959 offline episode-averaged; the latter is NOT a
deployable initial-time estimate. This new causal method must be tested.
New adaptation_context_memory.py wrapper isolates deployable producer inputs,
guards same-step repeated calls, snapshots memory into replay, per-world reset.
Policy/distillation/train/eval/probe integration added; tests and GPU smoke TODO.
Flat stateless export explicitly REJECTS memory actors pending stateful export.
No new full run yet. Existing jobs use defaultFalse and are unaffected.

Latest existing runs: teacher900(fullmix1)worst1.124/fail4; linear-all600
worst1.0806/fail1; linear-aux600worst1.1038/fail0; slowphysics1200
worst1.0744/fail1. Still incomplete; all4 trainers running. PD contract test
session46385 completed successfully. Final600xx untouched.

## Latest update —22:24 UTC

Additional DEVELOPMENT validation900xx COMPLETE, all nine evaluations successful.
Pre-run manifest `eval_v2/development_validation_900xx_freeze.json` timestamp
22:17:21UTC precedes launches22:18. Fixed seeds90001/90002/90003,42motions,
1008segments PER policy, fixed nominal unchanged. Candidates frozen before runs:
IMU2500 SHAa1976c...caf142ae2c and slowphysicsPPO750 SHA
8d2fe99124474c907d9cf52a6575892eb7b55113e66f687aa071bc0626d421c2.
900xx is NOT final confirmation and was NOT added to comparison-script
CONFIRMATION_SEED_SETS; output flags confirmfalse. 600xx remains unused.

`eval_v2/development_validation_900xx/{imu,physics}_comparison.json`:
IMU meanratios body1.02826,joint1.04832,root1.08610,anchorrot1.07045,
bodyrot1.08758,jointvel1.02921,anchorlin1.06664,anchorang1.06645,
bodylin1.06194,bodyang1.06799. Failure4/1008 vsnominal0.
PhysicsPPO750 meanratios body1.01569,joint1.04205,root1.07253,
anchorrot1.01958,bodyrot1.07654,jointvel1.02421,anchorlin1.05677,
anchorang1.05228,bodylin1.05182,bodyang1.05690. Failure5/1008 vsnominal0.
Thus all10 trackingmeans improved in this additional cohort, but stage2 is
STILL NOT achieved; worstmetricbodyrot1.07654. Physics root95%CI[1.00121,1.15152],
bodyrotCI[1.05072,1.10701], failure-differenceCI[0,+1.1905]pp. No cherry-picking
of seeds, metrics or failure episodes. Launch sessions93170/12673/2796 can be
polled if required; all nineJSON and comparisons exist.

Latest training at22:21: teacher600 mix1.0 root1.1744/body1.0538/joint1.0707,
bodyrot1.1180,fail4/336. Full replacement has NOT retained goodtracking; allow
post-transition training to assess convergence before changing again.
SlowphysicsPPO900 root1.0939/body1.0019/joint1.0363,fail1/336.
Linear-all300 root1.1074/body1.0048/joint1.0368,fail0.
Linear-aux300 root1.1297/body1.0070/joint1.0457,fail0.
All four main trainers and three watchers from22:04 still running.

Important CLOSED diagnostic: exact reason PD encoder-bias proxy is invalid.
Read actual MJLab code (read-only): JointPositionAction.apply_actions subtracts
encoder_bias from processed targets; EntityData.joint_pos_biased adds the SAME
bias to measurements. Therefore with q_meas=q+b and tau=K*(q_cmd-b-q)-D*qdot,
q_meas-q_cmd+(tau+D*qdot)/K =0, independent of b. This invalidates the proposed
direct-PD bias identification, not merely its tuning/timestamp alignment.
It does NOT prove bias unidentifiable from other geometric/contact information.
No actor ever used the proxy. `adaptation_identification.nominal_pd_contract`
now explicitly marks compensated targets, and sensor_pd_bias_proxy raises for
that contract. Its uncompensated toy case remains for documentation/testing.
Added guard test and explicit algebraic cancellation toy. Prior failed probe
JSON remains untouched as historical evidence. Relevant source read was
`.venv/.../mjlab/envs/mdp/actions/actions.py:222` and `entity/data.py:371`;
no external code edits. Test log`v2_tests_pd_contract_2224.log` session78481
predates final cancellation assertion; rerun this test after last patch.
All full tests at22:05 session20204 exited0, ruff passed before last toy test.

## Latest update —22:04 UTC

Still no second-stage pass; 600xx never used. Current best broad remains
IMU2500 max1.08553 on dev. Chinese `adaptation_status.md` updated22:01.

GPU0 curriculum teacher28275 continues, 300 evaluation at saved mix.502:
bodyratio.9748,joint1.0191,root1.0762,bodyrot1.0400,bodyangular1.0376,
failure2/336. At200 mix.302, all ten means<=1.0289. This decline during
replacement is NOT a successful end-point; must continue through mix1 at550
and subsequent training. No new teacher confirmation seeds consumed.

GPU1 high-LR physics PPO27633 stopped after stable350 checkpoint, files retained.
At300 body1.0542/joint1.0942/root1.1408/bodyrot1.1083,fail2/336. Failed.
GPU2 reference-only DAgger5448 completed1500; all watcher evaluations completed
with no errors. At1300 body1.0051/joint1.0390/root1.1036; no broad improvement.

NEW matched reward-shape trials (same initialized physics1500 actor as GPU3):
- GPU1 `context_physics_linear_all_ppo_seed69`, PID18150/session24519:
  primary AND auxiliary metric rewards use `linear`.
- GPU2 `context_physics_linear_aux_ppo_seed69`, PID17408/session23143:
  only auxiliary metric reward uses `linear`, primaries remainexp.
  Shared watcher session63328/log`v2_context_physics_linear_ppo_watch.log`,
  interval50. Both4096env1500updates, fixed actorLR3e-6,warmup50,std.1,
  base+context frozen, uniform/reference training starts, seed69,
  tracking4/primary2/aux8/rightarm8/offset1. Physical/sensor/evaluation configs
  unchanged. Matched GPU3 is original exp/exp, PID47254/session67456.
  GPU3 latest550 body1.0023/joint1.0376/root1.1172,bodyrot1.0515,fail0;
  partial improvements, still incomplete. Keep running to assess convergence.

New optional `auxiliary_tracking(...,shape='linear')` applies
`1-min(error/scale,5)` independently to all eight existing auxiliary metrics,
same scales and weights; defaultexp preserves old behavior. TrainCLI
`--auxiliary-reward-shape` records shape in metadata. Unit tests verify weighted
and unweighted values, invalid shape rejection. Smoke64env2steps passed:
only residual/std updated, base/context unchanged, correct linear metadata.
Relevant tests `v2_tests_2158.log` pass, ruff passes. Full previous tests
`v2_tests_full_2143.log` session70848 exited0. Latest subset session28987 may
be polled for exit if needed. All tracked files remain unmodified.

FAILED component combination: `context_physics400_reference1300_merge` swaps
only eight reference-encoder tensors from warmref1300 into slowphysicsPPO400.
All tracker normalizers verified bitwise equal before merge; all other actor
weights verified unchanged. 336-segment v2 result max1.113(root), body1.0052,
joint1.0424,fail1/336, not better than receiverroot1.0998. No further use yet.
Generalized existing `scripts/merge_adaptation_base.py` with allowlisted
`--module tracker.reference_encoder` (default tracker.mlp unchanged). Fraction1
copies donor exactly. For PPO checkpoints, `policy` and nested
`rsl_rl.actor_state_dict` aliases now synchronized; optimizer removed from both
top level and nested rsl_rl, weight-only warm start documented. All aliases
verified identical in actual merged checkpoint; merge tests pass. Old saved
artifacts were NOT rewritten. Evaluation log
`v2_context_physics400_reference1300_merge_eval.log` completed.

Read-only potential-bug check: context's 50-frame action term is `last_action`,
which returns `env.action_manager.action` (actual sampled/applied policy command),
NOT the separate legacy `prev_actions` mean-history helper. Thus PPO mean vs
sample history mismatch is ruled out for this context; no code change made.
No long-history encoder, foot odometry or concurrent-auxiliary PPO has been
implemented. These were only possible later hypotheses, not running methods.

## Latest update —21:44 UTC

Readable Chinese result/constraint/reproduction overview now in
`docs/adaptation_status.md`. Model SHAs were rechecked from actual checkpoint
bytes. No stage-two pass; do not use 600xx yet.

GPU0/1 seed68 trainers26974/27060 were SIGTERM-stopped after stable550
checkpoints; all files retained, old watcher finished remaining evaluations.
GPU2 frozen-feature trainer50677 completed3000 and its watcher exited cleanly
with no failed evaluations. Its final result is not successful.

CURRENT:
- GPU0 teacher `oracle_feature_curriculum_seed71`, PID28275/session81694,
  watcher PID29811/session70374, log`v2_oracle_feature_curriculum_watch.log`.
  Initializes payload250 teacher, 4096env2000updates/save100, fixed actor LR1e-5,
  base+privilege encoder+residual trainable, original training reset noise,
  uniform sampling, std.1, warmup50. Oracle feature mix rises0→1 over500
  postwarmup updates, then remains1. Tracking4/primary2/aux8/rightarm8/offset1.
  Smoke saved mixtures0/.5/1 for updates0/1/2, reloaded fullmix checkpoint in
  42env25step v2 smoke, all reset assertions passed. This is structural only.
- GPU1 student `context_physics_frozen_encoder_fast_ppo_seed69`,
  PID27633/session65969, watcher PID29827/session3523. Matched to GPU3
  physics-frozen run EXCEPT fixed actor LR3e-5 versus3e-6; same4096env,
  physical initialization, seed69, std.1, warmup50, uniform/reference starts,
  1500updates/save50. First50 evaluation completed,100 running.
- GPU2 reference-only student `context_reference_warm_seed70`,
  PID5448/session22461, watcher PID6744/session14840, interval100. ~900updates.
  Latest500 rootratio1.1286/body1.0074/joint1.0433, no broad improvement yet.
- GPU3 slow physics-frozen student PID47254/session67456,
  watcher PID47964/session53494. Latest250 root1.1157/body1.0089/joint1.0475,
  bodyrot1.0604/bodyangular1.0584; still no broad pass.

Added curriculum unit tests: exact mix endpoints and midpoint; warmup progression
bounded0/.5/1; strict initial/reload scalar preservation. DAgger warm initialization
unit test catches same-shape alignment changes. Full relevant adaptation +
residual policy/training + model-gradient tests were run to completion in
`v2_tests_full_2143.log`, no failure shown; session70848 can be polled for exit.
All modified Python files pass ruff. Tracked git diff remains empty.

Probe interpretation caveat: physics01 vs oldIMU1500 uses matched held-out
physical worlds, but the new architecture also enables right alignment. The
large COM probe gain is not a strictly isolated causal estimate of auxiliary
physics supervision alone; no identical-architecture no-physics control probe
has been completed. Tracking conclusions do not rely on that attribution.

## Latest update —21:36 UTC

Still no stage-two pass. Reserved 600xx untouched. All prior comparisons and
fixed nominal baseline remain unchanged.

GPU2 frozen-feature DAgger PID50677 completed3000; final watcher still flushing.
2750 root ratio1.2316, failures4/336: branch failed. NEW GPU2
`context_reference_warm_seed70`, PID5448/session22461, watcher session14840,
log`v2_context_reference_warm_watch.log`, interval100. Initializes BESTIMU2500
without any architecture/input changes; freezes base, residual AND context,
trains ONLY existing deployable reference encoder with action loss (no feature
loss), payload250 teacher. 1024env1500updates, LR1e-5, teacher mixture zero
from first step. At100 root1.1811/body1.0163/joint1.0473, no improvement yet.
Smoke64env2updates verified initial action max difference EXACT0 and only eight
`tracker.reference_encoder.*` tensors changed; all other tensors unchanged.

New DAgger flags `--student-initialization`, `--freeze-student-context`:
strict state-dict transfer plus semantic contract validation (alignment, sensor
flags, residual scale, gain etc). Architecture flags must explicitly match
source. Source SHA and initialization action equality saved. Context/nominal
initializations mutually exclusive. `--teacher-mix-updates 0` now means zero
teacher roll-ins including iteration0. `checkpoint_initial.pt` is recorded;
watcher now skips all nonnumeric/nonfinal checkpoint labels.

GPU0/1 uniform seed68 at500: root ratios1.1421/1.1710, no improvement across
ten checkpoints. Both still running at this timestamp; candidates for stopping
after next stable checkpoint to replace the failed method. GPU3 frozen physics
encoder PPO150: root1.1111/body1.0135/joint1.0503, slightly improving angular
metrics but still not better than best broad student. Continue trend check.

Next-method preparation: optional teacher observation curriculum. New
`oracle_feature_curriculum` actor flag adds saved scalar buffer
`oracle_feature_mix`; blends normalized true tracking features with existing
noisy/deployable tracking features. Defaults OFF preserve old checkpoint
behavior. At mix0/1 endpoints exactly oracle/deployable features (unit verified).
True physics still enters privileged encoder: this remains a TEACHER, never
a deployment claim, including at mix1. New `OracleFeatureCurriculumPPO`
inherits critic warmup, advances mixture only after warmup and saturates at1;
mixture is actor state so evaluations load actual checkpoint value. EvalJSON
records it. DAgger strips teacher-only constructor flag from student.
CLI `--oracle-feature-curriculum-updates`; requires initialized oracle teacher;
incompatible BC, must be nonnegative. Original tracked runner is UNMODIFIED.
GPU0 smoke3updates64env/warmup1/curriculum2 currently PID11354/session56067,
log`v2_oracle_feature_curriculum_smoke.log`; full teacher run NOT launched yet.
Must inspect saved mixtures0/.5/1 and actual reload before launching.

Research motivation, not an exact implementation or a performance guarantee:
[Student-Informed Teacher Training](https://arxiv.org/abs/2412.09149) explains
why a privileged teacher may rely on information unavailable to its student;
the paper jointly trains with action-mismatch penalties. Our proposed feature
curriculum is a simpler hypothesis prompted by our own reference ablations,
not a reproduction of that algorithm. Primary paper read 21:33 UTC.

## Latest update —21:28 UTC

Stage 1 tracking confirmation remains successful (all 10 upper tracking CIs
below 1.05); the failure-rate CI caveat remains. Stage 2 is NOT achieved.
Best development student remains IMU2500, worst ratio about 1.086. Reserved
60001/60002/60003 remain unused. No physical DR, sensor noise or evaluation
criteria changed. All changes remain inside this repository.

Current training jobs:
- GPU 0: `context_imu_uniform_original_seed68`, PID 26974, session 21938.
- GPU 1: `context_imu_uniform_reference_seed68`, PID 27060, session 91685.
  Both initialize IMU2500, frozen base, trainable context, 4096 environments,
  1000 PPO updates, fixed actor LR 1e-6, 50 critic-only warmup updates,
  exploration std .05. Both use uniform motion sampling without rewind.
  The only matched difference is training initial-state noise: original
  versus zero reference-state perturbation. This is a TRAINING curriculum;
  original/reference ranges are recorded, sensor and physics configs unchanged.
  Shared watcher PID 30694/session 83168, interval 50. At update 300, root
  ratios 1.1225/1.1330; no improvement yet. Old warm-fixed seed67 jobs stopped
  after 350, their evidence retained.
- GPU 2: `context_clean_frozen_feature_seed59`, PID 50677/session 63453.
  Clean500 teacher, IMU/orientation/right alignment/FK, feature and latent
  loss .05, frozen teacher base AND residual, only context/feature adapter
  train. Watcher PID 56956/session 84133. At 1750: root ratio 1.2725,
  body 1.0460, joint 1.0583, failures 3/336. Not successful.
- GPU 3: NEW `context_physics_frozen_encoder_ppo_seed69`, PID 47254,
  session 67456, watcher session 53494. Initializes physics01 checkpoint1500,
  frozen base AND context encoder/auxiliary head, residual policy trainable.
  4096 environments, 1500 updates, fixed LR 3e-6, 50 critic-only warmup,
  std .1, uniform/reference training starts; tracking4/primary2/aux8/rightarm8,
  reference offset1. Smoke64env2updates verified every base/context/aux tensor
  unchanged and residual weights updated after warmup. Initial diagnostic
  assertion incorrectly searched `mlp.` instead of `residual_mlp.`; corrected
  assertion passes, no policy change required.

Physics-label student completed 3000 (GPU3 previous PID1616); tracking still
not at goal. Checkpoint1500 SHA
fa4b74cf1df82d7fc1ec4d372804552d746ec5668cf789ea230922994695005d.
Matched held-out probe seed19647, 512 worlds, 384 train/128 test:
physics-supervised latent COM x/y R² .8109/.8474 versus old IMU1500
.0168/.2539; payload .8744 versus .8433. COM x/y RMSE .01816/.01607 m
versus .04140/.03554 m. Separate seed19649 corroborates. This is useful
identification evidence, NOT tracking acceptance. Other physics labels remain
poorly identified. Files `context_physics1500_probe_seed19647.json` and
`context_imu1500_probe_seed19647.json` under runs/adaptation_goal.

Read-only measured-PD encoder-bias proxy FAILED: held-out bias RMSE .011751
rad, worse than zero-prior .005784 rad. Diagnostic module
`adaptation_identification.py` is NOT connected to actor inference. It uses
only measured history and fixed nominal actuator calibration; simulator bias
is scoring-only. No benefit is claimed; controller/timestamp assumptions may
be responsible and are not yet resolved.

New `train_context_encoder` checkpoint/CLI flag preserves legacy default True;
False freezes context encoder and auxiliary head for PPO. Structural tests
completed without failures (`v2_tests_2124.log`). Frozen-feature CPU export
smoke SHA caf94d74939163128cdbec823581201dd33e3df7debc2a9bdee2d05f9ca232ab
passed eight fresh single-thread isolated Python processes, max action error
8.22544e-6. Nominal-prior single-thread full v2 CPU smoke completed all reset
assertions but failed tracking (8/336 failures); it is export plumbing evidence
only. Exact nominal-prior DAgger completed 3000 and did not improve tracking.

## Latest update —20:56 UTC

All goal constraints unchanged; stage2 NOT achieved,600xxstillunused.
CurrentGPU0/1safePPO52268/52355continue, now300checks started. At200 fullbase
rootratio1.1204/body1.0106/joint1.0474,fail1/336; frozenbase root1.1072/body1.0117/
joint1.05,fail0. Stable but no improvement over originalIMU2500max1.086 yet.
GPU2 nominalprior10292stillruns:1250root1.2219/body1.1013/joint1.0878,
bodyangular1.1497,fail1/336. No criterionpass. Itswatcher29933/log
`v2_context_nominal_exact_prior_watch.log` active.

GPU3 kinematicaux student31891 completed3000 and allwatcher evals. Finalroot
1.136/body1.0116/joint1.046,fail0. No overallimprovement; branchretained.
NEW GPU3 `context_payload_physics01_seed59`, PID1616/session90251,
watcher64735/log`v2_context_payload_physics01_watch.log`. Teacherpayload250,
1024env3000updates/LR1e-4,auxvelocity+payload.05, **physicsweight.01**,IMU+
rightalignment,noorientation/kinematics/latentloss. Contextauxhead10D: original
bodyvelocity3+payload1 followed by normalizedtorso mass/COMxyz/meanfootfriction/
meanarmature6. Existingphysical_labels supplies loss-onlylabels, normalized
centers[0,0,0,0,1.15,1],scales[1,.075,.075,.075,.85,.2]. No GTactioninputs.
Newprobehandles10Dhead andreportsfirst4rawRMSE plus6normalizedphysicsRMSE.
Smoke64env2updatesand64env100stepheldoutprobe passed; untrainedprobe accuracy
is of course poor. At446 physicsloss.18775 vsinitial.30454; nottrackingproof.
ContextnoGTtestsnowincludehead10D;95adaptation tests passed. Additional
comparison/fullresidualtests shouldbe rerun at nextstablecheckpoint.

IMPORTANT CPU export issue AND mitigation:
Old`context_nominal_exact_export_smoke`artifact4171... is NOT certified for
arbitrarythreadcount: fresh2-threadPython-I occasionallyreturns maxactiondiff
0.000683188 vsallowed2e-4 (reproduced1/8 and2/8 evenMKLDNNdisabled). Other
freshprocesses~6.18e-6. ThusnotexplainedsolelybyMKLDNN. Singlethread repeatedly
passes (16earlierfreshprocesses; subsequentformal8audits allpass). Rootcause
withinmultithreadexecution isnotyetfullyidentified; no tolerance relaxation.
NewexportCLI uses **torch.set_num_threads(1)** and REQUIRES8freshisolatedPython
processes, onlyTorch, withbatchsizes1/2/5/96 beforewritingmanifest. Manifest
recordsrequired_cpu_threads1 +each audit result. Evaluator honors this only
whenpresent innewmanifest; oldexportsretainhistoricalthreadsemantics.
Newartifact`context_nominal_single_thread_export_smoke/policy_cpu.ts`,SHA
be3950bec12f4744f3b2bcd4536a7b1e3714041d1085025b695d86ab4cb03faf,
GPUcomparisonmax4.82798e-6, all8independentprocessmax5.48363e-6. Singlethread
batch1 median6.56ms in100forwardcalls (NOThardware control latency/safetyproof).
Full336segmentv2exportevaljustlaunched, log
`v2_context_nominal_single_thread_export_eval_smoke.log`; inspectcompletion.
Alltraining/trackingevidenceabove uses originalGPUactors and is unaffected by
this CPUexport diagnostic. User hasbeeninformed ofissue andmitigation.

## Latest update —20:44 UTC

Stage2 STILLincomplete; no600xxconfirmation. Bestbroaddevelopment remains
IMU2500 max1.086. Do not claim success from growing code/test coverage.

Stopped old lowstdPPO53361 after250 (rootratio1.3322,body1.2074,joint1.1952,
fail3/336); stopped rootweighted5635 after125 (root1.4704,body1.1952,joint1.1937,
fail4/336). All outputs retained, user told. Replacements:
- GPU0 `context_imu_warm_fixed_ppo_seed67`, PID52268/session20968.
- GPU1 `context_imu_frozen_warm_fixed_ppo_seed67`, PID52355/session49884.
  Shared watcher2628/log `v2_context_imu_warm_fixed_ppo_watch.log`, interval50.
  Both initIMU2500,4096env2000updates/save50, std.05, entropy.0002, seed67,
  tracking4/primary2/aux8/rightarm8/refOffset1; **fixed actorLR1e-6**,50critic-only
  warmupupdates. Difference onlyGPU1 --no-train-base-policy. Frozen-base smoke
  confirms all tracker tensors identical through warmup AND postwarmup step.
  At100 updates: fullbase root1.1309/body1.0165/joint1.0495,fail0;
  frozenbase root1.1511/body1.0127/joint1.0488,fail0. Noearlycollapse, but NOT
  tracking improvement yet. Continue tochecktrend, not2hblindextension.
- GPU3 kinematicaux.5 student31891/session63332/watcher41230 ~2158/3000.
  Latest1500root1.1083/body1.0235/joint1.0537; no clear gain. Its1349? NO:
  inference input remains8349D. Full export336segment v2 smoke passed all
  isolated reset/history assertions,failure0. Original sensor/DRnoise unchanged.

GPU2 NEW `context_nominal_exact_prior_seed59`, PID10292/session72059,
watcher from latest launch log`v2_context_nominal_exact_prior_watch.log`.
Teacherpayload250,1024env3000updates,LR1e-4,aux.05, IMU+rightalignment,
NOorientation/kinematics/latentloss. Instead of inheriting the privileged
teacher's trunk, initializes whole deployable nominal trunk and residual from
FIXEDnominal1000 SHA800da8...d47ad8f1f. Context columns in first residual layer
are zero-padded; other nominal residual weights unchanged. To preserve nominal
actions while allowing a larger residual range, new optional29D trainable
`context_residual_log_gain` starts at log(nominal_scale/student_scale), bounded
[-4,0] then exponentiated on outputs. No extra network, selector or ensemble.
The function matches nominal mathematically; actual1024GPU-observation
initialization auditmaxdifference0.00011408 (passes2e-4 tolerance; NOT bitwise
closedloopidentity). Metadata records sourceSHA, actual actiondifference and
input contract. Teacher remains action/latent/aux labels only. Actorhaszero
privilegedinputs; evalbaseline checkpoint untouched. ~408updates/actionMSE.0424
frominitial.1168; muchharderfit thanteacher-trunkstudent, no trackingresultyet.
Smoke`context_nominal_exact_init_smoke`64env2updates passes; CPUexport96samples
maxdiff4.7088e-6, artifact`context_nominal_exact_export_smoke/policy_cpu.ts`SHA
4171cf366c643be9539aa5faf80ad1e2bd3f4c4fa6d86e3ca53da17bb1e0befe.
Earlier`context_nominal_init_smoke` used approximate logit scaling and is obsolete
diagnostic only; no trainingrunusesit. Gain tests verify exact even saturated
nominal functions and zero-padded context columns; all64noGT combinations now
covered, full135relevant tests passed.

Distillation config bookkeeping fixed for FUTUREruns: student_args.train_base_policy
now equals --train-student-base, not inherited teacher flag. Previously actual
DAgger MLPtrainingwas correctlydisabledviarequires_gradFalse, but savedcfgTrue
causedlater PPO totraincoreunlessoverridden. Oldcheckpoints retain oldsaved
behavior; explicit PPO tri-stateoverride isavailable. New nominalpriorcfgFalse
correctlyrecords frozen base. No old checkpoint rewrites.

## Latest update —20:31 UTC

Lowstd PPO currently FAILED: GPU0 `context_imu_low_noise_ppo_seed67`53361
eval125 bodyratio1.2174,joint1.1837,root1.4406,bodyrot1.2663,fail4/336. At231
updates; watch250 then stop if not substantially recovered. GPU1 matched root-
weighted PPO `context_imu_root_weighted_ppo_seed67`5635/session14678/watcher10990
~128updates. Only difference from GPU0: auxiliary metric reward weights
(4,1,2,1,1,1,1,1), emphasizing rootposition/bodyrotation. DefaultsNone preserve
original equal mean exactly. Training-only weights, evaluation unchanged.
Weightedreward unit test initially targeted wrong slot and failed; corrected
jointvelocity slot3 test and all101 relevant tests passed before next change.

New safer fine-tuning support (not yet fullrun): `CriticWarmupPPO` freezes all
currently trainable actor parameters for Nupdates, trains critic normally,
preserves originally frozen flags, restores schedule and trainability finally,
saves/restores updatecounter. CLI --critic-warmup-updates; incompatible with BC.
--actor-lr-schedule fixed bypasses old adaptive 1e-5 floor (default unchanged).
Smoke64env2updates/warmup1/fixedLR1e-6 verified: update0 actor bit-identical to
source except explicitstd.05 override;17critic tensors changed; update1 changes
37actor tensors; counters1/2 saved; LR1e-6. Unit tests pass69adaptation+comparison.
Base-training CLI now tri-stateNone: preserve checkpoint setting unless explicit
--no-train-base-policy/--train-base-policy; previous override did not affect
initialized actors. New warmup/freeze-base fullrun not launched yet.

GPU3 prior clean+aligned1894 stopped after750checkpoint (500rootratio1.2103,
fail6/336); all watcher evals succeeded. Now `context_payload_kinematic_aux05_seed59`
PID31891/session63332, watcher41230/log`v2_payload_kinematic_aux05_student_watch.log`.
Payloadteacher250,1024env3000updates,LR1e-4, **aux.5**, latentloss0, IMU+orientation+
rightalignment+**context-key-body**. Encoderhead now optionally receives existing
deployable195D noisyFK (position/rot scale1, linvelocity2, angular4). No new
sensor or groundtruth channel; 8349D exportinput unchanged. Replay/export/noGT
tests updated for32combinations;101 fullrelevant tests pass. Smoke64env2updates
and CPUexport96samples pass, maxdiff6.4969e-6; python-I max5.2452e-6. Artifact
`context_kinematic_export_smoke/policy_cpu.ts`SHA
b84e7a09e4181441fb4d52611a795b31420946844b1be1642143b93a701285d3.
This is an inductive-bias hypothesis, not evidence of better tracking yet.

IMU2500 ablations complete: zero body.031570/joint.509296/root.276831;
shuffle body.031656/joint.510649/root.275379; eachfailure1/336. Normal root~.2678,
so latent helps rootseveralpercent but affects primaryerrors onlyweakly.
Orientationfilter.1 givesroot.280377 despitebodyangular1.12941; .3root.270141;
both FAILED broad tracking and are not integrated into checkpoints. User told.
Stage1 strong800xx confirmation unchanged; stage2 stillincomplete,600xxunused.

## Latest update —20:19 UTC

Oriented+latent student15069 finished3000 normally; all watcher evaluations
passed. Best250 still max1.101. Latent-only matched control23124 ~2650/3000.
Rightaligned+IMU+oriented47179 now~900;250developmentmax1.103, not progress yet.
Its exported smoke passed full336segment v2 isolated-reset/history audits,
failure0/336; independent python-I numerical test maxdiff7.63e-6.

Stopped privileged deployable-front teacher48939 after750checkpoint: latest
rootratio1.1609, body1.0651, joint1.0569, failures2/336. Checkpoints/logs retained.
GPU3 now NEW `context_clean_aligned_imu_oriented_seed59`, session from tool
launch, PID to inspect. Same settings as GPU2 combined context (LR1e-4, aux.05,
latent.05, IMU+orientation+alignment), except stronger cleanteacher500 instead
of payloadteacher250. No reference encoder/feature adapter/base-MLP fitting.

GPU0 NEW `context_imu_low_noise_ppo_seed67`, PID53361/session7522, watcher
91301/log `v2_context_imu_low_noise_ppo_watch.log`. Init bestIMU2500,4096env,
2000updates/save125, tracking4/primary2/aux8/rightarm8/referenceoffset1,
initialLR3e-5 (adaptive), **initial exploration std.05**, entropy.0002. NoBC.
Actual savedstd~.0501 verified after update0; actortrainbaseTrue (full baseMLP
trainable during PPO). This changes policy exploration, not sensor noise/DR.

Added read-only orientation-filter diagnostic (not actor checkpoint/export):
`--orientation-filter-weight` default1=nochange. Complementary filter predicts
quaternion using trapezoidal measured gyro (history3197:3200) then normalized
hemisphere-corrected mixing with existing measured IMUquat. Only pastfiltered
q/gyro and current measurements, no simulator truth. Firstframe uses rawquat;
evaluation never restarts an active episode. TensorDict shallow copy leaves
raw measurements/history unchanged. Comparison/scoreboardexclude nondefault
weight; unit tests sign/dynamics/noise attenuation + exclusion pass52tests.
On bestIMU2500,weight.3 did not improve:body.031546,joint.509108,root.270141,
bodyangular1.13836,fail1/336. Weight.1 stillfinishing; no inference integration
unless a real gain appears. Output `eval_v2/imu2500_orientation_filter0*_seed10001.json`.

## Latest update —20:13 UTC

Stage1 remains independently confirmed clean teacher500 (800xx report below);
stage2 has NOT met the 2% all-tracking target. No student600xx confirmation used.

Confirmed architectural issue: with 50 frames, original valid-stride TCN drops
latest frame49 (gradient L1 exactly0; frame48 gradient0.05849). Optional
`context_right_aligned` now trims `(history_steps-21)%4` from LEFT before the
same convolutions (receptive field21, total stride4). Parameter names/shapes are
unchanged; defaultFalse preserves all historical checkpoint semantics. Tests
cover latest-frame gradients at history lengths49–52 and all16 combinations
of feature adapter/IMU/orientation/alignment with poisoned privileged keys.
83 relevant tests pass; ruff passes. Smoke64env2updates passed; CPU export96
samples maxactiondifference6.4075e-6, artifactSHA
7fddc946c95bfdc5ff03846f88145f69cbd8754d0b352d50b2ce02aecc7af6eb in
`context_aligned_imu_oriented_export_smoke`. Not yet tracking acceptance.

Current allocation (revalidate PIDs):
- GPU0 oriented+latent student15069, watcher20407/session65695. ~2550/3000.
  Best still250 max1.101;1750rootratio1.1023, body1.014, joint1.0496.
- GPU1 `context_payload_latent_only_seed59` PID23124/session33651, watcher
  session15643, log `v2_latent_only_student_watch.log`. Same seed59/teacher250/
  latentweight.05/aux.05/LR1e-4 as oriented run but NO orientation; ~1986/3000.
  Matched control separates latent supervision from orientation effect.
- GPU2 NEW `context_payload_aligned_imu_oriented_seed59`, PID47179/session70653,
  watcher59480/log `v2_aligned_imu_oriented_student_watch.log`. Payloadteacher250,
 1024env3000updates, LR1e-4, aux.05, latent.05, IMU+orientation+rightalignment.
  Previous clean_reference_kin student44690 completed3000 normally and all
  watcher evals succeeded; finalmax1.2354root/fail6of336. Branch unsuccessful.
- GPU3 deployable-front privileged teacher48939/session71964 continues;
  latest625 rootratio1.1558, body1.0474, joint1.0529, failure2of336. Still not
  competitive; truephysics alone has not compensated noisy-reference frontend.

Source inspection: noisy reference is a 50-frame(-42..7) window with random
reference joint biases, root drift and small additive noise, not just white
noise. Existing reference encoder is residual MLP1900→418 over11supportframes.
Only action/reference-feature fine-tuning tested so far; no alternative
temporal denoiser or kinematic velocity proxy implemented. No changes to noise
generation, DR, evaluation distribution, or nominal baseline.

## Latest update —19:55 UTC

Strongeststage1isnow independently confirmed cleanteacher500:
`oracle_clean_payload_seed58/checkpoint_500.pt`SHA
5a91d70ef4e8756796d074a7fbff75d8c5f9d5b325dc5121b8daf2267af31daf.
Frozen19:26:31before80001/2/3. AllTEN meanratios<1; allTEN95%uppertracking
ratios<1.05 (largest1.030720). Bodyratio.919807,joint.987757,root.936881,
bodyangular.931486. Failures5vs1/1008,+0.397pp,CI[0,+1.091]pp narrowly exceeds
failureguard; combinedallmetric+failureCIflagfalse. User informed. Report
`eval_v2/teacher_clean_confirmation.json`; manifests/docs/adaptation_goal.md
are updated. WholegoalNOTcomplete: beststudentstillmax~1.086vs2%target.

Currentallocation(revalidatePIDs):
- GPU0 `context_payload_oriented_latent_seed59`, PID15069/session40973,
  teacherpayload250,1024env3000updates,LR1e-4,aux.05,latentMSEweight.05,
  --context-root-orientation. Watcher`v2_oriented_latent_student_watch.log` just
  started19:55. Structuraldiagnosis: teacherstate_physics usesworldquat andworld
  linear/angularvelocity (`_robot_raw_state[:,2:]`), while oldcontext seeslocal
  gyro/gravityhistory only, soinitialheading ismissing. Newoptionalcontextflag
  appends existingDEPLOYABLErobot_root_quat4toTCNhead, notasimulatorquaternion.
  Noextraenvironmentgroup; exportinputstill8199D. Labelsremainloss-only.
  Smoke2updates64envpassed; CPU96sampleexportmaxdiff5.9009e-6,
  artifactSHA01c9f5f8779fea13dcef028d47f89c9fd9fc71df8af43327c52cb4f710875219.
  Current200updateactionMSE.00471,latentMSE.02358 (vs~.2+withoutlatentsupervision),
  nottrackingacceptance. Relevanttests71passed (8noGTfeature/IMU/orientationcases).
- GPU1 `context_payload_reference_action_seed59`, PID29251/session7045,
  watcher99613. ~2860updates,nearlydone3000; jointreferenceactiongradients alone
  notbetter:2250root.284657,body.031401,joint.508504,max~1.154.
- GPU2 `context_clean_reference_kin_seed59`, PID44690/session10526,
  watcher77447/log`v2_clean_reference_kin_student_watch.log`.
  Teacherclean500,1024env3000updates,referenceencodertrainable,corefrozen,
  actionKD +weight1.0teacher1645featureMSE +aux.05. NoIMU/noadapter/orientation.
  New`--reference-feature-weight` requires`--train-reference-encoder`; feature
  labelsincludevelocity/FK, notjustdecodedpositionMSE.1500evalstillpoor:
  body.032128,joint.513870,root.295474,fail4/336;~1934updatesat19:54.
  Considerstoppingifcontinuedplateau ratherthanblindlyextending.
- GPU3 `oracle_deploy_features_seed66`, PID48939/session71964,
  watcher60284/log`v2_oracle_deploy_features_watch.log`.
  Initpayload250, --no-oracle-tracking-features,4096env2000updates/save125,
  tracking4/actionrate1/primary2/aux8/rightarm8/offset1,initialLR2e-5/entropy.0005.
  PrivilegeONLYstate_physics bottleneck; frontusesdeployableheight/contact/ref.
  CLIoracle_tracking_features overridefixed totri-stateNonepreserveorboolean
  explicitoverrideoninitialize; smokesavedconfigconfirmsFalse,cleanFalse,
  state_physicsschema andtrainbaseTrue.250evalbody.031930,joint.509361,
  root.283997,fail2/336;so farbelowneed. Previouscleanteacher31097stoppedafter500,
  andpreviousstudentPPO60686stoppedafter250 (root.277446,max~1.125).

IMUstudent13167 completed3000andallwatcherevals withnoerrors. Best2500:
maxratio1.086,body.03139,joint.5081,fail1/336. Matchedseed19643heldoutworldprobe
completed forIMU2500andfrozencontrol2500:
- IMUvelocityRMSE[.10264,.09710,.08967]m/s,payloadridgeRMSE.19315kg,R².87574.
- controlvelocityRMSE[.12164,.11078,.09965],payloadRMSE.19432kg,R².87423.
IMUhelpsvelocity10–16%,notamajorload-estimationchange; stillnotpreciseenough.
Files`context_payload_imu2500_probe_seed19643.json` and`...frozen2500...json`.

Causalactionfilterdiagnostics allFAILED (notintegratedintopolicy):
new`adaptation_filters.py` reads ONLYtwoexecutedactions fromhistory[3200:4650],
resetzerosbypass,order1hold/order2linear prediction. EvalCLI flagsrecorded;
comparison/scoreboardexcludeallnonzeroaction_filter_strength fromacceptance.
TestedIMU2500:order1strength.1/.2,andorder2strength.2/.35. Alltrackingworse;
root~.279–.289,bodyangular1.159–1.297. Do notextendthisbranch.
Logs`v2_imu2500_filter_*`,JSONs`eval_v2/imu2500_filter_*`.

Allnewexportvariants(featureadapter,IMU,trainableref) also passed isolated
python-I CPUloading withonlytorch,no intact_tracking imports; maxdiff<9e-6.
Pastcodehas runsourcehashes but notautomaticarchivedsourcecopies; no claim
that hashes alone reconstructeveryhistoricalsourceversion. Currentloader is
backwardcompatible and independently evaluatesallfixedcheckpoints.
No600xxstudentconfirmationhasrun. Nooutsidecodeedits orGPU4–7use.

Earlier sections below are historical where inconsistent.

## Latest update —19:24 UTC

New dominant-bottleneck diagnosis on SAME frozenpayloadteacher250, v2devonly:
- `estimated_height_contact`: body0.028977,joint0.483206,root0.244462,
  bodyangular1.10758,failure1/336. Removing height/contact truth barely hurts.
- `estimated_reference`: body0.031533,joint0.509754,root0.276838,
  bodyangular1.13985,failure1/336, despite TRUE physical/state encoder.
- `estimated_all`: body0.031380,joint0.508191,root0.276511,
  bodyangular1.13984,failure0. These point toward the frozen noisy-reference
  front end, not just context's physical identification, as a major limitation.
Outputs`eval_v2/teacher_payload_estimated_*_seed10001.json`; CLI
`--teacher-feature-ablation` records diagnosticflag. Comparison/scoreboard
explicitly exclude these modes from acceptance. None modifies teachercheckpoint.

Action-trained deployable reference encoder now implemented:
`ContextAdaptationActor(train_reference_encoder=True)` estimates height/contact
fromdeployablehistory underno_grad, recomputes decodedreference from raw1900D
noisyreference with gradients, differentiable FK -> normalizedfeatures -> core
and actionresidual. It bypasses detachedbehaviorcache onlyinthisopt-inmode.
NoGTlabelsenterthispath. Distill option`--train-reference-encoder` replays raw
deployable8199D observations and recomputes currentfeatures to avoidstaletargets;
teacher actions remain loss-only. Defaultfalse preservesoldcheckpoints.
Real64-env2-updatesmoke passed; referenceencodermaxupdate0.000803,core and
normalizersbit-identical. CPU96sampleexportmaxdiff5.8413e-6;
artifact`context_reference_action_export_smoke/policy_cpu.ts`SHA
ab6accbdb15764cb0b33825ae9391a4c1b5f90fb298cadaa29fee0d68732cff4.
Latest full relevant tests65passed,includinggradient/noGTtests.

Currenttraining:
- GPU0 `context_payload_imu_seed59` PID13167/session30724,
  teacherpayload250,1024env3000updates,LR1e-4,aux0.05,IMUflagtrue.
  Same teacher/seed/loss settings as frozen-base control, exceptIMU channel.
  Watcher82959/log`v2_payload_imu_student_watch.log`.500developmentmax1.097,
  still not2%target. FullIMUsmokeeval passed allreset/historyinvariants; export
  8349D maxdiff7.6294e-6,rawnoisyIMU samplemean[.5785,.3954,9.4879]m/s²,
  RMS7.7509,range[-15.79,28.68],notanempty/noise-onlysensor.
- GPU1 `context_payload_reference_action_seed59` PID29251/session7045,
  teacherpayload250,1024env3000updates,LR1e-4,aux0.05,
  --train-reference-encoder; corefrozen,noIMU,nofeatureadapter.
  Watcher99613/log`v2_reference_action_student_watch.log`.
  ~153updates at19:24,actionMSE.0040,0.6s/update,first250evalpending.
- GPU2 `context_teacher_ppo_seed61` PID60686/session89554,watcher11434.
 125eval body0.031911,joint0.515589,root0.275864,maxratio1.118,failure1/336;
  worse thaninitialstudent so far. Wait250 beforedecidingcontinuation.
- GPU3 `oracle_clean_payload_seed58` PID31097/session85266,
  watcher46190.500evalnowrunningPID31837;250allmeanratios<1 butfailed5/336.

Completednormalexit with allwatcherevals successful:
- frozen-basedistill37779/session88690 (3000updates;watcher32406).
- featureadapterdistill47400/session49153 (3000updates;watcher71558).
Neither improvedbroaddevelopmentbeyond~10–11% gap. Trainablebasecontrol31015
was stopped after2000checkpoint (last1750best max1.136;rootgap remains).

No phase-two acceptancecohort600xx consumed. No goalcompletionclaimed.
Earlier sections below are historical where inconsistent.

## Latest update —19:11 UTC

Noisy IMU context branch IS NOW IMPLEMENTED (supersedes the proposed-only note
below). `adaptation_sensors.py` reads the existing XML `imu_lin_acc` raw
sensordata slice (unique3Dsensor assertion), NOT root acceleration/velocity.
Appends separate50-frame150D group with Gaussian0.2m/s² noise. Alloriginal
groups/DR events unchanged. Optional `context_imu_accel` actorflag extendsTCN
channels122->125; original6100 normalization is unchanged, IMU/10clamped±10
is concatenated. Defaultfalse keeps every previous checkpoint compatible.
Actor`deployable_observation_groups` explicitlyincludesIMUwhenenabled.
Distill/eval/PPO/probe/export configure the new group from actorcfg, and eval
still strips ALL nondeployablekeys. Flat CPUexport8349DwithIMU (old8199Dwithout).
`context_imu_smoke`64env2updates passed; tests59passed including bothhistory
schemas, fourfeature-adapter/IMU no-GT cases, and rawsensor(notrootstate)read.
Full v2smokeeval session30645/log`v2_context_imu_smoke.log`; independentCPUexport
session68427/log`context_imu_export_smoke.log`, bothonGPU0. Verifycompletion.
No fullIMUstudenttrainer launched yet. Currentfrozencontrolstudent~2550updates
willfinish3000 soon, potentially freeingGPU0.

Cleanteacher250 fullv2 result: body0.02875465,joint0.48245197,root0.24215685,
bodyangular1.018966rad/s, ALLten meanratios<1.0 nominaldevelopment. However
5/336failures versusnominal1/336 violates+1pp pointguard (+1.19pp), so NOT a
new acceptancepass. Continuechecking latercheckpoints; promising headroom for
futureprivilegeremoval, but don'tignorefailuresorreplaceconfirmedteacher250.

All newCLI helpers use onlyrepoedits. TeacherregularizedPPO metadata now hashes
adaptation_ppo.py; later launches also hashadaptation_sensors.py. Modelcheckpoint
actors never containteacherparameters. Source changes don't reload into already
runningprocesses; their recordedhashes identify the code snapshot they loaded.

Earlier sections below are historical where inconsistent.

## Latest update —19:06 UTC

Active GPU allocation (revalidate PIDs):
- GPU0 frozen-base action distill `context_payload_frozen_distill_seed59`,
  PID37779/session88690, watcher32406. Still training to3000.
- GPU1 feature-adapter distill `context_payload_features_seed59`,
  PID47400/session49153, watcher71558/log`v2_payload_feature_student_watch.log`.
  Same teacher250/seed59/LR1e-4/aux0.05 as other distills, plus featureloss0.05.
  Deployable1645features+64context -> zero-initialized residual feature adapter
  (256,128), then frozen teacher core and trainable action residual. No true
  features at inference. Absolute action replay avoids stale adapter/core targets.
  Feature module682861parameters; defaultsNone keeps old checkpoints compatible.
  Adapter smoke trained2updates64envs, corebit-identical; CPUexport96realstates
  maxdiff5.0068e-6,artifactSHA01fa9350f8b891d8c0596f64a38f900360d49d6b9c20c627a0d2e5b733e970c9.
  Unit tests exercise no-GT input and gradients through frozen core to adapter.
- GPU2 `context_teacher_ppo_seed61`, PID60686/session89554,4096envs2000updates,
  save125, initializedauxstudent2750. New `adaptation_ppo.py` alternates standard
  PPO with4teacher-BCsteps(weight0.1,batch2048) on same visited states. Teacher
  is frozenpayload250, algorithm-only (not a module in saved student). Wrapper
  reserves29D label field; act computes simultaneous teacher labels on separate
  TensorDictcache then restores student's deployable cache. PPOrewardtracking4/
  actionrate1/primary2/aux8/rightarm8/offset1,initialactorLR2e-5,entropy0.0005.
  Watcher`v2_teacher_regularized_student_watch.log` started19:06.
  Real2-update64-envPPO smoke passed and no-GT/cache/critic-gradient tests pass.
- GPU3 clean privileged teacher `oracle_clean_payload_seed58`, PID31097,
  session85266, watcher46190. First250evaluation not yet available.

Stopped TERM with allfilesretained:
- right-arm teacher3254 after375 (250 is independently confirmed/frozen;
  375development anchor0.260011 slightlyregressed).
- trainable-base distill31015 around1900updates after1750 checkpoint; broad
  trackingplateaued~18%rootgap despite lower trainingactionMSE. Watcher46190
  handles its remaining checkpoints and clean teacher; stopped entry harmless.

Learning-rate audit: PPO schedule is adaptive, CLI LR is INITIAL not fixed.
Inspected teacher250,oldstudentPPO750,newPPOsmokefinal all at actorLR1e-5 floor,
criticLR5e-4. Do not attribute failures to hypothetical exploding LR; measured
checkpoints do not support that. Relevant suite55passed after new PPO tests.

Possible next sensor experiment (NOT YET IMPLEMENTED): existing XML contains
pelvis `imu_lin_acc` accelerometer. Context currently only q/qdot/gravity/gyro/
knowncommand/torque histories. Original SPV5-2 features explicitly remove base
velocity and its error. Auxiliary velocity estimateRMSE~0.1m/s may contribute
to root drift. Could append noisy measuredIMUaccelerationhistory (not simulator
rootvelocity) to context; if implemented, update eval input selection, export,
distillation,probe and no-GT tests consistently. User allowsdeployableobschoice.

Earlier sections below are historical where inconsistent.

## Latest update —18:51 UTC

700xx confirmation completed successfully at18:52: allTEN mean ratios<=1.02902,
primary95%upper ratios0.98489(body),1.03334(joint). Average-tracking stage-one
milestone achieved, NOT statistical equality: anchor/bodyangularupperCI1.07423/
1.07021,failures6vs2outof1008,+0.397pp[-0.198,+1.190]pp. User explicitly informed
of uncertainty. Report`eval_v2/teacher_payload_confirmation.json`. Whole goal
still incomplete because broad student is outside2%. Frozen control student
PID37779/session88690 onGPU0, watcher session32406/log`v2_payload_frozen_student_watch.log`.
Latest relevant tests52passed (adaptation,comparison,residualpolicy,residualtraining).

Right-arm-targeted teacher250 development worst ten-metric ratio is now1.038,
body0.02904m,joint0.4835rad,failure1/336 equal nominal. Frozen SHA
`3a2ee99832feab927e7043c5980650d93d0c06455e80e1df75f1be46c00fec04`
BEFORE starting fresh70001/70002/70003 confirmation at18:47. Manifest
`eval_v2/teacher_payload_confirmation_freeze.json`. Three sequential nominal/
teacher pairs run onGPUs0/2/3; sessions35503,83354,25922 (parallel return order
not guaranteed; inspect logs rather than assume session mapping). Outputs
`nominal_seed7000X.json` and`teacher_payload_seed7000X.json` in eval_v2.
Do not tune frozen teacher250 on these confirmation data. Cohort600xx remains
reserved and unconsumed for the next frozen broad student.

Clean-proprio smoke full evaluation finished: body0.028393,joint0.475114,
root0.263678,maxratio1.069(root),failure1/336. Input substitution alone is not a
pass. Full clean teacher now trains from newer payload teacher250 instead:
GPU3 PID31097/session85266 `oracle_clean_payload_seed58`,4096envs2500updates,
tracking4/actionrate1/primary2/aux8/rightarm8,referenceoffset1,LR2e-5,seed58.

New student action-distillation option `--train-student-base` includes the
student MLP core in the optimizer, retaining frozen deployable preprocessing.
Replay stores ABSOLUTE teacher actions in this mode, avoiding targets tied to
stale pre-update base actions. Actual checkpoint0 shows base maxupdate0.000401,
all tracker normalizers bit-identical to teacher, no privileged module keys.
GPU2 PID31015/session88612 `context_payload_base_distill_seed59`,teacher250,
1024envs3000updates,LR1e-4,auxloss0.05,seed59. Matched frozen-base control launched
onGPU0 as`context_payload_frozen_distill_seed59` at18:51 (same settings except
no --train-student-base). Both remove ALL teacher-only input channels.
Watcher46190 `v2_payload_student_clean_teacher_watch.log` handlesGPU2/GPU3.

Stopped with TERM, retained checkpoints/logs (no deletion):
- Smooth teacher15463, latest500,maxratio1.068.
- Original teacher extension38406, latest2250,maxratio1.082,failure0.595%.
- Student PPO28013, latest750,maxratio1.095,failure1.190%.
- Time-alignment control55046, latest250,maxratio1.061,failure1.190%.
Right-arm teacher3254 continues onGPU1 with125-update evaluations.
Aux student38496 completed3000 normally; best2750 development maxratio1.098,
body0.030449,joint0.488497,root0.270955,failure0. The old best student is still
outside the2% broad tracking target; neither stage-two success nor whole-goal
completion has been claimed.

Earlier sections below are historical when inconsistent with this update.

## Latest update — v2 evaluations running normally, ~18:26 UTC

### Newest18:39UTC additions

Right-arm-targeted teacher125 now passes all TEN DEVELOPMENT point margins:
body0.02897m,joint0.4845rad,maxratio1.045(body angular velocity),failure0.595%.
This needs a fresh confirmation cohort (500xx was already consumed by a different
frozen model). Do not label the whole goal complete.

Added optional teacher-only `oracle_clean_proprio` flag, with separate clean
6100D history and195D FK observation groups. Original noisy/biased deployment
groups are copied, not modified. Agent substitutes only a separate TensorDict
view when computing privileged features. All history buffers are normal MJLab
buffers and included in v2 reset isolation audits. Config flag is saved/loaded;
distillation configures the teacher groups but removes this flag from student.
The full relevant32-test suite includes original-noise preservation, no mutation
of student observations during teacher feature substitution, and poisoned-GT
student inference with auxiliary head enabled.

`oracle_clean_proprio_smoke` trained2updates64envs with this mode from
oracle1750_merge020, passed GPU smoke. Its full336episode evaluation is now
running onGPU0, output`eval_v2/oracle_clean_proprio_smoke/eval_final_seed10001.json`,
log`v2_oracle_clean_proprio_smoke.log`. This is exploration, not final acceptance.

Auxiliary readout probe completed on newseed14643: payload ridgeR²0.8775,
instantRMSE0.210kg; episode-averagedR²0.9603/RMSE0.120kg. Frozen trained head
velocityRMSE[0.1026,0.1093,0.0927]m/s,payloadRMSE0.208kg onheldoutworlds.
So environment information is genuinely encoded, but velocity estimation has
meaningful error. No new sensor channels were added; existing XML does have
`imu_lin_acc`, a possible future deployable-input experiment, not implemented.

All four initial v2 full evaluations completed successfully, including all
survivor cursor, state and history assertions. The history-corruption regression
has now also been added; the relevant unit suite passes all 22 tests. The default
scoreboard root is `runs/adaptation_goal/eval_v2`. Comparisons gate acceptance on
the v2 protocol; old v1 values remain invalid for acceptance.

Fresh confirmation seeds 40001/40002/40003 were predeclared for the previously
frozen compact teacher250/student250 pair. Nominal and teacher results exist;
student evaluations completed (sessions44733,96293,72396; GPUs0,2,3).
Do not use these confirmation outputs for selecting a new candidate.

Valid v2 compact teacher250 confirmation: body ratio0.9793[0.9527,1.0118],
joint0.9708[0.9406,1.0017], failure difference+0.298pp[-0.298,+0.992]pp.
Student250: body0.9829[0.9548,1.0176],joint0.9721[0.9416,1.0034],
failure difference+0.298pp[-0.397,+1.290]pp. Primary point estimates and
intervals pass both prescribed primary margins, but root error is still
1.428x(teacher)/1.435x(student), so whole tracking equivalence is NOT established.
Student failure uncertainty also crosses the +1pp guard. User informed.

Current trainers (revalidate PIDs before action):
- GPU0 `oracle_merged_smooth_seed52`, PID15463/session42869: initialize from
  `oracle_base_merge025/checkpoint_0.pt`,4096envs2500updates, trackingx4,
  actionrate restoredx1, metricweight2/auxweight8,original sampling, actorLR2e-5.
  Watcher`v2_merged_smooth_watch.log` handles this run. Prior moderate run46737
  was stopped after250 (worst ratio1.098), checkpoint retained.
  A SECOND GPU0 trainer `oracle_merged_timealign_seed52`, PID55046/session86545,
  was launched at18:16,
  identical init/seed/envs/rewards/optimizer settings, except
  `--metric-reference-offset 1` (smooth control uses0). Explicit matched timing
  experiment, not a changed evaluation protocol. Both PPO jobs together fit
  comfortably in80GB; check CPU/GPU throughput before adding anything else.
- GPU1 `oracle_payload_arm_seed57`, PID3254/session86864:4096envs2500updates,
  save125,init`oracle1750_base_merge020/checkpoint_0.pt`,trackingx4/actionratex1,
  primary2/aux8/right-arm-angular8,referenceoffset1,actorLR2e-5,seed57.
  Watcher`v2_payload_arm_watch.log` withinterval125. A64-env2-update smoke passed.
  Prior `context_merged1750_seed53`, PID20795/session87311, completed3000updates
  and its watcher finished all evaluations without errors.
  Prior context_oracle1250_seed50 completed3000updates/session23891 exit0,
  and watcher50459/session86877 finished all evaluations without errors.
- GPU2 `context_merged_balanced_seed54`, PID28013/session32350,
  initializefrom`context_base_merge025/checkpoint_0.pt`,4096envs2500updates,
  trackingx4/actionratex1/metricweight2/auxweight16,actorLR2e-5,originalsampling.
  Entire actor remains deployable-only, including during PPO; privileged critic
  is allowed. Context encoder and policy core are trainable.
  Calibration run58407/session76367 was stopped: height estimation improved on
  training replay, but repeated closed-loop tracking evaluations stayed worse
  than uncalibrated distillation. All checkpoints and calibration logs retained.
- GPU3 TWO concurrent trainers (H100 memory ample):
  `oracle_features_tracking4_extend_seed44`, PID38406/session39003, resumes
  original teacher final with optimizer, same4096envs/rewards/seed44,2000extra
  updates. Original743/session86544 completed2000updates exit0; watcher30195
  completed all evals with no errors. Second GPU3 job:
  `context_aux_merged1750_seed53`, PID38496/session46257, same teacher andseed
  as GPU1's ordinary distillation, but weight0.05 loss-only supervision of a
  context-latent head for3D body-frame velocity and payloadkgminus2. Labels never
  enter inference. Default1024envs3000updates. Both watched by
  `v2_teacher_extend_context_aux_watch.log`.

### New development diagnosis and targeted response

Per-body velocity decomposition in adaptation_eval is read-only and asserts
that its body-wise means exactly reproduce the aggregate metric. V2 diagnostic
outputs are`nominal_diagnostics_seed10001.json` and`teacher_diagnostics_seed10001.json`.
The residual body-angular-error gap is concentrated in the payload arm:
right wrist yaw1.396->2.045rad/s,pitch1.230->1.686,elbow0.975->1.143.
Most left-side bodies improve. Correctly mapped actuator saturation fractions:
right wrist pitch0.133%->7.627%,yaw0.014%->3.880%. **Actuator order differs from
joint order; never zip joint_names with saturation fractions.** Raw diagnostic
JSON explicitly gives both name schemas. Payload-arm angular reward8 focuses
training on these six bodies, while unchanged evaluation still includes all22.
No force limits or DR ranges were relaxed. Latest relevant suite30passed.

Diagnostics reveal ordinary simulator numerical sensitivity: repeated nominal
root mean differs about1.4mm and teacher one repeat failed1/336 instead of0.
Do not claim bit-identical trajectories or zero population failure risk.

Stopped intentionally with TERM and retained all files: balanced teacher52800
(latest checkpoint750, v2 failure3.27%) and low-exploration-noise7844 (checkpoint500,
v2 anchor ratio1.369). These branches did not resolve the broad tracking gap.
Context balanced PPO64756 was also stopped around700 updates, retaining500:
anchor ratio1.220 and failure1.786%, versus nominal0.298%.
New watcher `v2_oracle_refinement_watch.log` handles GPUs0/1. Existing watcher
PID30195/session54964 continues handling GPUs2/3; stopped run entries are harmless.

Development v2 oracle-feature1250 is currently the strongest broad candidate:
body0.03145m,joint0.5304rad,max ten-metric ratio1.103(body angular velocity),
failure0.595%. New moderate-reward continuation attempts to close joint and
velocity gaps without the large root/failure regressions of metric-heavy runs.
Update1500 is now available: body0.03104m,joint0.5265rad,maxratio1.087(joint),
failure0.595%. Uncalibrated oracle1250 student500 maxratio1.175, failure0.893%.
Update1750: body0.03020m,joint0.5061rad,maxratio1.070(body angular velocity),
failure0%. New single-network base merge of oracle1500 with25%compact500 core
has body0.028352m,joint0.473266rad,anchor0.248616m,failure0.595%,maxratio1.060.
Only body angular velocity exceeds1.05; all other ratios<=1.027.
`oracle1750_base_merge025` is now being evaluated (session created after this
update; log`v2_oracle1750_base_merge025.log`). A single interpolated core, not an
ensemble. Exact source hashes and merged keys are in each run_config.json.
Its result is now complete: body0.0283372m,joint0.463796rad,anchor0.254713m,
failure0.298%,worst ratio1.054(body angular velocity). New1750-core fractions
0.20/0.30 are being evaluated in runs `oracle1750_base_merge020/030`.
Student-only core interpolation also being tested: receiver
context_oracle1250_seed50/checkpoint1500, donorcontext_action_distill_seed43/250,
fractions0.25/0.50, names`context_base_merge025/050`, sessions93511/89677.
All keep one fixed deployable actor and use v2 development only.
Student merges completed: 0.25 body0.031775,joint0.513465,anchor0.278228,
failure0.595%,maxratio1.128(anchor); 0.50 body0.030516,joint0.488117,
anchor0.294560,failure0%,maxratio1.194(anchor). GPU2 continuation uses0.25
for better broad balance, not the lower-primary but worse-root0.50 candidate.
Teacher1750 merges0.20/0.30 completed with zero failures and maximum ratios
1.051/1.052(body angular velocity), respectively. No new confirmation seeds yet.
Broad teacher confirmation seeds50001/2/3 and student60001/2/3 were reserved
at18:05UTC before any of those evaluations. Freeze SHA256 candidate beforeuse.
Teacher broad candidate FROZEN at18:13:53UTC:
`oracle1750_base_merge020/checkpoint_0.pt`, SHA256
`34cd8a4e8fc3a8d17c892335d3a14c6c42ddd2c67f18b6b000b0f5234ef520c7`.
Manifest`eval_v2/teacher_broad_confirmation_freeze.json`. Sessions38711/48114/57722
onGPUs0/1/2 sequentially run matching nominal then teacher for50001/50002/50003.
Do not tune this frozen candidate on these results. Development maximum1.051
is borderline; confirmation is explicitly a test, not a guaranteed pass.
These confirmations completed: body ratio0.9320,joint0.9680,root0.9869,zero
observed failures in1008; nine mean ratios<=1.025. Body angular remains1.05433
[1.02768,1.08329], narrowly failing the strict all-ten1.05 target. Do NOT loosen
the target or reuse500xx for selecting/tuning new candidates. Full comparison
`eval_v2/teacher_broad_confirmation.json`; earlier compact400xx is separate.
Original teacher2000(final) worst ratio1.079/failure0.893%, slightly worse than
1750. Its20%merged version is evaluating as`oracle2000_base_merge020`.
Smooth teacher continuation250 is also evaluating. Current best deployable
distill candidate1250: body0.03061,joint0.4908,maxratio1.098(root),failure0%.

Auxiliary-supervision implementation passed a real GPU smoke run and the full
relevant28-test suite. The actor auxiliary head reads only the context latent;
the action method does not call it. An input-poisoning test with this head enabled
and a rotation-invariant velocity-label test both pass. Old checkpoints default
to no auxiliary head and remain strict-load compatible.
Latest relevant suite29passed after adding optional next-frame reward target
and a no-cursor-mutation test. A64-env2-update GPU smoke also completed normally.
Offset0(default) retains existing behavior. Offset1 reads `gather_reference`
without modifying command clocks. Evaluation staysv2 and all priorv2 data remain
valid. No underlying MJLab or existing tracked repository source was changed.
The calibration smoke test completed: estimator/reference weights changed,
base-policy MLP and observation normalizers remained bit-identical. No privileged
module keys exist in the student checkpoint. Source hashes now also recorded
for new distillation runs. Comparison unit tests4passed enforce a fixed policy
hash across seeds, distinct seeds, reset audits, and exclude diagnostic ablations.

Everything below this section is historical and superseded where inconsistent.

## CRITICAL LATEST STATE — v1 evaluation invalidation, ~17:17 UTC

**Read this first. All v1 tracking acceptance claims later in this document are
superseded pending v2 re-evaluation.** Inspecting MJLab showed that public
`env.reset(env_ids=...)` calls global command.compute(dt=0), but our MotionCommand
advances time_steps even atdt0, and reset pushes ALL observation histories. Thus
resetting a terminated/inactive episode in the old evaluator corrupted surviving
worlds' reference clocks/history. User was promptly informed; old results are
retained as exploratory ONLY. No claim of goal completion is valid yet.

Training uses the standard auto-reset step, not this manual evaluator path, and
is unaffected. The physics readout probe uses standard auto-reset and is also
unaffected. Pointwise exported-model equivalence is valid; its old closed-loop
tracking evaluation must be redone under v2.

`adaptation_eval.py` now uses PROTOCOL=`balanced_fixed_starts_v2_isolated_resets`.
It masks inactive actions tozero and calls `reset_finished_worlds` after masking
done trajectories out of metrics. That helper calls ONLY env._reset_idx(done_ids)
and scene.write_data_to_sim(), NOT public reset(), NOT extra forward(), NOT
command/observation compute. Reset worlds need no fresh observation because they
are inactive; next normal simulator step computes their forward dynamics.
Every reset audits surviving qpos/qvel/qacc_warmstart/cursor and observation
history buffer pointer/push count/values. Every step asserts alive reference
cursors equal initial_steps+step+1. Resampling-time range forced1e9, no motion-end
resampling, original matched starts preserved. No outside-repo code changed.

All four OLD watchers were stopped (3396,3836,9898,55788), trainers untouched.
No v1 eval process was live when stopped. Watcher CLI now REQUIRES
`--evaluation-root runs/adaptation_goal/eval_v2`; output goes under that root's
run-name child, not original run directory. **Restart watchers only after the
initial v2 GPU checks below pass.** Existing v1 artifacts must never be overwritten
or mixed into v2 comparisons. `summarize_adaptation.py --root .../eval_v2` works
once its new nominal baseline exists. All future final confirmation needs fresh
matching v2 nominal/candidate results; don't mix v1 baseline with v2 candidate.

Initial v2 full evaluations RUNNING (all seed10001,42motions×8×500steps):
- GPU3 frozen DR tracker: session99864, log`runs/adaptation_goal/v2_frozen_dr_initial.log`,
  output`eval_v2/frozen_dr_seed10001.json`.
- GPU1 existing nominal: session88538, log`v2_existing_nominal_seed10001.log`,
  output`eval_v2/existing_nominal_seed10001.json`.
- GPU0 frozen teacher250: session94590, log`v2_teacher250_seed10001.log`,
  output`eval_v2/teacher250_seed10001.json`.
- GPU2 frozen student250: session47606, log`v2_student250_seed10001.log`,
  output`eval_v2/student250_seed10001.json`.

Latest relevant unit suite:22passed (reset helper test added; history snapshots
were added immediately afterward, real GPU runs above audit them). Add a forced-
history-reset regression if useful. Need inspect GPU completion and any assertion.

Current trainers still live: GPU0 balanced52800/session74003; GPU1 low-noise
balanced7844/session65372; GPU2 context balancedPPO64756/session92029;
GPU3 oracle-feature743/session86544. Exact launch args appear later in this file.

Updated 2026-09-06 ~16:02 UTC. User has approved starting the research, clarified
that stage two replaces **all** teacher privileged observations with context
encoder + deployable observations, and explicitly requested autonomous retries
until the objective is met. Do not ask for more permission. Code edits must stay
inside `/data_zcy/wxy/intact-tracking`. GPUs 0–3 only; GPUs 4–7 belong to other jobs.

The existing goal tool still returned `blocked` after user approval; its API does
not offer an active/resume update. Work is resumed by the user's explicit message.
Do not mark complete unless the entire two-stage research objective is proven.

Read `docs/adaptation_goal.md` for protocol, baseline and decisions. No stage has
passed yet. Student infrastructure is preparation and has not been trained.

## Running jobs (revalidate PIDs/cmdlines and sessions before acting)

| GPU | Run under runs/adaptation_goal/ | PID | exec session |
| --- | --- | ---: | ---: |
| 0 | nominal_scale1_seed42 | 8039 | 14026 |
| 1 | dr_control_scale1_seed42 | 8121 | 2146 |
| 2 | compact_unfrozen_tracking2_seed42 | 39942 | 97939 |
| 3 | oracle_features_tracking4_seed44 | 743 | 86544 |

Both initial controls target 2000 updates, 1024 envs. GPU2 targets 2000 updates,
2048 envs, compact 566-D privileges, full policy MLP training, tracking weights
x2 / action-rate x0.5. GPU3 targets 2000 updates, 4096 envs, adds true height,
contact and clean reference tracking features, tracking x4 / action-rate x0.25,
entropy 0.0005, seed44, initial actor LR5e-5. All retain full original DR plus 1–3kg
fixed hand payload. Startup kwargs and exact commands are in each run_config.json.

Watchers run `scripts/watch_adaptation_evals.py` and evaluate stable checkpoints
every 250 updates on development seed10001 on their respective GPUs:

- Initial batch watcher: session4331, `evaluation_watch.log`; stopped oracle runs
  are still in its list but it skips completed existing JSONs and checks trainer PIDs.
- GPU2 watcher: session35659, `compact_evaluation_watch.log`.
- GPU3 watcher: created after this handoff was written; inspect process list and
  `oracle_features_evaluation_watch.log` for its actual PID/session if not in context.

Initial oracle_scale2_seed42 (PID8193, session69246) was deliberately terminated,
checkpoint500 retained. Initial oracle_scale1_seed42 (PID6704, session14750) was
terminated after checkpoint1250: body0.039855m, joint0.688605rad, failure0.298%.
Both failed to meet target and stagnated despite training reward increasing.

## How to inspect

Training stdout logs may remain buffered even while TensorBoard events advance.
Check live process handles plus events, not just unchanged stdout. Use the project
`.venv/bin/python` and `tensorboard.backend.event_processing.event_accumulator.EventAccumulator`
to inspect last scalar steps (size_guidance={'scalars':0}). Key tags:
`Metrics/motion/error_body_pos`, `Metrics/motion/error_joint_pos`, `Train/mean_reward`,
`Loss/privilege_shuffle_action_delta_rms`, `Perf/collection_time`, `Perf/learning_time`.
Evaluation JSON files are `RUN/eval_ITER_seed10001.json` with per-episode arrays,
fixed start frames, coverage, failures and ten error metrics. Watcher logs record
the actual evaluation subprocess start/exit.

Compare with:
`.venv/bin/python scripts/compare_adaptation_evals.py --reference runs/adaptation_goal/existing_nominal_seed10001.json --candidate RUN/eval_ITER_seed10001.json --output RESULT.json`

Nominal baseline checkpoint is `runs/residual_policy_no_latent_nominal_v13_run2/checkpoint_1000.pt`.
Its development result is body0.030918m / joint0.484586rad / failure0.298%.
Confirmation results already exist for seeds20001,20002,20003. Do not use candidate
confirmation seeds for tuning; once a development candidate passes, test the three
confirmation seeds and assess paired motion/seed bootstrap intervals.

## Research code

- `adaptation_policy.py`: privileged actor with a replaceable 64-D latent; compact
  state/physics or full critic-privilege wrapper; optional full-MLP fine tuning and
  oracle tracking-feature path. Also deployable sensor-history TCN + student actor.
- `cli/adaptation_train.py`: PPO training, optional new reward weights, resume or
  actor/critic-only initialization from a teacher/student checkpoint with fresh PPO.
- `cli/adaptation_eval.py`: fixed all-motion evaluation, no adaptive sampler, strips
  all privileged keys before actual student actions, optional zero/shuffle ablations.
- `cli/adaptation_distill.py`: unrun stage-two implementation. Mix teacher and student
  on-policy actions, train context/residual head on full teacher actions. Supports
  different oracle/deployable base features. Defaults latent regression to0 because
  teacher latents may not be uniquely recoverable; action imitation is primary.
- `oracle_compensation.py`: rejected analytic torque-difference diagnostic. Bias gain1
  gave body0.05773/joint0.97618/failure8.63%. Nominal twin zero-correction audit passed.
- True tracking features alone (no retraining) gave body0.04377/joint0.62169/failure1.488%.

Most recent tests: `tests/test_adaptation.py tests/test_residual_policy.py`:17 passed.
Tests cover balanced deterministic starts, trainable teacher bottleneck, temporal
ordering, and student ignoring poisoned/absent privilege fields. GPU smoke verifies
teacher and oracle-feature training/checkpoints; the compact unfrozen smoke changed
16 policy-MLP tensors, while all frozen preprocessing weights stayed identical.

All research source files are newly added/untracked. Existing `wandb/` and a directory
whose name is a space are user-owned and untouched. No existing tracked code was edited.
`pytest` was installed only into this repository's `.venv`; use system `ruff` executable
to check/format explicitly named in-repo files (project venv has no ruff binary).

Keep pursuing the full goal. Current candidates may need more iterations, different
privileged conditioning, or sharper tracking objectives. Do not equate launching jobs,
reward gains, a smoke test, or development-only improvements with completion.

## Latest override: 16:27 UTC

- Nominal control PID 8039 finished normally (session 14026 exit 0), with
  `checkpoint_final.pt` containing iteration 1999. Its final evaluation is live
  under the original watcher. The old DR control PID 8121 was deliberately
  stopped after checkpoint 1750 (session 2146 exit 143), plateaued/worsening.
  Compare both controls at 1750 for equal update budgets. Nothing was deleted.
- GPU 0 now runs `metric_linear_compact_seed45`, PID **28185**, session **18292**.
  It starts from compact checkpoint 750, 4096 envs, 4000 new iterations, uniform
  motion sampling, original tracking weights, action rate x0.25, metric weight 10,
  `--metric-reward-shape linear`, entropy 0.0005, initial actor LR 3e-5, seed 45.
  Training started; **start its evaluation watcher once run_config.json exists**.
- GPU 1 runs `metric_aligned_compact_seed45`, PID **14830**, session **64186**.
  Same settings and initialization as the linear candidate, but default `exp`
  reward shape. Watcher PID **16292**, session **62589**, log
  `metric_aligned_evaluation_watch.log`. Eval 250 PID **28544** just started.
- GPU 2 compact unfrozen tracking2 PID 39942 continues, around 1200 updates.
  Update 1000: **0.036552 m body / 0.60698 rad joint / 0.893% failure**.
  Update 750 comparison JSON shows both ratios/95% CIs still fail the threshold.
- GPU 3 oracle features tracking4 PID 743 continues, around 480 updates.
  Update 250: **0.039247 m / 0.670797 rad / 1.190% failure**, worse than compact.
  Its watcher is PID **3836** (session unknown), running normally.

New module `adaptation_rewards.py`: two optional rewards aligned to joint L2
and the evaluation's yaw-aligned equal-body position error. Rewards are evaluated
at the environment's standard pre-command-advance reward time, not by changing
evaluation timing. Exponential scales: joint 0.4 rad, body 0.03 m. Linear option
is `1 - clamp(error/scale, max=5)`, unchanged slope below catastrophic outliers.
First GPU smoke hit a quaternion broadcasting bug for batch>1; fixed via explicit
body-axis expansion, now covered by a two-world test. Revised smoke completed.

Latest tests: **19 passed** across adaptation/residual tests. Resume smoke also
completed: `resume_metric_smoke/checkpoint_final.pt`, iteration 2, correctly
restores `state_physics` + oracle feature flags without CLI architecture flags.
`adaptation_train` now loads actor config for both `initialize-from` and `resume`.
For continuations, prefer a NEW output directory to preserve prior results and
avoid watcher ambiguity over overwritten `checkpoint_final.pt` / `eval_final`.
Resume is supported in a new directory; pass the intended reward/sampler arguments.

New optional `adaptation_eval --joint-diagnostics`: per-joint RMSE and final-
substep force >=95% force-limit fraction, without changing acceptance metrics.
Files `compact_750_joint_diagnostics_seed10001.json` and
`nominal_joint_diagnostics_seed10001.json` exist. DR right-wrist pitch/yaw RMSE
0.227/0.225 rad vs nominal 0.097/0.129; saturation 6.94%/4.33% vs 0.125%/0.016%.
Cannot infer impossibility from this. Repeated seeded GPU evals differ slightly
(nominal diagnostic body 0.030712 vs original 0.030918), so don't claim bitwise
determinism. Noisy-FK student key-body observations were source-audited: they use
measured biased q, noisy qdot and gyro, not privileged world body poses.

Comparison script now also reports paired failure-difference CI and requires the
full seed set in `confirmation_seed_set_complete`; no candidate has yet used the
confirmation seeds. New training runs record source SHA256 entries because the
research files are untracked and ordinary git diff doesn't capture them.

Goal-tool state remains stale `blocked` from the early review gate; user explicitly
approved and requested persistent autonomous research. `get_goal` confirmed this
again, but API has no resume operation. Do not mark complete or create a duplicate
unfinished goal to work around it. Continue the authorized research in this turn.

## Latest override: 16:36 UTC — primary teacher confirmed; student started

Teacher `metric_aligned_compact_seed45/checkpoint_250.pt` passes the two-primary-
metric + failure criterion on ALL three seeds 20001/2/3. Aggregate body
0.029834 m vs nominal 0.030444 (ratio 0.97998, 95% CI [0.95455,1.01472]);
joint 0.465322 vs 0.476425 (ratio 0.97669, CI [0.94534,1.00944]); failure
0.3968% vs 0.0992%, diff CI [-0.2976,+0.8929] percentage points. Result:
`runs/adaptation_goal/teacher_250_confirmation.json`. Teacher checkpoint is frozen
for stage-two supervision. **All-metric equivalence is NOT achieved:** anchor
position still 46.6% worse and velocity/orientation 9–19% worse. User was told
this clearly; don't mark whole goal complete or hide the limitations.

GPU 2 old compact PID 39942 was deliberately stopped after eval1250
(0.034792 m / 0.585389 rad / 0.5952% failures), retaining everything.
GPU 2 now runs **context_action_distill_seed43**, PID **37613**, session **74357**.
Teacher is the frozen confirmed checkpoint above. Student TCN uses only measured
50-frame history; complete teacher actions supervise its residual + context
encoder. Teacher fraction anneals 1 to 0 over 1500 of 3000 updates; 1024envs,
4 rollout / 4 gradient steps, batch2048, replay32768, latent MSE weight0,
uniform motion sampling, save250. New watcher session **from latest tool result**,
log `context_distillation_evaluation_watch.log` (find PID if needed). Watcher
script now recognizes both adaptation_train and adaptation_distill processes.

Student GPU smoke completed 3 updates; strict no-privilege 42-motion/16-step
inference smoke passed (`context_inference_smoke.json`). This is NOT tracking
acceptance, only runtime/input-contract evidence. Full student development
evaluations will be produced by watcher. Formal student confirmation will use
**30001,30002,30003**, selected before student results; corresponding existing
nominal evaluations were just launched (GPU0 session26091, GPU1 session26490,
GPU3 session36819). Comparator recognizes either full 200xx or full 300xx set.

GPU 0 linear candidate watcher is PID30448, session94563; running. GPU1 metric
teacher continues beyond frozen250 for improvement. GPU3 oracle-feature tracking4
at500: 0.035641 m / 0.602368 rad, anchor0.262525 m (closer to nominal than the
primary-passing teacher). All three teacher trainers continue. Consider using
their results for further balanced tracking improvements, but never replace the
frozen stage-two teacher silently or select student checkpoints on 300xx results.

## Latest override: 16:50 UTC — balanced training and latent probe

Student frozen250 confirmation on 30001/2/3: body0.030366 vs nominal0.030770,
ratio0.98687 CI[0.95821,1.01847]; joint0.475377 vs0.480585, ratio0.98916
CI[0.95507,1.02525]. Failure0.6944% vs0.0992%, diffCI[-0.0992,+1.4881]pp.
The originally specified POINT threshold passes, but the stricter joint95% CI
does not fit2%, and auxiliary errors still lag. User was explicitly told not
complete. Report `student_250_confirmation.json`. These confirmation seeds are
now consumed; don't select a new checkpoint on them. Future final confirmation
should predeclare a fresh seed set (e.g.400xx), retaining all earlier evidence.

Context250 ablations seed10001: normal0.029737/0.475436/failure0.893%; zero
0.030033/0.477931/1.190%; shuffle0.030048/0.478005/0.893%. Effect small, not
proof that adaptation is the main performance source. New read-only diagnostic
CLI `context_probe.py` probes latent physical identifiability with worlds split
384 train /128 heldout, all18 temporal samples of each world kept together.
Seed14643 independent of training/evaluation. Physics labels NEVER enter policy
actions. Current latent payload-mass R²=**0.71845**, RMSE0.318kg, compared with
current-sensor linear probe R²0.0450; episode-average latent probe R²0.8662.
Other parameter linear probes weak/negative. Artifact
`context_250_physics_probe.json`; completed GPU session69430. This establishes
some payload information, not recovery of all DR parameters or strong causal
policy dependence. Regression-unit test added; latest test session12945 pending
collection (previous20 adaptation/residual tests passed; one new test now added).

GPU0 linear trainer PID28185 was stopped after eval250 (0.030493m/0.480238rad,
anchor0.384854m), retained everything. GPU0 now **balanced_metrics_seed47**,
PID**52800**, session**74003**, watcher session**90721** (find its PID if needed),
log `balanced_metrics_evaluation_watch.log`. Initialized from metric teacher
checkpoint500, 4096envs4000updates, tracking×2, action-rate×0.25, metric weight10,
new auxiliary reward weight32, uniform sampling, seed47, actorLR3e-5, entropy.0005.

Auxiliary reward in `adaptation_rewards.py` normalizes all8 other evaluation
errors (anchorpos/orientation, bodyorientation, jointvelocity, anchorlin/angvel,
bodylin/angvel), averages exponentials, scales
(0.25,0.075,0.125,4.4,0.21,0.45,0.267,1.07). CLI flag
`--auxiliary-tracking-weight` defaults0. Smoke completed successfully, tests
cover qpos-only/full-reference velocity conventions. **No physics changed.**

GPU1 metric teacher PID14830 continues: eval500 **0.028599m /0.444436rad**, but
anchor worsened to0.371119m. This tradeoff is why balanced training is required;
do not finish merely because body/joint now beat nominal. GPU3 oracle-feature
teacher PID743 continues: eval750 **0.032791m/0.555117rad**, anchor0.256396m,
failure0.5952%; improves more broadly and may remain useful for another balanced
variant. GPU2 context distillation PID37613 is around2400/3000 updates; since
1500 teacher mix is zero. Development evaluations through1500 stay around
0.0298m/0.476rad, so more pure imitation alone is not improving task metrics.
Consider student PPO with `--initialize-from` after distillation finishes, using
the balanced reward and deployable-only actor. Resume architecture loading has
already been fixed; initialize-from copies actor/critic, fresh optimizer.

Current live training allocation: GPU0 balanced teacher52800, GPU1 metric
teacher14830, GPU2 student distillation37613, GPU3 oracle-feature teacher743.
Student watcher PID40273/session24612. Nominal300xx evaluations all completed.
Goal still incomplete. Continue improving broader tracking and robust student
confirmation. No tracked pre-existing code has been changed; all research code
is new/untracked. All outside-repo code remains untouched.

## Latest override: 17:03 UTC

GPU2 distillation completed all3000 updates normally (session74357 exit0).
Its watcher40273/session24612 finished all evals with zero errors. Final student
dev:0.029839m/0.476181rad/0.893%fail, anchor0.361250m; imitation alone plateaued.
GPU2 now **context_balanced_ppo_seed48**, PID**64756**, session**92029**, watcher
PID**3396**/session**87743**, log `context_balanced_evaluation_watch.log`. It
initializes actor/critic from distillation FINAL with a fresh PPO optimizer,
4096envs2000updates, same balanced objective as GPU0 (tracking2, actionrate.25,
metric10, auxiliary32), seed48, actorLR2e-5, entropy.0005. Actor remains a
ContextAdaptationActor; wrapper adds no physics/state privilege keys. Privileged
critic training is allowed, but student actions retain strict deployable path.
Context action-shuffle/zero diagnostics were added to ContextAdaptationActor's
policy_metrics; early PPO context-shuffle action RMS around0.02.

GPU1 former primary-only teacher PID14830 was stopped after evaluated750:
body0.028748/joint0.436543 but anchor0.388472 and failure1.786%, beyond guard.
Confirmed250 and500 checkpoints remain. GPU1 now **balanced_low_noise_seed47**,
PID**7844**, session**65372**, new watcher session from latest tool result
(log `balanced_low_noise_evaluation_watch.log`). Same initialization (metric500),
reward, seed47 and4096envs as GPU0 balanced teacher, except new CLI
`--initial-action-std 0.1`. This changes **policy exploration noise only**, NOT
sensor noise or DR. Distribution std remains trainable. Default flag is None,
so older experiments unchanged. Actual scalar/log std parameter override is
after checkpoint loading and before first rollout; validation rejects <=0.

GPU0 **balanced_metrics_seed47** remains PID52800/session74003, watcher
PID55788/session90721. Eval250:body0.029142/joint0.451884/anchor0.320440,
failure1.488% (slightly above guard), angular/velocity errors still exceed
baseline. Keep training to see whether stability/broader errors improve.
GPU3 **oracle_features_tracking4_seed44** PID743/session86544 continues
past1000; its eval1000 was finishing at this update. Keep this branch: it has
best broad tracking so far, despite worse joint error than the primary-only run.

New `scripts/summarize_adaptation.py` writes `development_scoreboard.json` and
ranks DR candidates by worst ratio across ALL10 metrics, failure guard first.
It enforces matched protocol/seed/motion IDs/starts/horizon/files and excludes
ablation and smoke evaluations. It reports both primary and all-metric point
criteria, never calls development a final confirmation. Original baseline remains
the old nominal checkpoint, not the weaker newly trained nominal control.

Probe unit test passed (8 adaptation tests; 13 residual tests passed earlier).
No model/weights in the physical probe were changed; only a diagnostic ridge
regressor was fitted with worlds held out. Current source has no export artifact:
inherited as_jit/as_onnx correctly raises NotImplementedError. Strict deployed-
observation evaluation works through the Python actor, but do not claim a
standalone hardware deployment package yet. Export can be added for final policy.

Current jobs: GPU0 balanced52800; GPU1 low-noise balanced7844;
GPU2 deployable balanced PPO64756; GPU3 oracle-feature teacher743.
Still pursue full goal and robust evidence; don't stop merely because primary
means passed. User was clearly told of broader tracking regressions and small
context ablation effect, along with the positive held-out payload R² probe.
