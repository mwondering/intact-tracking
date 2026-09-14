"""Train privileged DR policies and matched nominal/nonprivileged controls."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import OmegaConf

from intact_tracking.adaptation_correction import (
    CORRECTION_PHYSICS,
    correction_config_requires_privilege,
    correction_initialization_state,
)
from intact_tracking.adaptation_critic import CRITIC_PHYSICS, PhysicsCriticWrapper
from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.adaptation_policy import (
    ORACLE_HISTORY,
    ORACLE_KEY_BODY,
    PRIVILEGE,
    PrivilegedAdaptationWrapper,
    configure_oracle_proprioception,
)
from intact_tracking.adaptation_reward_contract import (
    assert_fixed_reward_checkpoint,
    assert_original_reward_arguments,
    assert_rewards_unchanged,
    capture_original_rewards,
)
from intact_tracking.cli.adaptation_eval import DATASET, TRACKER, configure_physics, load_actor
from intact_tracking.cli.residual_policy_train import (
    _audit_nominal_runtime,
    _build_train_configuration,
    _checkpoint_configuration,
    _prepare_output,
    _seed_everything,
)
from intact_tracking.environment.runtime import _load_saved_config, prepare_rollout
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.rollout.mjlab_adapter import _sha256


def run(args):
    assert_original_reward_arguments(args)
    preview_mode = getattr(args, "simulator_preview", None)
    if preview_mode:
        raise ValueError(
            "The v1 preview training setup is retired. Use python -m "
            "intact_tracking.cli.simulator_preview_train --variant preview|baseline "
            "--num-envs 4096 --iterations YOUR_UPDATES --output-dir NEW_DIRECTORY. "
            "V2 uses original inputs, no scalar distances, and a scratch single-path critic."
        )
    preview_critic = bool(preview_mode and getattr(args, "simulator_preview_critic", True))
    motion_file = getattr(args, "motion_file", None)
    if preview_mode and (
        not motion_file or not args.privileged or args.privilege_schema != "state_physics"
        or args.initialize_from or args.resume or args.oracle_tracking_features
        or args.oracle_clean_proprio or args.train_base_policy or args.physics_critic
        or args.frozen_nominal_prior or args.imitation_teacher
    ):
        raise ValueError("Preview requires a fresh single-motion state_physics actor with unchanged tracker; use its explicit preview-critic option")
    path = Path(args.tracker_checkpoint).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output must be new or empty: {output}")
    _seed_everything(args.seed)
    prepared = prepare_rollout(
        checkpoint_file=str(path),
        num_envs=args.num_envs,
        motion_path=None if motion_file else args.motion_path,
        motion_file=motion_file,
    )
    source = _load_saved_config(path)
    reward_contract = capture_original_rewards(prepared.env)
    physics = configure_physics(prepared.env, args.physics)
    prepared.env.seed = args.seed
    training_starts = configure_training_starts(prepared.env, args.training_start_mode)
    if args.sampling_mode != "original":
        prepared.env.commands["motion"].sampling_mode = args.sampling_mode
        prepared.env.commands["motion"].rewind.enabled = False
    reward_changes = {}
    train = _build_train_configuration(
        source,
        tracker_checkpoint=path,
        tracker_actor_kwargs=prepared.actor_kwargs,
        tracker_obs_groups=prepared.obs_groups,
        baseline="no-latent",
        dynamics_latent_dim=0,
        residual_hidden_dims=tuple(args.hidden_dims),
        residual_scale=args.residual_scale,
        iterations=args.iterations,
        num_steps_per_env=args.rollout_steps,
        save_interval=args.save_interval,
        seed=args.seed,
        logger="tensorboard",
        wandb_project="intact-adaptation",
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        check_for_nan=True,
    )
    train["algorithm"]["entropy_coef"] = args.entropy_coef
    if preview_critic:
        train["critic"]["class_name"] = "intact_tracking.simulator_preview:PreviewAwareHeftCritic"
        train["obs_groups"]["critic"].append(PRIVILEGE)
    train["algorithm"]["num_learning_epochs"] = args.epochs
    if args.actor_lr_schedule is not None:
        train["algorithm"]["schedule"] = args.actor_lr_schedule
    if args.critic_warmup_updates:
        train["algorithm"].update(
            class_name="intact_tracking.adaptation_ppo:CriticWarmupPPO",
            critic_warmup_updates=args.critic_warmup_updates,
        )
    initialization = None
    checkpoint_source = args.initialize_from or args.resume
    if args.frozen_nominal_prior_deployable_base and (
        checkpoint_source or not args.frozen_nominal_prior or not args.privileged
    ):
        raise ValueError("Explicit separated prior is a fresh privileged-actor architecture; initialized actors retain their saved contract")
    if checkpoint_source:
        initialization = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
        assert_fixed_reward_checkpoint(initialization, reward_contract)
        train["actor"] = OmegaConf.to_container(initialization["cfg"].agent.actor, resolve=True)
        is_privileged = train["actor"]["class_name"].endswith(":PrivilegedAdaptationActor")
        privilege_schema = train["actor"].get("privilege_schema", "critic")
    else:
        is_privileged = args.privileged
        privilege_schema = args.privilege_schema
    if args.add_teacher_refinement:
        if (
            not args.initialize_from or args.resume or not is_privileged
            or train["actor"].get("refinement_scale", 0.0)
            or train["actor"].get("train_base_policy", False)
            or args.train_base_policy
            or args.oracle_feature_curriculum_updates
            or args.imitation_teacher
        ):
            raise ValueError("Adding refinement requires an unmodified signed frozen-core teacher and a fresh optimizer")
        train["actor"].update(refinement_scale=args.refinement_scale, freeze_teacher_trunk=True)
    source_physics_critic = initialization is not None and str(
        initialization["cfg"].agent.critic.class_name
    ).endswith(":PhysicsAwareHeftCritic")
    physics_critic = args.physics_critic or source_physics_critic
    if args.physics_critic and initialization is not None and not source_physics_critic:
        raise ValueError("Adding the physics critic currently requires a fresh run; optimizer/state expansion is not implicit")
    if physics_critic:
        if source_physics_critic:
            train["critic"] = OmegaConf.to_container(initialization["cfg"].agent.critic, resolve=True)
        else:
            train["critic"]["class_name"] = "intact_tracking.adaptation_critic:PhysicsAwareHeftCritic"
        train["obs_groups"]["critic"].append(CRITIC_PHYSICS)
    if args.add_static_correction:
        if initialization is None or not train["actor"]["class_name"].endswith(":ContextAdaptationActor"):
            raise ValueError("Adding a static correction requires an ordinary initialized context policy")
        train["actor"].update(
            class_name="intact_tracking.adaptation_correction:ContextPhysicsCorrectionActor",
            correction_use_privilege=True,
            correction_scale=args.correction_scale,
            train_base_policy=False, train_context_encoder=False,
            train_residual_policy=False, train_reference_encoder=False,
        )
    is_correction = train["actor"]["class_name"].endswith(":ContextPhysicsCorrectionActor")
    if args.correction_use_privilege is not None:
        if not is_correction:
            raise ValueError("Correction privilege override requires a correction actor")
        train["actor"]["correction_use_privilege"] = args.correction_use_privilege
    correction_privileged = correction_config_requires_privilege(train["actor"])
    is_privileged = is_privileged or correction_privileged
    if is_correction and (args.imitation_teacher or args.context_aux_steps or args.oracle_feature_curriculum_updates):
        raise ValueError("Static correction cannot combine with teacher/auxiliary/curriculum algorithms")
    if args.train_base_policy is not None and checkpoint_source:
        if not train["actor"]["class_name"].endswith((":PrivilegedAdaptationActor", ":ContextAdaptationActor", ":ContextPhysicsCorrectionActor")):
            raise ValueError("Base-training override requires an adaptation actor")
        train["actor"]["train_base_policy"] = args.train_base_policy
    if args.train_context_encoder is not None:
        if not train["actor"]["class_name"].endswith((":ContextAdaptationActor", ":ContextPhysicsCorrectionActor")):
            raise ValueError("Context-training override requires an initialized context actor")
        train["actor"]["train_context_encoder"] = args.train_context_encoder
    if args.oracle_tracking_features is not None:
        if not is_privileged:
            raise ValueError("Oracle tracking-feature overrides require a privileged actor")
        if initialization is not None:
            train["actor"]["oracle_tracking_features"] = args.oracle_tracking_features
    clean_proprio = train["actor"].get("oracle_clean_proprio", False)
    if args.oracle_clean_proprio is not None:
        if not is_privileged:
            raise ValueError("Oracle proprioception is forbidden for a deployable student")
        clean_proprio = args.oracle_clean_proprio
        if initialization is not None:
            train["actor"]["oracle_clean_proprio"] = clean_proprio
    if clean_proprio:
        configure_oracle_proprioception(prepared.env)
    if args.oracle_feature_curriculum_updates:
        if not is_privileged or initialization is None or not train["actor"].get("oracle_tracking_features", False):
            raise ValueError("Observation curriculum requires an initialized oracle-feature teacher")
        train["actor"]["oracle_feature_curriculum"] = True
        train["algorithm"].update(
            class_name="intact_tracking.adaptation_ppo:OracleFeatureCurriculumPPO",
            critic_warmup_updates=args.critic_warmup_updates,
            oracle_feature_curriculum_updates=args.oracle_feature_curriculum_updates,
        )
    if train["actor"].get("context_imu_accel", False):
        from intact_tracking.adaptation_sensors import configure_context_sensors

        configure_context_sensors(prepared.env)
    supervision_schema = None
    if args.imitation_teacher:
        from intact_tracking.adaptation_ppo import TeacherSupervisionWrapper

        if not train["actor"]["class_name"].endswith(":ContextAdaptationActor"):
            raise ValueError("Teacher regularization requires an initialized context student")
        teacher_checkpoint = torch.load(
            args.imitation_teacher, map_location="cpu", weights_only=False
        )
        assert_fixed_reward_checkpoint(teacher_checkpoint, reward_contract)
        teacher_cfg = teacher_checkpoint["cfg"].agent.actor
        supervision_schema = teacher_cfg.get("privilege_schema", "critic")
        if teacher_cfg.get("oracle_clean_proprio", False):
            configure_oracle_proprioception(prepared.env)
        train["algorithm"].update(
            class_name="intact_tracking.adaptation_ppo:TeacherRegularizedPPO",
            teacher_bc_steps=args.teacher_bc_steps,
            teacher_bc_weight=args.teacher_bc_weight,
            teacher_bc_batch_size=args.teacher_bc_batch_size,
        )
    if args.context_aux_steps:
        if (
            is_privileged or not checkpoint_source or args.imitation_teacher
            or args.oracle_feature_curriculum_updates
            or train["actor"].get("context_latent_mean", False)
            or not train["actor"].get("train_context_encoder", True)
            or train["actor"].get("context_auxiliary_dim", 0) < 10
        ):
            raise ValueError("Auxiliary PPO requires an initialized trainable non-memory context actor with a physics readout")
        train["algorithm"].update(
            class_name="intact_tracking.adaptation_ppo:ContextAuxiliaryPPO",
            critic_warmup_updates=args.critic_warmup_updates,
            context_aux_steps=args.context_aux_steps,
            context_aux_batch_size=args.context_aux_batch_size,
            context_aux_weight=args.context_aux_weight,
            context_physics_weight=args.context_physics_weight,
        )
    assert_rewards_unchanged(reward_contract, prepared.env)
    env = ManagerBasedRlEnv(cfg=copy.deepcopy(prepared.env), device=args.device)
    preview_wrapper = None
    try:
        if args.physics == "nominal":
            physics["runtime_audit"] = _audit_nominal_runtime(env)
        if correction_privileged:
            from intact_tracking.adaptation_correction import StaticPhysicsCorrectionWrapper

            wrapped = StaticPhysicsCorrectionWrapper(RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions))
        elif args.imitation_teacher:
            wrapped = TeacherSupervisionWrapper(
                env, clip_actions=prepared.clip_actions, privilege_schema=supervision_schema
            )
        elif preview_mode:
            from intact_tracking.simulator_preview import SimulatorPreviewWrapper

            wrapped = preview_wrapper = SimulatorPreviewWrapper(
                env, clip_actions=prepared.clip_actions, preview_mode=preview_mode,
                privilege_schema=privilege_schema,
            )
        elif is_privileged:
            wrapped = PrivilegedAdaptationWrapper(
                env, clip_actions=prepared.clip_actions, privilege_schema=privilege_schema
            )
        else:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions)
        if physics_critic:
            wrapped = PhysicsCriticWrapper(wrapped)
        if args.context_aux_steps:
            from intact_tracking.adaptation_ppo import ContextSupervisionWrapper

            wrapped = ContextSupervisionWrapper(wrapped)
        if train["actor"].get("context_latent_mean", False):
            from intact_tracking.adaptation_context_memory import CausalContextMeanWrapper

            wrapped = CausalContextMeanWrapper(
                wrapped,
                train["actor"].get("adaptation_latent_dim", 64),
                train["actor"].get("context_mean_horizon", 250),
            )
        if is_privileged and initialization is None:
            train["actor"].update(
                {
                    "class_name": "intact_tracking.adaptation_policy:PrivilegedAdaptationActor",
                    "privilege_dim": int(wrapped.get_observations()[PRIVILEGE].shape[-1]),
                    "adaptation_latent_dim": args.latent_dim,
                    "privilege_schema": privilege_schema,
                    "actor_physics_input": args.actor_physics_input,
                    "train_base_policy": bool(args.train_base_policy),
                    "oracle_tracking_features": bool(args.oracle_tracking_features),
                    "oracle_clean_proprio": clean_proprio,
                    "latent_modulation": args.latent_modulation,
                    "physics_bilinear": args.physics_bilinear,
                    "physics_low_rank": args.physics_low_rank,
                    "latent_low_rank": args.latent_low_rank,
                    "rank_per_physics": args.rank_per_physics,
                    "shared_low_rank": args.shared_low_rank,
                    "frozen_nominal_prior": str(Path(args.frozen_nominal_prior).resolve()) if args.frozen_nominal_prior else None,
                    "frozen_nominal_prior_deployable_base": args.frozen_nominal_prior_deployable_base,
                }
            )
        metadata = {
            "version": "adaptation_teacher_v1",
            "physics": physics,
            "privileged": is_privileged,
            "oracle_clean_proprio": clean_proprio,
            "tracker_checkpoint": str(path),
            "tracker_sha256": _sha256(path),
            "motion_path": str(Path(motion_file or args.motion_path).resolve()),
            "motion_file": motion_file,
            "simulator_preview": preview_mode,
            "simulator_preview_horizon": 5 if preview_mode else None,
            "simulator_preview_critic": preview_critic,
            "simulator_preview_critic_contract": (
                "Actor and critic read the identical stored adaptation_privilege tensor; independent parameters; masked outcomes hidden from both; original PPO returns"
                if preview_critic else "Legacy actor-only preview ablation; original critic"
            ) if preview_mode else None,
            "arguments": vars(args),
            "motion_count": env.command_manager.get_term("motion").motion.num_files,
            "student_contract": "replace all privileged actor inputs by deployable context and observations",
            "initialization_sha256": _sha256(Path(checkpoint_source)) if checkpoint_source else None,
            "static_correction_contract": "frozen deployable source plus bounded learned correction; teacher uses only centered payload and normalized torso COMx/y; student uses copied context-head predictions only",
            "context_auxiliary_ppo_contract": "optional loss-only body velocity/payload/static-physics labels; PPO followed by encoder/readout-only supervised steps on identical rollout; disabled throughout critic warmup",
            "imitation_teacher_sha256": (
                _sha256(Path(args.imitation_teacher)) if args.imitation_teacher else None
            ),
            "imitation_contract": "teacher action is a training loss label only; alternating PPO and BC",
            "reward_changes": reward_changes,
            "reward_contract": reward_contract,
            "physics_critic": physics_critic,
            "actor_physics_input": train["actor"].get("actor_physics_input", "not_applicable"),
            "physics_critic_contract": "separate actual-physics critic group; zero-initialized additive value branch; unchanged PPO returns and actor inputs",
            "fixed_reward_lineage": {
                "source": "audited fixed-reward initialization" if checkpoint_source else "original common pretrained tracker",
                "original_tracker_pretraining_dr": True,
                "historical_shaped_checkpoint_used": False,
            },
            "training_starts": training_starts,
            "research_source_sha256": {
                str(p.relative_to(Path.cwd())): _sha256(p)
                for p in (
                    Path(__file__).resolve(),
                    Path(__file__).resolve().parents[1] / "adaptation_policy.py",
                    Path(__file__).resolve().parents[1] / "adaptation_rewards.py",
                    Path(__file__).resolve().parents[1] / "adaptation_reward_contract.py",
                    Path(__file__).resolve().parent / "adaptation_eval.py",
                    Path(__file__).resolve().parents[1] / "adaptation_ppo.py",
                    Path(__file__).resolve().parents[1] / "adaptation_sensors.py",
                    Path(__file__).resolve().parents[1] / "adaptation_curriculum.py",
                    Path(__file__).resolve().parents[1] / "adaptation_context_memory.py",
                    Path(__file__).resolve().parents[1] / "adaptation_correction.py",
                    Path(__file__).resolve().parents[1] / "adaptation_modulation.py",
                    Path(__file__).resolve().parents[1] / "adaptation_prior.py",
                    Path(__file__).resolve().parents[1] / "adaptation_critic.py",
                    Path(__file__).resolve().parents[1] / "simulator_preview.py",
                )
            },
        }
        cfg = _checkpoint_configuration(source, train, metadata)
        _prepare_output(output, resume=args.resume, run_config=metadata, checkpoint_config=cfg)
        if args.training_rng_seed is not None:
            # Separate model/rollout randomness from different startup-event
            # consumption in nominal versus DR environments.
            _seed_everything(args.training_rng_seed)
        runner = ResidualOnPolicyRunner(
            wrapped, train, str(output), args.device, checkpoint_cfg=cfg, residual_metadata=metadata
        )
        if preview_wrapper is not None:
            preview_wrapper.bind(runner.alg.actor)
        physics_critic_initial_difference = None
        if (physics_critic or preview_critic) and checkpoint_source is None:
            from intact_tracking.residual_policy import WarmStartedHeftCritic

            critic_observations = wrapped.get_observations()
            with torch.no_grad():
                actual = runner.alg.critic(critic_observations)
                expected = WarmStartedHeftCritic.forward(runner.alg.critic, critic_observations)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                physics_critic_initial_difference = float((actual - expected).abs().max())
        prior_audit = getattr(runner.alg.actor, "frozen_nominal_prior_audit", None)
        if prior_audit is not None:
            # Keep the pre-construction run config's arguments, and place the
            # executed exact-SHA/tensor audit in every saved checkpoint as well.
            metadata["frozen_nominal_prior_audit"] = prior_audit
            runner.residual_metadata["frozen_nominal_prior_audit"] = prior_audit
            metadata["fixed_reward_lineage"]["frozen_nominal_prior_sha256"] = prior_audit["sha256"]
            metadata["fixed_reward_lineage"]["source"] = "original tracker plus an exact-SHA audited original-reward nominal prior"
            cfg.residual_policy.frozen_nominal_prior_audit = OmegaConf.create(prior_audit)
            cfg.residual_policy.fixed_reward_lineage = OmegaConf.create(metadata["fixed_reward_lineage"])
            (output / "run_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            OmegaConf.save(cfg, output / "config.yaml")
        if args.initialize_from:
            initial_actor_state = dict(initialization["actor_state_dict"])
            if args.add_teacher_refinement:
                new_state = runner.alg.actor.state_dict()
                added = set(new_state) - set(initial_actor_state)
                if not added or any(not key.startswith("refinement_mlp.") for key in added):
                    raise ValueError("Refinement initialization may add ONLY its zero-output MLP")
                initial_actor_state.update({key: new_state[key].clone() for key in added})
            if is_correction:
                initial_actor_state = correction_initialization_state(runner.alg.actor, initial_actor_state)
            if args.oracle_feature_curriculum_updates:
                initial_actor_state.setdefault(
                    "oracle_feature_mix", runner.alg.actor.oracle_feature_mix.clone()
                )
            runner.alg.actor.load_state_dict(initial_actor_state, strict=True)
            runner.alg.critic.load_state_dict(initialization["critic_state_dict"], strict=True)
        initial_action_audit = None
        bilinear_audit = None
        if args.add_teacher_refinement:
            observations = wrapped.get_observations()
            source_actor = load_actor(args.initialize_from, prepared, observations.clone(recurse=True), wrapped).eval()
            with torch.no_grad():
                expected = source_actor(observations.clone(recurse=True))
                actual = runner.alg.actor(observations.clone(recurse=True))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                initial_action_audit = float((actual - expected).abs().max())
            del source_actor
        if is_privileged and (
            getattr(runner.alg.actor, "physics_bilinear", False)
            or getattr(runner.alg.actor, "physics_low_rank", False)
            or getattr(runner.alg.actor, "latent_low_rank", False)
            or (prior_audit is not None and checkpoint_source is None)
        ):
            observations = wrapped.get_observations()
            nominal_observations = observations.clone(recurse=False)
            nominal_observations.set(PRIVILEGE, torch.zeros_like(observations[PRIVILEGE]))
            with torch.no_grad():
                base_action = runner.alg.actor._base_features_and_action(observations)[1]
                nominal_action = runner.alg.actor(nominal_observations)
                physical_input_mode = getattr(runner.alg.actor, "actor_physics_input", "normal")
                nominal_identity_expected = (
                    physical_input_mode == "normal" and not getattr(runner.alg.actor, "shared_low_rank", 0)
                ) or checkpoint_source is None
                if nominal_identity_expected:
                    torch.testing.assert_close(nominal_action, base_action, atol=0, rtol=0)
                bilinear_audit = {
                    "nominal_code_action_max_difference": float((nominal_action - base_action).abs().max()),
                    "nominal_identity_asserted": nominal_identity_expected,
                    "actor_physics_input": physical_input_mode,
                    "baseline_frontend": "privileged oracle tracking features" if getattr(runner.alg.actor, "oracle_tracking_features", False) else "unchanged deployable tracking features",
                }
                if prior_audit is not None and nominal_identity_expected:
                    # Actor construction populates preprocessing caches in its
                    # input TensorDict. Audit construction must not replace the
                    # actual training observation's already captured caches.
                    source_actor = load_actor(prior_audit["checkpoint"], prepared, observations.clone(recurse=True), wrapped).eval()
                    differences = []
                    uncached_groups = list(source_actor.obs_groups) + [PRIVILEGE]
                    if runner.alg.actor.oracle_tracking_features:
                        model = runner.alg.actor.tracker
                        uncached_groups += [
                            model.estimator_target_group, model.foot_contact_target_group,
                            model.reference_encoder_target_group,
                        ]
                        if runner.alg.actor.oracle_clean_proprio:
                            uncached_groups += [ORACLE_HISTORY, ORACLE_KEY_BODY]
                    uncached = nominal_observations.select(*dict.fromkeys(uncached_groups))
                    for shared in (uncached, nominal_observations):
                        expected = source_actor(shared.clone(recurse=True))
                        actual = runner.alg.actor(shared.clone(recurse=True))
                        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
                        differences.append(float((actual - expected).abs().max()))
                    bilinear_audit["frozen_nominal_prior_source_action_max_difference"] = max(differences)
                    del source_actor
                if checkpoint_source is None:
                    initial_action = runner.alg.actor(observations)
                    torch.testing.assert_close(initial_action, base_action, atol=0, rtol=0)
                    bilinear_audit["initial_action_max_difference"] = float((initial_action - base_action).abs().max())
        if args.add_static_correction:
            observations = wrapped.get_observations()
            source_actor = load_actor(checkpoint_source, prepared, observations, wrapped).eval()
            with torch.no_grad():
                # Compare like-for-like input paths. A cached reference FK and
                # a freshly reconstructed reference can round differently.
                # Both cached training and uncached deployment must preserve
                # the source; do not compare one against the other.
                differences = []
                uncached = observations.select(*source_actor.deployable_observation_groups, CORRECTION_PHYSICS)
                for shared_observations in (uncached, observations):
                    reference_action = source_actor(shared_observations)
                    corrected_action = runner.alg.actor(shared_observations)
                    torch.testing.assert_close(corrected_action, reference_action, atol=2e-4, rtol=2e-4)
                    differences.append(float((corrected_action - reference_action).abs().max()))
                initial_action_audit = max(differences)
            del source_actor
        del initialization
        if args.resume:
            runner.load(args.resume, map_location=args.device)
        if getattr(runner.alg.actor, "context_latent_mean", False):
            wrapped.bind(runner.alg.actor)
        if args.imitation_teacher:
            runner.alg.teacher = load_actor(
                args.imitation_teacher, prepared, wrapped.get_observations(), wrapped
            ).eval().requires_grad_(False)
        if args.initial_action_std is not None:
            distribution = runner.alg.actor.distribution
            value = args.initial_action_std
            if distribution.std_type == "log":
                value = math.log(value)
            elif distribution.std_type != "scalar":
                raise ValueError("Action-noise override requires scalar/log Gaussian std")
            with torch.no_grad():
                distribution.std_param.fill_(value)
        if args.training_rng_seed is not None:
            _seed_everything(args.training_rng_seed + 1)
        if args.save_initial:
            runner.save(str(output / "checkpoint_initial.pt"), infos={
                "static_correction_initial_action_max_difference": initial_action_audit,
                "teacher_refinement_initial_action_max_difference": initial_action_audit if args.add_teacher_refinement else None,
                "bilinear_function_audit": bilinear_audit,
                "frozen_nominal_prior_audit": prior_audit,
                "physics_critic_initial_value_max_difference": physics_critic_initial_difference,
                "preview_critic_initial_value_max_difference": physics_critic_initial_difference if preview_critic else None,
            })
        runner.learn(args.iterations, init_at_random_ep_len=True)
    finally:
        if preview_wrapper is not None:
            preview_wrapper.close_preview()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-checkpoint", default=TRACKER)
    parser.add_argument("--motion-path", default=DATASET)
    parser.add_argument("--motion-file", help="Use exactly one motion instead of motion-path")
    parser.add_argument("--simulator-preview", choices=("true", "zero"), help="Five-step simulator outcome information control; original rewards unchanged")
    parser.add_argument("--simulator-preview-critic", action=argparse.BooleanOptionalAction, default=True,
                        help="Share actor preview information with an independent value branch (default); disable only for actor-only ablation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--physics", choices=("nominal", "dr"), required=True)
    parser.add_argument("--privileged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-rng-seed", type=int, help="Paired-control model construction seed; rollout is reseeded to this value plus one")
    parser.add_argument("--save-initial", action="store_true", help="Save the exact pre-update actor/critic for paired initialization audits")
    parser.add_argument("--add-teacher-refinement", action="store_true", help="Freeze an audited initialized teacher including its normalization; add a zero-output trainable action correction")
    parser.add_argument("--refinement-scale", type=float, default=1.0)
    parser.add_argument("--frozen-nominal-prior-deployable-base", action="store_true", help="Keep the exact nominal action branch on original deployable inputs; oracle/context-adapted features feed only the new residual")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(512, 256, 128))
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--latent-modulation", action="store_true", help="Condition each residual hidden layer on the same replaceable latent; original reward remains locked")
    parser.add_argument("--physics-bilinear", action="store_true", help="Feature-dependent corrections bilinear in seven nominal-centered physical coordinates; same original reward")
    parser.add_argument("--physics-low-rank", action="store_true", help="Physical code modulates low-rank adapters throughout the frozen controller; same original reward")
    parser.add_argument("--latent-low-rank", action="store_true", help="Learn nominal-centered nonlinear full-physics codes for low-rank controller adapters; no privileged current-state inputs")
    parser.add_argument("--shared-low-rank", type=int, default=0, help="Additional state-only low-rank update per frozen layer; zero initialized, but after learning nominal physics no longer guarantees zero residual")
    parser.add_argument("--rank-per-physics", type=int, default=2)
    parser.add_argument("--frozen-nominal-prior", help="Only the exact provenance-audited original-reward nominal baseline may be added as a frozen prior")
    parser.add_argument("--oracle-feature-curriculum-updates", type=int, default=0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr-schedule", choices=("fixed", "adaptive"))
    parser.add_argument("--critic-warmup-updates", type=int, default=0)
    parser.add_argument("--add-static-correction", action="store_true")
    parser.add_argument("--correction-scale", type=float, default=0.25)
    parser.add_argument("--correction-use-privilege", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--context-aux-steps", type=int, default=0)
    parser.add_argument("--context-aux-batch-size", type=int, default=2048)
    parser.add_argument("--context-aux-weight", type=float, default=0.05)
    parser.add_argument("--context-physics-weight", type=float, default=0.01)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--physics-critic", action="store_true", help="Add actual physical inputs to a zero-initialized value branch; actor/reward contracts unchanged")
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--initial-action-std", type=float)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--resume")
    parser.add_argument(
        "--initialize-from", help="Initialize actor/critic with a fresh PPO optimizer"
    )
    parser.add_argument("--tracking-multiplier", type=float, default=1.0)
    parser.add_argument("--action-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=0.0)
    parser.add_argument("--metric-tracking-weight", type=float, default=0.0)
    parser.add_argument("--metric-reward-shape", choices=("exp", "linear"), default="exp")
    parser.add_argument("--auxiliary-reward-shape", choices=("exp", "linear"), default="exp")
    parser.add_argument("--auxiliary-tracking-weight", type=float, default=0.0)
    parser.add_argument(
        "--auxiliary-metric-weights", type=float, nargs=8,
        help="Training reward weights: anchor position/rotation, body rotation, joint velocity, anchor linear/angular velocity, body linear/angular velocity",
    )
    parser.add_argument("--metric-reference-offset", type=int, choices=(0, 1), default=0)
    parser.add_argument("--right-arm-angular-weight", type=float, default=0.0)
    parser.add_argument("--right-arm-rotation-weight", type=float, default=0.0)
    parser.add_argument("--imitation-teacher")
    parser.add_argument("--teacher-bc-steps", type=int, default=2)
    parser.add_argument("--teacher-bc-weight", type=float, default=0.1)
    parser.add_argument("--teacher-bc-batch-size", type=int, default=2048)
    parser.add_argument("--sampling-mode", choices=("original", "uniform"), default="original")
    parser.add_argument("--training-start-mode", choices=("original", "reference"), default="original")
    parser.add_argument("--privilege-schema", choices=("critic", "state_physics", "compact_physics", "actuator_physics", "physics"), default="critic")
    parser.add_argument("--actor-physics-input", choices=("normal", "zero", "constant"), default="normal", help="Matched actor-information control before encoder and normalizer: zero for full-physics residual; constant ones for compact low-rank/bilinear so every rank stays active. Critic unchanged.")
    parser.add_argument("--train-base-policy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-context-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--oracle-tracking-features", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--oracle-clean-proprio", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args()
    if not math.isfinite(args.refinement_scale) or args.refinement_scale <= 0:
        parser.error("Refinement scale must be finite and positive")
    if not math.isfinite(args.failure_penalty) or args.failure_penalty < 0:
        parser.error("Failure penalty must be finite and nonnegative")
    if args.resume and args.initialize_from:
        parser.error("resume and initialize-from are mutually exclusive")
    if not math.isfinite(args.correction_scale) or args.correction_scale <= 0:
        parser.error("Correction scale must be finite and positive")
    if args.oracle_feature_curriculum_updates < 0 or (
        args.oracle_feature_curriculum_updates and args.imitation_teacher
    ):
        parser.error("Oracle-feature curriculum must be nonnegative and cannot combine with teacher BC")
    if args.critic_warmup_updates < 0 or (args.critic_warmup_updates and args.imitation_teacher):
        parser.error("Critic warmup must be nonnegative and cannot combine with teacher BC")
    if args.initial_action_std is not None and args.initial_action_std <= 0:
        parser.error("initial-action-std must be positive")
    if not math.isfinite(args.right_arm_rotation_weight) or args.right_arm_rotation_weight < 0:
        parser.error("Right-arm rotation weight must be finite and nonnegative")
    if (
        args.context_aux_steps < 0 or args.context_aux_batch_size < 1
        or not math.isfinite(args.context_aux_weight) or args.context_aux_weight < 0
        or not math.isfinite(args.context_physics_weight) or args.context_physics_weight < 0
        or (args.context_aux_steps and args.context_aux_weight + args.context_physics_weight <= 0)
    ):
        parser.error("Context auxiliary PPO requires nonnegative finite weights/steps, positive batch size and nonzero enabled loss")
    if args.auxiliary_metric_weights is not None and (
        any(not math.isfinite(weight) or weight < 0 for weight in args.auxiliary_metric_weights)
        or sum(args.auxiliary_metric_weights) <= 0
        or args.auxiliary_tracking_weight <= 0
    ):
        parser.error("Auxiliary metric weights require finite nonnegative values, positive sum and auxiliary reward")
    if args.teacher_bc_steps < 1 or args.teacher_bc_weight <= 0 or args.teacher_bc_batch_size < 1:
        parser.error("Teacher BC steps, weight and batch size must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    run(args)


if __name__ == "__main__":
    main()
