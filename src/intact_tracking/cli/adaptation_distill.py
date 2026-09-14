"""Replace teacher privileges by deployable history using on-policy action distillation.

Training with an unqualified teacher is preparation only, never stage-two
acceptance. A privileged critic may remain in the checkpoint for subsequent PPO,
but the saved actor contains no privileged encoder or privileged normalizer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
from omegaconf import OmegaConf
from tensordict import TensorDict

from intact_tracking.adaptation_context_memory import CONTEXT_MEAN, CausalContextMeanWrapper
from intact_tracking.adaptation_policy import (
    ContextAdaptationActor,
    PrivilegedAdaptationWrapper,
    _base_with_optional_update,
    configure_oracle_proprioception,
    expanded_model_field,
    normalized_context_kinematics,
)
from intact_tracking.adaptation_sensors import configure_context_sensors
from intact_tracking.cli.adaptation_eval import DATASET, TRACKER, configure_physics, load_actor
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.rollout.mjlab_adapter import _sha256


class DistillationReplay:
    """GPU replay of causal inputs and simultaneous teacher action targets."""

    def __init__(self, capacity, example):
        self.capacity = capacity
        self.data = {
            k: torch.empty((capacity, *v.shape[1:]), device=v.device, dtype=v.dtype)
            for k, v in example.items()
        }
        self.pointer = self.size = 0

    @torch.no_grad()
    def add(self, batch):
        n = min(len(next(iter(batch.values()))), self.capacity)
        ids = (
            torch.arange(n, device=next(iter(batch.values())).device) + self.pointer
        ) % self.capacity
        for key, value in batch.items():
            self.data[key][ids] = value[-n:]
        self.pointer = (self.pointer + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, size):
        ids = torch.randint(self.size, (size,), device=next(iter(self.data.values())).device)
        return {key: value[ids] for key, value in self.data.items()}


def context_auxiliary_targets(root_velocity_w, root_quat_w, payload_kg):
    """Loss labels: body-frame velocity in m/s and centered payload in kg.

    World heading is removed so no unobservable absolute orientation is imposed
    on a sensor-history encoder. These tensors must never be action inputs.
    """
    velocity_b = quat_apply_inverse(root_quat_w, root_velocity_w)
    return torch.cat((velocity_b, (payload_kg - 2.0)[:, None]), dim=-1)


def normalized_context_physics_targets(physical_labels):
    """Loss-only torso mass, COMxyz, foot friction and mean armature targets."""
    values = physical_labels[..., 1:]
    center = values.new_tensor((0, 0, 0, 0, 1.15, 1.0))
    scale = values.new_tensor((1, 0.075, 0.075, 0.075, 0.85, 0.2))
    return (values - center) / scale


def validate_context_initialization(
    student_args, source_args, *, allow_bias_head_expansion=False, allow_identity_feature_adapter=False,
):
    """Reject silent semantic changes even when state-dict tensor shapes match."""
    if not source_args["class_name"].endswith(":ContextAdaptationActor"):
        raise ValueError("Student initialization requires a deployable context actor")
    defaults = {
        "adaptation_latent_dim": 64,
        "context_history_steps": 50,
        "context_auxiliary_dim": 0,
        "context_feature_adapter": False,
        "context_imu_accel": False,
        "context_root_orientation": False,
        "context_right_aligned": False,
        "context_key_body": False,
        "context_learned_residual_gain": False,
        "context_mean_only": False,
        "latent_modulation": False,
        "physics_bilinear": False,
        "physics_low_rank": False,
        "latent_low_rank": False,
        "rank_per_physics": 2,
        "shared_low_rank": 0,
        "frozen_nominal_prior": None,
        "frozen_nominal_prior_deployable_base": False,
        "refinement_scale": 0.0,
        "residual_scale": 0.25,
    }
    for name, default in defaults.items():
        if (
            name == "context_feature_adapter" and allow_identity_feature_adapter
            and not source_args.get(name, default) and student_args.get(name, default)
        ):
            continue
        if (
            name == "context_auxiliary_dim" and allow_bias_head_expansion
            and source_args.get(name, default) == 10 and student_args.get(name, default) == 39
        ):
            continue
        if student_args.get(name, default) != source_args.get(name, default):
            raise ValueError(f"Student initialization changes input/action contract: {name}")
    if source_args.get("context_latent_mean", False):
        if not student_args.get("context_latent_mean", False) or student_args.get(
            "context_mean_horizon", 250
        ) != source_args.get("context_mean_horizon", 250):
            raise ValueError("Student initialization changes its existing causal memory contract")


def add_identity_feature_adapter_state(student, state):
    """Only introduce the zero-output additive frontend; preserve every source tensor."""
    from intact_tracking.residual_policy import _last_linear

    if student.feature_adapter is None or any(key.startswith("feature_adapter.") for key in state):
        raise ValueError("Identity adapter upgrade requires a new feature adapter")
    output = _last_linear(student.feature_adapter)
    if output.weight.count_nonzero() or output.bias.count_nonzero():
        raise ValueError("Identity feature adapter must have exactly zero output weights/bias")
    result = dict(state)
    result.update({key: value.clone() for key, value in student.state_dict().items() if key.startswith("feature_adapter.")})
    return result


def transfer_compact_context_encoder(student, source, student_args):
    """Reuse only a compatible physical estimator, never its old controller."""
    source_args = OmegaConf.to_container(source["cfg"].agent.actor, resolve=True)
    if not source_args["class_name"].endswith(":ContextAdaptationActor"):
        raise ValueError("Encoder-only transfer requires a deployable context actor")
    target_contract = source["residual_policy"].get("latent_target_contract", {})
    if target_contract.get("schema") != "compact_physics" or target_contract.get("actor_physics_input") != "normal":
        raise ValueError("Encoder-only transfer needs audited true-physics target semantics")
    for configuration in (student_args, source_args):
        if configuration.get("adaptation_latent_dim") != 7 or not (
            configuration.get("physics_low_rank") or configuration.get("physics_bilinear")
        ):
            raise ValueError("Encoder-only transfer requires the same explicit seven physical coordinates")
        if configuration.get("context_latent_mean"):
            raise ValueError("Encoder-only transfer requires a stateless instantaneous producer")
    for key, default in {
        "context_history_steps": 50, "context_imu_accel": False,
        "context_root_orientation": False, "context_right_aligned": False,
        "context_key_body": False,
    }.items():
        if source_args.get(key, default) != student_args.get(key, default):
            raise ValueError(f"Encoder-only transfer changes measured input semantics: {key}")
    state = source["actor_state_dict"]
    before = {key: value.detach().clone() for key, value in student.state_dict().items()}
    normalizer_keys = [key for key in before if key.startswith("tracker.history_normalizer.")]
    if not normalizer_keys or any(
        key not in state or not torch.equal(before[key].cpu(), state[key].cpu())
        for key in normalizer_keys
    ):
        raise ValueError("Encoder-only transfer changes frozen history normalization")
    encoder_state = {key.removeprefix("context_encoder."): value for key, value in state.items()
                     if key.startswith("context_encoder.")}
    student.context_encoder.load_state_dict(encoder_state, strict=True)
    after = student.state_dict()
    untouched = [key for key in before if not key.startswith("context_encoder.")]
    if any(not torch.equal(before[key], after[key]) for key in untouched):
        raise RuntimeError("Encoder-only transfer changed controller or preprocessing tensors")
    return {"encoder_tensors_transferred": len(encoder_state),
            "nonencoder_tensors_bitwise_unchanged": len(untouched),
            "history_normalization_bitwise_equal": len(normalizer_keys),
            "source_optimizer_controller_and_prior_transferred": False,
            "semantics": "payload/3, torso_delta_kg, COMxyz/.075, friction_delta/2, mean_armature_delta/.2"}


def expand_bias_readout_state(state, expected):
    """Preserve existing ten loss outputs; new bias outputs start at zero prior."""
    state = dict(state)
    for suffix in ("weight", "bias"):
        key = f"context_auxiliary_head.2.{suffix}"
        source, target = state[key], expected[key]
        if source.shape[0] != 10 or target.shape[0] != 39 or source.shape[1:] != target.shape[1:]:
            raise ValueError("Bias readout expansion must be exactly ten to39 outputs")
        state[key] = torch.cat((source, source.new_zeros((29, *source.shape[1:]))), 0)
    return state


@torch.no_grad()
def initialize_student_from_nominal(student, nominal_state, nominal_scale):
    """Transfer a deployable nominal trunk, padding only new context inputs.

    A trainable output gain preserves the exact nominal function at startup
    while allowing expansion toward the teacher's residual range. This is one
    network, not an inference ensemble or a changed nominal evaluation baseline.
    """
    student.tracker.load_state_dict(
        {key.removeprefix("tracker."): value for key, value in nominal_state.items() if key.startswith("tracker.")},
        strict=True,
    )
    layers = [name for name, module in student.residual_mlp.named_modules() if isinstance(module, torch.nn.Linear)]
    target = student.residual_mlp.state_dict()
    transferred = {}
    for key, expected in target.items():
        value = nominal_state[f"residual_mlp.{key}"].clone()
        if key == f"{layers[0]}.weight":
            if value.shape[1] != student.tracker.policy_input_dim:
                raise ValueError("Nominal residual must consume tracker features only")
            value = torch.cat((value, value.new_zeros(value.shape[0], expected.shape[1] - value.shape[1])), -1)
        transferred[key] = value
    student.residual_mlp.load_state_dict(transferred, strict=True)
    ratio = float(nominal_scale) / student.residual_scale
    if student.context_residual_log_gain is None or not math.exp(-4) <= ratio <= 1:
        raise ValueError("Exact nominal initialization needs a learnable bounded residual gain")
    student.context_residual_log_gain.fill_(math.log(ratio))


def calibrate_student_features(student, teacher, wrapped, obs, args, output):
    """Supervise deployable preprocessing before freezing it for action replay.

    Labels are loss targets only. The estimator and reference denoiser retain
    exactly their deployable input signatures and checkpoint normalizers.
    Separate phases prevent stale cached features while action replay is used.
    """
    model = student.tracker
    modules = (model.estimator, model.reference_encoder)
    parameters = [parameter for module in modules for parameter in module.parameters()]
    for module in modules:
        module.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=args.feature_calibration_lr)
    fields = (
        model.estimator_history_group,
        model.reference_encoder_input_group,
        model.estimator_target_group,
        model.foot_contact_target_group,
        model.reference_encoder_target_group,
    )
    replay = None
    start = time.time()
    action_term = wrapped.unwrapped.action_manager.get_term("joint_pos")
    with (output / "feature_calibration.jsonl").open("w", buffering=1) as log:
        for iteration in range(args.feature_calibration_updates):
            with torch.no_grad():
                for _ in range(args.rollout_steps):
                    batch = {key: obs[key] for key in fields}
                    if replay is None:
                        replay = DistillationReplay(args.replay_capacity, batch)
                    replay.add(batch)
                    teacher.populate_tracker_cache(obs)
                    actions = teacher(obs)
                    action_term.record_policy_mean(actions)
                    obs, _, _, _ = wrapped.step(actions)
            for _ in range(args.gradient_steps):
                batch = replay.sample(args.batch_size)
                height_mse, contact_bce, diagnostics = model.height_contact_losses(batch)
                reference_mse, reference_diagnostics = model.reference_encoder_losses(batch)
                loss = height_mse / 0.03**2 + contact_bce + reference_mse
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite feature-calibration loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
            record = {
                "phase": "feature_calibration",
                "iteration": iteration,
                "wall_seconds": time.time() - start,
                "height_rmse_m": float(height_mse.detach().sqrt()),
                "contact_bce": float(contact_bce.detach()),
                **{key: float(value.detach()) for key, value in diagnostics.items()},
                **{key: float(value.detach()) for key, value in reference_diagnostics.items()},
                "scope": "training replay diagnostics, not tracking acceptance",
            }
            log.write(json.dumps(record) + "\n")
            if iteration % 50 == 0:
                print(json.dumps(record), flush=True)
    for module in modules:
        module.requires_grad_(False)
    return obs


def run(args):
    from intact_tracking.adaptation_reward_contract import (
        assert_fixed_reward_checkpoint,
        assert_rewards_unchanged,
        capture_original_rewards,
    )

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)
    prepared = prepare_rollout(
        checkpoint_file=args.tracker_checkpoint,
        num_envs=args.num_envs,
        motion_path=args.motion_path,
        motion_file=None,
    )
    reward_contract = capture_original_rewards(prepared.env)
    physics = configure_physics(prepared.env, "dr")
    prepared.env.seed = args.seed
    prepared.env.commands["motion"].sampling_mode = args.sampling_mode
    prepared.env.commands["motion"].rewind.enabled = False
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    assert_fixed_reward_checkpoint(checkpoint, reward_contract)
    if checkpoint["cfg"].agent.actor.get("oracle_clean_proprio", False):
        configure_oracle_proprioception(prepared.env)
    if args.context_imu_accel:
        configure_context_sensors(prepared.env)
    assert_rewards_unchanged(reward_contract, prepared.env)
    env = ManagerBasedRlEnv(cfg=prepared.env, device=args.device)
    try:
        schema = str(checkpoint["cfg"].agent.actor.get("privilege_schema", "critic"))
        wrapped = PrivilegedAdaptationWrapper(
            env, clip_actions=prepared.clip_actions, privilege_schema=schema
        )
        if args.context_latent_mean:
            wrapped = CausalContextMeanWrapper(wrapped, horizon=args.context_mean_horizon)
        obs = wrapped.get_observations()
        teacher = load_actor(args.teacher_checkpoint, prepared, obs, wrapped)
        checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
        teacher_args = OmegaConf.to_container(checkpoint["cfg"].agent.actor, resolve=True)
        student_args = copy.deepcopy(teacher_args)
        student_args["class_name"] = "intact_tracking.adaptation_policy:ContextAdaptationActor"
        student_args.pop("privilege_dim", None)
        student_args.pop("privilege_hidden_dims", None)
        student_args.pop("privilege_schema", None)
        student_args.pop("actor_physics_input", None)
        student_args.pop("freeze_teacher_trunk", None)
        student_args.pop("oracle_tracking_features", None)
        student_args.pop("oracle_clean_proprio", None)
        student_args.pop("oracle_feature_curriculum", None)
        student_args["context_history_steps"] = 50
        # Record the actual distillation trainability, not the teacher's flag.
        # Old checkpoints retain their saved setting on reload.
        student_args["train_base_policy"] = args.train_student_base
        student_args["train_residual_policy"] = not args.freeze_student_residual
        student_args["train_context_encoder"] = not args.freeze_student_context
        student_args["context_latent_mean"] = args.context_latent_mean
        student_args["context_mean_horizon"] = args.context_mean_horizon
        student_args["context_auxiliary_dim"] = (
            39 if args.context_encoder_bias_weight else
            10 if args.context_physics_weight else 4 if args.context_auxiliary_weight else 0
        )
        student_args["context_feature_adapter"] = bool(args.feature_distillation_weight)
        student_args["context_imu_accel"] = args.context_imu_accel
        student_args["train_reference_encoder"] = args.train_reference_encoder
        student_args["context_root_orientation"] = args.context_root_orientation
        student_args["context_right_aligned"] = args.context_right_aligned
        student_args["context_key_body"] = args.context_key_body
        student_args["context_learned_residual_gain"] = bool(args.student_nominal_initialization)
        if teacher_args.get("latent_modulation", False) and (
            args.context_latent_mean or args.student_nominal_initialization
        ):
            raise ValueError("Modulated teacher currently supports direct current-latent distillation only")
        kwargs = {k: v for k, v in student_args.items() if k != "class_name"}
        groups = OmegaConf.to_container(checkpoint["cfg"].agent.obs_groups, resolve=True)
        student = ContextAdaptationActor(obs, groups, "actor", wrapped.num_actions, **kwargs).to(
            args.device
        )
        student.tracker.load_state_dict(teacher.tracker.state_dict(), strict=True)
        if not args.context_latent_mean:
            student.residual_mlp.load_state_dict(teacher.residual_mlp.state_dict(), strict=True)
        if student.refinement_mlp is not None:
            student.refinement_mlp.load_state_dict(teacher.refinement_mlp.state_dict(), strict=True)
        student.distribution.load_state_dict(teacher.distribution.state_dict(), strict=True)
        encoder_initialization_audit = None
        if getattr(args, "context_encoder_initialization", None):
            if schema != "compact_physics" or teacher_args.get("actor_physics_input", "normal") != "normal":
                raise ValueError("Encoder transfer destination must predict true compact physical coordinates")
            initial = torch.load(args.context_encoder_initialization, map_location="cpu", weights_only=False)
            assert_fixed_reward_checkpoint(initial, reward_contract)
            if "latent_target_contract" not in initial["residual_policy"]:
                # Older signed context checkpoints predate explicit target
                # metadata. Verify their exact recorded teacher, not a guessed
                # latent meaning based merely on matching tensor dimensions.
                source_teacher_path = Path(initial["residual_policy"]["arguments"]["teacher_checkpoint"])
                if _sha256(source_teacher_path) != initial["residual_policy"]["teacher_sha256"]:
                    raise ValueError("Encoder source teacher identity changed")
                source_teacher = torch.load(source_teacher_path, map_location="cpu", weights_only=False)
                assert_fixed_reward_checkpoint(source_teacher, reward_contract)
                source_teacher_args = source_teacher["cfg"].agent.actor
                initial["residual_policy"]["latent_target_contract"] = {
                    "schema": source_teacher_args.get("privilege_schema"),
                    "actor_physics_input": source_teacher_args.get("actor_physics_input", "normal"),
                }
                del source_teacher
            encoder_initialization_audit = transfer_compact_context_encoder(student, initial, student_args)
            encoder_initialization_audit["source_checkpoint_sha256"] = _sha256(Path(args.context_encoder_initialization))
            del initial
        student_initialization_sha256 = None
        student_initialization_action_max_difference = None
        if args.student_initialization:
            initial = torch.load(args.student_initialization, map_location="cpu", weights_only=False)
            assert_fixed_reward_checkpoint(initial, reward_contract)
            validate_context_initialization(
                student_args, OmegaConf.to_container(initial["cfg"].agent.actor, resolve=True),
                allow_bias_head_expansion=bool(args.context_encoder_bias_weight),
                allow_identity_feature_adapter=args.add_identity_feature_adapter,
            )
            initial_state = dict(initial["actor_state_dict"])
            if args.add_identity_feature_adapter:
                initial_state = add_identity_feature_adapter_state(student, initial_state)
            if args.context_encoder_bias_weight and initial["cfg"].agent.actor.get("context_auxiliary_dim") == 10:
                initial_state = expand_bias_readout_state(initial_state, student.state_dict())
            first_key = "residual_mlp.0.weight"
            if args.context_latent_mean and not initial["cfg"].agent.actor.get("context_latent_mean", False):
                value = initial_state[first_key]
                expected = student.state_dict()[first_key]
                extra = expected.shape[1] - value.shape[1]
                if extra != student.adaptation_latent_dim:
                    raise ValueError("Unexpected causal context input expansion")
                initial_state[first_key] = torch.cat((value, value.new_zeros(value.shape[0], extra)), -1)
            student.load_state_dict(initial_state, strict=True)
            student_initialization_sha256 = _sha256(Path(args.student_initialization))
            del initial
            initial_actor = load_actor(args.student_initialization, prepared, obs, wrapped)
            with torch.no_grad():
                deployable = obs.select(*student.deployable_observation_groups)
                actual, expected = student(deployable), initial_actor(deployable)
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
                student_initialization_action_max_difference = float((actual - expected).abs().max())
            del initial_actor
        if args.context_latent_mean:
            wrapped.bind(student)
            obs = wrapped.attach(obs)
        nominal_initialization_sha256 = None
        nominal_initialization_action_max_difference = None
        if args.student_nominal_initialization:
            nominal_checkpoint = torch.load(args.student_nominal_initialization, map_location="cpu", weights_only=False)
            assert_fixed_reward_checkpoint(nominal_checkpoint, reward_contract)
            nominal_cfg = nominal_checkpoint["cfg"].agent.actor
            if not nominal_cfg["class_name"].endswith(":FrozenTrackerResidualActor") or nominal_cfg.get("use_dynamics_latent", False):
                raise ValueError("Nominal initialization requires a deployable feature-only residual policy")
            initialize_student_from_nominal(
                student, nominal_checkpoint["actor_state_dict"], nominal_cfg["residual_scale"]
            )
            nominal_initialization_sha256 = _sha256(Path(args.student_nominal_initialization))
            del nominal_checkpoint
            nominal_actor = load_actor(args.student_nominal_initialization, prepared, obs, wrapped)
            with torch.no_grad():
                deployable = obs.select(*student.deployable_observation_groups)
                actual, expected = student(deployable), nominal_actor(deployable)
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
                nominal_initialization_action_max_difference = float((actual - expected).abs().max())
            del nominal_actor
        student.train()
        student.tracker.mlp.requires_grad_(args.train_student_base)
        parameters = list(student.context_encoder.parameters()) + [
            parameter for parameter in student.residual_mlp.parameters() if parameter.requires_grad
        ]
        if student.refinement_mlp is not None:
            parameters += [p for p in student.refinement_mlp.parameters() if p.requires_grad]
        if student.context_residual_log_gain is not None:
            parameters.append(student.context_residual_log_gain)
        if student.context_auxiliary_head is not None:
            parameters += list(student.context_auxiliary_head.parameters())
        if student.feature_adapter is not None:
            parameters += list(student.feature_adapter.parameters())
        if args.train_reference_encoder:
            parameters += list(student.tracker.reference_encoder.parameters())
        if args.train_student_base:
            student.tracker.mlp.requires_grad_(True)
            parameters += list(student.tracker.mlp.parameters())
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        if not parameters:
            raise ValueError("No trainable student parameters")
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
        cfg = copy.deepcopy(checkpoint["cfg"])
        cfg.agent.actor = OmegaConf.create(student_args)
        metadata = {
            "version": "deployable_context_distillation_v1",
            "reward_changes": {},
            "reward_contract": reward_contract,
            "fixed_reward_lineage": {
                "source": "audited fixed-reward teacher and optional audited student initialization",
                "historical_shaped_checkpoint_used": False,
                "supervised_losses": "action/latent/feature labels train the student; environment reward is neither used by distillation nor changed",
            },
            "arguments": vars(args),
            "teacher_sha256": _sha256(Path(args.teacher_checkpoint)),
            "latent_target_contract": {"schema": schema, "actor_physics_input": teacher_args.get("actor_physics_input", "normal")},
            "frozen_nominal_prior_audit": student.frozen_nominal_prior_audit,
            "student_initialization_sha256": student_initialization_sha256,
            "context_encoder_initialization_audit": encoder_initialization_audit,
            "student_initialization_action_max_difference": student_initialization_action_max_difference,
            "student_nominal_initialization_sha256": nominal_initialization_sha256,
            "student_nominal_initialization_action_max_difference": nominal_initialization_action_max_difference,
            "student_initialization_contract": "optional exact deployable nominal trunk transfer; context columns zero-padded; trainable per-joint log gain initialized to nominal/teacher residual scale, bounded[-4,0]; one shared network, no ensemble",
            "physics": physics,
            "actor_inputs": list(student.deployable_observation_groups),
            "additional_imu": {
                "enabled": args.context_imu_accel,
                "sensor": "existing XML pelvis imu_lin_acc, local specific force including gravity",
                "noise_std_m_s2": 0.2,
                "history_steps": 50,
                "encoder_scale_m_s2": 10.0,
            },
            "context_inputs": "50-frame measured q, qdot, gravity, gyro, known command and torque history",
            "privileged_actor_inputs": [],
            "action_distillation": "student-visited states with an annealed per-world teacher mixture",
            "latent_loss_weight": args.latent_loss_weight,
            "train_student_base": args.train_student_base,
            "freeze_student_residual": args.freeze_student_residual,
            "freeze_student_context": args.freeze_student_context,
            "context_latent_mean": args.context_latent_mean,
            "context_mean_horizon": args.context_mean_horizon,
            "context_memory_contract": "current latent plus causal prefix mean, EMA gain1/horizon after horizon samples; frozen producer encoder; reset per episode; no extra sensor draws or future samples",
            "feature_distillation_weight": args.feature_distillation_weight,
            "train_reference_encoder": args.train_reference_encoder,
            "reference_feature_weight": args.reference_feature_weight,
            "context_root_orientation": args.context_root_orientation,
            "context_right_aligned": args.context_right_aligned,
            "context_key_body": args.context_key_body,
            "context_kinematics_contract": "existing195D FK from measured joints and gyro, scaled positions/rotations1, linear velocities2, angular velocities4; no additional simulator observations",
            "context_alignment_contract": "optional left trim preserves newest causal sample; legacy checkpoints retain original alignment",
            "context_orientation_contract": "existing deployable robot_root_quat, not simulator quaternion; resolves absent initial world heading in local history",
            "reference_training_contract": "raw deployable observation replay; action gradients through reference kinematics; no true reference at inference",
            "feature_adapter_contract": "deployable normalized tracker features + context latent; teacher features only in loss",
            "add_identity_feature_adapter": args.add_identity_feature_adapter,
            "base_distillation_target": "absolute teacher action, avoiding stale replay base actions",
            "context_auxiliary_supervision": {
                "weight": args.context_auxiliary_weight,
                "labels": "body-frame linear velocity(m/s), payload mass minus2(kg)",
                "input_contract": "loss-only labels; the readout consumes context latent only",
            },
            "context_physics_supervision": {
                "weight": args.context_physics_weight,
                "labels": "torso_delta_kg, torso_COMxyz_m, mean_foot_friction, mean_armature_ratio",
                "centers": [0, 0, 0, 0, 1.15, 1.0],
                "scales": [1, 0.075, 0.075, 0.075, 0.85, 0.2],
                "contract": "six additional loss-only labels; no physics values enter the context encoder or action path",
            },
            "context_encoder_bias_supervision": {
                "weight": args.context_encoder_bias_weight,
                "labels": "29 true encoder biases divided by .01 rad, startup constant per world",
                "contract": "loss-only targets, never encoder or action inputs; ten old readout outputs preserved when expanding to39",
            },
            "feature_calibration": {
                "updates": args.feature_calibration_updates,
                "learning_rate": args.feature_calibration_lr,
                "loss": "height_MSE/0.03^2 + contact_BCE + normalized_reference_residual_MSE",
                "input_contract": "privileged targets only in training loss, never estimator inputs",
            },
            "research_source_sha256": {
                str(path.relative_to(Path.cwd())): _sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    Path(__file__).resolve().parents[1] / "adaptation_policy.py",
                    Path(__file__).resolve().parents[1] / "adaptation_reward_contract.py",
                    Path(__file__).resolve().parents[1] / "adaptation_modulation.py",
                    Path(__file__).resolve().parents[1] / "adaptation_prior.py",
                    Path(__file__).resolve().parents[1] / "adaptation_sensors.py",
                    Path(__file__).resolve().parents[1] / "adaptation_context_memory.py",
                )
            },
        }
        cfg.residual_policy = OmegaConf.create(metadata)
        (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        OmegaConf.save(cfg, output / "config.yaml")

        if args.feature_calibration_updates:
            obs = calibrate_student_features(student, teacher, wrapped, obs, args, output)
            student.tracker.reference_encoder.requires_grad_(args.train_reference_encoder)

        # The actor checkpoint is self-contained; teacher-only modules are absent.
        def save(label, iteration):
            torch.save(
                {
                    "actor_state_dict": student.state_dict(),
                    "critic_state_dict": checkpoint["critic_state_dict"],
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter": iteration,
                    "cfg": cfg,
                    "residual_policy": metadata,
                },
                output / f"checkpoint_{label}.pt",
            )

        replay = None
        if args.student_initialization or encoder_initialization_audit is not None:
            save("initial", -1)
        total_steps = 0
        start = time.time()
        action_term = env.action_manager.get_term("joint_pos")
        payload_kg = None
        physics_targets = None
        encoder_bias_targets = (
            env.scene["robot"].data.encoder_bias.detach().clone() / 0.01
            if args.context_encoder_bias_weight else None
        )
        if args.context_physics_weight:
            from intact_tracking.cli.context_probe import physical_labels

            _, physical_values = physical_labels(env)
            physics_targets = normalized_context_physics_targets(physical_values).to(env.device)
        if args.context_auxiliary_weight or args.context_physics_weight:
            model = env.sim.mj_model
            wrist_ids = [
                i
                for i in range(model.nbody)
                if model.body(i).name.split("/")[-1] == "right_wrist_yaw_link"
            ]
            if len(wrist_ids) != 1:
                raise ValueError("Ambiguous payload body for context supervision")
            mass, default_mass = expanded_model_field(env, "body_mass")
            payload_kg = (mass[:, wrist_ids[0]] - default_mass[wrist_ids[0]]).clone()
        # Optional paired-ablation stream, independent of added readout layers'
        # discarded random initializations. Does not change the physics sample.
        if args.training_rng_seed is not None:
            _seed_everything(args.training_rng_seed)
        with (output / "metrics.jsonl").open("w", buffering=1) as log:
            for iteration in range(args.iterations):
                env.command_manager.get_term("motion").begin_adaptive_sampling_iteration(iteration)
                fraction = (
                    max(0.0, 1.0 - iteration / args.teacher_mix_updates)
                    if args.teacher_mix_updates > 0 else 0.0
                )
                with torch.no_grad():
                    for _ in range(args.rollout_steps):
                        teacher.populate_tracker_cache(obs)
                        teacher_features, teacher_base = teacher._base_features_and_action(obs)
                        # Each actor must use its own deployable preprocessing;
                        # calibrated student estimators differ from the teacher.
                        student.populate_tracker_cache(obs)
                        features, base = _base_with_optional_update(student, obs)
                        teacher_z = teacher.privileged_latent(obs)
                        teacher_action = (
                            teacher_base
                            + teacher._residual(torch.cat((teacher_features, teacher_z), dim=-1))
                        )
                        history = student.normalized_context_history(obs)
                        orientation = obs["robot_root_quat"] if args.context_root_orientation else None
                        kinematics = (
                            normalized_context_kinematics(obs["robot_key_body"])
                            if args.context_key_body else None
                        )
                        z = student.context_encoder(history, orientation, kinematics)
                        control_z = (
                            torch.cat((z, obs[CONTEXT_MEAN]), -1)
                            if args.context_latent_mean else z
                        )
                        adapted = student.adapt_features(features, control_z)
                        if student.feature_adapter is not None:
                            base = student.base_action_from_adapted_features(features, adapted)
                        target = teacher_action - base
                        correction = student._residual(torch.cat((adapted, control_z), dim=-1))
                        batch = {
                            "history": history,
                            "features": features,
                            "target": target,
                            "latent": teacher_z,
                        }
                        if args.context_latent_mean:
                            batch["context_mean"] = obs[CONTEXT_MEAN]
                        if orientation is not None:
                            batch["context_orientation"] = orientation
                        if kinematics is not None:
                            batch["context_kinematics"] = kinematics
                        if (
                            args.train_student_base
                            or student.feature_adapter is not None
                            or args.train_reference_encoder
                        ):
                            batch["teacher_action"] = teacher_action
                        if args.train_reference_encoder:
                            for name in student.deployable_observation_groups:
                                batch[f"deployable/{name}"] = obs[name]
                        if student.feature_adapter is not None or args.reference_feature_weight:
                            batch["teacher_features"] = teacher_features
                        if payload_kg is not None:
                            robot = env.scene["robot"].data
                            batch["context_auxiliary_target"] = context_auxiliary_targets(
                                robot.root_link_lin_vel_w, robot.root_link_quat_w, payload_kg
                            )
                        if physics_targets is not None:
                            batch["context_physics_target"] = physics_targets
                        if encoder_bias_targets is not None:
                            batch["context_encoder_bias_target"] = encoder_bias_targets
                        if replay is None:
                            replay = DistillationReplay(args.replay_capacity, batch)
                        replay.add(batch)
                        use_teacher = torch.rand(args.num_envs, 1, device=args.device) < fraction
                        actions = base + torch.where(use_teacher, target, correction)
                        action_term.record_policy_mean(actions)
                        obs, _, dones, _ = wrapped.step(actions)
                        total_steps += 1
                assert replay is not None
                metrics = {
                    "action_mse": 0.0,
                    "latent_mse": 0.0,
                    "context_auxiliary_mse": 0.0,
                    "context_physics_mse": 0.0,
                    "context_encoder_bias_mse": 0.0,
                    "feature_mse": 0.0,
                }
                for _ in range(args.gradient_steps):
                    batch = replay.sample(args.batch_size)
                    z = student.context_encoder(
                        batch["history"], batch.get("context_orientation"), batch.get("context_kinematics")
                    )
                    control_z = (
                        torch.cat((z, batch["context_mean"]), -1)
                        if args.context_latent_mean else z
                    )
                    features = batch["features"]
                    if args.train_reference_encoder:
                        deployable_obs = TensorDict(
                            {
                                name: batch[f"deployable/{name}"]
                                for name in student.deployable_observation_groups
                            },
                            [len(features)],
                        )
                        features, _ = _base_with_optional_update(student, deployable_obs)
                    raw_features = features
                    features = student.adapt_features(raw_features, control_z)
                    prediction = student._residual(torch.cat((features, control_z), dim=-1))
                    if (
                        args.train_student_base
                        or student.feature_adapter is not None
                        or args.train_reference_encoder
                    ):
                        current_base = student.base_action_from_adapted_features(raw_features, features)
                        action_mse = (
                            current_base + prediction - batch["teacher_action"]
                        ).square().mean()
                    else:
                        action_mse = (prediction - batch["target"]).square().mean()
                    latent_mse = (z - batch["latent"]).square().mean()
                    loss = action_mse + args.latent_loss_weight * latent_mse
                    if student.feature_adapter is not None or args.reference_feature_weight:
                        feature_mse = (features - batch["teacher_features"]).square().mean()
                        loss = loss + (
                            args.feature_distillation_weight + args.reference_feature_weight
                        ) * feature_mse
                        metrics["feature_mse"] += float(feature_mse.detach()) / args.gradient_steps
                    if student.context_auxiliary_head is not None:
                        readout = student.context_auxiliary_head(z)
                        auxiliary_mse = (
                            (readout[..., :4] - batch["context_auxiliary_target"])
                            .square()
                            .mean()
                        )
                        loss = loss + args.context_auxiliary_weight * auxiliary_mse
                        metrics["context_auxiliary_mse"] += (
                            float(auxiliary_mse.detach()) / args.gradient_steps
                        )
                        if args.context_physics_weight:
                            physics_mse = (readout[..., 4:10] - batch["context_physics_target"]).square().mean()
                            loss = loss + args.context_physics_weight * physics_mse
                            metrics["context_physics_mse"] += float(physics_mse.detach()) / args.gradient_steps
                        if args.context_encoder_bias_weight:
                            bias_mse = (readout[..., 10:39] - batch["context_encoder_bias_target"]).square().mean()
                            loss = loss + args.context_encoder_bias_weight * bias_mse
                            metrics["context_encoder_bias_mse"] += float(bias_mse.detach()) / args.gradient_steps
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite distillation loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    metrics["action_mse"] += float(action_mse.detach()) / args.gradient_steps
                    metrics["latent_mse"] += float(latent_mse.detach()) / args.gradient_steps
                command = env.command_manager.get_term("motion")
                record = {
                    "iteration": iteration,
                    "environment_steps": total_steps,
                    "teacher_fraction": fraction,
                    "wall_seconds": time.time() - start,
                    **metrics,
                    "online_body_pos": float(command.metrics["error_body_pos"].mean()),
                    "online_joint_pos": float(command.metrics["error_joint_pos"].mean()),
                }
                log.write(json.dumps(record) + "\n")
                if iteration % 50 == 0:
                    print(json.dumps(record), flush=True)
                if iteration % args.save_interval == 0:
                    save(iteration, iteration)
            save("final", args.iterations - 1)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-nominal-initialization")
    parser.add_argument("--student-initialization")
    parser.add_argument("--context-encoder-initialization", help="Only transfer a compatible signed seven-physical-coordinate context encoder; keep the current teacher controller")
    parser.add_argument("--tracker-checkpoint", default=TRACKER)
    parser.add_argument("--motion-path", default=DATASET)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--training-rng-seed", type=int)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--replay-capacity", type=int, default=32768)
    parser.add_argument("--teacher-mix-updates", type=int, default=1500)
    parser.add_argument("--latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--context-auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--context-physics-weight", type=float, default=0.0)
    parser.add_argument("--context-encoder-bias-weight", type=float, default=0.0)
    parser.add_argument("--train-student-base", action="store_true")
    parser.add_argument("--freeze-student-residual", action="store_true")
    parser.add_argument("--freeze-student-context", action="store_true")
    parser.add_argument("--context-latent-mean", action="store_true")
    parser.add_argument("--context-mean-horizon", type=int, default=250)
    parser.add_argument("--feature-distillation-weight", type=float, default=0.0)
    parser.add_argument("--add-identity-feature-adapter", action="store_true", help="Allow a zero-output additive frontend on an otherwise bitwise-preserved initialized student")
    parser.add_argument("--context-imu-accel", action="store_true")
    parser.add_argument("--train-reference-encoder", action="store_true")
    parser.add_argument("--reference-feature-weight", type=float, default=0.0)
    parser.add_argument("--context-root-orientation", action="store_true")
    parser.add_argument("--context-right-aligned", action="store_true")
    parser.add_argument("--context-key-body", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--feature-calibration-updates", type=int, default=0)
    parser.add_argument("--feature-calibration-lr", type=float, default=3e-5)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--sampling-mode", choices=("uniform", "adaptive"), default="uniform")
    args = parser.parse_args()
    if args.student_initialization and args.student_nominal_initialization:
        parser.error("Choose context or nominal initialization, not both")
    if args.context_encoder_initialization and (
        args.student_initialization or args.student_nominal_initialization or args.context_latent_mean
    ):
        parser.error("Encoder-only initialization is separate from whole-actor/nominal/memory initialization")
    if args.freeze_student_context and not args.student_initialization:
        parser.error("Freezing context requires a trained student initialization")
    if args.context_latent_mean and (
        not args.student_initialization or not args.freeze_student_context or args.feature_distillation_weight
    ):
        parser.error("Causal context memory requires a frozen initialized encoder and no feature adapter")
    if args.context_mean_horizon < 1:
        parser.error("Context mean horizon must be positive")
    if args.feature_calibration_updates < 0 or args.feature_calibration_lr <= 0:
        parser.error("Feature-calibration updates must be nonnegative and LR positive")
    if args.context_auxiliary_weight < 0:
        parser.error("Context auxiliary weight must be nonnegative")
    if args.context_physics_weight < 0:
        parser.error("Context physics weight must be nonnegative")
    if args.context_encoder_bias_weight < 0 or (
        args.context_encoder_bias_weight and (
            not args.context_physics_weight or args.freeze_student_context
        )
    ):
        parser.error("Encoder-bias loss must be nonnegative and requires physics labels and a trainable encoder")
    if args.feature_distillation_weight < 0:
        parser.error("Feature distillation weight must be nonnegative")
    if args.add_identity_feature_adapter and (
        not args.student_initialization or not args.feature_distillation_weight or args.context_latent_mean
    ):
        parser.error("Identity frontend requires a non-memory initialized student and a positive feature loss")
    if args.reference_feature_weight < 0 or (
        args.reference_feature_weight and not args.train_reference_encoder
    ):
        parser.error("Reference feature weight must be nonnegative and requires train-reference-encoder")
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    run(args)


if __name__ == "__main__":
    main()
