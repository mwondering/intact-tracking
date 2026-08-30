"""SPV5 actor with supervised reference denoising and HEFT-compatible FK."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import quat_from_matrix
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from tensordict import TensorDict

from intact_tracking.environment.assets.robots.g1_tracking_bfm import G1_TRACKING_BFM_XML
from intact_tracking.environment.mdp import sp as sp_mdp
from intact_tracking.environment.mdp.keypoints import (
    KeypointKinematics,
    _rigid_points,
    parse_keypoint_specs,
)
from intact_tracking.environment.mdp.motion_fk import (
    MotionFKHelper,
    actor_angvel_from_quat_torch,
    actor_body_kinematics_torch,
    finite_diff_torch,
    normalize,
    quat_apply,
    quat_apply_inverse,
    smooth_avg5_torch,
)
from intact_tracking.environment.mdp.multi_commands import MUJOCO_JOINT_NAMES
from intact_tracking.environment.mdp.spv4 import (
    SPV4_KEY_BODY_COUNT,
    SPV4_KEY_BODY_STATE_DIM,
    RootFrameKeyBodyState,
    _key_body_error,
    _pack_state,
)
from intact_tracking.environment.mdp.spv5 import (
    SPV5_REFERENCE_FRAME_DIM,
    SPV5_REFERENCE_INPUT_DIM,
    SPV5_REFERENCE_SUPPORT_STEPS,
    SPV5_REFERENCE_TARGET_DIM,
    SPV5_ROBOT_ROOT_QUAT_DIM,
)

from .spv3_models import (
    PROPRIO_TERM_DIMS,
    SPV2_POLICY_HISTORY_LENGTH,
    SPV3_ESTIMATOR_HISTORY_LENGTH,
    SPV3_ESTIMATOR_OUTPUT_DIM,
    SPV3_EXISTING_ERROR_DIM,
    SPV3_POLICY_INPUT_DIM,
    SPV3_REFERENCE_DIM,
    SPV3EstimatorActor,
    _identity_or_normalizer,
)
from .spv4_models import _assemble_spv4_features

SPV5_REFERENCE_POLICY_START = SPV5_REFERENCE_SUPPORT_STEPS.index(0)
SPV5_REFERENCE_POLICY_LENGTH = 5
SPV5_POLICY_INPUT_DIM = SPV3_POLICY_INPUT_DIM + 3 * SPV4_KEY_BODY_STATE_DIM
SPV5_REFERENCE_CACHE_DIM = SPV3_REFERENCE_DIM + SPV4_KEY_BODY_STATE_DIM
SPV5_POLICY_CONTEXT_CACHE_DIM = SPV5_REFERENCE_CACHE_DIM + SPV3_ESTIMATOR_OUTPUT_DIM
SPV5_POLICY_CONTEXT_CACHE_GROUP = "spv5_policy_context_cache"
SPV5_RAW_ACTOR_OBS_DIM = (
    SPV5_ROBOT_ROOT_QUAT_DIM
    + SPV3_ESTIMATOR_HISTORY_LENGTH * sum(PROPRIO_TERM_DIMS)
    + SPV5_REFERENCE_INPUT_DIM
    + SPV4_KEY_BODY_STATE_DIM
)


def _rot6d_to_quat(value: torch.Tensor) -> torch.Tensor:
    first = normalize(value[..., :3])
    second_raw = value[..., 3:6]
    second = normalize(second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first)
    third = torch.linalg.cross(first, second, dim=-1)
    matrix = torch.stack((first, second, third), dim=-1)
    return normalize(quat_from_matrix(matrix))


def _normalizer_inverse(normalizer: nn.Module, value: torch.Tensor) -> torch.Tensor:
    if isinstance(normalizer, EmpiricalNormalization):
        return normalizer.inverse(value)
    return value


def _support_angvel_from_quat(quat_wxyz: torch.Tensor, fps: float, dim: int) -> torch.Tensor:
    """Compatibility wrapper for tests and older imports."""
    return actor_angvel_from_quat_torch(quat_wxyz, fps, dim)


def _current_proprio(history: torch.Tensor, history_length: int) -> tuple[torch.Tensor, ...]:
    values = []
    offset = 0
    for term_dim in PROPRIO_TERM_DIMS:
        size = history_length * term_dim
        values.append(history[..., offset + size - term_dim : offset + size])
        offset += size
    return tuple(values)


def _unpack_key_body(value: torch.Tensor) -> RootFrameKeyBodyState:
    batch = value.shape[0]
    pos_end = SPV4_KEY_BODY_COUNT * 3
    rot_end = pos_end + SPV4_KEY_BODY_COUNT * 6
    lin_end = rot_end + SPV4_KEY_BODY_COUNT * 3
    return RootFrameKeyBodyState(
        pos=value[..., :pos_end].reshape(batch, SPV4_KEY_BODY_COUNT, 3),
        quat=_rot6d_to_quat(value[..., pos_end:rot_end].reshape(batch, SPV4_KEY_BODY_COUNT, 6)),
        lin_vel=value[..., rot_end:lin_end].reshape(batch, SPV4_KEY_BODY_COUNT, 3),
        ang_vel=value[..., lin_end:].reshape(batch, SPV4_KEY_BODY_COUNT, 3),
    )


@dataclass(frozen=True)
class _ReferenceFeatures:
    standard: torch.Tensor
    key_body: torch.Tensor
    key_state: RootFrameKeyBodyState
    robot_from_reference_root: torch.Tensor
    joint_pos_current: torch.Tensor
    joint_vel_current: torch.Tensor
    gravity_current: torch.Tensor
    ang_vel_current: torch.Tensor
    height_current: torch.Tensor
    lin_vel_current_robot: torch.Tensor


class SPV5ReferenceKinematics:
    """Expand denoised minimal qpos with the same numerical recipe as HEFT."""

    def __init__(self, keypoint_specs: Sequence[dict[str, Any]], fps: float) -> None:
        self.specs = parse_keypoint_specs(keypoint_specs)
        if len(self.specs) != SPV4_KEY_BODY_COUNT:
            raise ValueError(
                f"SPV5 requires {SPV4_KEY_BODY_COUNT} semantic key bodies, got {len(self.specs)}"
            )
        physical_names: list[str] = []
        for spec in self.specs:
            for name in (
                spec.reference_body_name,
                spec.reference_correction_body_name,
            ):
                if name is not None and name not in physical_names:
                    physical_names.append(name)
        self.physical_names = tuple(physical_names)
        self.parent_indices = tuple(
            self.physical_names.index(spec.reference_body_name) for spec in self.specs
        )
        self.correction_indices = tuple(
            self.physical_names.index(
                spec.reference_correction_body_name or spec.reference_body_name
            )
            for spec in self.specs
        )
        self.local_pos = tuple(spec.reference_local_pos for spec in self.specs)
        self.local_quat = tuple(spec.reference_local_quat for spec in self.specs)
        self.correction_local_pos = tuple(
            spec.reference_correction_local_pos for spec in self.specs
        )
        self.fps = float(fps)
        self._helper: MotionFKHelper | None = None

    def _fk_helper(self, device: torch.device) -> MotionFKHelper:
        if self._helper is None or self._helper.device != device:
            self._helper = MotionFKHelper.from_mjcf_path(
                xml_path=G1_TRACKING_BFM_XML,
                dataset_joint_names=MUJOCO_JOINT_NAMES,
                output_body_names=self.physical_names,
                base_body_name="pelvis",
                device=device,
            )
        return self._helper

    def __call__(self, decoded: torch.Tensor, robot_root_quat: torch.Tensor) -> _ReferenceFeatures:
        batch = decoded.shape[0]
        support_length = len(SPV5_REFERENCE_SUPPORT_STEPS)
        frames = decoded.reshape(batch, support_length, SPV5_REFERENCE_FRAME_DIM)
        root_pos = frames[..., :3]
        root_quat = _rot6d_to_quat(frames[..., 3:9])
        joint_pos = frames[..., 9:]

        root_lin_vel_w = smooth_avg5_torch(finite_diff_torch(root_pos, self.fps, dim=1), dim=1)
        root_ang_vel_w = smooth_avg5_torch(
            _support_angvel_from_quat(root_quat, self.fps, dim=1), dim=1
        )
        root_lin_vel_b = quat_apply_inverse(root_quat, root_lin_vel_w)
        root_ang_vel_b = quat_apply_inverse(root_quat, root_ang_vel_w)
        joint_vel = smooth_avg5_torch(finite_diff_torch(joint_pos, self.fps, dim=1), dim=1)

        # Only the current key-body state reaches the policy.  HEFT's centered
        # difference plus avg5 at t=0 needs [-3,+3], not the full [-3,+7]
        # command support.  Limiting body FK to these seven frames cuts its batch
        # by 36% without changing the current pose or velocity.
        key_support_stop = SPV5_REFERENCE_POLICY_START + 4
        key_joint_pos = joint_pos[:, :key_support_stop]
        helper = self._fk_helper(decoded.device)
        body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b = actor_body_kinematics_torch(
            helper,
            key_joint_pos,
            root_ang_vel_b[:, :key_support_stop],
            self.fps,
            dim=1,
        )

        parent_ids = torch.as_tensor(self.parent_indices, device=decoded.device, dtype=torch.long)
        correction_ids = torch.as_tensor(
            self.correction_indices, device=decoded.device, dtype=torch.long
        )
        semantic: KeypointKinematics = _rigid_points(
            body_pos_b.index_select(-2, parent_ids),
            body_quat_b.index_select(-2, parent_ids),
            body_lin_vel_b.index_select(-2, parent_ids),
            body_ang_vel_b.index_select(-2, parent_ids),
            decoded.new_tensor(self.local_pos),
            decoded.new_tensor(self.local_quat),
            body_quat_b.index_select(-2, correction_ids),
            body_ang_vel_b.index_select(-2, correction_ids),
            decoded.new_tensor(self.correction_local_pos),
        )

        start = SPV5_REFERENCE_POLICY_START
        stop = start + SPV5_REFERENCE_POLICY_LENGTH
        current = start
        selected_root_quat = root_quat[:, start:stop]
        robot_quat = normalize(robot_root_quat).unsqueeze(1).expand_as(selected_root_quat)
        robot_from_reference = sp_mdp._quat_in_frame(robot_quat, selected_root_quat)
        root_offsets = quat_apply_inverse(
            root_quat[:, current : current + 1],
            root_pos[:, current + 1 : stop] - root_pos[:, current : current + 1],
        )
        gravity = sp_mdp._projected_gravity(root_quat)
        standard = torch.cat(
            (
                root_offsets.reshape(batch, -1),
                sp_mdp._rot6d(robot_from_reference).reshape(batch, -1),
                root_pos[:, start:stop, 2].reshape(batch, -1),
                root_lin_vel_b[:, start:stop].reshape(batch, -1),
                joint_pos[:, start:stop].reshape(batch, -1),
                joint_vel[:, start:stop].reshape(batch, -1),
                gravity[:, start:stop].reshape(batch, -1),
                root_ang_vel_b[:, start:stop].reshape(batch, -1),
            ),
            dim=-1,
        )
        if standard.shape[-1] != SPV3_REFERENCE_DIM:
            raise RuntimeError(
                f"SPV5 derived reference has {standard.shape[-1]} values, "
                f"expected {SPV3_REFERENCE_DIM}"
            )

        key_state = RootFrameKeyBodyState(
            pos=semantic.pos_w[:, current],
            quat=semantic.quat_w[:, current],
            lin_vel=semantic.lin_vel_w[:, current],
            ang_vel=semantic.ang_vel_w[:, current],
        )
        key_body = _pack_state(key_state)
        return _ReferenceFeatures(
            standard=standard,
            key_body=key_body,
            key_state=key_state,
            robot_from_reference_root=robot_from_reference[:, 0],
            joint_pos_current=joint_pos[:, current],
            joint_vel_current=joint_vel[:, current],
            gravity_current=gravity[:, current],
            ang_vel_current=root_ang_vel_b[:, current],
            height_current=root_pos[:, current, 2:3],
            lin_vel_current_robot=quat_apply(
                robot_from_reference[:, 0], root_lin_vel_b[:, current]
            ),
        )


def _pack_reference_cache(reference: _ReferenceFeatures) -> torch.Tensor:
    return torch.cat((reference.standard, reference.key_body), dim=-1)


def _reference_features_from_cache(value: torch.Tensor) -> _ReferenceFeatures:
    if value.shape[-1] != SPV5_REFERENCE_CACHE_DIM:
        raise ValueError(
            f"SPV5 reference cache has {value.shape[-1]} values, "
            f"expected {SPV5_REFERENCE_CACHE_DIM}"
        )
    standard = value[..., :SPV3_REFERENCE_DIM]
    key_body = value[..., SPV3_REFERENCE_DIM:]
    robot_from_reference = _rot6d_to_quat(standard[..., 12:18])
    key_state = _unpack_key_body(key_body)
    return _ReferenceFeatures(
        standard=standard,
        key_body=key_body,
        key_state=key_state,
        robot_from_reference_root=robot_from_reference,
        joint_pos_current=standard[..., 62:91],
        joint_vel_current=standard[..., 207:236],
        gravity_current=standard[..., 352:355],
        ang_vel_current=standard[..., 367:370],
        height_current=standard[..., 42:43],
        lin_vel_current_robot=quat_apply(robot_from_reference, standard[..., 47:50]),
    )


def _spv5_policy_features(
    *,
    history: torch.Tensor,
    latest_proprio: torch.Tensor,
    estimate: torch.Tensor,
    decoded_reference: torch.Tensor | None,
    robot_root_quat: torch.Tensor | None,
    robot_key_body: torch.Tensor,
    history_length: int,
    kinematics: SPV5ReferenceKinematics,
    reference_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    if reference_cache is None:
        if decoded_reference is None or robot_root_quat is None:
            raise ValueError("SPV5 uncached policy features require decoded reference")
        reference = kinematics(decoded_reference, robot_root_quat)
    else:
        reference = _reference_features_from_cache(reference_cache)
    joint_pos, joint_vel, gravity, ang_vel, _, _ = _current_proprio(history, history_length)
    existing_error = torch.cat(
        (
            reference.joint_pos_current - joint_pos,
            reference.joint_vel_current - joint_vel,
            reference.gravity_current - gravity,
            reference.ang_vel_current - ang_vel,
        ),
        dim=-1,
    )
    if existing_error.shape[-1] != SPV3_EXISTING_ERROR_DIM:
        raise RuntimeError(
            f"SPV5 existing error has {existing_error.shape[-1]} values, expected 64"
        )
    root_state_error = torch.cat(
        (
            reference.height_current - estimate[..., :1],
            reference.lin_vel_current_robot - estimate[..., 1:],
        ),
        dim=-1,
    )
    spv3_features = torch.cat(
        (
            latest_proprio,
            estimate,
            reference.standard,
            existing_error,
            root_state_error,
        ),
        dim=-1,
    )
    robot_key_state = _unpack_key_body(robot_key_body)
    key_error = _key_body_error(
        robot_key_state,
        reference.key_state,
        reference.robot_from_reference_root,
    )
    features = _assemble_spv4_features(spv3_features, robot_key_body, reference.key_body, key_error)
    if features.shape[-1] != SPV5_POLICY_INPUT_DIM:
        raise RuntimeError(
            f"SPV5 policy features have {features.shape[-1]} values, "
            f"expected {SPV5_POLICY_INPUT_DIM}"
        )
    return features


class SPV5ReferenceEncoderActor(SPV3EstimatorActor):
    """SPV4 information layout whose reference side is reconstructed from qpos."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: Sequence[int] = (2048, 2048, 1024, 1024, 512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        estimator_hidden_dims: Sequence[int] = (512, 256, 128),
        estimator_activation: str = "elu",
        reference_encoder_hidden_dims: Sequence[int] = (512, 256, 128),
        reference_encoder_activation: str = "elu",
        actor_core_group: str = "actor_core",
        robot_root_quat_group: str = "robot_root_quat",
        estimator_history_group: str = "estimator_history",
        estimator_target_group: str = "estimator_target",
        reference_encoder_input_group: str = "reference_encoder_input",
        reference_encoder_target_group: str = "reference_encoder_target",
        robot_key_body_group: str = "robot_key_body",
        estimator_history_length: int = SPV3_ESTIMATOR_HISTORY_LENGTH,
        policy_history_length: int = SPV2_POLICY_HISTORY_LENGTH,
        reference_fps: float = 50.0,
        keypoint_specs: Sequence[dict[str, Any]] = (),
        extra_actor_groups: Sequence[str] = (),
        extra_policy_input_dim: int = 0,
        raw_actor_obs_extra_dim: int = 0,
    ) -> None:
        del actor_core_group
        self.robot_root_quat_group = str(robot_root_quat_group)
        self.reference_encoder_input_group = str(reference_encoder_input_group)
        self.reference_encoder_target_group = str(reference_encoder_target_group)
        self.robot_key_body_group = str(robot_key_body_group)
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            estimator_hidden_dims=estimator_hidden_dims,
            estimator_activation=estimator_activation,
            actor_core_group=self.robot_root_quat_group,
            estimator_history_group=estimator_history_group,
            estimator_target_group=estimator_target_group,
            estimator_history_length=estimator_history_length,
            policy_history_length=policy_history_length,
            extra_actor_groups=(
                self.reference_encoder_input_group,
                self.robot_key_body_group,
                *(str(name) for name in extra_actor_groups),
            ),
            extra_policy_input_dim=(3 * SPV4_KEY_BODY_STATE_DIM + int(extra_policy_input_dim)),
            actor_core_expected_dim=SPV5_ROBOT_ROOT_QUAT_DIM,
        )
        expected_sizes = {
            self.reference_encoder_input_group: SPV5_REFERENCE_INPUT_DIM,
            self.robot_key_body_group: SPV4_KEY_BODY_STATE_DIM,
        }
        for name, expected in expected_sizes.items():
            actual = int(obs[name].shape[-1])
            if actual != expected:
                raise ValueError(
                    f"SPV5 observation group {name!r} has {actual} values, expected {expected}"
                )
        target_dim = int(obs[self.reference_encoder_target_group].shape[-1])
        if target_dim != SPV5_REFERENCE_TARGET_DIM:
            raise ValueError(
                f"SPV5 reference target has {target_dim} values, "
                f"expected {SPV5_REFERENCE_TARGET_DIM}"
            )
        expected_raw_obs_dim = SPV5_RAW_ACTOR_OBS_DIM + int(raw_actor_obs_extra_dim)
        if self.obs_dim != expected_raw_obs_dim:
            raise RuntimeError(
                f"SPV5 raw actor observation has {self.obs_dim} values, "
                f"expected {expected_raw_obs_dim}"
            )

        self.reference_input_normalizer = _identity_or_normalizer(
            self.obs_normalization, SPV5_REFERENCE_INPUT_DIM
        )
        self.reference_residual_normalizer = _identity_or_normalizer(
            self.obs_normalization, SPV5_REFERENCE_TARGET_DIM
        )
        self.reference_encoder = MLP(
            SPV5_REFERENCE_INPUT_DIM,
            SPV5_REFERENCE_TARGET_DIM,
            tuple(int(value) for value in reference_encoder_hidden_dims),
            reference_encoder_activation,
        )
        linear_layers = [
            module for module in self.reference_encoder.modules() if isinstance(module, nn.Linear)
        ]
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)
        self.reference_kinematics = SPV5ReferenceKinematics(keypoint_specs, reference_fps)

    @staticmethod
    def _noisy_support(reference_input: torch.Tensor) -> torch.Tensor:
        return reference_input[..., -SPV5_REFERENCE_TARGET_DIM:]

    def encode_reference(self, obs: TensorDict) -> torch.Tensor:
        reference_input = obs[self.reference_encoder_input_group]
        residual_normalized = self.reference_encoder(
            self.reference_input_normalizer(reference_input)
        )
        residual = _normalizer_inverse(self.reference_residual_normalizer, residual_normalized)
        return self._noisy_support(reference_input) + residual

    def _spv5_features(
        self,
        obs: TensorDict,
        estimate: torch.Tensor,
        decoded: torch.Tensor | None,
        reference_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        history = obs[self.estimator_history_group]
        return _spv5_policy_features(
            history=history,
            latest_proprio=self._latest_policy_proprio(history),
            estimate=estimate,
            decoded_reference=decoded,
            robot_root_quat=(obs[self.robot_root_quat_group] if reference_cache is None else None),
            robot_key_body=obs[self.robot_key_body_group],
            history_length=self.estimator_history_length,
            kinematics=self.reference_kinematics,
            reference_cache=reference_cache,
        )

    @torch.no_grad()
    def populate_policy_context_cache(self, obs: TensorDict) -> torch.Tensor:
        """Materialize one behavior-time context for rollout storage.

        PPO treats the supervised encoder and estimator outputs as observations,
        not policy parameters.  Storing this compact context makes those
        observations fixed for the rollout and removes all FK/encoder/estimator
        inference from the repeated learning minibatches.
        """
        estimate = self.estimate_root_state(obs)
        decoded = self.encode_reference(obs)
        reference = self.reference_kinematics(decoded, obs[self.robot_root_quat_group])
        context = torch.cat((_pack_reference_cache(reference), estimate), dim=-1).detach()
        if context.shape[-1] != SPV5_POLICY_CONTEXT_CACHE_DIM:
            raise RuntimeError(
                f"SPV5 policy context has {context.shape[-1]} values, "
                f"expected {SPV5_POLICY_CONTEXT_CACHE_DIM}"
            )
        obs.set(SPV5_POLICY_CONTEXT_CACHE_GROUP, context)
        return context

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        context = obs.get(SPV5_POLICY_CONTEXT_CACHE_GROUP)
        if context is not None:
            reference_cache = context[..., :SPV5_REFERENCE_CACHE_DIM]
            estimate = context[..., SPV5_REFERENCE_CACHE_DIM:]
            features = self._spv5_features(obs, estimate, None, reference_cache=reference_cache)
        else:
            estimate = self.estimate_root_state(obs).detach()
            # PPO must not update either supervised representation network.
            decoded = self.encode_reference(obs).detach()
            features = self._spv5_features(obs, estimate, decoded)
        return self.policy_normalizer(features)

    def reference_encoder_losses(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reference_input = obs[self.reference_encoder_input_group]
        target = obs[self.reference_encoder_target_group]
        noisy_support = self._noisy_support(reference_input)
        residual_target = target - noisy_support
        residual_prediction_normalized = self.reference_encoder(
            self.reference_input_normalizer(reference_input)
        )
        target_normalized = self.reference_residual_normalizer(residual_target)
        total_mse = (residual_prediction_normalized - target_normalized).square().mean()
        decoded = noisy_support + _normalizer_inverse(
            self.reference_residual_normalizer, residual_prediction_normalized
        )
        decoded_frames = decoded.reshape(
            *decoded.shape[:-1], len(SPV5_REFERENCE_SUPPORT_STEPS), SPV5_REFERENCE_FRAME_DIM
        )
        target_frames = target.reshape_as(decoded_frames)
        diagnostics = {
            "reference_encoder_mse": total_mse,
            "reference_root_pos_mse": (decoded_frames[..., :3] - target_frames[..., :3])
            .square()
            .mean(),
            "reference_root_rot6d_mse": (decoded_frames[..., 3:9] - target_frames[..., 3:9])
            .square()
            .mean(),
            "reference_joint_pos_mse": (decoded_frames[..., 9:] - target_frames[..., 9:])
            .square()
            .mean(),
        }
        return total_mse, diagnostics

    @torch.no_grad()
    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            self.populate_policy_context_cache(obs)
            return
        history = obs[self.estimator_history_group]
        reference_input = obs[self.reference_encoder_input_group]
        target = obs[self.reference_encoder_target_group]
        self.history_normalizer.update(history)  # type: ignore[attr-defined]
        self.reference_input_normalizer.update(reference_input)  # type: ignore[attr-defined]
        self.reference_residual_normalizer.update(  # type: ignore[attr-defined]
            target - self._noisy_support(reference_input)
        )
        context = self.populate_policy_context_cache(obs)
        estimate = context[..., SPV5_REFERENCE_CACHE_DIM:]
        self.policy_normalizer.update(  # type: ignore[attr-defined]
            self._spv5_features(
                obs,
                estimate,
                None,
                reference_cache=context[..., :SPV5_REFERENCE_CACHE_DIM],
            )
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _SPV5ActorExport(self)

    def as_jit(self) -> nn.Module:
        return _SPV5ActorExport(self)


class _SPV5ActorExport(nn.Module):
    is_recurrent = False

    def __init__(self, model: SPV5ReferenceEncoderActor) -> None:
        super().__init__()
        self.history_normalizer = copy.deepcopy(model.history_normalizer)
        self.policy_normalizer = copy.deepcopy(model.policy_normalizer)
        self.estimator = copy.deepcopy(model.estimator)
        self.reference_input_normalizer = copy.deepcopy(model.reference_input_normalizer)
        self.reference_residual_normalizer = copy.deepcopy(model.reference_residual_normalizer)
        self.reference_encoder = copy.deepcopy(model.reference_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.reference_kinematics = copy.deepcopy(model.reference_kinematics)
        self.input_size = model.obs_dim
        self.history_length = model.estimator_history_length
        self.policy_history_length = model.policy_history_length
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def _latest_policy_proprio(self, history: torch.Tensor) -> torch.Tensor:
        terms = []
        offset = 0
        for term_dim in PROPRIO_TERM_DIMS:
            size = self.history_length * term_dim
            term = history[..., offset : offset + size].reshape(
                *history.shape[:-1], self.history_length, term_dim
            )
            terms.append(
                term[..., -self.policy_history_length :, :].reshape(
                    *history.shape[:-1], self.policy_history_length * term_dim
                )
            )
            offset += size
        return torch.cat(terms, dim=-1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        quat_end = SPV5_ROBOT_ROOT_QUAT_DIM
        history_end = quat_end + self.history_length * sum(PROPRIO_TERM_DIMS)
        reference_end = history_end + SPV5_REFERENCE_INPUT_DIM
        robot_root_quat = value[..., :quat_end]
        history = value[..., quat_end:history_end]
        reference_input = value[..., history_end:reference_end]
        robot_key_body = value[..., reference_end:]

        estimate = self.estimator(self.history_normalizer(history))
        residual_normalized = self.reference_encoder(
            self.reference_input_normalizer(reference_input)
        )
        decoded = reference_input[..., -SPV5_REFERENCE_TARGET_DIM:] + (
            _normalizer_inverse(self.reference_residual_normalizer, residual_normalized)
        )
        features = _spv5_policy_features(
            history=history,
            latest_proprio=self._latest_policy_proprio(history),
            estimate=estimate,
            decoded_reference=decoded,
            robot_root_quat=robot_root_quat,
            robot_key_body=robot_key_body,
            history_length=self.history_length,
            kinematics=self.reference_kinematics,
        )
        return self.deterministic_output(self.mlp(self.policy_normalizer(features)))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["spv5_observation"]

    @property
    def deploy_input_names(self) -> list[str]:
        return ["spv5_observation"]

    @property
    def output_names(self) -> list[str]:
        return ["action"]
