"""SPV5-2 actor with height/contact estimation and no velocity estimator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict

from intact_tracking.environment.mdp.spv4 import _key_body_error

from .spv3_models import (
    PROPRIO_TERM_DIMS,
    SPV2_POLICY_HISTORY_LENGTH,
    SPV3_ESTIMATOR_HISTORY_LENGTH,
)
from .spv5_1_models import SPV5_1_CONTACT_DIM
from .spv5_models import (
    SPV5_POLICY_INPUT_DIM,
    SPV5_REFERENCE_CACHE_DIM,
    SPV5_REFERENCE_INPUT_DIM,
    SPV5_REFERENCE_TARGET_DIM,
    SPV5_ROBOT_ROOT_QUAT_DIM,
    SPV5ReferenceEncoderActor,
    _current_proprio,
    _normalizer_inverse,
    _pack_reference_cache,
    _reference_features_from_cache,
    _SPV5ActorExport,
    _unpack_key_body,
)

SPV5_2_HEIGHT_DIM = 1
SPV5_2_BASE_POLICY_INPUT_DIM = SPV5_POLICY_INPUT_DIM - 6
SPV5_2_POLICY_INPUT_DIM = SPV5_2_BASE_POLICY_INPUT_DIM + SPV5_1_CONTACT_DIM
SPV5_2_POLICY_CONTEXT_CACHE_DIM = SPV5_REFERENCE_CACHE_DIM + SPV5_2_HEIGHT_DIM + SPV5_1_CONTACT_DIM
SPV5_2_POLICY_CONTEXT_CACHE_GROUP = "spv5_2_policy_context_cache"


class SPV52HeightContactEstimator(nn.Module):
    """Shared history backbone with height and foot-contact heads only."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in hidden_dims)
        if len(widths) < 3:
            raise ValueError(
                "SPV5-2 estimator_hidden_dims needs at least three entries: "
                "shared input, shared latent, and per-head hidden width"
            )
        shared_dim = widths[-2]
        head_dim = widths[-1]
        self.shared_backbone = MLP(
            input_dim,
            shared_dim,
            widths[:-2],
            activation,
            last_activation=activation,
        )
        self.height_head = MLP(
            shared_dim,
            SPV5_2_HEIGHT_DIM,
            (head_dim,),
            activation,
        )
        self.contact_head = MLP(
            shared_dim,
            SPV5_1_CONTACT_DIM,
            (head_dim,),
            activation,
        )

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_backbone(history)
        return self.height_head(shared), self.contact_head(shared)


def _spv5_2_base_policy_features(
    *,
    history: torch.Tensor,
    latest_proprio: torch.Tensor,
    height_estimate: torch.Tensor,
    decoded_reference: torch.Tensor | None,
    robot_root_quat: torch.Tensor | None,
    robot_key_body: torch.Tensor,
    history_length: int,
    kinematics,
    reference_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build SPV5 features after removing estimated velocity and its error."""
    if height_estimate.shape[-1] != SPV5_2_HEIGHT_DIM:
        raise ValueError(
            f"SPV5-2 height estimate has {height_estimate.shape[-1]} values, "
            f"expected {SPV5_2_HEIGHT_DIM}"
        )
    if reference_cache is None:
        if decoded_reference is None or robot_root_quat is None:
            raise ValueError("SPV5-2 uncached policy features require decoded reference")
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
    height_error = reference.height_current - height_estimate

    # Keep SPV5's semantic ordering while shortening the estimator-dependent
    # sections from [height, vx, vy, vz] to [height].
    robot_key_state = _unpack_key_body(robot_key_body)
    key_error = _key_body_error(
        robot_key_state,
        reference.key_state,
        reference.robot_from_reference_root,
    )
    features = torch.cat(
        (
            latest_proprio,
            height_estimate,
            robot_key_body,
            reference.standard,
            reference.key_body,
            existing_error,
            height_error,
            key_error,
        ),
        dim=-1,
    )
    if features.shape[-1] != SPV5_2_BASE_POLICY_INPUT_DIM:
        raise RuntimeError(
            f"SPV5-2 base policy features have {features.shape[-1]} values, "
            f"expected {SPV5_2_BASE_POLICY_INPUT_DIM}"
        )
    return features


class SPV52HeightContactEstimatorActor(SPV5ReferenceEncoderActor):
    """SPV5-1 successor that estimates height and contact, but not velocity."""

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
        foot_contact_target_group: str = "foot_contact_target",
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
        self.foot_contact_target_group = str(foot_contact_target_group)
        policy_dim_delta = SPV5_2_POLICY_INPUT_DIM - SPV5_POLICY_INPUT_DIM
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
            reference_encoder_hidden_dims=reference_encoder_hidden_dims,
            reference_encoder_activation=reference_encoder_activation,
            actor_core_group=actor_core_group,
            robot_root_quat_group=robot_root_quat_group,
            estimator_history_group=estimator_history_group,
            estimator_target_group=estimator_target_group,
            reference_encoder_input_group=reference_encoder_input_group,
            reference_encoder_target_group=reference_encoder_target_group,
            robot_key_body_group=robot_key_body_group,
            estimator_history_length=estimator_history_length,
            policy_history_length=policy_history_length,
            reference_fps=reference_fps,
            keypoint_specs=keypoint_specs,
            extra_actor_groups=extra_actor_groups,
            extra_policy_input_dim=policy_dim_delta + int(extra_policy_input_dim),
            raw_actor_obs_extra_dim=int(raw_actor_obs_extra_dim),
        )
        target_dim = int(obs[self.estimator_target_group].shape[-1])
        if target_dim != SPV5_2_HEIGHT_DIM:
            raise ValueError(
                f"SPV5-2 estimator target has {target_dim} values, expected {SPV5_2_HEIGHT_DIM}"
            )
        contact_dim = int(obs[self.foot_contact_target_group].shape[-1])
        if contact_dim != SPV5_1_CONTACT_DIM:
            raise ValueError(
                f"SPV5-2 contact target has {contact_dim} values, expected {SPV5_1_CONTACT_DIM}"
            )
        expected_policy_input_dim = SPV5_2_POLICY_INPUT_DIM + int(extra_policy_input_dim)
        if self.policy_input_dim != expected_policy_input_dim:
            raise RuntimeError(
                f"SPV5-2 policy input has {self.policy_input_dim} values, "
                f"expected {expected_policy_input_dim}"
            )
        self.estimator = SPV52HeightContactEstimator(
            self.estimator_history_dim,
            hidden_dims=estimator_hidden_dims,
            activation=estimator_activation,
        )

    def estimate_height_and_contact(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        history = obs[self.estimator_history_group]
        return self.estimator(self.history_normalizer(history))

    def estimate_root_state(self, obs: TensorDict) -> torch.Tensor:
        height, _ = self.estimate_height_and_contact(obs)
        return height

    def _spv5_2_features(
        self,
        obs: TensorDict,
        height_estimate: torch.Tensor,
        contact_probability: torch.Tensor,
        decoded: torch.Tensor | None,
        reference_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        history = obs[self.estimator_history_group]
        base = _spv5_2_base_policy_features(
            history=history,
            latest_proprio=self._latest_policy_proprio(history),
            height_estimate=height_estimate,
            decoded_reference=decoded,
            robot_root_quat=(obs[self.robot_root_quat_group] if reference_cache is None else None),
            robot_key_body=obs[self.robot_key_body_group],
            history_length=self.estimator_history_length,
            kinematics=self.reference_kinematics,
            reference_cache=reference_cache,
        )
        features = torch.cat((base, contact_probability), dim=-1)
        if features.shape[-1] != SPV5_2_POLICY_INPUT_DIM:
            raise RuntimeError(
                f"SPV5-2 policy features have {features.shape[-1]} values, "
                f"expected {SPV5_2_POLICY_INPUT_DIM}"
            )
        return features

    @torch.no_grad()
    def populate_policy_context_cache(self, obs: TensorDict) -> torch.Tensor:
        height, contact_logits = self.estimate_height_and_contact(obs)
        decoded = self.encode_reference(obs)
        reference = self.reference_kinematics(decoded, obs[self.robot_root_quat_group])
        context = torch.cat(
            (
                _pack_reference_cache(reference),
                height,
                contact_logits.sigmoid(),
            ),
            dim=-1,
        ).detach()
        if context.shape[-1] != SPV5_2_POLICY_CONTEXT_CACHE_DIM:
            raise RuntimeError(
                f"SPV5-2 policy context has {context.shape[-1]} values, "
                f"expected {SPV5_2_POLICY_CONTEXT_CACHE_DIM}"
            )
        obs.set(SPV5_2_POLICY_CONTEXT_CACHE_GROUP, context)
        return context

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        context = obs.get(SPV5_2_POLICY_CONTEXT_CACHE_GROUP)
        if context is not None:
            reference_cache = context[..., :SPV5_REFERENCE_CACHE_DIM]
            height_end = SPV5_REFERENCE_CACHE_DIM + SPV5_2_HEIGHT_DIM
            height = context[..., SPV5_REFERENCE_CACHE_DIM:height_end]
            contact_probability = context[..., height_end:]
            features = self._spv5_2_features(
                obs,
                height,
                contact_probability,
                None,
                reference_cache=reference_cache,
            )
        else:
            height, contact_logits = self.estimate_height_and_contact(obs)
            decoded = self.encode_reference(obs)
            features = self._spv5_2_features(
                obs,
                height.detach(),
                contact_logits.sigmoid().detach(),
                decoded.detach(),
            )
        return self.policy_normalizer(features)

    def height_contact_losses(
        self, obs: TensorDict
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        height, contact_logits = self.estimate_height_and_contact(obs)
        height_target = obs[self.estimator_target_group]
        contact_target = obs[self.foot_contact_target_group].float()
        if height_target.shape[-1] != SPV5_2_HEIGHT_DIM:
            raise ValueError(
                f"SPV5-2 height target has {height_target.shape[-1]} values, expected 1"
            )
        if contact_target.shape[-1] != SPV5_1_CONTACT_DIM:
            raise ValueError(
                f"SPV5-2 contact target has {contact_target.shape[-1]} values, expected 2"
            )
        height_mse = (height - height_target).square().mean()
        contact_bce = F.binary_cross_entropy_with_logits(contact_logits, contact_target)

        with torch.no_grad():
            predicted = contact_logits >= 0.0
            positive = contact_target >= 0.5
            true_positive = (predicted & positive).float().sum()
            predicted_positive = predicted.float().sum()
            target_positive = positive.float().sum()
            precision = true_positive / predicted_positive.clamp_min(1.0)
            recall = true_positive / target_positive.clamp_min(1.0)
            diagnostics = {
                "estimator_foot_contact_accuracy": (predicted == positive).float().mean(),
                "estimator_foot_contact_precision": precision,
                "estimator_foot_contact_recall": recall,
                "estimator_foot_contact_f1": (
                    2.0 * precision * recall / (precision + recall).clamp_min(1.0e-8)
                ),
                "estimator_foot_contact_target_rate": positive.float().mean(),
                "estimator_foot_contact_pred_rate": predicted.float().mean(),
            }
        return height_mse, contact_bce, diagnostics

    @torch.no_grad()
    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            self.populate_policy_context_cache(obs)
            return
        history = obs[self.estimator_history_group]
        reference_input = obs[self.reference_encoder_input_group]
        target = obs[self.reference_encoder_target_group]
        self.history_normalizer.update(history)  # type: ignore[attr-defined]
        self.reference_input_normalizer.update(  # type: ignore[attr-defined]
            reference_input
        )
        self.reference_residual_normalizer.update(  # type: ignore[attr-defined]
            target - self._noisy_support(reference_input)
        )
        context = self.populate_policy_context_cache(obs)
        height_end = SPV5_REFERENCE_CACHE_DIM + SPV5_2_HEIGHT_DIM
        self.policy_normalizer.update(  # type: ignore[attr-defined]
            self._spv5_2_features(
                obs,
                context[..., SPV5_REFERENCE_CACHE_DIM:height_end],
                context[..., height_end:],
                None,
                reference_cache=context[..., :SPV5_REFERENCE_CACHE_DIM],
            )
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _SPV52ActorExport(self)

    def as_jit(self) -> nn.Module:
        return _SPV52ActorExport(self)


class _SPV52ActorExport(_SPV5ActorExport):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        quat_end = SPV5_ROBOT_ROOT_QUAT_DIM
        history_end = quat_end + self.history_length * sum(PROPRIO_TERM_DIMS)
        reference_end = history_end + SPV5_REFERENCE_INPUT_DIM
        robot_root_quat = value[..., :quat_end]
        history = value[..., quat_end:history_end]
        reference_input = value[..., history_end:reference_end]
        robot_key_body = value[..., reference_end:]

        height, contact_logits = self.estimator(self.history_normalizer(history))
        residual_normalized = self.reference_encoder(
            self.reference_input_normalizer(reference_input)
        )
        decoded = reference_input[..., -SPV5_REFERENCE_TARGET_DIM:] + (
            _normalizer_inverse(self.reference_residual_normalizer, residual_normalized)
        )
        base_features = _spv5_2_base_policy_features(
            history=history,
            latest_proprio=self._latest_policy_proprio(history),
            height_estimate=height,
            decoded_reference=decoded,
            robot_root_quat=robot_root_quat,
            robot_key_body=robot_key_body,
            history_length=self.history_length,
            kinematics=self.reference_kinematics,
        )
        features = torch.cat((base_features, contact_logits.sigmoid()), dim=-1)
        return self.deterministic_output(self.mlp(self.policy_normalizer(features)))

    @property
    def input_names(self) -> list[str]:
        return ["spv5_2_observation"]

    @property
    def deploy_input_names(self) -> list[str]:
        return ["spv5_2_observation"]
