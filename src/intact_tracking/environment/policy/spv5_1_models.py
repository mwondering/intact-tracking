"""SPV5-1 actor with a jointly supervised root/contact estimator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict

from .residual_moe import ObservationConditionedResidualMoE
from .spv3_models import (
    PROPRIO_TERM_DIMS,
    SPV2_POLICY_HISTORY_LENGTH,
    SPV3_ESTIMATOR_HISTORY_LENGTH,
    SPV3_ESTIMATOR_OUTPUT_DIM,
)
from .spv5_models import (
    SPV5_POLICY_INPUT_DIM,
    SPV5_REFERENCE_CACHE_DIM,
    SPV5_REFERENCE_INPUT_DIM,
    SPV5_REFERENCE_TARGET_DIM,
    SPV5_ROBOT_ROOT_QUAT_DIM,
    SPV5ReferenceEncoderActor,
    _normalizer_inverse,
    _pack_reference_cache,
    _spv5_policy_features,
    _SPV5ActorExport,
)

SPV5_1_CONTACT_DIM = 2
SPV5_1_POLICY_INPUT_DIM = SPV5_POLICY_INPUT_DIM + SPV5_1_CONTACT_DIM
SPV5_1_POLICY_CONTEXT_CACHE_DIM = (
    SPV5_REFERENCE_CACHE_DIM + SPV3_ESTIMATOR_OUTPUT_DIM + SPV5_1_CONTACT_DIM
)
SPV5_1_POLICY_CONTEXT_CACHE_GROUP = "spv5_1_policy_context_cache"


class SPV51ContactEstimator(nn.Module):
    """Shared history backbone with regression and contact-classification heads."""

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
                "SPV5-1 estimator_hidden_dims needs at least three entries: "
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
        self.root_head = MLP(
            shared_dim,
            SPV3_ESTIMATOR_OUTPUT_DIM,
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
        return self.root_head(shared), self.contact_head(shared)


class SPV51ContactEstimatorActor(SPV5ReferenceEncoderActor):
    """SPV5 policy augmented with estimated left/right contact probabilities."""

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
    ) -> None:
        self.foot_contact_target_group = str(foot_contact_target_group)
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
            extra_policy_input_dim=SPV5_1_CONTACT_DIM,
        )
        target_dim = int(obs[self.foot_contact_target_group].shape[-1])
        if target_dim != SPV5_1_CONTACT_DIM:
            raise ValueError(
                f"SPV5-1 contact target has {target_dim} values, expected {SPV5_1_CONTACT_DIM}"
            )
        if self.policy_input_dim != SPV5_1_POLICY_INPUT_DIM:
            raise RuntimeError(
                f"SPV5-1 policy input has {self.policy_input_dim} values, "
                f"expected {SPV5_1_POLICY_INPUT_DIM}"
            )
        self.estimator = SPV51ContactEstimator(
            self.estimator_history_dim,
            hidden_dims=estimator_hidden_dims,
            activation=estimator_activation,
        )

    def estimate_root_and_contact(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        history = obs[self.estimator_history_group]
        return self.estimator(self.history_normalizer(history))

    def estimate_root_state(self, obs: TensorDict) -> torch.Tensor:
        estimate, _ = self.estimate_root_and_contact(obs)
        return estimate

    def _spv5_1_features(
        self,
        obs: TensorDict,
        estimate: torch.Tensor,
        contact_probability: torch.Tensor,
        decoded: torch.Tensor | None,
        reference_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self._spv5_features(obs, estimate, decoded, reference_cache=reference_cache)
        features = torch.cat((base, contact_probability), dim=-1)
        if features.shape[-1] != SPV5_1_POLICY_INPUT_DIM:
            raise RuntimeError(
                f"SPV5-1 policy features have {features.shape[-1]} values, "
                f"expected {SPV5_1_POLICY_INPUT_DIM}"
            )
        return features

    @torch.no_grad()
    def populate_policy_context_cache(self, obs: TensorDict) -> torch.Tensor:
        estimate, contact_logits = self.estimate_root_and_contact(obs)
        decoded = self.encode_reference(obs)
        reference = self.reference_kinematics(decoded, obs[self.robot_root_quat_group])
        context = torch.cat(
            (
                _pack_reference_cache(reference),
                estimate,
                contact_logits.sigmoid(),
            ),
            dim=-1,
        ).detach()
        if context.shape[-1] != SPV5_1_POLICY_CONTEXT_CACHE_DIM:
            raise RuntimeError(
                f"SPV5-1 policy context has {context.shape[-1]} values, "
                f"expected {SPV5_1_POLICY_CONTEXT_CACHE_DIM}"
            )
        obs.set(SPV5_1_POLICY_CONTEXT_CACHE_GROUP, context)
        return context

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        context = obs.get(SPV5_1_POLICY_CONTEXT_CACHE_GROUP)
        if context is not None:
            reference_cache = context[..., :SPV5_REFERENCE_CACHE_DIM]
            estimate_end = SPV5_REFERENCE_CACHE_DIM + SPV3_ESTIMATOR_OUTPUT_DIM
            estimate = context[..., SPV5_REFERENCE_CACHE_DIM:estimate_end]
            contact_probability = context[..., estimate_end:]
            features = self._spv5_1_features(
                obs,
                estimate,
                contact_probability,
                None,
                reference_cache=reference_cache,
            )
        else:
            estimate, contact_logits = self.estimate_root_and_contact(obs)
            decoded = self.encode_reference(obs)
            # PPO must not update either supervised representation through policy
            # gradients; both heads are optimized only by their explicit losses.
            features = self._spv5_1_features(
                obs,
                estimate.detach(),
                contact_logits.sigmoid().detach(),
                decoded.detach(),
            )
        return self.policy_normalizer(features)

    def estimator_losses(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        height_mse, lin_vel_mse, _, _ = self.estimator_contact_losses(obs)
        return height_mse, lin_vel_mse

    def estimator_contact_losses(
        self, obs: TensorDict
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        estimate, contact_logits = self.estimate_root_and_contact(obs)
        root_target = obs[self.estimator_target_group]
        contact_target = obs[self.foot_contact_target_group].float()
        if root_target.shape[-1] != SPV3_ESTIMATOR_OUTPUT_DIM:
            raise ValueError(f"SPV5-1 root target has {root_target.shape[-1]} values, expected 4")
        if contact_target.shape[-1] != SPV5_1_CONTACT_DIM:
            raise ValueError(
                f"SPV5-1 contact target has {contact_target.shape[-1]} values, "
                f"expected {SPV5_1_CONTACT_DIM}"
            )
        height_mse = (estimate[..., :1] - root_target[..., :1]).square().mean()
        lin_vel_mse = (estimate[..., 1:] - root_target[..., 1:]).square().mean()
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
        return height_mse, lin_vel_mse, contact_bce, diagnostics

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
        estimate_end = SPV5_REFERENCE_CACHE_DIM + SPV3_ESTIMATOR_OUTPUT_DIM
        self.policy_normalizer.update(  # type: ignore[attr-defined]
            self._spv5_1_features(
                obs,
                context[..., SPV5_REFERENCE_CACHE_DIM:estimate_end],
                context[..., estimate_end:],
                None,
                reference_cache=context[..., :SPV5_REFERENCE_CACHE_DIM],
            )
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _SPV51ActorExport(self)

    def as_jit(self) -> nn.Module:
        return _SPV51ActorExport(self)


class SPV51ContactEstimatorMoEActor(SPV51ContactEstimatorActor):
    """SPV5-1 actor whose policy MLP is replaced by a residual MoE core."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        moe_context_hidden_dim: int = 1472,
        moe_hidden_dim: int = 608,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_expansion: int = 4,
        moe_router_temperature: float = 1.5,
        moe_router_init_std: float = 1.0e-2,
        moe_output_init_gain: float = 5.0e-2,
        **kwargs,
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        baseline_parameter_count = sum(parameter.numel() for parameter in self.mlp.parameters())
        policy_output_dim = (
            self.distribution.input_dim if self.distribution is not None else int(output_dim)
        )
        self.mlp = ObservationConditionedResidualMoE(
            self.policy_input_dim,
            policy_output_dim,
            context_hidden_dim=moe_context_hidden_dim,
            hidden_dim=moe_hidden_dim,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            expansion=moe_expansion,
            router_temperature=moe_router_temperature,
            router_init_std=moe_router_init_std,
            output_init_gain=moe_output_init_gain,
        )
        self.baseline_policy_parameter_count = baseline_parameter_count

    def routing_probabilities(self, obs: TensorDict) -> torch.Tensor:
        """Return dense probabilities over all experts for routing losses."""
        return self.mlp.routing_probabilities(self.get_latent(obs))

    @property
    def moe_policy_parameter_count(self) -> int:
        return self.mlp.dense_parameter_count


class _SPV51ActorExport(_SPV5ActorExport):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        quat_end = SPV5_ROBOT_ROOT_QUAT_DIM
        history_end = quat_end + self.history_length * sum(PROPRIO_TERM_DIMS)
        reference_end = history_end + SPV5_REFERENCE_INPUT_DIM
        robot_root_quat = value[..., :quat_end]
        history = value[..., quat_end:history_end]
        reference_input = value[..., history_end:reference_end]
        robot_key_body = value[..., reference_end:]

        estimate, contact_logits = self.estimator(self.history_normalizer(history))
        residual_normalized = self.reference_encoder(
            self.reference_input_normalizer(reference_input)
        )
        decoded = reference_input[..., -SPV5_REFERENCE_TARGET_DIM:] + (
            _normalizer_inverse(self.reference_residual_normalizer, residual_normalized)
        )
        base_features = _spv5_policy_features(
            history=history,
            latest_proprio=self._latest_policy_proprio(history),
            estimate=estimate,
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
        return ["spv5_1_observation"]

    @property
    def deploy_input_names(self) -> list[str]:
        return ["spv5_1_observation"]
