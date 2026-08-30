"""SPV4 privileged key-body actor built on the SPV3 estimator policy."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from rsl_rl.modules import HiddenState
from tensordict import TensorDict

from intact_tracking.environment.mdp.spv4 import SPV4_KEY_BODY_STATE_DIM

from .spv3_models import (
    PROPRIO_TERM_DIMS,
    SPV2_POLICY_HISTORY_LENGTH,
    SPV3_ESTIMATOR_HISTORY_LENGTH,
    SPV3_POLICY_INPUT_DIM,
    SPV3EstimatorActor,
    _SPV3ActorExport,
)

SPV3_PROPRIO_WITH_ESTIMATE_DIM = 614
SPV3_REFERENCE_COMMAND_DIM = 382
SPV4_POLICY_INPUT_DIM = SPV3_POLICY_INPUT_DIM + 3 * SPV4_KEY_BODY_STATE_DIM


def _assemble_spv4_features(
    spv3_features: torch.Tensor,
    robot_key_body: torch.Tensor,
    ref_key_body: torch.Tensor,
    key_body_error: torch.Tensor,
) -> torch.Tensor:
    proprio_end = SPV3_PROPRIO_WITH_ESTIMATE_DIM
    reference_end = proprio_end + SPV3_REFERENCE_COMMAND_DIM
    return torch.cat(
        (
            spv3_features[..., :proprio_end],
            robot_key_body,
            spv3_features[..., proprio_end:reference_end],
            ref_key_body,
            spv3_features[..., reference_end:],
            key_body_error,
        ),
        dim=-1,
    )


class SPV4KeyBodyActor(SPV3EstimatorActor):
    """SPV3 plus current privileged robot/reference/error key-body states."""

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
        actor_core_group: str = "actor_core",
        estimator_history_group: str = "estimator_history",
        estimator_target_group: str = "estimator_target",
        estimator_history_length: int = SPV3_ESTIMATOR_HISTORY_LENGTH,
        policy_history_length: int = SPV2_POLICY_HISTORY_LENGTH,
        robot_key_body_group: str = "robot_key_body",
        ref_key_body_group: str = "ref_key_body",
        key_body_error_group: str = "key_body_error",
    ) -> None:
        self.robot_key_body_group = str(robot_key_body_group)
        self.ref_key_body_group = str(ref_key_body_group)
        self.key_body_error_group = str(key_body_error_group)
        extra_groups = (
            self.robot_key_body_group,
            self.ref_key_body_group,
            self.key_body_error_group,
        )
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
            actor_core_group=actor_core_group,
            estimator_history_group=estimator_history_group,
            estimator_target_group=estimator_target_group,
            estimator_history_length=estimator_history_length,
            policy_history_length=policy_history_length,
            extra_actor_groups=extra_groups,
            extra_policy_input_dim=3 * SPV4_KEY_BODY_STATE_DIM,
        )
        for name in extra_groups:
            size = int(obs[name].shape[-1])
            if size != SPV4_KEY_BODY_STATE_DIM:
                raise ValueError(
                    f"SPV4 observation group {name!r} has {size} values, "
                    f"expected {SPV4_KEY_BODY_STATE_DIM}"
                )
        if self.policy_input_dim != SPV4_POLICY_INPUT_DIM:
            raise RuntimeError(
                f"SPV4 policy input has {self.policy_input_dim} values, "
                f"expected {SPV4_POLICY_INPUT_DIM}"
            )

    def _spv4_features(
        self,
        obs: TensorDict,
        estimate: torch.Tensor,
    ) -> torch.Tensor:
        spv3_features = self._policy_features(
            obs[self.actor_core_group],
            obs[self.estimator_history_group],
            estimate,
        )
        features = _assemble_spv4_features(
            spv3_features,
            obs[self.robot_key_body_group],
            obs[self.ref_key_body_group],
            obs[self.key_body_error_group],
        )
        if features.shape[-1] != SPV4_POLICY_INPUT_DIM:
            raise RuntimeError(
                f"SPV4 policy features have {features.shape[-1]} values, "
                f"expected {SPV4_POLICY_INPUT_DIM}"
            )
        return features

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        estimate = self.estimate_root_state(obs).detach()
        return self.policy_normalizer(self._spv4_features(obs, estimate))

    @torch.no_grad()
    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return
        history = obs[self.estimator_history_group]
        self.history_normalizer.update(history)  # type: ignore[attr-defined]
        estimate = self.estimate_root_state(obs)
        self.policy_normalizer.update(  # type: ignore[attr-defined]
            self._spv4_features(obs, estimate)
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _SPV4ActorExport(self)

    def as_jit(self) -> nn.Module:
        return _SPV4ActorExport(self)


class _SPV4ActorExport(_SPV3ActorExport):
    def __init__(self, model: SPV4KeyBodyActor) -> None:
        super().__init__(model)
        self.key_body_dim = SPV4_KEY_BODY_STATE_DIM

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        core_end = self.core_dim
        history_end = core_end + self.estimator_history_length * sum(PROPRIO_TERM_DIMS)
        robot_end = history_end + self.key_body_dim
        reference_end = robot_end + self.key_body_dim
        core = value[..., :core_end]
        history = value[..., core_end:history_end]
        robot_key_body = value[..., history_end:robot_end]
        ref_key_body = value[..., robot_end:reference_end]
        key_body_error = value[..., reference_end:]

        estimate = self.estimator(self.history_normalizer(history))
        spv3_features = self._policy_features(core, history, estimate)
        features = _assemble_spv4_features(
            spv3_features,
            robot_key_body,
            ref_key_body,
            key_body_error,
        )
        output = self.mlp(self.policy_normalizer(features))
        return self.deterministic_output(output)

    @property
    def input_names(self) -> list[str]:
        return ["spv4_privileged_observation"]

    @property
    def deploy_input_names(self) -> list[str]:
        return ["spv4_privileged_observation"]
