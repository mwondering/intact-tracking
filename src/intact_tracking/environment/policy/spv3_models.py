"""SPV3 actor with a supervised root-state estimator."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories
from tensordict import TensorDict

PROPRIO_TERM_DIMS = (29, 29, 3, 3, 29, 29)
SPV2_POLICY_HISTORY_LENGTH = 5
SPV3_ESTIMATOR_HISTORY_LENGTH = 50
SPV3_REFERENCE_DIM = 382
SPV3_EXISTING_ERROR_DIM = 64
SPV3_ACTOR_CORE_DIM = SPV3_REFERENCE_DIM + SPV3_EXISTING_ERROR_DIM
SPV3_ESTIMATOR_OUTPUT_DIM = 4
SPV3_POLICY_INPUT_DIM = 1064


def _identity_or_normalizer(enabled: bool, size: int) -> nn.Module:
    return EmpiricalNormalization(size) if enabled else nn.Identity()


class SPV3EstimatorActor(nn.Module):
    """SPV2 policy augmented with supervised height and body-velocity estimates.

    The environment exposes a single 50-frame, term-major proprioceptive history.
    The estimator consumes all 50 frames while the policy reuses only its newest
    five frames.  PPO sees detached estimates; only the explicit estimator MSE
    loss updates the estimator network.
    """

    is_recurrent = False

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
        extra_actor_groups: Sequence[str] = (),
        extra_policy_input_dim: int = 0,
        actor_core_expected_dim: int = SPV3_ACTOR_CORE_DIM,
    ) -> None:
        super().__init__()
        self.obs_groups = list(obs_groups[obs_set])
        self.actor_core_group = str(actor_core_group)
        self.estimator_history_group = str(estimator_history_group)
        self.estimator_target_group = str(estimator_target_group)
        expected_groups = [
            self.actor_core_group,
            self.estimator_history_group,
            *(str(name) for name in extra_actor_groups),
        ]
        if self.obs_groups != expected_groups:
            raise ValueError(
                f"SPV3EstimatorActor requires actor observation groups {expected_groups}, "
                f"got {self.obs_groups}"
            )

        self.estimator_history_length = int(estimator_history_length)
        self.policy_history_length = int(policy_history_length)
        if self.estimator_history_length != SPV3_ESTIMATOR_HISTORY_LENGTH:
            raise ValueError("SPV3 estimator history must contain exactly 50 frames")
        if self.policy_history_length != SPV2_POLICY_HISTORY_LENGTH:
            raise ValueError("SPV3 policy history must contain exactly 5 frames")

        self.proprio_frame_dim = sum(PROPRIO_TERM_DIMS)
        self.estimator_history_dim = self.proprio_frame_dim * self.estimator_history_length
        self.actor_core_dim = int(obs[self.actor_core_group].shape[-1])
        history_dim = int(obs[self.estimator_history_group].shape[-1])
        if self.actor_core_dim != int(actor_core_expected_dim):
            raise ValueError(
                f"SPV3 actor_core has {self.actor_core_dim} values, "
                f"expected {int(actor_core_expected_dim)}"
            )
        if history_dim != self.estimator_history_dim:
            raise ValueError(
                f"SPV3 estimator_history has {history_dim} values, "
                f"expected {self.estimator_history_dim}"
            )
        self.obs_dim = sum(int(obs[name].shape[-1]) for name in self.obs_groups)
        self.policy_input_dim = SPV3_POLICY_INPUT_DIM + int(extra_policy_input_dim)
        self.obs_normalization = bool(obs_normalization)
        self.history_normalizer = _identity_or_normalizer(
            self.obs_normalization, self.estimator_history_dim
        )
        self.policy_normalizer = _identity_or_normalizer(
            self.obs_normalization, self.policy_input_dim
        )

        self.estimator = MLP(
            self.estimator_history_dim,
            SPV3_ESTIMATOR_OUTPUT_DIM,
            tuple(int(v) for v in estimator_hidden_dims),
            estimator_activation,
        )

        cfg = dict(distribution_cfg or {})
        distribution_class = cfg.pop("class_name", None)
        if distribution_class is not None:
            dist_cls: type[Distribution] = resolve_callable(distribution_class)
            self.distribution: Distribution | None = dist_cls(output_dim, **cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim
        self.mlp = MLP(
            self.policy_input_dim,
            mlp_output_dim,
            tuple(int(v) for v in hidden_dims),
            activation,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def _latest_policy_proprio(self, history: torch.Tensor) -> torch.Tensor:
        terms = []
        offset = 0
        for term_dim in PROPRIO_TERM_DIMS:
            size = self.estimator_history_length * term_dim
            term = history[..., offset : offset + size].reshape(
                *history.shape[:-1], self.estimator_history_length, term_dim
            )
            terms.append(
                term[..., -self.policy_history_length :, :].reshape(
                    *history.shape[:-1], self.policy_history_length * term_dim
                )
            )
            offset += size
        return torch.cat(terms, dim=-1)

    @staticmethod
    def _reference_lin_vel_in_robot_frame(core: torch.Tensor) -> torch.Tensor:
        # actor_core starts with root_pos(12), then five relative rotations(5*6).
        rot6d = core[..., 12:18]
        first_col = rot6d[..., :3]
        second_col = rot6d[..., 3:6]
        third_col = torch.linalg.cross(first_col, second_col, dim=-1)
        robot_from_reference = torch.stack((first_col, second_col, third_col), dim=-1)
        # ref_root_lin_vel starts after root_pos(12), root_ori(30), height(5).
        ref_lin_vel_reference = core[..., 47:50]
        return torch.matmul(robot_from_reference, ref_lin_vel_reference.unsqueeze(-1)).squeeze(-1)

    def estimate_root_state(self, obs: TensorDict) -> torch.Tensor:
        history = obs[self.estimator_history_group]
        return self.estimator(self.history_normalizer(history))

    def _policy_features(
        self,
        core: torch.Tensor,
        history: torch.Tensor,
        estimate: torch.Tensor,
    ) -> torch.Tensor:
        latest_proprio = self._latest_policy_proprio(history)
        ref_height = core[..., 42:43]
        ref_lin_vel_robot = self._reference_lin_vel_in_robot_frame(core)
        root_state_error = torch.cat(
            (
                ref_height - estimate[..., :1],
                ref_lin_vel_robot - estimate[..., 1:],
            ),
            dim=-1,
        )
        reference = core[..., :SPV3_REFERENCE_DIM]
        existing_error = core[..., SPV3_REFERENCE_DIM:]
        features = torch.cat(
            (
                latest_proprio,
                estimate,
                reference,
                existing_error,
                root_state_error,
            ),
            dim=-1,
        )
        if features.shape[-1] != SPV3_POLICY_INPUT_DIM:
            raise RuntimeError(
                f"SPV3 policy features have {features.shape[-1]} values, "
                f"expected {SPV3_POLICY_INPUT_DIM}"
            )
        return features

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        core = obs[self.actor_core_group]
        history = obs[self.estimator_history_group]
        # This detach is the explicit boundary that prevents PPO gradients from
        # changing the physical estimator.  estimator_losses() retains its graph.
        estimate = self.estimate_root_state(obs).detach()
        return self.policy_normalizer(self._policy_features(core, history, estimate))

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        output = self.mlp(self.get_latent(obs, hidden_state=hidden_state))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def estimator_losses(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        estimate = self.estimate_root_state(obs)
        target = obs[self.estimator_target_group]
        if target.shape[-1] != SPV3_ESTIMATOR_OUTPUT_DIM:
            raise ValueError(f"SPV3 estimator target has {target.shape[-1]} values, expected 4")
        height_mse = (estimate[..., :1] - target[..., :1]).square().mean()
        lin_vel_mse = (estimate[..., 1:] - target[..., 1:]).square().mean()
        return height_mse, lin_vel_mse

    @torch.no_grad()
    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return
        history = obs[self.estimator_history_group]
        self.history_normalizer.update(history)  # type: ignore[attr-defined]
        estimate = self.estimate_root_state(obs)
        features = self._policy_features(obs[self.actor_core_group], history, estimate)
        self.policy_normalizer.update(features)  # type: ignore[attr-defined]

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore[union-attr]

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(  # type: ignore[union-attr]
            old_params, new_params
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _SPV3ActorExport(self)

    def as_jit(self) -> nn.Module:
        return _SPV3ActorExport(self)


class _SPV3ActorExport(nn.Module):
    """One-input deployment wrapper; supervision targets are intentionally absent."""

    is_recurrent = False

    def __init__(self, model: SPV3EstimatorActor) -> None:
        super().__init__()
        # Copy only inference modules.  Copying the live distribution after a
        # rollout can fail because it caches non-leaf tensors from its last update.
        self.history_normalizer = copy.deepcopy(model.history_normalizer)
        self.policy_normalizer = copy.deepcopy(model.policy_normalizer)
        self.estimator = copy.deepcopy(model.estimator)
        self.mlp = copy.deepcopy(model.mlp)
        self.input_size = model.obs_dim
        self.core_dim = model.actor_core_dim
        self.estimator_history_length = model.estimator_history_length
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
            size = self.estimator_history_length * term_dim
            term = history[..., offset : offset + size].reshape(
                *history.shape[:-1], self.estimator_history_length, term_dim
            )
            terms.append(
                term[..., -self.policy_history_length :, :].reshape(
                    *history.shape[:-1], self.policy_history_length * term_dim
                )
            )
            offset += size
        return torch.cat(terms, dim=-1)

    def _policy_features(
        self,
        core: torch.Tensor,
        history: torch.Tensor,
        estimate: torch.Tensor,
    ) -> torch.Tensor:
        latest_proprio = self._latest_policy_proprio(history)
        ref_height = core[..., 42:43]
        ref_lin_vel_robot = SPV3EstimatorActor._reference_lin_vel_in_robot_frame(core)
        root_state_error = torch.cat(
            (
                ref_height - estimate[..., :1],
                ref_lin_vel_robot - estimate[..., 1:],
            ),
            dim=-1,
        )
        return torch.cat(
            (
                latest_proprio,
                estimate,
                core[..., :SPV3_REFERENCE_DIM],
                core[..., SPV3_REFERENCE_DIM:],
                root_state_error,
            ),
            dim=-1,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        core = value[..., : self.core_dim]
        history = value[..., self.core_dim :]
        normalized_history = self.history_normalizer(history)
        estimate = self.estimator(normalized_history)
        features = self._policy_features(core, history, estimate)
        output = self.mlp(self.policy_normalizer(features))
        return self.deterministic_output(output)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["actor_core_estimator_history"]

    @property
    def deploy_input_names(self) -> list[str]:
        return ["actor_core_estimator_history"]

    @property
    def output_names(self) -> list[str]:
        return ["action"]
