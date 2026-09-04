"""Frozen-tracker residual PPO models compatible with SPV5-2A checkpoints."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer, unpad_trajectories
from tensordict import TensorDict

from intact_tracking.environment.policy import SPV52HeightContactEstimatorActor

DYNAMICS_LATENT_GROUP = "dynamics_latent"


def _checkpoint_state(checkpoint: Mapping[str, Any], name: str) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get(name)
    if name == "actor_state_dict" and not isinstance(state, Mapping):
        state = checkpoint.get("policy")
    if not isinstance(state, Mapping):
        raise KeyError(f"Checkpoint has no mapping-valued {name!r}")
    return state


def _last_linear(module: nn.Module) -> nn.Linear:
    for child in reversed(tuple(module.modules())):
        if isinstance(child, nn.Linear):
            return child
    raise TypeError("Residual network has no Linear output layer")


class FrozenTrackerResidualActor(nn.Module):
    """SPV5-2A tracker plus a trainable, bounded residual action head.

    The Gaussian is over the final action actually sent to MJLab.  Its mean is
    ``frozen_tracker_action + residual``.  This preserves PPO's action/log-prob
    contract while keeping the trainable correction explicit.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        tracker_checkpoint: str,
        tracker_actor_kwargs: Mapping[str, Any],
        tracker_obs_groups: Mapping[str, Sequence[str]],
        use_dynamics_latent: bool,
        dynamics_latent_group: str = DYNAMICS_LATENT_GROUP,
        dynamics_latent_dim: int = 64,
        residual_hidden_dims: Sequence[int] = (512, 256, 128),
        residual_activation: str = "elu",
        residual_scale: float = 0.25,
        distribution_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.obs_groups = list(obs_groups[obs_set])
        expected_groups = list(tracker_obs_groups["actor"])
        if self.obs_groups != expected_groups:
            raise ValueError(
                "Residual actor observations must exactly match the frozen tracker: "
                f"{self.obs_groups} != {expected_groups}"
            )
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")

        tracker_path = Path(tracker_checkpoint).expanduser().resolve()
        if not tracker_path.is_file():
            raise FileNotFoundError(tracker_path)
        self.tracker_checkpoint = str(tracker_path)
        self.use_dynamics_latent = bool(use_dynamics_latent)
        self.dynamics_latent_group = str(dynamics_latent_group)
        self.dynamics_latent_dim = int(dynamics_latent_dim)
        self.residual_scale = float(residual_scale)

        actor_kwargs = copy.deepcopy(dict(tracker_actor_kwargs))
        self.tracker = SPV52HeightContactEstimatorActor(
            obs,
            {name: list(groups) for name, groups in tracker_obs_groups.items()},
            "actor",
            output_dim,
            **actor_kwargs,
        )
        checkpoint = torch.load(tracker_path, map_location="cpu", weights_only=False)
        self.tracker.load_state_dict(_checkpoint_state(checkpoint, "actor_state_dict"), strict=True)
        del checkpoint
        self.tracker.requires_grad_(False)
        self.tracker.eval()

        if self.use_dynamics_latent:
            latent = obs.get(self.dynamics_latent_group)
            if not isinstance(latent, torch.Tensor):
                raise KeyError(
                    f"Latent residual baseline requires observation {self.dynamics_latent_group!r}"
                )
            if latent.ndim != 2 or latent.size(-1) != self.dynamics_latent_dim:
                raise ValueError(
                    f"Dynamics latent must be [N,{self.dynamics_latent_dim}], got "
                    f"{tuple(latent.shape)}"
                )

        tracker_feature_dim = int(self.tracker.policy_input_dim)
        residual_input_dim = tracker_feature_dim + (
            self.dynamics_latent_dim if self.use_dynamics_latent else 0
        )
        widths = tuple(int(width) for width in residual_hidden_dims)
        if not widths or any(width < 1 for width in widths):
            raise ValueError("residual_hidden_dims must contain positive widths")
        self.residual_mlp = MLP(
            residual_input_dim,
            output_dim,
            list(widths),
            residual_activation,
        )
        output_layer = _last_linear(self.residual_mlp)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

        cfg = copy.deepcopy(dict(distribution_cfg or {}))
        class_name = str(cfg.pop("class_name", "GaussianDistribution"))
        if not class_name.endswith("GaussianDistribution"):
            raise ValueError("Residual policy currently requires GaussianDistribution")
        cfg.setdefault("init_std", 1.0)
        self.distribution = GaussianDistribution(output_dim, **cfg)
        tracker_std = getattr(self.tracker.distribution, "std_param", None)
        residual_std = getattr(self.distribution, "std_param", None)
        if isinstance(tracker_std, torch.Tensor) and isinstance(residual_std, torch.Tensor):
            residual_std.data.copy_(tracker_std.detach().to(residual_std))

        self.last_base_action: torch.Tensor | None = None
        self.last_residual_mean: torch.Tensor | None = None
        self.last_dynamics_latent: torch.Tensor | None = None
        self.populate_tracker_cache(obs)

    @torch.no_grad()
    def populate_tracker_cache(self, obs: TensorDict) -> None:
        self.tracker.populate_policy_context_cache(obs)

    def train(self, mode: bool = True):
        super().train(mode)
        # ``nn.Module.train`` recurses, so restore the immutable tracker to eval.
        self.tracker.eval()
        return self

    def _base_features_and_action(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.tracker.get_latent(obs).detach()
            tracker_output = self.tracker.mlp(features)
            base_action = self.tracker.distribution.deterministic_output(tracker_output).detach()
        return features, base_action

    def _residual_input(self, obs: TensorDict, tracker_features: torch.Tensor) -> torch.Tensor:
        if not self.use_dynamics_latent:
            self.last_dynamics_latent = None
            return tracker_features
        latent = obs[self.dynamics_latent_group].to(dtype=tracker_features.dtype).detach()
        self.last_dynamics_latent = latent
        return torch.cat((tracker_features, latent), dim=-1)

    def _residual(self, value: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * torch.tanh(self.residual_mlp(value))

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        tracker_features, base_action = self._base_features_and_action(obs)
        residual = self._residual(self._residual_input(obs, tracker_features))
        mean = base_action + residual
        self.last_base_action = base_action
        self.last_residual_mean = residual.detach()
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    @torch.no_grad()
    def policy_metrics(self, obs: TensorDict) -> dict[str, float]:
        tracker_features, base_action = self._base_features_and_action(obs)
        normal = self._residual(self._residual_input(obs, tracker_features))
        metrics = {
            "base_action_rms": float(base_action.square().mean().sqrt().item()),
            "residual_action_rms": float(normal.square().mean().sqrt().item()),
            "residual_action_abs_max": float(normal.abs().max().item()),
        }
        if not self.use_dynamics_latent or obs.batch_size[0] < 2:
            metrics.update(
                {
                    "latent_shuffle_action_delta_rms": 0.0,
                    "latent_zero_action_delta_rms": 0.0,
                }
            )
            return metrics
        latent = obs[self.dynamics_latent_group].to(dtype=tracker_features.dtype)
        shuffled = self._residual(torch.cat((tracker_features, latent.roll(1, dims=0)), dim=-1))
        zeroed = self._residual(torch.cat((tracker_features, torch.zeros_like(latent)), dim=-1))
        metrics.update(
            {
                "latent_shuffle_action_delta_rms": float(
                    (normal - shuffled).square().mean().sqrt().item()
                ),
                "latent_zero_action_delta_rms": float(
                    (normal - zeroed).square().mean().sqrt().item()
                ),
            }
        )
        return metrics

    def update_normalization(self, obs: TensorDict) -> None:
        # The checkpoint's normalizers are frozen; only cache its deterministic
        # estimator/reference preprocessing for storage and PPO minibatches.
        self.populate_tracker_cache(obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("Residual policy export needs the online context-history wrapper")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        raise NotImplementedError("Residual policy export needs the online context-history wrapper")


class DecayVecNorm(nn.Module):
    """Exact normalization used by the SPV5-2A HEFT critic."""

    def __init__(self, size: int, decay: float = 0.9999, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.decay = float(decay)
        self.eps = float(eps)
        self.register_buffer("sum", torch.zeros(size))
        self.register_buffer("ssq", torch.zeros(size))
        self.register_buffer("count", torch.zeros(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        count = self.count.clamp_min(1.0)
        mean = self.sum / count
        variance = (self.ssq / count - mean.square()).clamp_min(self.eps)
        return (value - mean) / variance.sqrt().clamp_min(self.eps)

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        if not self.training:
            return
        flat = value.reshape(-1, value.shape[-1])
        self.sum.mul_(self.decay).add_(flat.sum(dim=0))
        self.ssq.mul_(self.decay).add_(flat.square().sum(dim=0))
        self.count.mul_(self.decay).add_(float(flat.shape[0]))


def _make_heft_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(input_dim)
    for width in hidden_dims:
        width = int(width)
        layers.extend((nn.Linear(current, width), nn.LayerNorm(width), nn.Mish()))
        current = width
    layers.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*layers)


class WarmStartedHeftCritic(nn.Module):
    """Checkpoint-compatible HEFT critic initialized from the frozen tracker run."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        initial_checkpoint: str,
        hidden_dims: Sequence[int] = (1024, 512, 512),
        activation: str = "mish",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        vecnorm_decay: float = 0.9999,
    ) -> None:
        super().__init__()
        del distribution_cfg
        if activation.lower() != "mish":
            raise ValueError("WarmStartedHeftCritic requires Mish activation")
        self.obs_groups = list(obs_groups[obs_set])
        self.obs_dim = sum(int(obs[name].shape[-1]) for name in self.obs_groups)
        self.obs_normalization = bool(obs_normalization)
        self.obs_normalizer = (
            DecayVecNorm(self.obs_dim, decay=vecnorm_decay)
            if self.obs_normalization
            else nn.Identity()
        )
        self.mlp = _make_heft_mlp(self.obs_dim, hidden_dims, output_dim)
        checkpoint_path = Path(initial_checkpoint).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.load_state_dict(_checkpoint_state(checkpoint, "critic_state_dict"), strict=True)
        del checkpoint

    def _flat_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups], dim=-1)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        return self.mlp(self.obs_normalizer(self._flat_obs(obs)))

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(self._flat_obs(obs))  # type: ignore[union-attr]

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones


class ResidualPPO(PPO):
    """Standard PPO with SP's split actor/critic learning-rate contract."""

    _SPLIT_LR_STATE_KEY = "tracking_split_lr_state"

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *args,
        actor_learning_rate: float = 1.0e-3,
        critic_learning_rate: float = 5.0e-4,
        adaptive_critic_learning_rate: bool = False,
        learning_rate: float = 1.0e-3,
        optimizer: str = "adam",
        **kwargs,
    ) -> None:
        super().__init__(
            actor,
            critic,
            storage,
            *args,
            learning_rate=learning_rate,
            optimizer=optimizer,
            **kwargs,
        )
        self.actor_learning_rate = float(actor_learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.adaptive_critic_learning_rate = bool(adaptive_critic_learning_rate)
        self.learning_rate = self.actor_learning_rate
        actor_parameters = [
            parameter for parameter in self.actor.parameters() if parameter.requires_grad
        ]
        critic_parameters = [
            parameter for parameter in self.critic.parameters() if parameter.requires_grad
        ]
        self.optimizer = resolve_optimizer(optimizer)(
            [
                {"params": actor_parameters, "lr": self.actor_learning_rate},
                {"params": critic_parameters, "lr": self.critic_learning_rate},
            ]
        )

    def _set_adaptive_learning_rate(self, kl_mean: torch.Tensor) -> None:
        if self.gpu_global_rank == 0:
            old_actor_lr = self.actor_learning_rate
            if kl_mean > self.desired_kl * 2.0:
                self.actor_learning_rate = max(1.0e-5, self.actor_learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.actor_learning_rate = min(1.0e-2, self.actor_learning_rate * 1.5)
            if self.adaptive_critic_learning_rate and old_actor_lr > 0.0:
                self.critic_learning_rate *= self.actor_learning_rate / old_actor_lr
        if self.is_multi_gpu:
            rates = torch.tensor(
                (self.actor_learning_rate, self.critic_learning_rate), device=self.device
            )
            torch.distributed.broadcast(rates, src=0)
            self.actor_learning_rate = float(rates[0].item())
            self.critic_learning_rate = float(rates[1].item())
        self.learning_rate = self.actor_learning_rate
        self.optimizer.param_groups[0]["lr"] = self.actor_learning_rate
        self.optimizer.param_groups[1]["lr"] = self.critic_learning_rate

    def update(self) -> dict[str, float]:
        if self.rnd is not None or self.symmetry is not None:
            raise NotImplementedError("ResidualPPO intentionally supports plain SPV5-2A PPO only")
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = tuple(
                parameter[:original_batch_size]
                for parameter in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = kl.mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean)
                        kl_mean /= self.gpu_world_size
                    self._set_adaptive_learning_rate(kl_mean)

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (values - batch.returns).square(),
                    (value_clipped - batch.returns).square(),
                ).mean()
            else:
                value_loss = (batch.returns - values).square().mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(
                (parameter for parameter in self.actor.parameters() if parameter.requires_grad),
                self.max_grad_norm,
            )
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())
            mean_entropy += float(entropy.mean().item())

        count = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / count,
            "surrogate": mean_surrogate_loss / count,
            "entropy": mean_entropy / count,
        }

    def save(self) -> dict:
        state = super().save()
        state[self._SPLIT_LR_STATE_KEY] = {
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "learning_rate": self.learning_rate,
        }
        return state

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        state = loaded_dict.get(self._SPLIT_LR_STATE_KEY, {})
        if isinstance(state, Mapping):
            self.actor_learning_rate = float(
                state.get("actor_learning_rate", self.optimizer.param_groups[0]["lr"])
            )
            self.critic_learning_rate = float(
                state.get("critic_learning_rate", self.optimizer.param_groups[1]["lr"])
            )
            self.learning_rate = float(state.get("learning_rate", self.actor_learning_rate))
            self.optimizer.param_groups[0]["lr"] = self.actor_learning_rate
            self.optimizer.param_groups[1]["lr"] = self.critic_learning_rate
        return load_iteration

    def reduce_parameters(self) -> None:
        parameters = [
            parameter
            for parameter in chain(self.actor.parameters(), self.critic.parameters())
            if parameter.grad is not None
        ]
        if not parameters:
            return
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
        torch.distributed.all_reduce(flat)
        flat.div_(self.gpu_world_size)
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad.copy_(flat[offset : offset + count].view_as(parameter.grad))
            offset += count
