"""Frozen dynamics-context inference for residual-policy training."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict

from intact_tracking.forward_predictor import DynamicsContextEncoder, ForwardPredictorConfig
from intact_tracking.forward_predictor_inputs import ACTION_DIM, ROBOT_STATE_DIM
from intact_tracking.rollout.mjlab_adapter import _robot_raw_state, _sha256
from intact_tracking.rollout.online import _read_motion_resample_boundary

from .residual_policy import DYNAMICS_LATENT_GROUP


@dataclass(frozen=True)
class FrozenContextCheckpoint:
    """Loaded inference-only portion of a Forward Predictor checkpoint."""

    encoder: DynamicsContextEncoder
    config: ForwardPredictorConfig
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    path: str
    sha256: str
    tracker_sha256: str | None


def _as_vector(
    normalization: Mapping[str, Any],
    name: str,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if name not in normalization:
        raise KeyError(f"Forward Predictor checkpoint normalization has no {name!r}")
    value = torch.as_tensor(normalization[name], dtype=torch.float32, device=device)
    if value.shape != (width,):
        raise ValueError(
            f"Normalization {name} must have shape [{width}], got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Normalization {name} contains NaN or Inf")
    if name.endswith("_std") and not bool((value > 0.0).all()):
        raise ValueError(f"Normalization {name} must be strictly positive")
    return value


def load_frozen_context_checkpoint(
    checkpoint_file: str | Path,
    *,
    device: torch.device | str,
    expected_tracker_sha256: str | None = None,
) -> FrozenContextCheckpoint:
    """Strictly load only the Context Encoder and its normalization statistics."""

    path = Path(checkpoint_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    target_device = torch.device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Forward Predictor checkpoint must be mapping-valued")
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise KeyError("Forward Predictor checkpoint has no mapping-valued model_config")
    config = ForwardPredictorConfig(**dict(raw_config))
    architecture = str(checkpoint.get("architecture_version", config.architecture_version))
    if architecture != config.architecture_version:
        raise ValueError(
            "Forward Predictor architecture metadata disagrees with model_config: "
            f"{architecture!r} != {config.architecture_version!r}"
        )
    if config.state_dim != ROBOT_STATE_DIM or config.action_dim != ACTION_DIM:
        raise ValueError("Context Encoder requires the 71-D state and 29-D applied-target contract")

    raw_state = checkpoint.get("model")
    if not isinstance(raw_state, Mapping):
        raise KeyError("Forward Predictor checkpoint has no mapping-valued model state")
    prefix = "context_encoder."
    context_state = {
        str(name)[len(prefix) :]: value
        for name, value in raw_state.items()
        if str(name).startswith(prefix)
    }
    if not context_state:
        raise KeyError("Forward Predictor model state has no context_encoder parameters")
    encoder = DynamicsContextEncoder(config)
    encoder.load_state_dict(context_state, strict=True)
    encoder.to(target_device)
    encoder.requires_grad_(False)
    encoder.eval()

    raw_normalization = checkpoint.get("normalization")
    if not isinstance(raw_normalization, Mapping):
        raise KeyError("Forward Predictor checkpoint has no mapping-valued normalization")
    tracker = checkpoint.get("tracker")
    tracker_sha256 = (
        str(tracker.get("checkpoint_sha256"))
        if isinstance(tracker, Mapping) and tracker.get("checkpoint_sha256")
        else None
    )
    if (
        expected_tracker_sha256 is not None
        and tracker_sha256 is not None
        and tracker_sha256 != expected_tracker_sha256
    ):
        raise ValueError(
            "Context Encoder and frozen tracker checkpoints do not match: "
            f"predictor={tracker_sha256}, tracker={expected_tracker_sha256}"
        )
    result = FrozenContextCheckpoint(
        encoder=encoder,
        config=config,
        state_mean=_as_vector(raw_normalization, "state_mean", ROBOT_STATE_DIM, target_device),
        state_std=_as_vector(raw_normalization, "state_std", ROBOT_STATE_DIM, target_device),
        action_mean=_as_vector(raw_normalization, "action_mean", ACTION_DIM, target_device),
        action_std=_as_vector(raw_normalization, "action_std", ACTION_DIM, target_device),
        path=str(path),
        sha256=_sha256(path),
        tracker_sha256=tracker_sha256,
    )
    del checkpoint
    return result


class DynamicsContextInference:
    """Per-world causal ring buffer feeding a frozen Context Encoder."""

    def __init__(
        self,
        checkpoint: FrozenContextCheckpoint,
        *,
        num_envs: int,
        device: torch.device | str,
        use_bfloat16: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.checkpoint = checkpoint
        self.encoder = checkpoint.encoder
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.history_steps = int(checkpoint.config.context_history_steps)
        self.latent_dim = int(checkpoint.config.dynamics_latent_dim)
        self.use_bfloat16 = bool(
            use_bfloat16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        self.history_state = torch.zeros(
            (self.history_steps, self.num_envs, ROBOT_STATE_DIM),
            device=self.device,
        )
        self.history_action = torch.zeros(
            (self.history_steps, self.num_envs, ACTION_DIM),
            device=self.device,
        )
        self.history_valid = torch.zeros(
            (self.history_steps, self.num_envs),
            dtype=torch.bool,
            device=self.device,
        )
        self.pointer = 0

    @torch.no_grad()
    def clear(self, boundary: torch.Tensor | None = None) -> None:
        if boundary is None:
            self.history_valid.zero_()
            self.pointer = 0
            return
        boundary = boundary.to(device=self.device, dtype=torch.bool)
        if boundary.shape != (self.num_envs,):
            raise ValueError(
                f"Context boundary must have shape [{self.num_envs}], got {tuple(boundary.shape)}"
            )
        self.history_valid[:, boundary] = False

    @torch.no_grad()
    def append(
        self,
        state: torch.Tensor,
        applied_target: torch.Tensor,
        boundary: torch.Tensor,
    ) -> None:
        if state.shape != (self.num_envs, ROBOT_STATE_DIM):
            raise ValueError(f"Context state has unexpected shape {tuple(state.shape)}")
        if applied_target.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(f"Context action has unexpected shape {tuple(applied_target.shape)}")
        boundary = boundary.to(device=self.device, dtype=torch.bool)
        if boundary.shape != (self.num_envs,):
            raise ValueError(f"Context boundary has unexpected shape {tuple(boundary.shape)}")
        self.clear(boundary)
        valid = ~boundary
        self.history_state[self.pointer].copy_(state)
        self.history_action[self.pointer].copy_(applied_target)
        self.history_valid[self.pointer].copy_(valid)
        self.pointer = (self.pointer + 1) % self.history_steps

    def _ordered(self, value: torch.Tensor) -> torch.Tensor:
        order = torch.arange(self.history_steps, device=self.device)
        order = (order + self.pointer).remainder(self.history_steps)
        return value.index_select(0, order).transpose(0, 1)

    @torch.no_grad()
    def encode(self, current_state: torch.Tensor) -> torch.Tensor:
        if current_state.shape != (self.num_envs, ROBOT_STATE_DIM):
            raise ValueError(f"Current state has unexpected shape {tuple(current_state.shape)}")
        history_state = self._ordered(self.history_state)
        history_action = self._ordered(self.history_action)
        history_valid = self._ordered(self.history_valid)
        state_mean = self.checkpoint.state_mean
        state_std = self.checkpoint.state_std
        action_mean = self.checkpoint.action_mean
        action_std = self.checkpoint.action_std
        normalized_history_state = (history_state - state_mean) / state_std
        normalized_history_action = (history_action - action_mean) / action_std
        normalized_current_state = (current_state - state_mean) / state_std
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_bfloat16
            else nullcontext()
        )
        with context:
            latent = self.encoder(
                normalized_history_state,
                normalized_history_action,
                normalized_current_state,
                history_valid,
            )
        return latent.float()

    @property
    def metrics(self) -> dict[str, float]:
        counts = self.history_valid.sum(dim=0)
        return {
            "context_valid_fraction": float(self.history_valid.float().mean().item()),
            "context_full_fraction": float((counts == self.history_steps).float().mean().item()),
        }


class ResidualLatentVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL wrapper that appends a frozen, history-only dynamics latent."""

    def __init__(
        self,
        env: Any,
        *,
        clip_actions: float | None,
        context_checkpoint: FrozenContextCheckpoint,
        use_bfloat16: bool = True,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)
        self.context = DynamicsContextInference(
            context_checkpoint,
            num_envs=self.num_envs,
            device=self.device,
            use_bfloat16=use_bfloat16,
        )
        self.motion_command = self.unwrapped.command_manager.get_term("motion")
        self._current_state = _robot_raw_state(self.unwrapped).detach().clone()
        self._current_latent = self.context.encode(self._current_state)
        self._latent_rms = float(self._current_latent.square().mean().sqrt().item())

    def _attach_latent(self, observations: TensorDict) -> TensorDict:
        observations.set(DYNAMICS_LATENT_GROUP, self._current_latent)
        return observations

    def get_observations(self) -> TensorDict:
        return self._attach_latent(super().get_observations())

    def reset(self) -> tuple[TensorDict, dict]:
        observations, extras = super().reset()
        self.context.clear()
        self._current_state = _robot_raw_state(self.unwrapped).detach().clone()
        self._current_latent = self.context.encode(self._current_state)
        return self._attach_latent(observations), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        state = self._current_state
        observations, rewards, dones, extras = super().step(actions)
        target = getattr(self.unwrapped.scene["robot"].data, "joint_pos_target", None)
        if not isinstance(target, torch.Tensor) or target.shape != (self.num_envs, ACTION_DIM):
            shape = tuple(target.shape) if isinstance(target, torch.Tensor) else None
            raise RuntimeError(
                "Residual Context Encoder requires simulator joint_pos_target with shape "
                f"[{self.num_envs},{ACTION_DIM}], got {shape}"
            )
        environment_boundary = dones.to(dtype=torch.bool)
        motion_boundary = _read_motion_resample_boundary(
            self.motion_command,
            environment_boundary,
        )
        boundary = environment_boundary | motion_boundary
        next_state = _robot_raw_state(self.unwrapped).detach().clone()
        self.context.append(state, target.detach(), boundary)
        self._current_state = next_state
        self._current_latent = self.context.encode(next_state)
        self._latent_rms = float(self._current_latent.square().mean().sqrt().item())
        return self._attach_latent(observations), rewards, dones, extras

    @property
    def latent_metrics(self) -> dict[str, float]:
        return {**self.context.metrics, "dynamics_latent_rms": self._latent_rms}
