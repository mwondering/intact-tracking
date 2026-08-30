"""Build the MJLab environment and frozen SPV5-2 policy from one checkpoint."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from .config import build_env_cfg
from .policy import SPV52HeightContactEstimatorActor


@dataclass(frozen=True)
class PreparedRollout:
    """Checkpoint-derived environment and actor configuration."""

    checkpoint_path: Path
    checkpoint_task_id: str
    env: Any
    actor_kwargs: dict[str, Any]
    obs_groups: dict[str, list[str]]
    clip_actions: float | None


@dataclass
class RolloutRuntime:
    """Live simulator, wrapper, actor and callable inference policy."""

    env: Any
    wrapped: Any
    actor: SPV52HeightContactEstimatorActor
    policy: Any

    def close(self) -> None:
        self.env.close()


class _InferencePolicy:
    def __init__(self, actor: SPV52HeightContactEstimatorActor, stochastic: bool) -> None:
        self.actor = actor
        self.stochastic = bool(stochastic)

    def __call__(self, observations: Any) -> torch.Tensor:
        with torch.inference_mode():
            return self.actor(observations, stochastic_output=self.stochastic)


def _load_saved_config(checkpoint_path: Path) -> DictConfig:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("cfg")
    del checkpoint
    if isinstance(raw, DictConfig):
        container = OmegaConf.to_container(raw, resolve=True)
    elif isinstance(raw, Mapping):
        container = dict(raw)
    else:
        raise TypeError(
            f"The rollout checkpoint must contain a mapping-valued 'cfg'; got {type(raw).__name__}"
        )
    config = OmegaConf.create(container)
    if "task" not in config or "agent" not in config:
        raise ValueError("Checkpoint cfg must contain both task and agent sections")
    return config


def _apply_motion(env_cfg: Any, motion_path: str | None, motion_file: str | None) -> None:
    if bool(motion_path) == bool(motion_file):
        raise ValueError("Provide exactly one of motion_path or motion_file")
    command = env_cfg.commands["motion"]
    if motion_file:
        path = Path(motion_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        command.motion_path = ""
        command.motion_file = str(path)
    else:
        path = Path(str(motion_path)).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        command.motion_file = ""
        command.motion_path = str(path)


def _actor_configuration(config: DictConfig) -> tuple[dict[str, Any], dict[str, list[str]], Any]:
    overrides = config.task.get("agent_overrides", {})
    merged = OmegaConf.merge(config.agent, overrides)
    data = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Merged checkpoint agent configuration must be a mapping")
    actor = dict(data["actor"])
    class_name = str(actor.pop("class_name", ""))
    if not class_name.endswith(":SPV52HeightContactEstimatorActor"):
        raise ValueError(
            "This rollout runtime currently supports SPV52HeightContactEstimatorActor, "
            f"got {class_name!r}"
        )
    signature = inspect.signature(SPV52HeightContactEstimatorActor.__init__)
    reserved = {"self", "obs", "obs_groups", "obs_set", "output_dim"}
    accepted = set(signature.parameters).difference(reserved)
    legacy_keys = {"key_body_error_group", "ref_key_body_group"}
    unsupported = set(actor).difference(accepted, legacy_keys)
    if unsupported:
        raise ValueError(f"Unsupported SPV5-2 actor configuration keys: {sorted(unsupported)}")
    actor_kwargs = {name: value for name, value in actor.items() if name in accepted}
    raw_groups = data.get("obs_groups", {})
    obs_groups = {
        str(name): [str(group) for group in groups] for name, groups in raw_groups.items()
    }
    if obs_groups.get("actor") != [
        "robot_root_quat",
        "estimator_history",
        "reference_encoder_input",
        "robot_key_body",
    ]:
        raise ValueError(f"Unexpected SPV5-2 actor observation groups: {obs_groups.get('actor')}")
    return actor_kwargs, obs_groups, data.get("clip_actions")


def prepare_rollout(
    *,
    checkpoint_file: str,
    num_envs: int,
    motion_path: str | None,
    motion_file: str | None,
    task_id: str | None = None,
) -> PreparedRollout:
    """Reconstruct the exact inference configuration embedded in a checkpoint."""
    checkpoint_path = Path(checkpoint_file).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    config = _load_saved_config(checkpoint_path)
    checkpoint_task_id = str(config.get("task_id", config.task.get("name", "")))
    if task_id is not None and str(task_id) != checkpoint_task_id:
        raise ValueError(
            f"Requested task {task_id!r} does not match checkpoint task {checkpoint_task_id!r}"
        )
    env_cfg = build_env_cfg(config.task)
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(config.get("seed", config.agent.get("seed", env_cfg.seed)))
    _apply_motion(env_cfg, motion_path, motion_file)
    actor_kwargs, obs_groups, raw_clip_actions = _actor_configuration(config)
    clip_actions = None if raw_clip_actions is None else float(raw_clip_actions)
    return PreparedRollout(
        checkpoint_path=checkpoint_path,
        checkpoint_task_id=checkpoint_task_id,
        env=env_cfg,
        actor_kwargs=actor_kwargs,
        obs_groups=obs_groups,
        clip_actions=clip_actions,
    )


def create_runtime(
    prepared: PreparedRollout,
    *,
    device: str,
    stochastic_policy: bool = False,
) -> RolloutRuntime:
    """Instantiate MJLab and strictly load the checkpoint actor weights."""
    from copy import deepcopy

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    env = ManagerBasedRlEnv(cfg=deepcopy(prepared.env), device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions)
        observations = wrapped.get_observations()
        actor = SPV52HeightContactEstimatorActor(
            observations,
            prepared.obs_groups,
            "actor",
            wrapped.num_actions,
            **prepared.actor_kwargs,
        ).to(device)
        checkpoint = torch.load(prepared.checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint.get("actor_state_dict", checkpoint.get("policy"))
        if not isinstance(state, Mapping):
            raise KeyError("Checkpoint has neither actor_state_dict nor policy weights")
        actor.load_state_dict(state, strict=True)
        del state, checkpoint
        actor.requires_grad_(False)
        actor.eval()
        if any(parameter.requires_grad for parameter in actor.parameters()):
            raise RuntimeError("Frozen tracker still has trainable parameters")
    except BaseException:
        env.close()
        raise
    return RolloutRuntime(
        env=env,
        wrapped=wrapped,
        actor=actor,
        policy=_InferencePolicy(actor, stochastic_policy),
    )
