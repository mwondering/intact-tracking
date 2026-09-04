"""SPV5-2-compatible on-policy runner for frozen-tracker residual policies."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.runners.on_policy_runner import check_nan
from tensordict import TensorDict

from .residual_policy import FrozenTrackerResidualActor


class ResidualOnPolicyRunner(MjlabOnPolicyRunner):
    """Plain PPO loop plus SP's adaptive-motion and action-mean hooks."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str,
        device: str,
        *,
        checkpoint_cfg: Any,
        residual_metadata: Mapping[str, Any],
    ) -> None:
        self.checkpoint_cfg = checkpoint_cfg
        self.residual_metadata = dict(residual_metadata)
        super().__init__(env, train_cfg, log_dir, device)

    def _configure_multi_gpu(self) -> None:
        """Accept the process group initialized by our CLI before MJLab construction."""

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1 or not torch.distributed.is_initialized():
            super()._configure_multi_gpu()
            return
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        global_rank = int(os.environ.get("RANK", "0"))
        if torch.distributed.get_world_size() != world_size:
            raise RuntimeError("Preinitialized process group and WORLD_SIZE disagree")
        if torch.distributed.get_rank() != global_rank:
            raise RuntimeError("Preinitialized process group and RANK disagree")
        if self.device != f"cuda:{local_rank}":
            raise ValueError(f"Distributed residual training requires device cuda:{local_rank}")
        self.gpu_world_size = world_size
        self.is_distributed = True
        self.gpu_local_rank = local_rank
        self.gpu_global_rank = global_rank
        self.cfg["multi_gpu"] = {
            "global_rank": global_rank,
            "local_rank": local_rank,
            "world_size": world_size,
        }
        torch.cuda.set_device(local_rank)

    def _begin_adaptive_sampling_iteration(self, iteration: int) -> None:
        command = self.env.unwrapped.command_manager.get_term("motion")
        begin = getattr(command, "begin_adaptive_sampling_iteration", None)
        if callable(begin):
            begin(iteration)

    def _record_policy_action_mean(self) -> None:
        mean = getattr(self.alg.actor, "output_mean", None)
        if not isinstance(mean, torch.Tensor):
            return
        action_manager = self.env.unwrapped.action_manager
        try:
            action_term = action_manager.get_term("joint_pos")
        except KeyError:
            return
        record = getattr(action_term, "record_policy_mean", None)
        if callable(record):
            record(mean)

    def _policy_diagnostics(self, obs: TensorDict) -> dict[str, float]:
        actor = self.alg.get_policy()
        if not isinstance(actor, FrozenTrackerResidualActor):
            return {}
        with torch.inference_mode():
            diagnostics = actor.policy_metrics(obs)
        latent_metrics = getattr(self.env, "latent_metrics", None)
        if isinstance(latent_metrics, Mapping):
            diagnostics.update({str(name): float(value) for name, value in latent_metrics.items()})
        if self.is_distributed and diagnostics:
            names = tuple(diagnostics)
            values = torch.tensor(
                [diagnostics[name] for name in names],
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(values)
            values.div_(self.gpu_world_size)
            diagnostics = {
                name: float(value) for name, value in zip(names, values.cpu().tolist(), strict=True)
            }
        return diagnostics

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(
                f"Synchronizing trainable parameters for rank {self.gpu_global_rank}...",
                flush=True,
            )
            synchronization_start = time.perf_counter()
            self.alg.broadcast_parameters()
            synchronization_seconds = time.perf_counter() - synchronization_start
            synchronized_mib = int(getattr(self.alg, "last_parameter_broadcast_bytes", 0)) / 2**20
            synchronized_tensors = int(
                getattr(self.alg, "last_parameter_broadcast_tensor_count", 0)
            )
            print(
                "Synchronized "
                f"{synchronized_tensors} trainable tensors "
                f"({synchronized_mib:.2f} MiB) on rank {self.gpu_global_rank} "
                f"in {synchronization_seconds:.3f}s without a device-wide barrier.",
                flush=True,
            )
        if self.gpu_global_rank == 0:
            print("Initializing logging writer...", flush=True)
        self.logger.init_logging_writer()
        if self.gpu_global_rank == 0:
            print("Logging writer initialized.", flush=True)

        start_iteration = self.current_learning_iteration
        final_iteration = start_iteration + num_learning_iterations
        for iteration in range(start_iteration, final_iteration):
            self._begin_adaptive_sampling_iteration(iteration)
            start = time.time()
            with torch.inference_mode():
                for _ in range(int(self.cfg["num_steps_per_env"])):
                    actions = self.alg.act(obs)
                    self._record_policy_action_mean()
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)
                collect_time = time.time() - start
                self.alg.compute_returns(obs)

            start = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - start
            loss_dict.update(self._policy_diagnostics(obs))
            self.current_learning_iteration = iteration
            self.logger.log(
                it=iteration,
                start_it=start_iteration,
                total_it=final_iteration,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )
            if self.logger.writer is not None and iteration % int(self.cfg["save_interval"]) == 0:
                self.save(str(Path(self.logger.log_dir) / f"checkpoint_{iteration}.pt"))

        if self.logger.writer is not None:
            self.save(str(Path(self.logger.log_dir) / "checkpoint_final.pt"))
            self.logger.stop_logging_writer()

    def _checkpoint_payload(self, infos: dict[str, Any] | None = None) -> dict[str, Any]:
        env_state = {"common_step_counter": int(self.env.unwrapped.common_step_counter)}
        infos = {**(infos or {}), "env_state": env_state}
        rsl_state = self.alg.save()
        return {
            **rsl_state,
            "policy": rsl_state["actor_state_dict"],
            "rsl_rl": rsl_state,
            "env": env_state,
            "iter": int(self.current_learning_iteration),
            "infos": infos,
            "cfg": self.checkpoint_cfg,
            "residual_policy": self.residual_metadata,
        }

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        torch.save(self._checkpoint_payload(infos), path)
        if self.cfg.get("upload_model", False):
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict[str, Any] | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Residual checkpoint must be mapping-valued")
        state = dict(checkpoint.get("rsl_rl", checkpoint))
        if "actor_state_dict" not in state and isinstance(checkpoint.get("policy"), Mapping):
            state["actor_state_dict"] = checkpoint["policy"]
        load_iteration = self.alg.load(state, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = int(checkpoint.get("iter", state.get("iter", 0)))
        infos = checkpoint.get("infos")
        if not isinstance(infos, dict):
            infos = {}
        env_state = checkpoint.get("env", infos.get("env_state"))
        if isinstance(env_state, Mapping) and "common_step_counter" in env_state:
            self.env.unwrapped.common_step_counter = int(env_state["common_step_counter"])
        return infos
