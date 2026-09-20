"""SPV5-2-compatible on-policy runner for frozen-tracker residual policies."""

from __future__ import annotations

import copy
import itertools
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
from .wandb_logger import RslWandbLogWriter


def _initialize_logging_writer_collectively(
    logger: Any,
    *,
    is_distributed: bool,
    device: str,
) -> None:
    """Initialize rank-zero logging and propagate startup failure to every rank."""

    initialization_error: Exception | None = None
    try:
        logger.init_logging_writer()
    except Exception as error:
        initialization_error = error

    if initialization_error is None and isinstance(
        getattr(logger, "writer", None), RslWandbLogWriter
    ):
        # RSL-RL uses this exact marker to avoid mixing iteration and wall-time
        # values in W&B's single monotonically increasing step axis.
        logger.logger_type = "WandbLogWriter"

    if is_distributed:
        initialized = torch.tensor(
            initialization_error is None,
            dtype=torch.int32,
            device=device,
        )
        torch.distributed.all_reduce(initialized, op=torch.distributed.ReduceOp.MIN)
        if not bool(initialized.item()):
            if initialization_error is not None:
                raise initialization_error
            raise RuntimeError("Logging writer initialization failed on another distributed rank")
    elif initialization_error is not None:
        raise initialization_error


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
        frozen_dependencies: Mapping[str, Any] | None = None,
    ) -> None:
        self.checkpoint_cfg = checkpoint_cfg
        self.residual_metadata = dict(residual_metadata)
        # CPU inference tensors, cached once; never optimizer parameters.
        self.frozen_dependencies = dict(frozen_dependencies or {})
        self.stop_requested = False
        self.completed_learning_updates = 0
        # RSL-RL's factory pops class_name and other constructor options. Never
        # let that mutate the caller's serializable checkpoint configuration.
        super().__init__(env, copy.deepcopy(train_cfg), log_dir, device)

    def request_stop(self) -> None:
        """Finish the current PPO update before saving a resumable checkpoint."""
        self.stop_requested = True

    def _collective_stop_requested(self) -> bool:
        if self.is_distributed:
            value = torch.tensor(int(self.stop_requested), device=self.device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
            self.stop_requested = bool(value.item())
        return self.stop_requested

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
            action_manager = getattr(self.env.unwrapped, "action_manager", None)
            if action_manager is not None and actor.last_residual_mean is not None:
                try:
                    action_term = action_manager.get_term("joint_pos")
                except KeyError:
                    action_term = None
                if action_term is not None:
                    # The joint-position offset induced by the mean residual,
                    # before SP's delay/smoothing and motor/joint offsets.
                    target_delta = actor.last_residual_mean.float() * action_term._scale
                    diagnostics["residual_target_rms_rad"] = float(
                        target_delta.square().mean().sqrt().item())
                    diagnostics["residual_target_abs_max_rad"] = float(
                        target_delta.abs().max().item())
        latent_metrics = getattr(self.env, "latent_metrics", None)
        if isinstance(latent_metrics, Mapping):
            diagnostics.update({str(name): float(value) for name, value in latent_metrics.items()})
        if self.is_distributed and diagnostics:
            names = tuple(diagnostics)
            rms_names = {name for name in names if name in (
                "base_action_rms", "residual_action_rms", "residual_target_rms_rad"
            ) or name.endswith("_action_delta_rms")}
            max_names = tuple(name for name in names if name.endswith(("_abs_max", "_abs_max_rad")))
            values = torch.tensor(
                [diagnostics[name] ** 2 if name in rms_names else diagnostics[name] for name in names],
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(values)
            values.div_(self.gpu_world_size)
            # Every training rank has the same observation/action count. Pool
            # mean squares before sqrt; a mean of rank RMSs underestimates RMS.
            maxima = torch.tensor([diagnostics[name] for name in max_names],
                                  dtype=torch.float64, device=self.device)
            if max_names:
                torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
            diagnostics = {
                name: float(value ** 0.5 if name in rms_names else value)
                for name, value in zip(names, values.cpu().tolist(), strict=True)
            }
            diagnostics.update(zip(max_names, maxima.cpu().tolist(), strict=True))
        if "residual_to_base_rms_ratio" in diagnostics:
            diagnostics["residual_to_base_rms_ratio"] = (
                diagnostics["residual_action_rms"] / max(diagnostics["base_action_rms"], 1e-12))
        return diagnostics

    def learn(self, num_learning_iterations: int | None, init_at_random_ep_len: bool = False) -> None:
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
        _initialize_logging_writer_collectively(
            self.logger,
            is_distributed=self.is_distributed,
            device=self.device,
        )
        if self.gpu_global_rank == 0:
            print("Logging writer initialized.", flush=True)

        start_iteration = self.current_learning_iteration
        final_iteration = None if num_learning_iterations is None else start_iteration + num_learning_iterations
        iterations = itertools.count(start_iteration) if final_iteration is None else range(start_iteration, final_iteration)
        for iteration in iterations:
            if self._collective_stop_requested():
                break
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
            self.completed_learning_updates = iteration + 1
            checkpoint_evaluator = getattr(self, "checkpoint_evaluator", None)
            if checkpoint_evaluator is not None:
                checkpoint_evaluator(self)
            self.logger.log(
                it=iteration,
                start_it=start_iteration,
                # RSL's console logger needs finite arithmetic for its ETA;
                # this display value does not bound the count() training loop.
                total_it=final_iteration if final_iteration is not None else iteration + 1,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )
            if checkpoint_evaluator is None and iteration % int(self.cfg["save_interval"]) == 0:
                preparer = getattr(self, "checkpoint_state_preparer", None)
                if preparer is not None:
                    preparer(self)
                if self.logger.writer is not None:
                    self.save(str(Path(self.logger.log_dir) / f"checkpoint_{iteration}.pt"))

        preparer = getattr(self, "checkpoint_state_preparer", None)
        if preparer is not None:
            preparer(self)
        if self.logger.writer is not None:
            if self.stop_requested:
                self.save(str(Path(self.logger.log_dir) / "checkpoint_interrupted.pt"),
                          infos={"stop_reason": "user_requested_at_update_boundary"})
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
            "completed_updates": int(self.completed_learning_updates),
            "infos": infos,
            "cfg": self.checkpoint_cfg,
            "residual_policy": self.residual_metadata,
            "motion_sampling_state": getattr(self, "motion_sampling_state", None),
            **getattr(self, "frozen_dependencies", {}),
        }

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        # A signal or reader must not observe a partially overwritten checkpoint.
        destination = Path(path)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        torch.save(self._checkpoint_payload(infos), temporary)
        os.replace(temporary, destination)
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
        self.loaded_motion_sampling_state = checkpoint.get("motion_sampling_state")
        state = dict(checkpoint.get("rsl_rl", checkpoint))
        if "actor_state_dict" not in state and isinstance(checkpoint.get("policy"), Mapping):
            state["actor_state_dict"] = checkpoint["policy"]
        load_iteration = self.alg.load(state, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = int(checkpoint.get("iter", state.get("iter", 0)))
            self.completed_learning_updates = int(
                checkpoint.get("completed_updates", self.current_learning_iteration + 1)
            )
        infos = checkpoint.get("infos")
        if not isinstance(infos, dict):
            infos = {}
        env_state = checkpoint.get("env", infos.get("env_state"))
        if isinstance(env_state, Mapping) and "common_step_counter" in env_state:
            self.env.unwrapped.common_step_counter = int(env_state["common_step_counter"])
        return infos
