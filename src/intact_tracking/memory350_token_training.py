"""Bind collection history and audit checkpoint boundaries for temporal PPO."""

from pathlib import Path
import time

import torch

from intact_tracking.residual_runner import ResidualOnPolicyRunner


def audit_physics(env, metadata):
    if "anchor_fraction" in metadata:
        from intact_tracking.payload_prototype_physics import audit_physics as audit
    else:
        from intact_tracking.residual_uniform_protocol import audit_physics as audit
    return audit(env, metadata)


class TemporalTokenRunner(ResidualOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bind = getattr(self.env, "bind_policy", None)
        if bind is not None:
            bind(self.alg.actor)
        if self.env.clip_actions is not None:
            raise ValueError("The residual comparison requires an unclipped policy action interface")

    def _policy_diagnostics(self, obs):
        result = super()._policy_diagnostics(obs)
        if self.alg.actor.architecture == "transformer" and torch.device(self.device).type == "cuda":
            # The 49152-sample FP32 update fits on an H100, but PyTorch keeps
            # its freed attention workspaces reserved. Warp's next CUDA graph
            # launch uses a separate allocator and cannot reclaim that cache.
            # Hand unused memory back after the complete update/diagnostics;
            # leave samples, gradients, optimizer steps, and live tensors intact.
            before = torch.cuda.memory_reserved(self.device)
            peak = torch.cuda.max_memory_reserved(self.device)
            start = time.perf_counter()
            torch.cuda.empty_cache()
            elapsed = time.perf_counter() - start
            after = torch.cuda.memory_reserved(self.device)
            free, _ = torch.cuda.mem_get_info(self.device)
            result.update(
                cuda_peak_reserved_gib=peak / 2**30,
                cuda_reserved_before_release_gib=before / 2**30,
                cuda_reserved_after_release_gib=after / 2**30,
                cuda_cache_released_gib=(before - after) / 2**30,
                cuda_allocated_after_update_gib=torch.cuda.memory_allocated(self.device) / 2**30,
                cuda_device_free_after_release_gib=free / 2**30,
                cuda_cache_release_seconds=elapsed,
            )
            torch.cuda.reset_peak_memory_stats(self.device)
        return result

    def save(self, *args, **kwargs):
        self.alg.actor.assert_tracker_frozen()
        return super().save(*args, **kwargs)

    def learn(self, *args, **kwargs):
        def save_completed(runner):
            if runner.completed_learning_updates % runner.cfg["save_interval"]:
                return
            audit_physics(runner.env.unwrapped, runner.residual_metadata["physics"])
            preparer = getattr(runner, "checkpoint_state_preparer", None)
            if preparer is not None:
                preparer(runner)
            if runner.logger.writer is not None:
                runner.save(str(Path(runner.logger.log_dir) /
                                f"checkpoint_update_{runner.completed_learning_updates:06d}.pt"))
        self.checkpoint_evaluator = save_completed
        result = super().learn(*args, **kwargs)
        self.alg.actor.assert_tracker_frozen()
        return result
