"""Optional history warmup and checkpoint audits for direct tracker FiLM PPO."""

import hashlib
from pathlib import Path

import torch

from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.residual_uniform_protocol import audit_physics


class TrackerFiLMRunner(ResidualOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        settings = self.residual_metadata["arguments"]
        steps = int(settings["tracker_warmup_steps"])
        if steps < 0:
            raise ValueError("Tracker warmup steps must be nonnegative")
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        seed = int(settings["seed"]) + 1000003 * rank + 2300000
        trace = hashlib.sha256()
        if steps:
            self._warmup_tracker(steps, seed, trace)
        self.residual_metadata["tracker_warmup"] = {
            "steps": steps, "enabled": bool(steps),
            "policy": "unmodified frozen tracker" if steps else None,
            "actor_conditioning": settings["actor_conditioning"],
            "trajectory_sha256": trace.hexdigest() if steps else None, "seed": seed,
            "context_metrics": dict(self.env.latent_metrics),
            "ppo_updates": 0,
        }
        self.alg.actor.tracker_warmup = self.residual_metadata["tracker_warmup"]

    @torch.inference_mode()
    def _warmup_tracker(self, steps, seed, trace):
        obs = self.env.get_observations()
        for step in range(steps):
            _seed_everything(seed + step + 1)
            action = self.alg.actor.tracker(obs)
            self.env.unwrapped.action_manager.get_term("joint_pos").record_policy_mean(action)
            obs, _, _, _ = self.env.step(action)
            if (step + 1) % 100 == 0 or step + 1 == steps:
                for value in (self.env.unwrapped.sim.data.qpos, self.env.unwrapped.sim.data.qvel, action):
                    trace.update(value.contiguous().cpu().numpy().tobytes())
                print(f"Matched tracker FiLM warmup {step + 1}/{steps}", flush=True)

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
