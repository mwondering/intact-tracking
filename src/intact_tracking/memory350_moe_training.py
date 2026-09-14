"""Online router updates at PPO boundaries, with matched warm-history bootstrap."""

import hashlib
import time

import torch

from intact_tracking.limb_context_distributed import DistributedResidualPPO, tensor_digest
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.cli.residual_policy_train import _seed_everything


class OnlineMoEPPO(DistributedResidualPPO):
    def copy_router_to_critic(self):
        source, target = self.actor.residual_mlp.router, self.critic.mlp.router
        if source is not None:
            target.load_state_dict(source.state_dict(), strict=True)

    def update(self):
        router = self.actor.residual_mlp.router
        # Storage.clear resets its cursor; observations remain until the next
        # rollout. Hold the exact saved inputs while centers remain unchanged.
        latent = self.storage.observations["dynamics_latent"] if router is not None else None
        result = super().update()
        if router is not None:
            result.update(router.update_centers(latent))
            self.copy_router_to_critic()
        return result


class OnlineMoERunner(ResidualOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        settings = self.residual_metadata["arguments"]
        steps = int(settings.get("router_bootstrap_steps", 500))
        if steps < 350:
            raise ValueError("Bootstrap must cover at least the full Memory350 history")
        actor = self.alg.actor
        samples = []
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        bootstrap_seed = int(settings["seed"]) + 1000003 * rank + 2300000
        _seed_everything(bootstrap_seed)
        trace = hashlib.sha256()
        start = time.monotonic()
        with torch.inference_mode():
            obs = self.env.get_observations()
            for step in range(steps):
                # Keep simulator noise/force draws independent of architecture
                # construction and context-inference RNG consumption.
                _seed_everything(bootstrap_seed + 1 + step)
                action = actor.tracker(obs)
                self.env.unwrapped.action_manager.get_term("joint_pos").record_policy_mean(action)
                obs, _, _, _ = self.env.step(action)
                if actor.residual_mlp.router is not None and step + 1 >= 350 and (step + 1) % 50 == 0:
                    samples.append(obs["dynamics_latent"].clone())
                if (step + 1) % 100 == 0:
                    for value in (self.env.unwrapped.sim.data.qpos, self.env.unwrapped.sim.data.qvel, action):
                        trace.update(value.contiguous().cpu().numpy().tobytes())
                    print(f"Matched tracker bootstrap {step+1}/{steps}; {time.monotonic()-start:.1f}s", flush=True)
            report = {"steps": steps, "policy": "deterministic frozen tracker",
                      "ppo_updates": 0, "same_bootstrap_in_baseline": True,
                      "rank_seed": bootstrap_seed, "trajectory_sha256": trace.hexdigest()}
            if actor.residual_mlp.router is not None:
                report.update(actor.residual_mlp.router.initialize(torch.cat(samples), seed=settings["seed"] + 31009))
                self.alg.copy_router_to_critic()
                report["centers_sha256"] = tensor_digest(actor.residual_mlp.router.state_dict().items())
            actor.routing_bootstrap = report
        print(f"History/router bootstrap complete: {report}", flush=True)
