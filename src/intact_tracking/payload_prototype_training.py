"""PPO bookkeeping for current observation routes and checkpointed load MoEs."""

from pathlib import Path

import torch

from intact_tracking.limb_context_distributed import DistributedResidualPPO, tensor_digest
from intact_tracking.payload_prototype_moe import PrototypeGate
from intact_tracking.payload_prototype_physics import audit_physics
from intact_tracking.residual_runner import ResidualOnPolicyRunner


class PayloadMoEPPO(DistributedResidualPPO):
    def act(self,obs):
        self.transition.hidden_states = (self.actor.get_hidden_state(),self.critic.get_hidden_state())
        # Actor populates raw tracker action and current detached routes BEFORE V.
        self.transition.actions = self.actor(obs,stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def compute_returns(self,obs):
        with torch.no_grad():
            self.actor(obs)
        return super().compute_returns(obs)


class PayloadMoERunner(ResidualOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fixed_gate_digest = (tensor_digest(self.alg.actor.gate.state_dict().items())
                                 if isinstance(self.alg.actor.gate,PrototypeGate) else None)

    def audit_models(self):
        self.alg.actor.assert_tracker_frozen()
        digest = tensor_digest(self.alg.actor.gate.state_dict().items())
        if self.fixed_gate_digest is not None and digest != self.fixed_gate_digest:
            raise RuntimeError("Fixed prototype routing changed during PPO")
        if torch.distributed.is_initialized():
            values = [None]*torch.distributed.get_world_size()
            torch.distributed.all_gather_object(values,digest)
            if len(set(values)) != 1: raise RuntimeError("Router differs between ranks")

    def learn(self,*args,**kwargs):
        def checkpoint(runner):
            if runner.completed_learning_updates % runner.cfg["save_interval"]: return
            audit_physics(runner.env.unwrapped,runner.residual_metadata["physics"])
            runner.audit_models()
            if runner.checkpoint_state_preparer is not None:
                runner.checkpoint_state_preparer(runner)
            if runner.logger.writer is not None:
                runner.save(str(Path(runner.logger.log_dir)/f"checkpoint_update_{runner.completed_learning_updates:06d}.pt"))
        self.checkpoint_evaluator = checkpoint
        result = super().learn(*args,**kwargs)
        self.audit_models()
        return result
