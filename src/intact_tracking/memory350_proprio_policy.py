"""Policy adapter for the noisy-proprio122 Memory350 input version."""

import torch

from intact_tracking.memory350_native_policy import NativePolicyWrapper, validate_context
from intact_tracking.memory350_policy_env import Memory350PolicyWrapper
from intact_tracking.memory350_proprio_inputs import (
    INPUT_CONTRACT, current_proprio, control_action, validate_proprio_observations,
    validate_input_checkpoint, PROPRIO_ARCHITECTURE,
)
from intact_tracking.residual_dr_aux import (
    DR_TARGET_GROUP, DR_HISTORY_WEIGHT_GROUP, capture_dr_aux_targets, dr_history_weight,
)


def validate_proprio_context(state, profile):
    validate_context(state, profile)
    validate_input_checkpoint(state)


class ProprioNativePolicyWrapper(NativePolicyWrapper):
    def __init__(self, env, clip_actions, checkpoint=None, *, dr_aux_schema=None,
                 dr_aux_allow_extra_parameters=False, latent_input_mode="learned", **kwargs):
        if latent_input_mode not in ("learned", "zero"):
            raise ValueError("latent_input_mode must be learned or zero")
        self.latent_input_mode = latent_input_mode
        if checkpoint is not None and checkpoint.config.architecture_version != PROPRIO_ARCHITECTURE:
            raise ValueError("Noisy proprio inputs cannot be passed to the old truth71 encoder")
        validate_proprio_observations(env.unwrapped)
        self._dr_aux_targets = None
        super().__init__(env, clip_actions, checkpoint, **kwargs)
        if dr_aux_schema is not None:
            if self.context is None:
                raise ValueError("DR auxiliary supervision requires encoder interaction history")
            # Native physics is static after nominal-world restoration and is
            # already audited by NativePolicyWrapper throughout training.
            self._dr_aux_targets = capture_dr_aux_targets(self.unwrapped, dr_aux_schema,
                allow_extra_parameters=dr_aux_allow_extra_parameters)

    def _encode(self):
        if self.latent_input_mode == "zero":
            # Retain identical history bookkeeping and auxiliary-loss validity.
            # The frozen encoder forward is unnecessary for a constant input.
            return torch.zeros(self.num_envs, 64, device=self.device)
        return super()._encode()

    def _attach(self, obs):
        obs = super()._attach(obs)
        if self._dr_aux_targets is not None:
            obs.set(DR_TARGET_GROUP, self._dr_aux_targets)
            # A fresh tensor is essential: old observations are still pending
            # in the PPO transition while the next history is being advanced.
            obs.set(DR_HISTORY_WEIGHT_GROUP, dr_history_weight(self.context.memory))
        return obs

    def _context_state(self):
        # compute() returns the already cached policy observation; no history
        # update and no second noise draw. It also handles external env resets.
        return current_proprio(self.unwrapped.observation_manager.compute())

    def _context_action(self):
        return control_action(self.unwrapped)

    @property
    def latent_metrics(self):
        return {**Memory350PolicyWrapper.latent_metrics.fget(self),
                "latent_input_is_zero": float(self.latent_input_mode == "zero"),
                "context_proprio_dim": INPUT_CONTRACT["state_dim"],
                "nominal_physics_max_error": self.unwrapped.native_policy_runtime_audit["last_parameter_audit_max_error"]}
