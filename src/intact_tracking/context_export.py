"""Flat-tensor student inference, including all learned preprocessing and context."""

from __future__ import annotations

import copy

import torch

from intact_tracking.adaptation_context_memory import update_context_mean
from intact_tracking.adaptation_policy import ContextAdaptationActor, normalized_context_kinematics
from intact_tracking.adaptation_sensors import IMU_HISTORY

DEPLOYABLE_FIELDS = (
    ("robot_root_quat", 4),
    ("estimator_history", 6100),
    ("reference_encoder_input", 1900),
    ("robot_key_body", 195),
)


def deployable_fields(actor):
    return DEPLOYABLE_FIELDS + (((IMU_HISTORY, 150),) if actor.context_imu_accel else ())


class _ContextInferenceCore(torch.nn.Module):
    """Shared sensor preprocessing and context-conditioned action head."""

    def __init__(self, actor: ContextAdaptationActor):
        super().__init__()
        if not isinstance(actor, ContextAdaptationActor):
            raise TypeError("Only the privilege-free context actor may be exported")
        if getattr(actor, "correction_use_privilege", False):
            raise ValueError("A true-physics correction teacher cannot be exported as deployable")
        self.preprocessing = actor.tracker.as_jit()
        # The tracker export's final two operations are normalization -> MLP ->
        # distribution mean. Expose its normalized features by replacing only
        # the latter two, then compose the complete context-conditioned policy.
        self.preprocessing.mlp = torch.nn.Identity()
        self.preprocessing.deterministic_output = torch.nn.Identity()
        self.base_mlp = copy.deepcopy(actor.tracker.mlp)
        self.base_output = actor.tracker.distribution.as_deterministic_output_module()
        self.context_encoder = copy.deepcopy(actor.context_encoder)
        self.feature_adapter = copy.deepcopy(actor.feature_adapter)
        self.context_imu_accel = actor.context_imu_accel
        self.context_root_orientation = actor.context_root_orientation
        self.context_key_body = actor.context_key_body
        self.history_normalizer = copy.deepcopy(actor.tracker.history_normalizer)
        self.residual_mlp = copy.deepcopy(actor.residual_mlp)
        self.refinement_mlp = copy.deepcopy(getattr(actor, "refinement_mlp", None))
        self.refinement_scale = float(getattr(actor, "refinement_scale", 0.0))
        self.frozen_nominal_prior_mlp = copy.deepcopy(getattr(actor, "frozen_nominal_prior_mlp", None))
        self.frozen_nominal_prior_scale = float(getattr(actor, "frozen_nominal_prior_scale", 0.0))
        self.frozen_nominal_prior_deployable_base = bool(getattr(actor, "frozen_nominal_prior_deployable_base", False))
        self.residual_scale = actor.residual_scale
        self.correction_mlp = copy.deepcopy(getattr(actor, "correction_mlp", None))
        self.correction_physics_decoder = copy.deepcopy(getattr(actor, "correction_physics_decoder", None))
        self.correction_scale = float(getattr(actor, "correction_scale", 0.0))
        self.register_buffer(
            "context_residual_log_gain",
            None if actor.context_residual_log_gain is None else actor.context_residual_log_gain.detach().clone(),
        )

    def encode(self, observation):
        features = self.preprocessing(observation[..., :8199])
        history = self.history_normalizer(observation[..., 4:6104]).clamp(-10, 10)
        if self.context_imu_accel:
            history = torch.cat(
                (history, (observation[..., 8199:8349] / 10.0).clamp(-10, 10)), dim=-1
            )
        latent = self.context_encoder(
            history, observation[..., :4] if self.context_root_orientation else None,
            normalized_context_kinematics(observation[..., 8004:8199]) if self.context_key_body else None,
        )
        return features, latent

    def action(self, features, latent):
        raw_features = features
        if self.feature_adapter is not None:
            features = features + self.feature_adapter(torch.cat((features, latent), dim=-1))
        action_features = raw_features if self.frozen_nominal_prior_deployable_base else features
        base = self.base_output(self.base_mlp(action_features))
        if self.frozen_nominal_prior_mlp is not None:
            base = base + self.frozen_nominal_prior_scale * self.frozen_nominal_prior_mlp(action_features).tanh()
        residual = self.residual_scale * torch.tanh(
            self.residual_mlp(torch.cat((features, latent), dim=-1))
        )
        if self.context_residual_log_gain is not None:
            residual = residual * self.context_residual_log_gain.clamp(-4.0, 0.0).exp()
        if self.refinement_mlp is not None:
            residual = residual + self.refinement_scale * self.refinement_mlp(torch.cat((features, latent), -1)).tanh()
        if self.correction_mlp is not None:
            physics = self.correction_physics_decoder(latent)[..., [3, 5, 6]].clamp(-1, 1)
            residual = residual + self.correction_scale * self.correction_mlp(
                torch.cat((features, latent, physics), -1)
            ).tanh()
        return base + residual


class ContextInferenceModule(_ContextInferenceCore):
    """One 8199-D tensor (8349-D with noisy IMU), action targets out."""

    def __init__(self, actor):
        if actor.context_latent_mean:
            raise ValueError("Causal context memory requires a stateful export; flat stateless export is invalid")
        super().__init__(actor)

    def forward(self, observation):
        features, latent = self.encode(observation)
        return self.action(features, latent)


class StatefulContextInferenceModule(_ContextInferenceCore):
    """Explicit memory input/output: one invocation per sensor control step.

    The caller owns mean[B, latent_dim], count[B, 1] and reset[B]. Initialize
    both state tensors to zero; reset only worlds starting a new episode.
    No simulator or training-only model is needed at inference.
    """

    def __init__(self, actor):
        if not actor.context_latent_mean:
            raise ValueError("Stateful context export requires a causal-memory actor")
        super().__init__(actor)
        self.horizon = actor.context_mean_horizon
        self.mean_only = actor.context_mean_only

    def forward(self, observation, previous_mean, count, reset):
        features, latent = self.encode(observation)
        mean, count = update_context_mean(previous_mean, count, latent, reset, self.horizon)
        action = self.action(features, mean if self.mean_only else torch.cat((latent, mean), -1))
        return action, mean, count
