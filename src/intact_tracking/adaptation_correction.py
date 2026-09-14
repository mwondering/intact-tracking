"""Small static-physics correction around an unchanged deployable policy."""

from __future__ import annotations

import copy

import torch
from rsl_rl.modules import MLP

from intact_tracking.adaptation_policy import ContextAdaptationActor, expanded_model_field
from intact_tracking.residual_policy import _last_linear

CORRECTION_PHYSICS = "adaptation_correction_physics"


def correction_config_requires_privilege(cfg):
    return cfg["class_name"].endswith(":ContextPhysicsCorrectionActor") and bool(
        cfg.get("correction_use_privilege", True)
    )


class StaticPhysicsCorrectionWrapper:
    """Only three true static quantities: payload mass and torso COMx/y.

    Values are freshly read after every observation, not stale reset labels.
    This wrapper must never be used by a deployed correction policy.
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped
        model = self.unwrapped.sim.mj_model
        self.body_ids = []
        for name in ("right_wrist_yaw_link", "torso_link"):
            indices = [i for i in range(model.nbody) if model.body(i).name.split("/")[-1] == name]
            if len(indices) != 1:
                raise ValueError(f"Ambiguous static correction body: {name}")
            self.body_ids.append(indices[0])

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def attach(self, obs):
        wrist, torso = self.body_ids
        mass, default_mass = expanded_model_field(self.unwrapped, "body_mass")
        com, default_com = expanded_model_field(self.unwrapped, "body_ipos")
        code = torch.stack((
            mass[:, wrist] - default_mass[wrist] - 2.0,
            (com[:, torso, 0] - default_com[torso, 0]) / 0.075,
            (com[:, torso, 1] - default_com[torso, 1]) / 0.075,
        ), -1)
        obs.set(CORRECTION_PHYSICS, code.detach())
        return obs

    def get_observations(self):
        return self.attach(self.wrapped.get_observations())

    def step(self, actions):
        obs, reward, done, extras = self.wrapped.step(actions)
        return self.attach(obs), reward, done, extras

    def reset(self):
        obs, extras = self.wrapped.reset()
        return self.attach(obs), extras


class ContextPhysicsCorrectionActor(ContextAdaptationActor):
    """One actor: frozen source policy plus a trainable bounded correction.

    Teacher mode uses ONLY three additional static truths, never clean state
    or clean reference. Student mode obtains the same three coordinates solely
    from its context latent through a copied, deployable prediction head.
    """

    def __init__(
        self, *args, correction_use_privilege=True, correction_scale=0.25,
        correction_hidden_dims=(256, 128), **kwargs,
    ):
        for name in ("train_base_policy", "train_context_encoder", "train_residual_policy", "train_reference_encoder"):
            if kwargs.get(name, False):
                raise ValueError("Static correction requires the original policy to remain frozen")
            kwargs[name] = False
        super().__init__(*args, **kwargs)
        if self.context_latent_mean or self.feature_adapter is not None or self.context_residual_log_gain is not None:
            raise ValueError("Static correction currently requires an ordinary context policy")
        if self.context_auxiliary_head is None or self.context_auxiliary_head[-1].out_features < 10:
            raise ValueError("Static correction requires the trained physics prediction head")
        if correction_scale <= 0:
            raise ValueError("Correction scale must be positive")
        self.correction_use_privilege = bool(correction_use_privilege)
        self.correction_scale = float(correction_scale)
        self.correction_base_width = self.tracker.policy_input_dim + self.adaptation_latent_dim
        self.correction_physics_decoder = copy.deepcopy(self.context_auxiliary_head).requires_grad_(False)
        self.correction_mlp = MLP(
            self.correction_base_width + 3,
            self.residual_mlp[-1].out_features,
            correction_hidden_dims,
            "elu",
        )
        output = _last_linear(self.correction_mlp)
        torch.nn.init.zeros_(output.weight)
        torch.nn.init.zeros_(output.bias)

    def estimated_correction_physics(self, latent):
        # Head coordinates: body velocity[0:3], payload-2[3], then
        # torso mass[4], COMx/y/z[5:8], friction[8], armature[9].
        return self.correction_physics_decoder(latent)[..., [3, 5, 6]].clamp(-1, 1)

    def _residual_input(self, obs, tracker_features, base_action, *, latent_override=None):
        source_input = super()._residual_input(
            obs, tracker_features, base_action, latent_override=latent_override
        )
        code = (
            obs[CORRECTION_PHYSICS] if self.correction_use_privilege
            else self.estimated_correction_physics(source_input[..., -self.adaptation_latent_dim:])
        )
        return torch.cat((source_input, code), -1)

    def _residual(self, value):
        if value.shape[-1] != self.correction_base_width + 3:
            raise ValueError("Correction input must explicitly include its three physics coordinates")
        original = super()._residual(value[..., :self.correction_base_width].contiguous())
        correction = self.correction_scale * self.correction_mlp(value).tanh()
        return original + correction

    @torch.no_grad()
    def policy_metrics(self, obs):
        features, base = self._base_features_and_action(obs)
        value = self._residual_input(obs, features, base)
        correction = self.correction_scale * self.correction_mlp(value).tanh()
        shuffled = value.clone()
        shuffled[..., -3:] = value[..., -3:].roll(1, 0)
        alternative = self.correction_scale * self.correction_mlp(shuffled).tanh()
        return {
            "correction_action_rms": float(correction.square().mean().sqrt()),
            "correction_action_abs_max": float(correction.abs().max()),
            "correction_saturation_fraction": float((correction.abs() > 0.95 * self.correction_scale).float().mean()),
            "correction_physics_shuffle_action_delta_rms": float((alternative - correction).square().mean().sqrt()),
        }


def correction_initialization_state(actor, state):
    """Upgrade weights, rejecting any missing source component or unexpected key."""
    state = dict(state)
    if any(key.startswith("correction_") for key in state):
        return state
    for key, value in actor.state_dict().items():
        if key.startswith("correction_physics_decoder."):
            source_key = key.replace("correction_physics_decoder.", "context_auxiliary_head.", 1)
            state[key] = state[source_key].clone()
        elif key.startswith("correction_mlp."):
            state[key] = value.clone()
    return state
