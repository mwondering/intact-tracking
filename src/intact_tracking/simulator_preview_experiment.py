"""Strict preview-vs-original-input experiment (v2).

Only the preview arm receives future target/state/coordinate-wise error.
Neither arm receives the old current-state/physics privilege group. Both
critics are single, randomly initialized networks with the same hidden widths.
Legacy v1 classes remain in simulator_preview solely for old artifacts.
"""

from __future__ import annotations

import copy

import torch
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from intact_tracking.residual_policy import FrozenTrackerResidualActor, _make_heft_mlp
from intact_tracking.simulator_preview import HORIZON, STATE_DIM, FrozenTrackerPreview

PREVIEW_GROUP = "simulator_preview"
PREVIEW_DIM = HORIZON * STATE_DIM * 3  # 355 targets + 710 states/errors; no scalars.
CRITIC_HIDDEN_DIMS = (1024, 512, 512, 256)
EXPERIMENT_VERSION = "simulator_preview_original_inputs_v2"


class PreviewObservationWrapper(RslRlVecEnvWrapper):
    """Original observations plus 1065 future features; never current privileges."""

    def __init__(self, env, clip_actions=None, *, preview_mode="true"):
        if preview_mode not in ("true", "zero", "shuffle"):
            raise ValueError("Expected true/zero/shuffle preview intervention")
        self.preview_mode = preview_mode
        self.preview = self.tracker = None
        super().__init__(env, clip_actions=clip_actions)

    def bind(self, actor):
        if any(p.requires_grad for p in actor.tracker.parameters()):
            raise ValueError("Preview requires a fully frozen tracker")
        self.tracker = actor.tracker
        if self.preview is None:
            self.preview = FrozenTrackerPreview(
                self.unwrapped, self.clip_actions, include_distances=False,
            )

    def attach(self, obs):
        if self.preview is None:
            value = torch.zeros(self.num_envs, PREVIEW_DIM, device=self.device)
        else:
            reference, outcome = self.preview.query(self.tracker, obs)
            value = torch.cat((reference, outcome), -1)
            if self.preview_mode == "zero":
                value = torch.zeros_like(value)
            elif self.preview_mode == "shuffle":
                value = value.roll(1, 0)
        if value.shape != (self.num_envs, PREVIEW_DIM):
            raise ValueError(f"Unexpected v2 preview shape: {value.shape}")
        return obs.set(PREVIEW_GROUP, value)

    def get_observations(self):
        return self.attach(super().get_observations())

    def step(self, actions):
        obs, reward, done, extras = super().step(actions)
        return self.attach(obs), reward, done, extras

    def reset(self):
        obs, extras = super().reset()
        return self.attach(obs), extras

    def close_preview(self):
        if self.preview is not None:
            self.preview.close()


class PreviewResidualActor(FrozenTrackerResidualActor):
    """Frozen original frontend; normalized raw preview concatenated to its features.

    No 64-D bottleneck, privileged height/contact replacement, or direct physics
    input. Only the residual MLP and exploration distribution learn.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        groups = copy.deepcopy(obs_groups)
        if groups[obs_set].count(PREVIEW_GROUP) != 1:
            raise ValueError("Preview actor requires exactly one simulator_preview group")
        groups[obs_set].remove(PREVIEW_GROUP)
        if (kwargs.get("use_dynamics_latent") is not True
                or kwargs.get("dynamics_latent_group") != PREVIEW_GROUP
                or kwargs.get("dynamics_latent_dim") != PREVIEW_DIM
                or kwargs.get("residual_input_mode") != "tracker_features"):
            raise ValueError("Preview actor must concatenate all 1065 future features")
        super().__init__(obs, groups, obs_set, output_dim, **kwargs)
        self.obs_groups = list(obs_groups[obs_set])
        self.preview_normalizer = EmpiricalNormalization(PREVIEW_DIM)

    def _residual_input(self, obs, tracker_features, base_action, *, latent_override=None):
        del base_action
        value = obs[PREVIEW_GROUP] if latent_override is None else latent_override
        value = self.preview_normalizer(value.to(tracker_features.dtype)).clamp(-10, 10)
        return torch.cat((tracker_features, value), -1)

    def update_normalization(self, obs):
        super().update_normalization(obs)
        self.preview_normalizer.update(obs[PREVIEW_GROUP])


class ScratchHeftCritic(torch.nn.Module):
    """One randomly initialized input->1024->512->512->256->1 HEFT value MLP.

    The checkpoint supplies the observation schema, NOT parameters or running
    statistics. Actor and critic do not share encoders or trainable parameters.
    """

    is_recurrent = False

    def __init__(self, obs, obs_groups, obs_set, output_dim, *,
                 hidden_dims=CRITIC_HIDDEN_DIMS, activation="mish",
                 obs_normalization=True, distribution_cfg=None):
        super().__init__()
        if tuple(hidden_dims) != CRITIC_HIDDEN_DIMS or activation != "mish":
            raise ValueError("This comparison locks critic widths to 1024,512,512,256 with Mish")
        if output_dim != 1 or distribution_cfg is not None:
            raise ValueError("Critic must be a scalar deterministic value")
        self.obs_groups = list(obs_groups[obs_set])
        self.obs_dim = sum(int(obs[name].shape[-1]) for name in self.obs_groups)
        self.obs_normalization = obs_normalization
        self.obs_normalizer = (
            EmpiricalNormalization(self.obs_dim) if obs_normalization else torch.nn.Identity()
        )
        self.mlp = _make_heft_mlp(self.obs_dim, hidden_dims, output_dim)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        value = torch.cat([obs[name] for name in self.obs_groups], -1)
        return self.mlp(self.obs_normalizer(value))

    def update_normalization(self, obs):
        if self.obs_normalization:
            self.obs_normalizer.update(torch.cat([obs[name] for name in self.obs_groups], -1))

    def reset(self, dones=None, hidden_state=None):
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        pass


def configure_preview_comparison(train, variant):
    """Apply an explicit architecture/input contract to a fresh residual config."""
    if variant not in ("preview", "baseline"):
        raise ValueError("variant must be preview or baseline")
    train = copy.deepcopy(train)
    original = train["actor"]["tracker_obs_groups"]
    if train["obs_groups"] != original:
        raise ValueError("Start from the original checkpoint observation groups")
    train["critic"] = {
        "class_name": "intact_tracking.simulator_preview_experiment:ScratchHeftCritic",
        "hidden_dims": list(CRITIC_HIDDEN_DIMS),
        "activation": "mish", "obs_normalization": True, "distribution_cfg": None,
    }
    if variant == "preview":
        train["actor"].update(
            class_name="intact_tracking.simulator_preview_experiment:PreviewResidualActor",
            use_dynamics_latent=True, dynamics_latent_group=PREVIEW_GROUP,
            dynamics_latent_dim=PREVIEW_DIM,
        )
        for name in ("actor", "critic"):
            train["obs_groups"][name].append(PREVIEW_GROUP)
    return train


def audit_input_contract(actor, critic, obs: TensorDict, variant):
    original = actor.tracker.obs_groups
    expected_actor = list(original) + ([PREVIEW_GROUP] if variant == "preview" else [])
    if actor.obs_groups != expected_actor:
        raise ValueError(f"Actor input contract mismatch: {actor.obs_groups}")
    expected_critic = ["policy", "priv"] + ([PREVIEW_GROUP] if variant == "preview" else [])
    if critic.obs_groups != expected_critic:
        raise ValueError(f"Critic input contract mismatch: {critic.obs_groups}")
    if any(p.requires_grad for p in actor.tracker.parameters()):
        raise ValueError("Frozen tracker has trainable parameters")
    if "adaptation_privilege" in obs:
        raise ValueError("Old current state/physics privileges must not enter either arm")
    if (PREVIEW_GROUP in obs) != (variant == "preview"):
        raise ValueError("Baseline must have no preview tensor, even a zero-filled one")
    base_dim = sum(int(obs[name].shape[-1]) for name in expected_critic if name != PREVIEW_GROUP)
    if base_dim != 6330:
        raise ValueError(f"Expected original checkpoint critic dimension 6330, got {base_dim}")
    return {
        "actor_groups": list(actor.obs_groups), "critic_groups": list(critic.obs_groups),
        "original_critic_dim": base_dim, "critic_input_dim": critic.obs_dim,
        "extra_input_dim": PREVIEW_DIM if variant == "preview" else 0,
        "actor_residual_input_dim": actor.residual_input_dim,
        "critic_hidden_dims": list(CRITIC_HIDDEN_DIMS),
        "critic_initialization": "random; no checkpoint weights or normalization statistics",
        "critic_additive_branches": False, "scalar_distance_features": False,
        "current_state_physics_privilege_group": False, "tracker_frozen": True,
    }
