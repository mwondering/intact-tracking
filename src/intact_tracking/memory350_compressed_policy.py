"""Matched learned observation compression followed by frozen 64-D context concatenation."""

import torch
from torch import nn
from rsl_rl.modules import MLP

from intact_tracking.limb_context_policy import (
    LimbContextResidualActor, LimbContextCritic, configure_context_models, SCRATCH_ACTION_STD,
)
from intact_tracking.limb_context_distributed import tensor_digest

VERSION = "memory350_compressed_concat_residual_v1"
ACTION_VERSION = "memory350_compressed_tracker_action_concat_residual_v1"
ACTOR_CLASS = "intact_tracking.memory350_compressed_policy:CompressedContextActor"
CRITIC_CLASS = "intact_tracking.memory350_compressed_policy:CompressedContextCritic"


class CompressedConcatMLP(nn.Module):
    def __init__(self, feature_dim, output_dim, *, compression_dims, head_dims=(256, 128),
                 fusion="concat", seed=0, zero_output=False, tracker_action_dim=0, latent_dim=64):
        super().__init__()
        if fusion not in ("baseline", "concat") or tuple(compression_dims)[-1] != 128:
            raise ValueError("Use a 128-D compressed observation and baseline/concat fusion")
        self.feature_dim, self.latent_dim, self.fusion = feature_dim, int(latent_dim), fusion
        if self.latent_dim not in (64, 320):
            raise ValueError("Compressed context supports one or five 64-D frames")
        self.tracker_action_dim = int(tracker_action_dim)
        if self.tracker_action_dim not in (0, 29):
            raise ValueError("The optional tracker action must have 29 dimensions")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.observation = nn.Sequential(
                MLP(feature_dim, 128, list(compression_dims[:-1]), "elu"), nn.ELU())
            self.head = MLP(128 + self.latent_dim, output_dim, list(head_dims), "elu")
            # Initial values and actions are identical across the two arms.
            # These columns are trainable and receive gradients immediately.
            with torch.no_grad():
                self.head[0].weight[:, 128:].zero_()
            if self.tracker_action_dim:
                # Preserve every original randomly initialized parameter. The
                # new action columns start at zero and learn through PPO.
                original = self.head[0]
                expanded = nn.Linear(original.in_features + self.tracker_action_dim,
                                     original.out_features,
                                     device=original.weight.device, dtype=original.weight.dtype)
                with torch.no_grad():
                    expanded.weight.zero_()
                    expanded.weight[:, :128].copy_(original.weight[:, :128])
                    expanded.weight[:, 128 + self.tracker_action_dim:].copy_(original.weight[:, 128:])
                    expanded.bias.copy_(original.bias)
                self.head[0] = expanded
            if zero_output:
                nn.init.zeros_(self.head[-1].weight)
                nn.init.zeros_(self.head[-1].bias)

    def forward(self, value):
        input_dim = self.feature_dim + self.tracker_action_dim
        if value.shape[-1] not in (input_dim, input_dim + self.latent_dim):
            raise ValueError("Incorrect observation-plus-latent dimensions")
        features = value[..., :self.feature_dim]
        tracker_action = value[..., self.feature_dim:input_dim].detach()
        if self.fusion == "baseline":
            latent = features.new_zeros(*features.shape[:-1], self.latent_dim)
        else:
            if value.shape[-1] != input_dim + self.latent_dim:
                raise ValueError("Latent arm requires the configured frozen context dimensions")
            latent = value[..., input_dim:].detach()
        compressed = self.observation(features)
        return self.head(torch.cat((compressed, tracker_action, latent), dim=-1))


class CompressedContextActor(LimbContextResidualActor):
    def __init__(self, *args, fusion_mode="baseline", compression_dims=(512, 256, 128),
                 head_dims=(256, 128), tracker_action_input=False, latent_history_frames=1, **kwargs):
        if latent_history_frames not in (1, 5):
            raise ValueError("Use one or five latent frames")
        kwargs["dynamics_latent_dim"] = 64 * latent_history_frames
        super().__init__(*args, fusion_mode="baseline", **kwargs)
        self.latent_history_frames = latent_history_frames
        if self.tracker.policy_input_dim != 1645 or tuple(compression_dims) != (512, 256, 128):
            raise ValueError("Actor compression must be 1645 -> 512 -> 256 -> 128")
        self.tracker_action_input = bool(tracker_action_input)
        self.residual_mlp = CompressedConcatMLP(
            1645, self.distribution.output_dim, compression_dims=compression_dims,
            head_dims=head_dims, fusion=fusion_mode, seed=self.initialization_seed, zero_output=True,
            tracker_action_dim=29 if self.tracker_action_input else 0, latent_dim=64 * latent_history_frames)
        self.fusion_mode = fusion_mode
        self.use_dynamics_latent = fusion_mode == "concat"
        self.residual_input_dim = 1645 + (29 if self.tracker_action_input else 0) + (64 * latent_history_frames if self.use_dynamics_latent else 0)

    def _residual_input(self, obs, tracker_features, base_action, *, latent_override=None):
        value = super()._residual_input(obs, tracker_features, base_action,
                                        latent_override=latent_override)
        if not self.tracker_action_input:
            return value
        # Reuse the exact current deterministic tracker output that forward()
        # subsequently adds to the residual. No sample, actuator processing,
        # stale action cache, or additional tracker forward is involved.
        if base_action.shape[:-1] != tracker_features.shape[:-1] or base_action.shape[-1] != 29:
            raise ValueError("Tracker action does not match the residual observation batch")
        features = tracker_features.shape[-1]
        return torch.cat((value[..., :features], base_action.to(tracker_features).detach(),
                          value[..., features:]), dim=-1)

    def configure_grouped_diagnostics(self, num_envs):
        from intact_tracking.limb_context_grouped_dr import GROUPS, group_ids
        ids = group_ids(num_envs)
        if num_envs < 2 * GROUPS:
            raise ValueError("Grouped policy diagnostics need at least two replicas per DR")
        generator = torch.Generator().manual_seed(73521)
        permutation = torch.randperm(GROUPS, generator=generator)
        while (permutation == torch.arange(GROUPS)).any():
            permutation = torch.randperm(GROUPS, generator=generator)
        same = (torch.arange(num_envs) + GROUPS) % num_envs
        cross = (same // GROUPS) * GROUPS + permutation[ids]
        device = next(self.residual_mlp.parameters()).device
        self.register_buffer("_same_dr_donors", same.to(device), persistent=False)
        self.register_buffer("_cross_dr_donors", cross.to(device), persistent=False)

    @torch.no_grad()
    def policy_metrics(self, obs):
        metrics = super().policy_metrics(obs)
        if not hasattr(self, "_same_dr_donors"):
            return metrics
        names = ("latent_same_dr_action_delta_rms", "latent_cross_dr_action_delta_rms",
                 "latent_cross_dr_to_residual_rms", "latent_same_dr_distance", "latent_cross_dr_distance")
        if not self.use_dynamics_latent:
            return {**metrics, **dict.fromkeys(names, 0.0)}
        latent = obs[self.dynamics_latent_group]
        if latent.shape[0] != len(self._same_dr_donors):
            raise ValueError("Grouped diagnostics must use the current full world batch")
        features, action = self._base_features_and_action(obs)
        normal = self._residual(self._residual_input(obs, features, action))
        for kind, donors in (("same", self._same_dr_donors), ("cross", self._cross_dr_donors)):
            alternate = self._residual(self._residual_input(obs, features, action, latent_override=latent[donors]))
            metrics[f"latent_{kind}_dr_action_delta_rms"] = float((normal - alternate).square().mean().sqrt())
            metrics[f"latent_{kind}_dr_distance"] = float((latent - latent[donors]).norm(dim=-1).mean())
        metrics["latent_cross_dr_to_residual_rms"] = metrics["latent_cross_dr_action_delta_rms"] / max(metrics["residual_action_rms"], 1e-8)
        return metrics


class CompressedContextCritic(LimbContextCritic):
    def __init__(self, *args, fusion_mode="baseline", compression_dims=(1024, 512, 256, 128),
                 head_dims=(256, 128), latent_history_frames=1, **kwargs):
        if latent_history_frames not in (1, 5):
            raise ValueError("Use one or five latent frames")
        kwargs["dynamics_latent_dim"] = 64 * latent_history_frames
        super().__init__(*args, fusion_mode="baseline", **kwargs)
        self.latent_history_frames = latent_history_frames
        self.mlp = CompressedConcatMLP(
            self.obs_dim, 1, compression_dims=compression_dims, head_dims=head_dims,
            fusion=fusion_mode, seed=self.initialization_seed, latent_dim=64 * latent_history_frames)
        self.fusion_mode = fusion_mode


def configure_compressed_models(train, fusion, *, scratch_seed=None, tracker_action_input=False, latent_history_frames=1):
    result = configure_context_models(train, fusion, scratch_seed=scratch_seed)
    result["actor"].update(class_name=ACTOR_CLASS, compression_dims=[512, 256, 128], head_dims=[256, 128])
    result["critic"].update(class_name=CRITIC_CLASS, compression_dims=[1024, 512, 256, 128], head_dims=[256, 128])
    if latent_history_frames not in (1, 5):
        raise ValueError("Use one or five latent frames")
    if latent_history_frames != 1:
        for role in ("actor", "critic"):
            result[role].update(latent_history_frames=latent_history_frames, dynamics_latent_dim=64 * latent_history_frames)
    if tracker_action_input:
        result["actor"]["tracker_action_input"] = True
    return result


def audit_initial_models(actor, critic, obs, fusion):
    with torch.no_grad():
        features, base_action = actor._base_features_and_action(obs)
        action = actor(obs)
        torch.testing.assert_close(action, base_action, atol=0, rtol=0)
        torch.testing.assert_close(action, actor.tracker(obs), atol=0, rtol=0)
        actor.distribution.update(action)
        torch.testing.assert_close(actor.output_std, torch.full_like(actor.output_std, SCRATCH_ACTION_STD), atol=0, rtol=0)
        if actor.tracker_action_input:
            residual_input = actor._residual_input(obs, features, base_action)
            torch.testing.assert_close(residual_input[..., 1645:1674], base_action, atol=0, rtol=0)
            assert actor.residual_mlp.head[0].in_features == 128 + 29 + actor.residual_mlp.latent_dim
        if fusion == "concat":
            alternate = obs.clone()
            alternate["dynamics_latent"] = torch.randn_like(obs["dynamics_latent"])
            torch.testing.assert_close(actor(alternate), action, atol=0, rtol=0)
            torch.testing.assert_close(critic(alternate), critic(obs), atol=0, rtol=0)
    return {"actor_original_features": features.shape[-1], "critic_original_features": critic.obs_dim,
            "actor_compression": [1645, 512, 256, 128],
            "critic_compression": [critic.obs_dim, 1024, 512, 256, 128],
            "fusion_input_dim": actor.residual_mlp.head[0].in_features, "latent_slot_dim": actor.residual_mlp.latent_dim,
            "actor_fusion_input_dim": actor.residual_mlp.head[0].in_features,
            "critic_fusion_input_dim": critic.mlp.head[0].in_features,
            "tracker_action_input_dim": 29 if actor.tracker_action_input else 0,
            "tracker_action_input": ("current deterministic raw tracker mean; exact same tensor values as additive base action"
                                     if actor.tracker_action_input else "absent"),
            "latent_dimensions": actor.residual_mlp.latent_dim if fusion == "concat" else 0,
            "latent_history_frames": actor.residual_mlp.latent_dim // 64,
            "baseline_latent_slot": "zeros before the common residual/value heads",
            "identical_initial_action": True, "initial_value_independent_of_latent": True,
            "action_std_initialization": SCRATCH_ACTION_STD,
            "actor_initialization_seed": actor.initialization_seed,
            "critic_initialization_seed": critic.initialization_seed,
            "actor_common_trunk_sha256": tensor_digest(actor.residual_mlp.named_parameters()),
            "critic_common_trunk_sha256": tensor_digest(critic.mlp.named_parameters()),
            "critic_initial_normalizer_sha256": tensor_digest(critic.obs_normalizer.state_dict().items()),
            "critic_initial_normalizer_count": float(critic.obs_normalizer.count),
            "actor_trainable_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
            "critic_trainable_parameters": sum(p.numel() for p in critic.parameters() if p.requires_grad)}
