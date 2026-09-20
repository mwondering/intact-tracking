"""Direct concatenation of five context frames and the current frozen tracker action."""

import math

import torch
from torch import nn
from rsl_rl.utils import unpad_trajectories

from intact_tracking.limb_context_distributed import DistributedResidualPPO
from intact_tracking.limb_context_policy import (
    ConditionedMLP, LimbContextResidualActor, LimbContextCritic, configure_context_models,
)
from intact_tracking.residual_policy import DYNAMICS_LATENT_GROUP, FrozenTrackerResidualActor
from intact_tracking.residual_dr_aux import (
    DR_DIM, DR_TARGET_GROUP, DR_HISTORY_WEIGHT_GROUP, GROUPS, MAE_UNITS,
    DRAuxiliaryObjective,
)

VERSION = "memory350_native_proprio122_history5_tracker_action_residual_v2"
ACTION_GROUP = "frozen_tracker_action"
LATENT_HISTORY_FRAMES = 5
LATENT_FRAME_DIM = 64
LATENT_HISTORY_DIM = LATENT_HISTORY_FRAMES * LATENT_FRAME_DIM
TRACKER_ACTION_DIM = 29
MOTION_BOUNDARY_CONTRACT = {
    "version": "spv5_3_motion_boundary_v1",
    "signal": "post-step extras.motion_resample_boundary",
    "gae": "cut at environment done or motion boundary",
    "bootstrap": "gamma * pre-transition value at timeout OR motion boundary, once",
    "environment_episode_and_context_memory": "unchanged",
}


def _input(obs, name, width):
    value = obs.get(name)
    if not isinstance(value, torch.Tensor) or value.shape != (*obs.batch_size, width):
        raise ValueError(f"Observation {name!r} must have shape {(*obs.batch_size, width)}")
    return value.detach()


class TrackerActionResidualActor(LimbContextResidualActor):
    def __init__(self, obs, obs_groups, obs_set, output_dim, *, fusion_mode="concat",
                 dynamics_latent_dim=LATENT_HISTORY_DIM, tracker_action_dim=TRACKER_ACTION_DIM,
                 dr_aux_schema=None, dr_aux_motor_weight=0.0,
                 **kwargs):
        if fusion_mode != "concat" or output_dim != tracker_action_dim:
            raise ValueError("Tracker-action policy requires concat and one tracker command per action")
        _input(obs, DYNAMICS_LATENT_GROUP, dynamics_latent_dim)
        super().__init__(obs, obs_groups, obs_set, output_dim, fusion_mode="baseline",
                         dynamics_latent_dim=dynamics_latent_dim, **kwargs)
        self.tracker_action_dim = tracker_action_dim
        self.fusion_mode = "concat"
        self.use_dynamics_latent = True
        self.residual_mlp = ConditionedMLP(
            self.residual_mlp, dynamics_latent_dim + tracker_action_dim, "concat")
        self.residual_input_dim += dynamics_latent_dim + tracker_action_dim
        self.dr_aux_head = None
        self.dr_aux_objective = None
        self.dr_aux_features = None
        if dr_aux_schema is not None:
            self.dr_aux_objective = DRAuxiliaryObjective(dr_aux_schema, dr_aux_motor_weight)
            output = self.residual_mlp.base[-1]
            # Preserve the action/critic/environment RNG streams. A nonzero DR
            # head gives the shared trunk gradients even with the zero action head.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(self.initialization_seed or 0) + 30001)
                self.dr_aux_head = nn.Linear(output.in_features, DR_DIM)
                nn.init.constant_(self.dr_aux_head.bias, 0.5)
            self.dr_aux_head.to(device=output.weight.device, dtype=output.weight.dtype)
            self.dr_aux_objective.to(device=output.weight.device)

    def _residual(self, value):
        if self.dr_aux_head is None:
            return super()._residual(value)
        features = self.residual_mlp.forward_features(value)
        self.dr_aux_features = features if torch.is_grad_enabled() else None
        output = self.residual_mlp.base[-1](features)
        return output if self.residual_output_mode == "unbounded" else self.residual_scale * output.tanh()

    def predict_dr(self, obs):
        """Predict normalized parameters without requiring privileged labels."""
        if self.dr_aux_head is None:
            raise RuntimeError("This checkpoint has no DR auxiliary head")
        features, action = self._base_features_and_action(obs)
        hidden = self.residual_mlp.forward_features(self._residual_input(obs, features, action))
        return self.dr_aux_head(hidden)

    @torch.no_grad()
    def populate_tracker_cache(self, obs):
        # The base constructor also calls this before RolloutStorage determines
        # its schema. Store a real current action, never a zero placeholder.
        super().populate_tracker_cache(obs)
        _, action = FrozenTrackerResidualActor._base_features_and_action(self, obs)
        obs.set(ACTION_GROUP, action.detach().clone())

    @torch.no_grad()
    def _base_features_and_action(self, obs):
        # Reuse the frozen output captured with this observation, including in
        # shuffled PPO minibatches. It is independent of last_base_action.
        if ACTION_GROUP not in obs:
            self.populate_tracker_cache(obs)
        features = self.tracker.get_latent(obs).detach()
        return features, _input(obs, ACTION_GROUP, self.tracker_action_dim).to(features)

    def _residual_input(self, obs, tracker_features, base_action, *, latent_override=None):
        value = super()._residual_input(obs, tracker_features, base_action,
                                        latent_override=latent_override)
        return torch.cat((value, base_action.to(tracker_features).detach()), -1)


class TrackerActionCritic(LimbContextCritic):
    def __init__(self, obs, *args, fusion_mode="concat", dynamics_latent_dim=LATENT_HISTORY_DIM,
                 tracker_action_dim=TRACKER_ACTION_DIM, **kwargs):
        if fusion_mode != "concat":
            raise ValueError("Tracker-action critic requires concat")
        _input(obs, DYNAMICS_LATENT_GROUP, dynamics_latent_dim)
        _input(obs, ACTION_GROUP, tracker_action_dim)
        super().__init__(obs, *args, fusion_mode="baseline", **kwargs)
        self.fusion_mode = "concat"
        self.dynamics_latent_dim = dynamics_latent_dim
        self.tracker_action_dim = tracker_action_dim
        self.value_input_dim = self.obs_dim + dynamics_latent_dim + tracker_action_dim
        self.mlp = ConditionedMLP(self.mlp, dynamics_latent_dim + tracker_action_dim, "concat")

    def value_input(self, obs):
        features = self.obs_normalizer(self._flat_obs(obs))
        latent = _input(obs, DYNAMICS_LATENT_GROUP, self.dynamics_latent_dim).to(features)
        action = _input(obs, ACTION_GROUP, self.tracker_action_dim).to(features)
        return torch.cat((features, latent, action), -1)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        return self.mlp(self.value_input(obs))


class TrackerActionPPO(DistributedResidualPPO):
    motion_boundary_contract = MOTION_BOUNDARY_CONTRACT

    def __init__(self, *args, dr_aux_coef=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(dr_aux_coef) or dr_aux_coef < 0:
            raise ValueError("DR auxiliary coefficient must be finite and nonnegative")
        self.dr_aux_coef = float(dr_aux_coef)
        if (self.dr_aux_coef > 0) != (self.actor.dr_aux_head is not None):
            raise ValueError("Enable the DR actor head and a positive PPO DR coefficient together")
        if self.dr_aux_coef > 0:
            for key in (DR_TARGET_GROUP, DR_HISTORY_WEIGHT_GROUP):
                if key not in self.storage.observations:
                    raise ValueError(f"DR auxiliary PPO requires rollout observation {key!r}")
            self._dr_aux_stats = torch.zeros(20, device=self.device)
            self._dr_aux_batches = 0
        # Allocate outside the rollout's inference_mode: update() clears these
        # counters after leaving that context.
        self._motion_boundary_counts = torch.zeros(4, dtype=torch.long, device=self.device)

    def _prepare_observations(self, obs):
        if ACTION_GROUP not in obs:
            self.actor.populate_tracker_cache(obs)

    def act(self, obs):
        self._prepare_observations(obs)
        return super().act(obs)

    def compute_returns(self, obs):
        # This also works for a fresh next-state observation with no preceding
        # actor forward. Ordinary process_env_step has already populated it via
        # actor.update_normalization, without changing frozen normalizers.
        self._prepare_observations(obs)
        return super().compute_returns(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        # Match SP_Tracking.SPV53MotionBoundaryPPO. The base RSL PPO adds
        # gamma * transition.values to timeout rewards, then uses gae_dones
        # to exclude the newly sampled motion's value and future advantages.
        boundary = extras.get("motion_resample_boundary")
        if not isinstance(boundary, torch.Tensor):
            raise KeyError("TrackerActionPPO requires extras.motion_resample_boundary")
        boundary = boundary.to(device=dones.device, dtype=torch.bool)
        if boundary.shape != dones.shape:
            raise ValueError("motion_resample_boundary shape must match dones")
        gae_dones = (dones.bool() | boundary).to(dtype=dones.dtype)
        ppo_extras = dict(extras)
        time_outs = ppo_extras.get("time_outs")
        if time_outs is None:
            time_outs = torch.zeros_like(boundary)
        else:
            time_outs = time_outs.to(device=dones.device, dtype=torch.bool)
            if time_outs.shape != boundary.shape:
                raise ValueError("time_outs shape must match motion_resample_boundary")
        ppo_extras["time_outs"] = time_outs | boundary
        counts = torch.stack((boundary.sum(), (boundary & ~dones.bool()).sum(),
                              (boundary & time_outs).sum(),
                              boundary.new_tensor(boundary.numel(), dtype=torch.long)))
        self._motion_boundary_counts.add_(counts)
        return super().process_env_step(obs, rewards, gae_dones, ppo_extras)

    def update(self):
        if self.dr_aux_coef > 0:
            self._dr_aux_stats.zero_()
            self._dr_aux_batches = 0
        result = super().update()
        if self.dr_aux_coef > 0:
            if self.is_multi_gpu:
                torch.distributed.all_reduce(self._dr_aux_stats)
            stats = self._dr_aux_stats
            support, positive, count, kl, ppo_sq, aux_sq = stats[14:].tolist()
            raw = float((stats[:7] @ self.actor.dr_aux_objective.weights).item()) / max(count, 1)
            result.update({
                "AuxDR/loss": raw,
                "AuxDR/weighted_loss": self.dr_aux_coef * raw,
                "AuxDR/coefficient": self.dr_aux_coef,
                "AuxDR/history_weight_mean": support / max(count, 1),
                "AuxDR/valid_fraction": positive / max(count, 1),
                "AuxDR/policy_kl": kl / max(count, 1),
                # Activation gradients at the last shared hidden layer, sampled
                # on the first minibatch; the critic has no path to this layer.
                "AuxDR/shared_ppo_gradient_norm": math.sqrt(ppo_sq / self.gpu_world_size),
                "AuxDR/shared_aux_gradient_norm": math.sqrt(aux_sq / self.gpu_world_size),
                "AuxDR/shared_gradient_ratio": math.sqrt(aux_sq / ppo_sq) if ppo_sq > 0 else 0.,
                "AuxDR/shared_gradient_ratio_valid": float(ppo_sq > 0),
            })
            mse, mae = (stats[:14] / max(support, 1e-12)).split(7)
            for group, unit, error, absolute in zip(GROUPS, MAE_UNITS, mse.tolist(), mae.tolist(), strict=True):
                if group not in self.actor.dr_aux_objective.supervised_groups:
                    continue
                result[f"AuxDR/{group}_mse_normalized"] = error
                result[f"AuxDR/{group}_mae_{unit}"] = absolute
            self.actor.dr_aux_features = None
        counts = getattr(self, "_motion_boundary_counts", None)
        if counts is not None:
            total, nonterminal, overlap, transitions = counts.tolist()
            result.update(motion_boundary_fraction=total / max(transitions, 1),
                          motion_boundary_nonterminal_count=float(nonterminal),
                          motion_boundary_timeout_overlap_count=float(overlap))
            counts.zero_()
        return result

    def auxiliary_loss(self, batch, policy_loss):
        if self.dr_aux_coef == 0:
            return None
        hidden = self.actor.dr_aux_features
        if hidden is None:
            raise RuntimeError("DR supervision requires the current actor's shared hidden features")
        prediction = self.actor.dr_aux_head(hidden)
        loss, stats = self.actor.dr_aux_objective(
            prediction, batch.observations[DR_TARGET_GROUP], batch.observations[DR_HISTORY_WEIGHT_GROUP])
        weighted = self.dr_aux_coef * loss
        grad_stats = prediction.new_zeros(2)
        if self._dr_aux_batches == 0:
            ppo_grad = torch.autograd.grad(policy_loss, hidden, retain_graph=True)[0]
            aux_grad = torch.autograd.grad(weighted, hidden, retain_graph=True)[0]
            grad_stats = torch.stack((ppo_grad.detach().square().sum(), aux_grad.detach().square().sum()))
        with torch.no_grad():
            kl = self.actor.get_kl_divergence(batch.old_distribution_params,
                                             self.actor.output_distribution_params).sum()
            self._dr_aux_stats.add_(torch.cat((stats, kl.view(1), grad_stats)))
        self._dr_aux_batches += 1
        return weighted


def configure_tracker_action_models(train, fusion, *, scratch_seed=None,
                                    dr_aux_schema=None, dr_aux_coef=0.0, dr_aux_motor_weight=0.0):
    if fusion != "concat":
        raise ValueError("This experiment uses direct concatenation")
    result = configure_context_models(train, fusion, scratch_seed=scratch_seed)
    for name, class_name in (("actor", "TrackerActionResidualActor"), ("critic", "TrackerActionCritic")):
        result[name].update(
            class_name=f"intact_tracking.memory350_tracker_action_policy:{class_name}",
            dynamics_latent_dim=LATENT_HISTORY_DIM, tracker_action_dim=TRACKER_ACTION_DIM)
    if not math.isfinite(dr_aux_coef) or dr_aux_coef < 0:
        raise ValueError("DR auxiliary coefficient must be finite and nonnegative")
    if dr_aux_coef > 0:
        if dr_aux_schema is None:
            raise ValueError("DR auxiliary training requires the native context checkpoint schema")
        DRAuxiliaryObjective(dr_aux_schema, dr_aux_motor_weight)
        result["actor"].update(dr_aux_schema=dr_aux_schema, dr_aux_motor_weight=dr_aux_motor_weight)
        result.setdefault("algorithm", {})["dr_aux_coef"] = dr_aux_coef
    return result


def audit_tracker_action_models(actor, critic, obs, fusion, *, original_audit):
    actor.populate_tracker_cache(obs)
    result = original_audit(actor, critic, obs, fusion)
    with torch.no_grad():
        features, action = actor._base_features_and_action(obs)
        expected = actor.tracker.distribution.deterministic_output(actor.tracker.mlp(features))
        torch.testing.assert_close(action, expected, atol=0, rtol=0)
        actor_input = actor._residual_input(obs, features, action)
        critic_input = critic.value_input(obs)
        torch.testing.assert_close(actor_input[:, features.shape[-1]:-TRACKER_ACTION_DIM],
                                   obs[DYNAMICS_LATENT_GROUP], atol=0, rtol=0)
        torch.testing.assert_close(critic_input[:, critic.obs_dim:-TRACKER_ACTION_DIM],
                                   obs[DYNAMICS_LATENT_GROUP], atol=0, rtol=0)
        for value in (actor_input, critic_input):
            torch.testing.assert_close(value[:, -TRACKER_ACTION_DIM:], action, atol=0, rtol=0)
        if any(p.requires_grad for p in actor.tracker.parameters()) or actor.tracker.training:
            raise RuntimeError("Tracker weights and normalization must remain frozen")
    result.update(
        latent_dimensions=LATENT_HISTORY_DIM, latent_frame_dimensions=LATENT_FRAME_DIM,
        latent_history_frames=LATENT_HISTORY_FRAMES,
        latent_history_order="oldest to newest; left zero padding; clear at episode/motion boundary",
        encoder_long_memory="preserved across episode/motion boundary under existing physics rules",
        actor_input_dimensions=actor.residual_input_dim, critic_input_dimensions=critic.value_input_dim,
        actor_tracker_action_input_dim=TRACKER_ACTION_DIM, critic_tracker_action_input_dim=TRACKER_ACTION_DIM,
        input_order="original features, five latent frames, current frozen tracker action",
        tracker_action_source="current deterministic tracker output before residual, exploration or SP processing",
        tracker_action_storage="detached observation tensor shared by actor and critic; stored in rollout buffer",
        tracker_action_normalization="raw command appended after original feature normalization",
        tracker_action_matches_actor_and_critic=True, tracker_frozen=True)
    if actor.dr_aux_head is not None:
        result["dr_auxiliary"] = {
            "shared_hidden_dim": actor.dr_aux_head.in_features, "output_dim": DR_DIM,
            "groups": list(GROUPS), "mass_supervised": False,
            "supervised_groups": list(actor.dr_aux_objective.supervised_groups),
            "group_weights_normalized": actor.dr_aux_objective.weights.tolist(),
            "normalization": "fixed physical range; no encoder distance weights",
            "history_weight": "available short steps plus valid archived steps, divided by 350",
            "targets_are_actor_inputs": False,
        }
    return result
