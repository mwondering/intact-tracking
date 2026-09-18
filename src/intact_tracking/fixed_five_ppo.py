"""Four fixed environment specialists and one independently acting PPO baseline."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from rsl_rl.storage import RolloutStorage

from intact_tracking.memory350_moe_critic_action import attach_tracker_action, reserve_tracker_action
from intact_tracking.residual_policy import ResidualPPO
from intact_tracking.limb_context_distributed import synchronize_critic_normalization, tensor_digest


VERSION = "fixed_radial_four_plus_on_policy_baseline_v1"
ROLES = ("expert_0", "expert_1", "expert_2", "expert_3", "baseline")


@torch.no_grad()
def randomize_paired_episode_phase(env, *, seed):
    """Stagger first timeouts after calibration/resume; keep paired clocks equal."""
    if env.num_envs % 2 or env.max_episode_length < 1:
        raise ValueError("Episode phase randomization needs paired worlds and a positive horizon")
    n = env.num_envs // 2
    generator = torch.Generator(device=env.episode_length_buf.device).manual_seed(seed)
    age = torch.randint(env.max_episode_length, (n,), generator=generator,
                        device=env.episode_length_buf.device, dtype=env.episode_length_buf.dtype)
    env.episode_length_buf[:n].copy_(age)
    env.episode_length_buf[n:].copy_(age)
    return age


def split_episode_endings(dones, timeouts, terminated):
    """Mutually exclusive endings; a failure at the timeout boundary is a failure."""
    done, timeout, failure = dones.bool(), timeouts.bool(), terminated.bool()
    if done.shape != timeout.shape or done.shape != failure.shape:
        raise ValueError("Termination flags must match the world batch")
    failures = done & failure
    normal_timeouts = done & timeout & ~failure
    if not torch.equal(done, normal_timeouts | failures):
        raise ValueError("An episode ending has no timeout or failure flag")
    return normal_timeouts, failures


def dispatch_indices(labels: torch.Tensor) -> list[torch.Tensor]:
    """First N slots are specialist worlds; last N are matched baseline copies."""
    if labels.ndim != 1 or labels.dtype != torch.long or not len(labels):
        raise ValueError("Expected nonempty one-dimensional int64 expert IDs")
    if not bool(((labels >= 0) & (labels < 4)).all()):
        raise ValueError("Expert IDs must be in [0, 3]")
    groups = [(labels == i).nonzero().flatten() for i in range(4)]
    return groups + [torch.arange(len(labels), 2 * len(labels), device=labels.device)]


@dataclass
class FixedPartition:
    centers: torch.Tensor
    reference: torch.Tensor
    boundaries: torch.Tensor
    query_count: torch.Tensor
    labels: torch.Tensor

    @classmethod
    def from_sums(cls, sums, count, reference, boundaries):
        if sums.ndim != 2 or count.shape != sums.shape[:1] or reference.shape != sums.shape[1:]:
            raise ValueError("Invalid per-environment center dimensions")
        if not bool((count > 0).all()) or not bool(torch.isfinite(sums).all()):
            raise ValueError("Every environment needs finite, mature latent queries")
        if boundaries.shape != (3,) or not bool(torch.isfinite(boundaries).all()):
            raise ValueError("Four distance classes require three finite boundaries")
        if not bool((boundaries > 0).all() and (boundaries[1:] > boundaries[:-1]).all()):
            raise ValueError("Distance boundaries must be strictly increasing and positive")
        centers = sums / count[:, None]
        radius = torch.linalg.vector_norm(centers - reference, dim=-1)
        if not bool(torch.isfinite(radius).all()):
            raise ValueError("Nonfinite nominal reference or environment distance")
        # Exact boundaries enter the upper bin, as in the archived NumPy audit.
        labels = torch.bucketize(radius, boundaries, right=True)
        return cls(centers.clone(), reference.clone(), boundaries.clone(), count.clone(), labels)

    def state_dict(self):
        return {k: getattr(self, k).detach().cpu().clone() for k in
                ("centers", "reference", "boundaries", "query_count", "labels")}

    @classmethod
    def from_state_dict(cls, state, device):
        values = {k: v.to(device) for k, v in state.items()}
        result = cls(**values)
        expected = torch.bucketize(torch.linalg.vector_norm(result.centers - result.reference, dim=-1),
                                   result.boundaries, right=True)
        torch.testing.assert_close(result.labels, expected, atol=0, rtol=0)
        dispatch_indices(result.labels)
        return result


@torch.no_grad()
def mirror_baseline_physics(env):
    """Pair physical parameters without coupling actions, resets, or noise."""
    from mjlab.managers.event_manager import RecomputeLevel
    from intact_tracking.fixed_dr_profiles import MOTOR_OBSERVATIONS
    from intact_tracking.independent_nominal_dr import EVENT
    from intact_tracking.limb_context_protocol import PAYLOAD_EVENT

    if env.num_envs % 2:
        raise ValueError("Paired baseline requires an even number of simulator slots")
    n = env.num_envs // 2
    copied = []
    for name in sorted(env.sim.expanded_fields):
        value = getattr(env.sim.model, name)
        if value.shape[0] != env.num_envs:
            raise ValueError(f"Expanded field lacks a world axis: {name}")
        value[n:].copy_(value[:n])
        copied.append(name)
    bias = env.scene["robot"].data.encoder_bias
    bias[n:].copy_(bias[:n])
    payload = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
    payload.mass[n:].copy_(payload.mass[:n])
    motor = env.event_manager.get_term_cfg("motor_params_implicit").func
    for name in MOTOR_OBSERVATIONS:
        value = getattr(motor, name)
        value[n:].copy_(value[:n])
    mixture = env.event_manager.get_term_cfg(EVENT).func
    for tensors in (mixture.masks, mixture.original):
        for value in tensors.values():
            value[n:].copy_(value[:n])
    env.sim.recompute_constants(RecomputeLevel.set_const)
    # GPU reductions used by set_const can differ by a few ulps between
    # otherwise identical worlds. Give each replica the source world's exact
    # derived constants as well as its independent physical parameters.
    for name in copied:
        value = getattr(env.sim.model, name)
        value[n:].copy_(value[:n])
    return {"paired_worlds": n, "expanded_fields": copied,
            "encoder_bias_payload_motor_observations_paired": True,
            "actions_noise_and_motion_not_copied": True}


class FixedExpertPPO(ResidualPPO):
    """Latent-free PPO with the current frozen tracker mean supplied to critic."""

    def act(self, obs):
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        attach_tracker_action(obs, self.actor.last_base_action)
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def compute_returns(self, obs):
        with torch.no_grad():
            _, action = self.actor._base_features_and_action(obs)
            attach_tracker_action(obs, action)
        super().compute_returns(obs)
        raw = self.storage.returns - self.storage.values
        if self.is_multi_gpu:
            flat = raw.double()
            moments = torch.stack((flat.sum(), flat.square().sum(), flat.new_tensor(flat.numel())))
            torch.distributed.all_reduce(moments)
            total, squares, count = moments
            mean = total / count
            variance = ((squares - total * mean) / (count - 1).clamp_min(1)).clamp_min(0)
            self.storage.advantages = (raw - mean.to(raw.dtype)) / (variance.sqrt().to(raw.dtype) + 1e-8)
            self.raw_advantage_std = float(variance.sqrt())
        else:
            self.raw_advantage_std = float(raw.std())

    def reduce_parameters(self):
        # Correct rank averaging for unequal numbers of expert environments.
        for model in (self.actor, self.critic):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(self.gradient_rank_weight)
        super().reduce_parameters()


class FixedFivePPO:
    """Five optimizers and disjoint on-policy trajectories in one vector simulator.

    The baseline gets one replica of EVERY specialist world and acts there
    itself. No expert transition is reused as a baseline PPO transition.
    """

    def __init__(self, actor, critic, obs, labels, *, rollout_steps, algorithm_cfg, device):
        if len(obs) != 2 * len(labels):
            raise ValueError("Expected N expert slots plus N baseline slots")
        if "dynamics_latent" in obs or "expert_id" in obs:
            raise ValueError("Policy observations must not include routing inputs")
        self.labels = labels.detach().clone()
        self.indices = dispatch_indices(self.labels)
        self.algorithms = []
        self.completed_updates = 0
        self.role_updates = [0] * 5
        self.transitions = [0] * 5
        self.device = device
        self.distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        self.world_size = torch.distributed.get_world_size() if self.distributed else 1
        self.rank = torch.distributed.get_rank() if self.distributed else 0
        local_counts = torch.tensor([len(ids) for ids in self.indices], device=device, dtype=torch.long)
        global_counts = local_counts.clone()
        if self.distributed:
            counts = [torch.empty_like(local_counts) for _ in range(self.world_size)]
            torch.distributed.all_gather(counts, local_counts)
            if bool((torch.stack(counts) == 0).any()):
                raise ValueError("Distributed five-PPO requires all four classes on each rank; increase --num-envs")
            torch.distributed.all_reduce(global_counts)
        self.global_counts = global_counts.tolist()
        reserve_tracker_action(obs)
        actor.populate_tracker_cache(obs)
        options = copy.deepcopy(algorithm_cfg)
        options.pop("class_name", None)
        options.pop("share_cnn_encoders", None)
        if self.distributed:
            if options.get("normalize_advantage_per_mini_batch", False):
                raise ValueError("Distributed five-PPO uses global per-expert advantage normalization")
            if rollout_steps % options["num_mini_batches"]:
                raise ValueError("rollout_steps must divide evenly into mini-batches on every rank")
            options["multi_gpu_cfg"] = {"global_rank": self.rank, "world_size": self.world_size}
        for role_index, ids in enumerate(self.indices):
            a, c = copy.deepcopy(actor), copy.deepcopy(critic)
            # Frozen modules may be shared; every trainable tensor and all
            # critic running statistics remain independently allocated.
            a.tracker = actor.tracker
            a.train()
            c.train()
            if self.distributed:
                synchronize_critic_normalization(c)
            selected = obs[ids]
            if len(ids):
                c.update_normalization(selected)
            storage = RolloutStorage("rl", len(ids), rollout_steps, selected, [29], device)
            local = {**options, "num_mini_batches": min(options["num_mini_batches"], max(1, len(ids)*rollout_steps))}
            algorithm = FixedExpertPPO(a, c, storage, device=device, **local)
            algorithm.gradient_rank_weight = self.world_size * len(ids) / max(1, self.global_counts[role_index])
            if self.distributed:
                algorithm.broadcast_parameters()
            self.algorithms.append(algorithm)
        self.initial_audit = self.audit_independence()

    def prepare_observations(self, obs):
        if "dynamics_latent" in obs or "expert_id" in obs:
            raise ValueError("Routing inputs reached latent-free actors/critics")
        self.algorithms[0].actor.populate_tracker_cache(obs)
        return reserve_tracker_action(obs)

    @torch.no_grad()
    def act(self, obs):
        actions = torch.empty(len(obs), 29, device=self.device)
        means = torch.empty_like(actions)
        for ids, alg in zip(self.indices, self.algorithms, strict=True):
            if len(ids):
                actions[ids] = alg.act(obs[ids])
                means[ids] = alg.actor.output_mean
        return actions, means

    @torch.no_grad()
    def process_step(self, obs, rewards, dones, extras):
        for i, (ids, alg) in enumerate(zip(self.indices, self.algorithms, strict=True)):
            if not len(ids):
                continue
            selected_extras = {"time_outs": extras["time_outs"][ids]} if "time_outs" in extras else {}
            alg.process_env_step(obs[ids], rewards[ids], dones[ids], selected_extras)
            self.transitions[i] += len(ids)

    def update(self, obs):
        result = {}
        for i, (role, ids, alg) in enumerate(zip(ROLES, self.indices, self.algorithms, strict=True)):
            if not len(ids):
                result[role] = {"worlds": 0, "skipped_empty_class": True}
                continue
            with torch.no_grad():
                alg.compute_returns(obs[ids])
                advantage_std = alg.raw_advantage_std
                if not bool(torch.isfinite(alg.storage.returns).all() & torch.isfinite(alg.storage.advantages).all()):
                    raise RuntimeError(f"Nonfinite returns for {role}")
            losses = alg.update()
            if not all(torch.isfinite(p).all() for model in (alg.actor, alg.critic)
                       for p in model.parameters() if p.requires_grad):
                raise RuntimeError(f"Nonfinite parameters for {role}")
            self.role_updates[i] += 1
            result[role] = {**losses, "worlds": len(ids), "raw_advantage_std": advantage_std,
                            "actor_lr": alg.actor_learning_rate, "critic_lr": alg.critic_learning_rate,
                            "updates": self.role_updates[i], "transitions": self.transitions[i]}
        self.completed_updates += 1
        return result

    def audit_independence(self):
        seen, normalizers = set(), set()
        for role, alg in zip(ROLES, self.algorithms, strict=True):
            if alg.actor.use_dynamics_latent or alg.actor.fusion_mode != "baseline" or alg.critic.fusion_mode != "baseline":
                raise AssertionError(f"{role} consumes latent")
            if any(p.requires_grad for p in alg.actor.tracker.parameters()):
                raise AssertionError("Tracker must be frozen")
            owned = set()
            for model in (alg.actor, alg.critic):
                for p in model.parameters():
                    if not p.requires_grad:
                        continue
                    ptr = p.data_ptr()
                    if ptr in seen:
                        raise AssertionError("Trainable parameters shared between actors, critics or PPOs")
                    seen.add(ptr)
                    owned.add(id(p))
            optimizer_owned = {id(p) for g in alg.optimizer.param_groups for p in g["params"]}
            if owned != optimizer_owned:
                raise AssertionError("Optimizer owns foreign or missing parameters")
            for b in alg.critic.obs_normalizer.buffers():
                if b.numel() and b.data_ptr() in normalizers:
                    raise AssertionError("Critic normalization is shared")
                if b.numel():
                    normalizers.add(b.data_ptr())
        merged = torch.cat(self.indices)
        torch.testing.assert_close(merged.sort().values, torch.arange(2*len(self.labels), device=self.labels.device))
        return {"independent_ppo_instances": 5, "independent_optimizers": 5,
                "disjoint_trainable_parameters_and_normalizers": True,
                "actor_critic_latent_inputs": False,
                "world_counts": [len(ids) for ids in self.indices],
                "baseline_owns_all_class_replicas": True, "cross_policy_transition_reuse": False}

    def audit_rank_agreement(self):
        if not self.distributed:
            return {"passed": True, "world_size": 1}
        local = {role: {
            "actor": tensor_digest((name, p) for name, p in alg.actor.named_parameters() if p.requires_grad),
            "critic": tensor_digest(alg.critic.named_parameters()),
            "critic_normalizer": tensor_digest(alg.critic.obs_normalizer.state_dict().items()),
        } for role, alg in zip(ROLES, self.algorithms, strict=True)}
        states = [None] * self.world_size
        torch.distributed.all_gather_object(states, local)
        if any(state != states[0] for state in states[1:]):
            raise RuntimeError("Five-PPO parameters or critic normalization diverged across ranks")
        return {"passed": True, "world_size": self.world_size,
                "global_world_counts": self.global_counts, "model_sha256": local}

    def state_dict(self):
        return {"version": VERSION, "labels": self.labels.detach().cpu().clone(),
                "completed_updates": self.completed_updates, "role_updates": self.role_updates,
                "transitions": self.transitions,
                "algorithms": {role: alg.save() for role, alg in zip(ROLES, self.algorithms, strict=True)}}

    def load_state_dict(self, state):
        if state["version"] != VERSION:
            raise ValueError("Incompatible five-PPO checkpoint")
        torch.testing.assert_close(self.labels.cpu(), state["labels"], atol=0, rtol=0)
        if tuple(state["algorithms"]) != ROLES:
            raise ValueError("Checkpoint must contain all five named PPO states")
        for role, alg in zip(ROLES, self.algorithms, strict=True):
            alg.load(state["algorithms"][role], None, True)
        self.completed_updates = int(state["completed_updates"])
        self.role_updates = list(state["role_updates"])
        self.transitions = list(state["transitions"])
        self.audit_independence()
