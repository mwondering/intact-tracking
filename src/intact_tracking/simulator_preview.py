"""Nondeployable five-step frozen-tracker simulator oracle, stored at behavior time.

This module does not alter the real task's rewards, actions or physics. The
shadow is an independent environment, never a rewind of the training world.
"""

from __future__ import annotations

import copy
import random
from contextlib import contextmanager

import numpy as np
import torch
import warp as wp
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from intact_tracking.adaptation_policy import PRIVILEGE, PrivilegedAdaptationWrapper
from intact_tracking.residual_policy import WarmStartedHeftCritic
from intact_tracking.rollout.mjlab_adapter import _robot_raw_state

HORIZON = 5
STATE_DIM = 71
# Next state, signed next-state error, body-position and joint-position distances.
OUTCOME_DIM = HORIZON * (2 * STATE_DIM + 2)
REFERENCE_DIM = HORIZON * STATE_DIM
REFERENCE_OFFSETS = tuple(range(1, HORIZON + 1))


@contextmanager
def isolated_rng(device):
    """Branch construction and noisy shadow observations cannot consume real RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    index = torch.device(device).index
    try:
        with torch.random.fork_rng(devices=[index if index is not None else 0]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def copy_runtime(source, target):
    """Copy owned tensors/scalars, not environment/model/dataset object references.

    Explicit callers below select the stateful manager/buffer objects. In
    particular this must not recursively traverse an object's `_env` pointer.
    """
    for name, value in vars(source).items():
        if isinstance(value, torch.Tensor):
            old = getattr(target, name, None)
            if isinstance(old, torch.Tensor) and old.shape == value.shape:
                old.copy_(value)
            else:
                setattr(target, name, value.clone())
        elif isinstance(value, (int, float, bool)) or value is None:
            setattr(target, name, value)


def reference_states(env, offsets=REFERENCE_OFFSETS):
    command = env.command_manager.get_term("motion")
    values = [command.gather_root_reference(field, offsets) for field in (
        "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"
    )]
    values += [command.gather_reference(field, offsets) for field in ("joint_pos", "joint_vel")]
    return torch.cat(values, dim=-1)


def preview_features(states, targets, initial, body_errors=None):
    """Shared local translation origin; align quaternion signs before subtraction."""
    states, targets = states.clone(), targets.clone()
    states[..., :2] -= initial[:, None, :2]
    targets[..., :2] -= initial[:, None, :2]
    # Canonicalize reference sign, then express the predicted quaternion nearby.
    targets[..., 3:7] *= torch.where(targets[..., 3:4] < 0, -1.0, 1.0)
    dots = (states[..., 3:7] * targets[..., 3:7]).sum(-1, keepdim=True)
    states[..., 3:7] *= torch.where(dots < 0, -1.0, 1.0)
    error = states - targets
    # New experiments use only the full state and coordinate-wise error.
    # Scalar distances are retained ONLY to read historical v1 checkpoints.
    blocks = [states, error]
    if body_errors is not None:
        blocks += [body_errors[..., None], error[..., 13:42].norm(dim=-1, keepdim=True)]
    outcomes = torch.cat(blocks, dim=-1)
    return targets.flatten(1), outcomes.flatten(1)


class PreviewAwareHeftCritic(WarmStartedHeftCritic):
    """Value receives exactly the actor's behavior-time privileged observation.

    Independent value parameters/normalizer; no critic gradient into the actor.
    The original value is preserved at initialization by a zero-output branch.
    True/zero controls mask the outcome before either network can read it.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        groups = {key: list(value) for key, value in obs_groups.items()}
        if PRIVILEGE not in groups[obs_set]:
            raise ValueError("Preview critic requires the actor's privileged observation group")
        groups[obs_set].remove(PRIVILEGE)
        super().__init__(obs, groups, obs_set, output_dim, **kwargs)
        self.preview_input_dim = int(obs[PRIVILEGE].shape[-1])
        self.preview_normalizer = EmpiricalNormalization(self.preview_input_dim, until=20_000_000)
        self.preview_value = MLP(self.obs_dim + self.preview_input_dim, output_dim, (256, 128), "mish")
        torch.nn.init.zeros_(self.preview_value[-1].weight)
        torch.nn.init.zeros_(self.preview_value[-1].bias)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        state = self.obs_normalizer(self._flat_obs(obs))
        preview = self.preview_normalizer(obs[PRIVILEGE]).clamp(-10, 10)
        return self.mlp(state) + self.preview_value(torch.cat((state, preview), -1))

    def update_normalization(self, obs):
        super().update_normalization(obs)
        self.preview_normalizer.update(obs[PRIVILEGE])


class FrozenTrackerPreview:
    """Five deterministic tracker actions in an independently synchronized world."""

    # Integrator inputs, warm start and observation/sensor kinematics. Workspaces
    # (contacts, Jacobians, factorizations) are rebuilt by the first mj_step.
    DATA_FIELDS = (
        "time", "qpos", "qvel", "qacc", "qacc_warmstart", "act", "act_dot", "ctrl",
        "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "userdata",
        "xpos", "xquat", "xmat", "xipos", "ximat", "cvel", "cacc",
        "subtree_com", "sensordata", "qfrc_actuator", "actuator_force",
    )

    def __init__(self, env, clip_actions=None, *, full_sim_state=False, include_distances=True):
        from mjlab.envs import ManagerBasedRlEnv

        from intact_tracking.environment.mdp.actions import ObservationHistoryJointPositionAction
        from intact_tracking.environment.mdp.shared_motion import share_motion_arrays

        self.source = env
        self.full_sim_state = full_sim_state
        self.include_distances = include_distances
        self.clip_actions = clip_actions
        source_command = env.command_manager.get_term("motion")
        if any(not isinstance(term, ObservationHistoryJointPositionAction)
               for term in env.action_manager._terms.values()):
            raise ValueError("Preview audit currently supports the original memoryless BFM PD action only")
        if any(mode in env.event_manager.available_modes for mode in ("step", "interval")):
            raise ValueError("Preview does not yet branch step/interval events")
        cfg = copy.deepcopy(env.cfg)
        cfg.auto_reset = False
        cfg.terminations = {}
        cfg.rewards = {}
        cfg.recorders = {}
        cfg.curriculum = {}
        cfg.commands["motion"].sampling_mode = "uniform"
        cfg.commands["motion"].rewind.enabled = False
        cfg.commands["motion"].resample_on_motion_end = False
        cfg.commands["motion"].resampling_time_range = (1e9, 1e9)
        with isolated_rng(env.device), share_motion_arrays(source_command):
            self.shadow = ManagerBasedRlEnv(cfg=cfg, device=env.device)
            self.shadow.reset()
        if self.shadow.command_manager.get_term("motion").motion is not source_command.motion:
            self.shadow.close()
            raise RuntimeError("Preview did not share the real environment's immutable motion catalog")
        missing = env.sim.expanded_fields - self.shadow.sim.expanded_fields
        if missing:
            self.shadow.sim.expand_model_fields(tuple(sorted(missing)))
        self.model_fields = tuple(sorted(env.sim.expanded_fields))
        self.data_fields = tuple(name for name in self.DATA_FIELDS if hasattr(env.sim.data, name))
        self.full_data_pairs = []
        if full_sim_state:
            def visit(source, target, depth=0):
                for name, value in vars(source).items():
                    other = getattr(target, name)
                    if isinstance(value, wp.array):
                        if value.shape != other.shape:
                            raise ValueError(f"Shadow allocation mismatch for {name}")
                        if value.size:
                            self.full_data_pairs.append((value, other))
                    elif depth < 2 and hasattr(value, "__dict__") and value.__class__.__module__.startswith("mujoco_warp"):
                        visit(value, other, depth + 1)
            visit(env.sim.wp_data, self.shadow.sim.wp_data)
        self.last_states = self.last_actions = self.last_targets = None

    def synchronize(self):
        src, dst = self.source, self.shadow
        torch.cuda.synchronize(src.device)
        for name in self.model_fields:
            value = getattr(src.sim.model, name)
            if value.numel():
                getattr(dst.sim.model, name)[:] = value
        if self.full_sim_state:
            for source_array, target_array in self.full_data_pairs:
                wp.copy(target_array, source_array)
        else:
            for name in self.data_fields:
                value = getattr(src.sim.data, name)
                if value.numel():
                    getattr(dst.sim.data, name)[:] = value
        copy_runtime(src, dst)
        # Neither task-side reset flags nor the shadow's previous termination
        # may reset the five-step physical branch.
        dst._manual_reset_pending.zero_()
        for name in src.scene.entities:
            copy_runtime(src.scene[name], dst.scene[name])
            copy_runtime(src.scene[name].data, dst.scene[name].data)
        for name in src.scene.sensors:
            source_sensor, target_sensor = src.scene[name], dst.scene[name]
            copy_runtime(source_sensor, target_sensor)
            for attribute in ("_air_time_state",):
                value = getattr(source_sensor, attribute, None)
                if value is not None:
                    copy_runtime(value, getattr(target_sensor, attribute))
            if getattr(source_sensor, "_history_state", None) is not None:
                target_sensor._history_state = {
                    key: value.clone() for key, value in source_sensor._history_state.items()
                }
        copy_runtime(src.action_manager, dst.action_manager)
        for name, term in src.action_manager._terms.items():
            copy_runtime(term, dst.action_manager._terms[name])
        command = dst.command_manager.get_term("motion")
        copy_runtime(src.command_manager.get_term("motion"), command)
        command.time_left.fill_(1e9)
        command._invalidate_reference_cache()
        for attribute in ("_group_obs_term_history_buffer", "_group_obs_term_delay_buffer"):
            for group, terms in getattr(src.observation_manager, attribute).items():
                for name, buffer in terms.items():
                    other = getattr(dst.observation_manager, attribute)[group][name]
                    copy_runtime(buffer, other)
                    if hasattr(buffer, "_circular_buffer"):
                        copy_runtime(buffer._circular_buffer, other._circular_buffer)
                    elif hasattr(getattr(buffer, "_buffer", None), "__dict__"):
                        # MJLab DelayBuffer owns a CircularBuffer in `_buffer`.
                        copy_runtime(buffer._buffer, other._buffer)
        for group, cfgs in src.observation_manager._group_obs_class_term_cfgs.items():
            for cfg, other in zip(cfgs, dst.observation_manager._group_obs_class_term_cfgs[group], strict=True):
                copy_runtime(cfg.func, other.func)
        for group, terms in src.observation_manager._group_obs_class_instances.items():
            for name, instance in terms.items():
                copy_runtime(instance, dst.observation_manager._group_obs_class_instances[group][name])
        if hasattr(src, "_sp_substep_tracking_cache"):
            copy_runtime(src._sp_substep_tracking_cache, dst._sp_substep_tracking_cache)
        torch.cuda.synchronize(src.device)

    @torch.no_grad()
    def query(self, tracker, observations):
        # Tracker remains frozen/eval; never call the residual to obtain a
        # branch action. First observation is an independent copy of the exact
        # real observation; subsequent ones come from the shadow's own history.
        with isolated_rng(self.source.device):
            self.synchronize()
            obs = observations.select(*tracker.obs_groups).clone(recurse=True)
            initial = _robot_raw_state(self.source)
            targets = reference_states(self.shadow)
            states, actions, errors, bodies = [], [], [], []
            for _ in range(HORIZON):
                action = tracker(obs)
                for term in self.shadow.action_manager._terms.values():
                    term.record_policy_mean(action)
                applied = action if self.clip_actions is None else action.clamp(-self.clip_actions, self.clip_actions)
                raw, *_ = self.shadow.step(applied)
                obs = TensorDict(raw, batch_size=[self.source.num_envs])
                command = self.shadow.command_manager.get_term("motion")
                states.append(_robot_raw_state(self.shadow))
                actions.append(action.clone())
                if self.include_distances:
                    errors.append((command.body_pos_relative_w - command.robot_body_pos_w).norm(dim=-1).mean(-1))
                bodies.append(command.robot_body_pos_w.clone())
            states, actions = torch.stack(states, 1), torch.stack(actions, 1)
            self.last_states, self.last_actions, self.last_targets = states, actions, targets
            self.last_body_positions = torch.stack(bodies, 1)
            return preview_features(states, targets, initial, torch.stack(errors, 1) if errors else None)

    def close(self):
        self.shadow.close()


class SimulatorPreviewWrapper(PrivilegedAdaptationWrapper):
    """Same-capacity true/zero future controls; common state, physics and targets."""

    def __init__(self, env, clip_actions=None, *, preview_mode="true", privilege_schema="state_physics"):
        if privilege_schema != "state_physics" or preview_mode not in ("true", "zero", "shuffle"):
            raise ValueError("Five-step preview requires state_physics and true/zero/shuffle outcomes")
        self.preview_mode = preview_mode
        self.preview = self.tracker = None
        super().__init__(env, clip_actions, privilege_schema=privilege_schema)

    def bind(self, actor):
        if actor.train_base_policy or actor.oracle_tracking_features or actor.oracle_clean_proprio:
            raise ValueError("Preview needs an unchanged frozen/deployable tracker frontend")
        self.tracker = actor.tracker
        if self.preview is None:
            self.preview = FrozenTrackerPreview(self.unwrapped, self.clip_actions)

    def attach(self, obs):
        obs = super().attach(obs)
        if self.preview is None:
            reference = obs[PRIVILEGE].new_zeros((self.num_envs, REFERENCE_DIM))
            outcome = obs[PRIVILEGE].new_zeros((self.num_envs, OUTCOME_DIM))
        else:
            reference, outcome = self.preview.query(self.tracker, obs)
            if self.preview_mode == "zero":
                outcome = torch.zeros_like(outcome)
            elif self.preview_mode == "shuffle":
                outcome = outcome.roll(1, dims=0)
        return obs.set(PRIVILEGE, torch.cat((obs[PRIVILEGE], reference, outcome), -1))

    def close_preview(self):
        if self.preview is not None:
            self.preview.close()
