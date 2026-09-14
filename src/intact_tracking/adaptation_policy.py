"""Privileged teacher with a replaceable information bottleneck for adaptation."""

from __future__ import annotations

import copy

import torch
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.modules import MLP, EmpiricalNormalization
from tensordict import TensorDict

from intact_tracking.adaptation_context_memory import CONTEXT_MEAN
from intact_tracking.adaptation_modulation import (
    LatentModulatedResidual,
    PhysicsBilinearResidual,
    PhysicsLowRankResidual,
)
from intact_tracking.adaptation_prior import load_frozen_nominal_prior
from intact_tracking.adaptation_sensors import IMU_HISTORY
from intact_tracking.residual_policy import FrozenTrackerResidualActor, _last_linear
from intact_tracking.rollout.mjlab_adapter import _robot_raw_state

PRIVILEGE = "adaptation_privilege"
ORACLE_HISTORY = "adaptation_clean_history"
ORACLE_KEY_BODY = "adaptation_clean_key_body"


def normalized_context_kinematics(key_body):
    """Existing noisy FK only:13 bodies, position/rotation/linear/angular speed."""
    if key_body.shape[-1] != 195:
        raise ValueError("Context FK requires the existing 195D deployable key-body observation")
    return torch.cat((key_body[..., :117], key_body[..., 117:156] / 2, key_body[..., 156:] / 4), -1)


def configure_oracle_proprioception(cfg):
    """Add separate privileged channels; never alter the noisy deployment groups."""
    for original, privileged in (
        ("estimator_history", ORACLE_HISTORY),
        ("robot_key_body", ORACLE_KEY_BODY),
    ):
        group = copy.deepcopy(cfg.observations[original])
        group.enable_corruption = False
        for term in group.terms.values():
            term.noise = None
            for name in tuple(term.params):
                if "noise_std" in name:
                    term.params[name] = 0.0
                elif name == "biased":
                    term.params[name] = False
        cfg.observations[privileged] = group


def base_action_from_features(actor, features):
    """Evaluate the frozen/current core and its optional audited nominal head."""
    action = actor.tracker.distribution.deterministic_output(actor.tracker.mlp(features))
    if getattr(actor, "frozen_nominal_prior_mlp", None) is not None:
        action = action + actor.frozen_nominal_prior_scale * actor.frozen_nominal_prior_mlp(features).tanh()
    return action


def make_refinement_mlp(input_dim, output_dim, scale):
    if scale < 0:
        raise ValueError("Refinement scale must be nonnegative")
    if scale == 0:
        return None
    model = MLP(input_dim, output_dim, (512, 256, 128), "elu")
    output = _last_linear(model)
    torch.nn.init.zeros_(output.weight)
    torch.nn.init.zeros_(output.bias)
    return model


def additional_refinement(actor, value):
    if actor.refinement_mlp is None:
        return 0.0
    return actor.refinement_scale * actor.refinement_mlp(value).tanh()


def _base_with_optional_update(actor, obs):
    if getattr(actor, "train_reference_encoder", False):
        model = actor.tracker
        with torch.no_grad():
            height, contact_logits = model.estimate_height_and_contact(obs)
        # Bypass detached behavior-time caches: recompute from the raw noisy
        # reference, so action loss can train the deployable reference encoder.
        decoded = model.encode_reference(obs)
        features = model.policy_normalizer(
            model._spv5_2_features(obs, height, contact_logits.sigmoid(), decoded)
        )
        action = model.distribution.deterministic_output(model.mlp(features))
        return features, action
    with torch.no_grad():
        if getattr(actor, "oracle_tracking_features", False):
            model = actor.tracker
            ablation = getattr(actor, "oracle_feature_ablation", "normal")
            height = obs[model.estimator_target_group]
            contacts = obs[model.foot_contact_target_group]
            reference = obs[model.reference_encoder_target_group]
            if ablation in ("estimated_height_contact", "estimated_all"):
                height, logits = model.estimate_height_and_contact(obs)
                contacts = logits.sigmoid()
            if ablation in ("estimated_reference", "estimated_all"):
                reference = model.encode_reference(obs)
            remove = getattr(actor, "oracle_state_ablation", "normal")
            if remove != "normal":
                estimated_height, logits = model.estimate_height_and_contact(obs)
                if remove in ("height", "height_contact"):
                    height = estimated_height
                if remove in ("contact", "height_contact"):
                    contacts = logits.sigmoid()
            feature_obs = obs
            if getattr(actor, "oracle_clean_proprio", False):
                feature_obs = obs.select(*model.obs_groups)
                feature_obs.set(model.estimator_history_group, obs[ORACLE_HISTORY])
                feature_obs.set(model.robot_key_body_group, obs[ORACLE_KEY_BODY])
            raw = model._spv5_2_features(
                feature_obs,
                height,
                contacts,
                reference,
            )
            features = model.policy_normalizer(raw).detach()
            mixture = getattr(actor, "oracle_feature_mix", None)
            if mixture is not None:
                # Training observation curriculum only. At one, the entire
                # tracking frontend is exactly the existing noisy/deployable
                # frontend. Physical privileges still enter the teacher latent.
                estimated = model.get_latent(obs).detach()
                features = features * (1 - mixture) + estimated * mixture
        else:
            features = actor.tracker.get_latent(obs).detach()
    action_features = features
    if getattr(actor, "frozen_nominal_prior_deployable_base", False):
        # Oracle features are inputs to the NEW residual only. The audited
        # nominal source always sees its original noisy/deployable frontend.
        with torch.no_grad():
            action_features = actor.tracker.get_latent(obs).detach()
    with torch.set_grad_enabled(torch.is_grad_enabled() and actor.train_base_policy):
        action = base_action_from_features(actor, action_features)
    return features, action


class DeployableHistoryEncoder(torch.nn.Module):
    """Temporal encoder over only measured sensors and known action history.

    The input is the existing term-major estimator history: joint position,
    joint velocity, gravity, gyro, previous command, and joint torque. Every
    frame is at or before the current action query; no reference/simulator
    state or environment labels enter this module.
    """

    TERM_DIMS = (29, 29, 3, 3, 29, 29)

    def __init__(
        self, latent_dim=64, history_steps=50, imu_accel=False,
        root_orientation=False, right_aligned=False, key_body=False,
    ):
        super().__init__()
        self.history_steps = int(history_steps)
        self.root_orientation = bool(root_orientation)
        self.key_body = bool(key_body)
        # Three valid convolutions have receptive field 21 and total stride 4.
        # Trim unused samples from the LEFT, so the last output includes the
        # current measurement. Default zero preserves all legacy checkpoints.
        self.history_left_trim = (self.history_steps - 21) % 4 if right_aligned else 0
        self.term_dims = self.TERM_DIMS + ((3,) if imu_accel else ())
        self.frame_dim = sum(self.term_dims)
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(self.frame_dim, 128, kernel_size=5, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(128, 128, kernel_size=5, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(128, 64, kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Flatten(),
        )
        with torch.no_grad():
            width = self.network(torch.zeros(1, self.frame_dim, history_steps)).shape[-1]
        self.head = MLP(
            width + (4 if self.root_orientation else 0) + (195 if self.key_body else 0),
            latent_dim, (256, 128), "elu", last_activation="tanh",
        )

    def frames(self, history):
        if history.shape[-1] != self.frame_dim * self.history_steps:
            raise ValueError("Deployable history has an unexpected shape")
        terms = history.split([d * self.history_steps for d in self.term_dims], dim=-1)
        return torch.cat(
            [
                x.reshape(*x.shape[:-1], self.history_steps, d)
                for x, d in zip(terms, self.term_dims, strict=True)
            ],
            dim=-1,
        )

    def forward(self, history, orientation=None, kinematics=None):
        frames = self.frames(history).transpose(-1, -2)
        features = self.network(frames[..., self.history_left_trim :])
        if self.root_orientation:
            if orientation is None:
                raise ValueError("This context encoder requires the deployable IMU orientation")
            features = torch.cat((features, orientation), dim=-1)
        if self.key_body:
            if kinematics is None:
                raise ValueError("This context encoder requires deployable noisy FK")
            features = torch.cat((features, kinematics), dim=-1)
        return self.head(features)


class ContextAdaptationActor(FrozenTrackerResidualActor):
    """The complete student action path never reads privileged observation keys."""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        *,
        adaptation_latent_dim=64,
        context_history_steps=50,
        context_auxiliary_dim=0,
        context_feature_adapter=False,
        context_imu_accel=False,
        train_reference_encoder=False,
        context_root_orientation=False,
        context_right_aligned=False,
        context_key_body=False,
        context_learned_residual_gain=False,
        train_residual_policy=True,
        train_context_encoder=True,
        context_latent_mean=False,
        context_mean_only=False,
        context_mean_horizon=250,
        train_base_policy=False,
        latent_modulation=False,
        physics_bilinear=False,
        physics_low_rank=False,
        latent_low_rank=False,
        rank_per_physics=2,
        shared_low_rank=0,
        frozen_nominal_prior=None,
        frozen_nominal_prior_deployable_base=False,
        refinement_scale=0.0,
        **kwargs,
    ):
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.frozen_nominal_prior_deployable_base = bool(frozen_nominal_prior_deployable_base)
        if self.frozen_nominal_prior_deployable_base and frozen_nominal_prior is None:
            raise ValueError("Separated deployable base requires the audited nominal prior")
        if frozen_nominal_prior is not None and (
            train_base_policy or train_reference_encoder
            or (context_feature_adapter and not self.frozen_nominal_prior_deployable_base)
        ):
            raise ValueError("Frozen nominal prior requires an unchanged deployable frontend")
        self.frozen_nominal_prior_mlp, self.frozen_nominal_prior_scale, self.frozen_nominal_prior_audit = load_frozen_nominal_prior(frozen_nominal_prior, self.tracker, output_dim)
        self.train_base_policy = bool(train_base_policy)
        self.latent_modulation = bool(latent_modulation)
        self.physics_bilinear = bool(physics_bilinear)
        self.physics_low_rank = bool(physics_low_rank)
        self.latent_low_rank = bool(latent_low_rank)
        self.shared_low_rank = int(shared_low_rank)
        if self.shared_low_rank < 0 or (self.shared_low_rank and not (physics_low_rank or latent_low_rank)):
            raise ValueError("Shared low-rank updates require a nonnegative rank and low-rank controller")
        if self.latent_low_rank and (
            physics_bilinear or physics_low_rank or latent_modulation or train_base_policy
            or train_reference_encoder or context_feature_adapter or context_latent_mean
        ):
            raise ValueError("Learned low-rank code requires a frozen frontend and one current context latent")
        self.context_mean_only = bool(context_mean_only)
        if self.context_mean_only and (
            not context_latent_mean or not (physics_bilinear or physics_low_rank)
            or train_context_encoder or adaptation_latent_dim != 7
        ):
            raise ValueError("Mean-only context requires frozen seven-coordinate physical encoder and causal memory")
        if physics_bilinear and physics_low_rank:
            raise ValueError("Choose bilinear or low-rank physical conditioning")
        if (self.physics_bilinear or self.physics_low_rank) and (
            adaptation_latent_dim not in (7, 64) or train_base_policy or train_reference_encoder
            or (context_latent_mean and not context_mean_only) or latent_modulation
        ):
            raise ValueError("Physical-code student needs seven compact or64 actuator coordinates and a frozen core; an explicit deployable feature adapter is allowed")
        if self.latent_modulation and context_latent_mean:
            raise ValueError("Modulated residual currently requires a single current context latent")
        self.adaptation_latent_dim = int(adaptation_latent_dim)
        self.context_latent_mean = bool(context_latent_mean)
        self.context_mean_horizon = int(context_mean_horizon)
        if self.context_latent_mean and (train_context_encoder or context_feature_adapter):
            raise ValueError("Causal latent mean requires a frozen encoder and no feature adapter")
        if self.context_mean_horizon < 1:
            raise ValueError("Context mean horizon must be positive")
        self.context_imu_accel = bool(context_imu_accel)
        self.train_reference_encoder = bool(train_reference_encoder)
        self.context_root_orientation = bool(context_root_orientation)
        self.context_right_aligned = bool(context_right_aligned)
        self.context_key_body = bool(context_key_body)
        self.context_residual_log_gain = (
            torch.nn.Parameter(torch.zeros(output_dim))
            if context_learned_residual_gain else None
        )
        self.tracker.mlp.requires_grad_(self.train_base_policy)
        if self.train_reference_encoder:
            self.tracker.reference_encoder.requires_grad_(True)
        self.context_encoder = DeployableHistoryEncoder(
            adaptation_latent_dim, context_history_steps,
            self.context_imu_accel, self.context_root_orientation,
            self.context_right_aligned,
            self.context_key_body,
        )
        self.context_encoder.requires_grad_(bool(train_context_encoder))
        self.feature_adapter = (
            MLP(
                self.tracker.policy_input_dim + adaptation_latent_dim,
                self.tracker.policy_input_dim,
                (256, 128),
                "elu",
            )
            if context_feature_adapter
            else None
        )
        if self.feature_adapter is not None:
            output = _last_linear(self.feature_adapter)
            torch.nn.init.zeros_(output.weight)
            torch.nn.init.zeros_(output.bias)
        # Training-only readout; actions use the latent, never its labels.
        self.context_auxiliary_head = (
            MLP(adaptation_latent_dim, context_auxiliary_dim, (64,), "elu")
            if context_auxiliary_dim
            else None
        )
        if self.context_auxiliary_head is not None:
            self.context_auxiliary_head.requires_grad_(bool(train_context_encoder))
        self.residual_mlp = MLP(
            self.tracker.policy_input_dim + adaptation_latent_dim * (2 if self.context_latent_mean and not self.context_mean_only else 1),
            output_dim,
            kwargs.get("residual_hidden_dims", (512, 256, 128)),
            "elu",
        )
        if self.latent_modulation:
            self.residual_mlp = LatentModulatedResidual(
                self.tracker.policy_input_dim + adaptation_latent_dim,
                output_dim, kwargs.get("residual_hidden_dims", (512, 256, 128)),
                adaptation_latent_dim,
            )
        if self.physics_bilinear:
            self.residual_mlp = PhysicsBilinearResidual(
                self.tracker.policy_input_dim + adaptation_latent_dim,
                output_dim, kwargs.get("residual_hidden_dims", (512, 256, 128)),
                adaptation_latent_dim,
            )
        if self.physics_low_rank or self.latent_low_rank:
            self.residual_mlp = PhysicsLowRankResidual(
                self.tracker.mlp, adaptation_latent_dim, rank_per_physics, self.shared_low_rank,
            )
        else:
            output = _last_linear(self.residual_mlp)
            torch.nn.init.zeros_(output.weight)
            torch.nn.init.zeros_(output.bias)
        self.residual_mlp.requires_grad_(bool(train_residual_policy))
        if self.physics_low_rank or self.latent_low_rank:
            self.residual_mlp.freeze_reference()
        self.refinement_scale = float(refinement_scale)
        if self.refinement_scale and context_latent_mean:
            raise ValueError("Teacher refinement currently requires one current context latent")
        self.refinement_mlp = make_refinement_mlp(
            self.tracker.policy_input_dim + adaptation_latent_dim, output_dim, self.refinement_scale,
        )
        if self.refinement_mlp is not None:
            self.refinement_mlp.requires_grad_(bool(train_residual_policy))

    @property
    def deployable_observation_groups(self):
        return (
            tuple(self.obs_groups)
            + ((IMU_HISTORY,) if self.context_imu_accel else ())
            + ((CONTEXT_MEAN,) if self.context_latent_mean else ())
        )

    def _residual(self, value):
        result = super()._residual(value)
        if self.context_residual_log_gain is not None:
            result = result * self.context_residual_log_gain.clamp(-4.0, 0.0).exp()
        return result if self.refinement_mlp is None else result + additional_refinement(self, value)

    def normalized_context_history(self, obs):
        history = self.tracker.history_normalizer(obs["estimator_history"]).clamp(-10, 10)
        if self.context_imu_accel:
            history = torch.cat((history, (obs[IMU_HISTORY] / 10.0).clamp(-10, 10)), dim=-1)
        return history

    def instant_context_latent(self, obs):
        return self.context_encoder(
            self.normalized_context_history(obs),
            obs["robot_root_quat"] if self.context_root_orientation else None,
            normalized_context_kinematics(obs["robot_key_body"]) if self.context_key_body else None,
        )

    def context_latent(self, obs):
        latent = self.instant_context_latent(obs)
        if self.context_latent_mean:
            latent = obs[CONTEXT_MEAN] if self.context_mean_only else torch.cat((latent, obs[CONTEXT_MEAN]), -1)
        ablation = getattr(self, "context_ablation", "normal")
        if ablation == "zero":
            return torch.zeros_like(latent)
        if ablation == "shuffle":
            return latent.roll(1, 0)
        return latent

    def _base_features_and_action(self, obs):
        features, action = _base_with_optional_update(self, obs)
        if self.feature_adapter is not None:
            raw_features = features
            features = self.adapt_features(raw_features, self.context_latent(obs))
            # Frozen core weights must still propagate input gradients to the
            # adapter in the legacy shared-frontend mode. In separated mode,
            # only the new residual receives the adapted features.
            action = self.base_action_from_adapted_features(raw_features, features)
        return features, action

    def base_action_from_adapted_features(self, raw_features, adapted_features):
        features = (
            raw_features if self.frozen_nominal_prior_deployable_base else adapted_features
        )
        return base_action_from_features(self, features)

    def adapt_features(self, features, latent):
        if self.feature_adapter is None:
            return features
        return features + self.feature_adapter(torch.cat((features, latent), dim=-1))

    def _residual_input(self, obs, tracker_features, base_action, *, latent_override=None):
        del base_action
        latent = self.context_latent(obs) if latent_override is None else latent_override
        return torch.cat((tracker_features, latent), dim=-1)

    @torch.no_grad()
    def policy_metrics(self, obs):
        raw_features, _ = _base_with_optional_update(self, obs)
        latent = self.context_latent(obs)
        features = self.adapt_features(raw_features, latent)
        residual = self._residual(torch.cat((features, latent), dim=-1))

        def action_for(z):
            f = self.adapt_features(raw_features, z)
            base = self.base_action_from_adapted_features(raw_features, f)
            return base + self._residual(torch.cat((f, z), dim=-1))

        normal = action_for(latent)
        shuffled = action_for(latent.roll(1, 0))
        zeroed = action_for(torch.zeros_like(latent))
        return {
            "residual_action_rms": float(residual.square().mean().sqrt()),
            "residual_action_abs_max": float(residual.abs().max()),
            "residual_saturation_fraction": float(
                (residual.abs() > 0.95 * self.residual_scale).float().mean()
            ),
            "context_shuffle_action_delta_rms": float((normal - shuffled).square().mean().sqrt()),
            "context_zero_action_delta_rms": float((normal - zeroed).square().mean().sqrt()),
            "context_latent_rms": float(latent.square().mean().sqrt()),
        }


def physics_observation(env) -> torch.Tensor:
    """Compiled-model-relative physical values, including sensor bias.

    Fixed dimensions in nominal and DR worlds. Read actual current values, so a
    reset callback changing parameters cannot leave a stale privileged label.
    """
    blocks = []
    for name in ("body_mass", "body_ipos", "body_inertia", "dof_armature", "geom_friction"):
        actual, default = expanded_model_field(env, name)
        if name == "body_ipos":
            value = (actual - default) / 0.075
        else:
            value = (actual - default) / default.abs().clamp_min(0.01)
        blocks.append(value.flatten(1).clamp(-100.0, 100.0))
    blocks.append(env.scene["robot"].data.encoder_bias / 0.01)
    return torch.cat(blocks, dim=-1)


def expanded_model_field(env, name):
    actual = getattr(env.sim.model, name)
    default = env.sim.get_default_field(name).to(actual)
    if actual.shape == default.shape:
        actual = actual.unsqueeze(0)
    if actual.shape == (1, *default.shape):
        actual = actual.expand(env.num_envs, *default.shape)
    if actual.shape != (env.num_envs, *default.shape):
        raise ValueError(f"Unexpected {name} shape {actual.shape}; nominal {default.shape}")
    return actual, default


def compact_physics_observation(env):
    """Seven static, nominal-centered truths; no privileged motion/state inputs."""
    model = env.sim.mj_model
    indices = getattr(env, "_adaptation_compact_physics_indices", None)
    if indices is None:
        bodies = []
        for suffix in ("right_wrist_yaw_link", "torso_link"):
            matches = [i for i in range(model.nbody) if model.body(i).name.split("/")[-1] == suffix]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous compact physics body: {suffix}")
            bodies.append(matches[0])
        feet = [i for i in range(model.ngeom) if "_foot" in model.geom(i).name]
        if not feet:
            raise ValueError("Missing foot friction geometry")
        indices = (*bodies, feet)
        env._adaptation_compact_physics_indices = indices
    wrist, torso, feet = indices
    mass, mass0 = expanded_model_field(env, "body_mass")
    com, com0 = expanded_model_field(env, "body_ipos")
    friction, friction0 = expanded_model_field(env, "geom_friction")
    armature, armature0 = expanded_model_field(env, "dof_armature")
    dofs = armature0 > 0
    return torch.cat((
        (mass[:, wrist:wrist + 1] - mass0[wrist]) / 3.0,
        mass[:, torso:torso + 1] - mass0[torso],
        (com[:, torso] - com0[torso]) / 0.075,
        (friction[:, feet, 0] - friction0[feet, 0]).mean(-1, keepdim=True) / 2.0,
        ((armature[:, dofs] / armature0[dofs]) - 1).mean(-1, keepdim=True) / 0.2,
    ), -1).detach()


def actuator_physics_observation(env):
    """64 static truths: six compact coordinates,29 armatures,29 encoder biases.

    No learned compression or true robot state. The six compact coordinates
    preserve payload, torso mass/COM and mean foot friction; individual foot
    geometry frictions are still averaged, not claimed to be fully retained.
    Armature and bias coordinates share the robot's canonical joint order.
    """
    compact = compact_physics_observation(env)
    robot = env.scene["robot"]
    armature, armature0 = expanded_model_field(env, "dof_armature")
    dofs = robot.indexing.joint_v_adr.long()
    bias = robot.data.encoder_bias
    if dofs.shape != (29,) or bias.shape != (env.num_envs, 29) or not bool((armature0[dofs] > 0).all()):
        raise ValueError("Actuator physics code requires29 ordered positive-armature robot joints")
    ratios = (armature[:, dofs] / armature0[dofs] - 1) / 0.2
    return torch.cat((compact[:, :6], ratios, bias / 0.01), -1).detach()


class PrivilegedAdaptationWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions=None, *, privilege_schema="critic"):
        self.privilege_schema = privilege_schema
        super().__init__(env, clip_actions=clip_actions)

    def attach(self, obs: TensorDict) -> TensorDict:
        if self.privilege_schema == "compact_physics":
            return obs.set(PRIVILEGE, compact_physics_observation(self.unwrapped))
        if self.privilege_schema == "actuator_physics":
            return obs.set(PRIVILEGE, actuator_physics_observation(self.unwrapped))
        if self.privilege_schema == "physics":
            # Full static parameters, including every encoder bias and motor
            # armature. No privileged current state, height, contact or future.
            return obs.set(PRIVILEGE, physics_observation(self.unwrapped))
        if self.privilege_schema == "critic":
            state = obs["priv"]
        elif self.privilege_schema == "state_physics":
            # Exclude absolute x/y and privileged future reference windows.
            # The student can estimate this current-state information from
            # proprioception, while the existing actor observes its reference.
            state = torch.cat(
                (_robot_raw_state(self.unwrapped)[:, 2:], obs["foot_contact_target"]), dim=-1
            )
        else:
            raise ValueError(f"Unknown privilege schema {self.privilege_schema}")
        obs.set(PRIVILEGE, torch.cat((state, physics_observation(self.unwrapped)), dim=-1))
        return obs

    def get_observations(self):
        return self.attach(super().get_observations())

    def reset(self):
        obs, extras = super().reset()
        return self.attach(obs), extras

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        return self.attach(obs), rewards, dones, extras


class PrivilegedAdaptationActor(FrozenTrackerResidualActor):
    """One policy across DR worlds; privileged encoder is the teacher bottleneck.

    In the default variant, privileged inputs pass through `privilege_encoder`.
    The optional oracle tracking-feature variant additionally uses true height,
    contact and reference values. Stage two must replace that route as well.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        *,
        privilege_dim: int | None = None,
        adaptation_latent_dim: int = 64,
        privilege_hidden_dims=(256, 128),
        privilege_schema="critic",
        actor_physics_input="normal",
        train_base_policy=False,
        oracle_tracking_features=False,
        oracle_clean_proprio=False,
        oracle_feature_curriculum=False,
        latent_modulation=False,
        physics_bilinear=False,
        physics_low_rank=False,
        latent_low_rank=False,
        rank_per_physics=2,
        shared_low_rank=0,
        frozen_nominal_prior=None,
        frozen_nominal_prior_deployable_base=False,
        refinement_scale=0.0,
        freeze_teacher_trunk=False,
        **kwargs,
    ):
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.frozen_nominal_prior_deployable_base = bool(frozen_nominal_prior_deployable_base)
        if self.frozen_nominal_prior_deployable_base and frozen_nominal_prior is None:
            raise ValueError("Separated deployable base requires the audited nominal prior")
        if frozen_nominal_prior is not None and (
            train_base_policy or (
                (oracle_tracking_features or oracle_clean_proprio)
                and not self.frozen_nominal_prior_deployable_base
            )
        ):
            raise ValueError("Frozen nominal prior requires an unchanged deployable frontend")
        self.frozen_nominal_prior_mlp, self.frozen_nominal_prior_scale, self.frozen_nominal_prior_audit = load_frozen_nominal_prior(frozen_nominal_prior, self.tracker, output_dim)
        self.privilege_schema = privilege_schema
        self.actor_physics_input = str(actor_physics_input)
        self.shared_low_rank = int(shared_low_rank)
        if self.shared_low_rank < 0 or (self.shared_low_rank and not (physics_low_rank or latent_low_rank)):
            raise ValueError("Shared low-rank updates require a nonnegative rank and low-rank controller")
        if self.actor_physics_input not in ("normal", "zero", "constant"):
            raise ValueError("Unknown actor physical-input ablation")
        if self.actor_physics_input == "zero" and not (physics_low_rank and self.shared_low_rank > 0) and (
            privilege_schema != "physics" or latent_low_rank
        ):
            raise ValueError("Zero-physics training control requires full physical-only inputs and a non-centered trainable residual architecture")
        if self.actor_physics_input == "constant" and not (
            privilege_schema in ("compact_physics", "actuator_physics") and (physics_low_rank or physics_bilinear)
        ):
            raise ValueError("Constant-code control requires compact physical low-rank or bilinear conditioning")
        self.latent_modulation = bool(latent_modulation)
        self.physics_bilinear = bool(physics_bilinear)
        self.physics_low_rank = bool(physics_low_rank)
        self.latent_low_rank = bool(latent_low_rank)
        if self.latent_low_rank and (
            privilege_schema != "physics" or physics_bilinear or physics_low_rank
            or latent_modulation or train_base_policy or oracle_tracking_features
            or oracle_clean_proprio
        ):
            raise ValueError("Learned low-rank teacher requires full physical inputs and an unchanged frozen frontend")
        if physics_bilinear and physics_low_rank:
            raise ValueError("Choose bilinear or low-rank physical conditioning")
        if (self.physics_bilinear or self.physics_low_rank) and (
            privilege_schema not in ("compact_physics", "actuator_physics")
            or adaptation_latent_dim != {"compact_physics": 7, "actuator_physics": 64}.get(privilege_schema)
            or train_base_policy or latent_modulation
        ):
            raise ValueError("Physical-code teacher needs7 compact or64 actuator nominal-centered truths and a frozen core; the explicit oracle frontend is stage-one privilege")
        self.oracle_tracking_features = bool(oracle_tracking_features)
        self.oracle_clean_proprio = bool(oracle_clean_proprio)
        if oracle_feature_curriculum and not self.oracle_tracking_features:
            raise ValueError("Oracle-feature curriculum requires oracle tracking features")
        if oracle_feature_curriculum:
            self.register_buffer("oracle_feature_mix", torch.tensor(0.0))
        if self.oracle_clean_proprio and not self.oracle_tracking_features:
            raise ValueError("Clean privileged proprioception requires oracle tracking features")
        self.train_base_policy = bool(train_base_policy)
        self.tracker.mlp.requires_grad_(self.train_base_policy)
        observed_dim = int(obs[PRIVILEGE].shape[-1])
        if privilege_dim is not None and privilege_dim != observed_dim:
            raise ValueError(f"Privileged schema mismatch: {privilege_dim} != {observed_dim}")
        self.privilege_dim = observed_dim
        self.adaptation_latent_dim = int(adaptation_latent_dim)
        self.privilege_normalizer = EmpiricalNormalization(observed_dim, until=20_000_000)
        self.privilege_encoder = MLP(
            observed_dim,
            adaptation_latent_dim,
            privilege_hidden_dims,
            "elu",
            last_activation="tanh",
        )
        if self.physics_bilinear or self.physics_low_rank:
            self.privilege_normalizer = torch.nn.Identity()
            self.privilege_encoder = torch.nn.Identity()
        self.residual_mlp = MLP(
            self.tracker.policy_input_dim + adaptation_latent_dim,
            output_dim,
            kwargs.get("residual_hidden_dims", (512, 256, 128)),
            "elu",
        )
        if self.latent_modulation:
            self.residual_mlp = LatentModulatedResidual(
                self.tracker.policy_input_dim + adaptation_latent_dim,
                output_dim, kwargs.get("residual_hidden_dims", (512, 256, 128)),
                adaptation_latent_dim,
            )
        if self.physics_bilinear:
            self.residual_mlp = PhysicsBilinearResidual(
                self.tracker.policy_input_dim + adaptation_latent_dim,
                output_dim, kwargs.get("residual_hidden_dims", (512, 256, 128)),
                adaptation_latent_dim,
            )
        if self.physics_low_rank or self.latent_low_rank:
            self.residual_mlp = PhysicsLowRankResidual(
                self.tracker.mlp, adaptation_latent_dim, rank_per_physics, self.shared_low_rank,
            )
        else:
            output = _last_linear(self.residual_mlp)
            torch.nn.init.zeros_(output.weight)
            torch.nn.init.zeros_(output.bias)
        self.refinement_scale = float(refinement_scale)
        self.freeze_teacher_trunk = bool(freeze_teacher_trunk)
        if self.freeze_teacher_trunk and (
            self.refinement_scale <= 0 or train_base_policy or oracle_feature_curriculum
        ):
            raise ValueError("Frozen teacher refinement requires a new positive-scale branch and an unchanged teacher frontend")
        self.refinement_mlp = make_refinement_mlp(
            self.tracker.policy_input_dim + adaptation_latent_dim, output_dim, self.refinement_scale,
        )
        if self.freeze_teacher_trunk:
            self.residual_mlp.requires_grad_(False)
            self.privilege_encoder.requires_grad_(False)

    def _residual(self, value):
        result = super()._residual(value)
        return result if self.refinement_mlp is None else result + additional_refinement(self, value)

    def _actor_privilege_input(self, obs):
        value = obs[PRIVILEGE]
        if getattr(self, "actor_physics_input", "normal") == "zero":
            value = torch.zeros_like(value)
        elif getattr(self, "actor_physics_input", "normal") == "constant":
            # Every rank remains active, but no world-specific physical
            # information survives. Unlike a zero code, this is a trainable
            # state-only low-rank controller, not a disabled residual.
            value = torch.ones_like(value)
        return value

    def privileged_latent(self, obs):
        value = PrivilegedAdaptationActor._actor_privilege_input(self, obs)
        remove = getattr(self, "oracle_state_ablation", "normal")
        if remove != "normal":
            if self.privilege_schema != "state_physics":
                raise ValueError("All-path height/contact ablation requires state_physics schema")
            # _robot_raw_state is 71D; removing x/y leaves 69D, followed by
            # two contact labels, then physics. Replace BOTH actor routes.
            height, logits = self.tracker.estimate_height_and_contact(obs)
            value = value.clone()
            if remove in ("height", "height_contact"):
                value[..., :1] = height
            if remove in ("contact", "height_contact"):
                value[..., 69:71] = logits.sigmoid()
        latent = self.privilege_encoder(self.privilege_normalizer(value).clamp(-10, 10))
        if getattr(self, "latent_low_rank", False):
            nominal = self.privilege_encoder(
                self.privilege_normalizer(torch.zeros_like(value)).clamp(-10, 10)
            )
            # Nonlinear physical coordinates with structural zero at the true
            # nominal parameter vector. Range stays [-1,1] for context targets.
            latent = 0.5 * (latent - nominal)
        return latent

    def _base_features_and_action(self, obs):
        return _base_with_optional_update(self, obs)

    def _residual_input(self, obs, tracker_features, base_action, *, latent_override=None):
        del base_action
        latent = self.privileged_latent(obs) if latent_override is None else latent_override
        return torch.cat((tracker_features, latent), dim=-1)

    def update_normalization(self, obs):
        super().update_normalization(obs)
        if not (self.physics_bilinear or self.physics_low_rank or self.freeze_teacher_trunk):
            self.privilege_normalizer.update(self._actor_privilege_input(obs))

    @torch.no_grad()
    def policy_metrics(self, obs):
        features, base = self._base_features_and_action(obs)
        latent = self.privileged_latent(obs)
        residual = self._residual(self._residual_input(obs, features, base, latent_override=latent))
        shuffled = self._residual(
            self._residual_input(obs, features, base, latent_override=latent.roll(1, 0))
        )
        return {
            "residual_action_rms": float(residual.square().mean().sqrt()),
            "residual_action_abs_max": float(residual.abs().max()),
            "residual_saturation_fraction": float(
                (residual.abs() > 0.95 * self.residual_scale).float().mean()
            ),
            "privilege_shuffle_action_delta_rms": float(
                (residual - shuffled).square().mean().sqrt()
            ),
            "privilege_latent_rms": float(latent.square().mean().sqrt()),
        }
