"""Alternating PPO and teacher-action regularization for a deployable student."""

from __future__ import annotations

import torch

from intact_tracking.adaptation_policy import (
    ContextAdaptationActor,
    PrivilegedAdaptationWrapper,
)
from intact_tracking.residual_policy import ResidualPPO

TEACHER_ACTION = "adaptation_teacher_action_target"
CONTEXT_TARGET = "adaptation_context_auxiliary_target"


class CriticWarmupPPO(ResidualPPO):
    """Fit the value function to the initialized policy before actor updates."""

    def __init__(self, *args, critic_warmup_updates=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_warmup_updates = int(critic_warmup_updates)
        self.adaptation_update_count = 0

    def update(self):
        warming = self.adaptation_update_count < self.critic_warmup_updates
        parameters = [p for p in self.actor.parameters() if p.requires_grad]
        schedule = self.schedule
        if warming:
            for parameter in parameters:
                parameter.requires_grad_(False)
            self.schedule = "fixed"
        try:
            result = super().update()
        finally:
            for parameter in parameters:
                parameter.requires_grad_(True)
            self.schedule = schedule
        self.adaptation_update_count += 1
        result["critic_warmup_active"] = float(warming)
        return result

    def save(self):
        state = super().save()
        state["adaptation_update_count"] = self.adaptation_update_count
        return state

    def load(self, loaded_dict, load_cfg, strict):
        result = super().load(loaded_dict, load_cfg, strict)
        self.adaptation_update_count = int(loaded_dict.get("adaptation_update_count", 0))
        return result


class OracleFeatureCurriculumPPO(CriticWarmupPPO):
    """Gradually replace oracle tracking features without changing sensor noise.

    The scalar is actor state, so every saved checkpoint evaluates at its actual
    observation mixture. A completed curriculum still has privileged physics
    in its encoder and is a teacher, never a stage-two deployment claim.
    """

    def __init__(self, *args, oracle_feature_curriculum_updates=500, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_feature_curriculum_updates = int(oracle_feature_curriculum_updates)
        if self.oracle_feature_curriculum_updates <= 0:
            raise ValueError("Oracle feature curriculum needs a positive duration")
        if not hasattr(self.actor, "oracle_feature_mix"):
            raise ValueError("Oracle feature curriculum requires a compatible privileged actor")

    def update(self):
        result = super().update()
        progress = min(
            1.0,
            max(0, self.adaptation_update_count - self.critic_warmup_updates)
            / self.oracle_feature_curriculum_updates,
        )
        self.actor.oracle_feature_mix.fill_(progress)
        result["oracle_feature_mix"] = progress
        return result


class ContextSupervisionWrapper:
    """Training-only velocity/physics labels, disjoint from actor inputs."""

    def __init__(self, wrapped):
        from intact_tracking.cli.adaptation_distill import normalized_context_physics_targets
        from intact_tracking.cli.context_probe import physical_labels

        self.wrapped = wrapped
        _, physical = physical_labels(wrapped.unwrapped)
        self.payload = physical[:, 0].to(wrapped.device)
        self.physics = normalized_context_physics_targets(physical).to(wrapped.device)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def attach(self, obs):
        from intact_tracking.cli.adaptation_distill import context_auxiliary_targets

        robot = self.unwrapped.scene["robot"].data
        dynamic = context_auxiliary_targets(
            robot.root_link_lin_vel_w, robot.root_link_quat_w, self.payload
        )
        obs.set(CONTEXT_TARGET, torch.cat((dynamic, self.physics), -1).detach())
        return obs

    def get_observations(self):
        return self.attach(self.wrapped.get_observations())

    def step(self, actions):
        obs, reward, done, extras = self.wrapped.step(actions)
        return self.attach(obs), reward, done, extras

    def reset(self):
        obs, extras = self.wrapped.reset()
        return self.attach(obs), extras


class ContextAuxiliaryPPO(CriticWarmupPPO):
    """Keep context state/physics prediction trained alongside task PPO.

    Extra steps reuse the PPO optimizer; only encoder/readout receive gradients.
    Their labels never reach either encoder or controller inputs. Actor warmup
    also disables these steps, preserving the initial policy during critic fit.
    """

    def __init__(
        self, *args, context_aux_steps=2, context_aux_batch_size=2048,
        context_aux_weight=0.05, context_physics_weight=0.01, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(self.actor, ContextAdaptationActor) or self.actor.context_latent_mean:
            raise TypeError("Auxiliary PPO requires a non-memory deployable context actor")
        if self.actor.context_auxiliary_head is None or not any(
            parameter.requires_grad for parameter in self.actor.context_encoder.parameters()
        ):
            raise ValueError("Auxiliary PPO requires a trained readout and trainable encoder")
        self.context_aux_steps = int(context_aux_steps)
        self.context_aux_batch_size = int(context_aux_batch_size)
        self.context_aux_weight = float(context_aux_weight)
        self.context_physics_weight = float(context_physics_weight)

    def update(self):
        observations = self.storage.observations.flatten(0, 1)
        result = super().update()
        if result["critic_warmup_active"]:
            return result
        dynamic_total = physics_total = 0.0
        for _ in range(self.context_aux_steps):
            ids = torch.randint(len(observations), (self.context_aux_batch_size,), device=self.device)
            batch = observations[ids]
            inputs = batch.select(*self.actor.deployable_observation_groups)
            latent = self.actor.instant_context_latent(inputs)
            prediction = self.actor.context_auxiliary_head(latent)
            target = batch[CONTEXT_TARGET].detach()
            dynamic_mse = (prediction[..., :4] - target[..., :4]).square().mean()
            physics_mse = (prediction[..., 4:10] - target[..., 4:10]).square().mean()
            loss = self.context_aux_weight * dynamic_mse + self.context_physics_weight * physics_mse
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite context auxiliary PPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.actor.parameters() if parameter.requires_grad],
                self.max_grad_norm,
            )
            self.optimizer.step()
            dynamic_total += float(dynamic_mse.detach())
            physics_total += float(physics_mse.detach())
        result["context_supervision_dynamic_mse"] = dynamic_total / max(self.context_aux_steps, 1)
        result["context_supervision_physics_mse"] = physics_total / max(self.context_aux_steps, 1)
        return result


class TeacherSupervisionWrapper(PrivilegedAdaptationWrapper):
    """Reserve a loss-label field in storage before its schema is constructed."""

    def attach(self, obs):
        obs = super().attach(obs)
        obs.set(TEACHER_ACTION, obs["priv"].new_zeros((self.num_envs, self.num_actions)))
        return obs


@torch.no_grad()
def teacher_action_targets(teacher, student, obs):
    # Separate TensorDict structure: teacher cache writes cannot replace the
    # student's cached deployable estimator/reference features.
    teacher_obs = obs.clone(recurse=False)
    teacher.populate_tracker_cache(teacher_obs)
    target = teacher(teacher_obs).detach()
    student.populate_tracker_cache(obs)
    return target


class TeacherRegularizedPPO(ResidualPPO):
    """Standard PPO followed by small BC updates on the same visited states.

    Teacher actions are simultaneous loss labels, not action inputs. The
    teacher is attached to the algorithm, never to the saved student module.
    """

    def __init__(
        self,
        *args,
        teacher_bc_steps=2,
        teacher_bc_weight=0.1,
        teacher_bc_batch_size=2048,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(self.actor, ContextAdaptationActor):
            raise TypeError("Teacher regularization requires a deployable context student")
        self.teacher = None
        self.teacher_bc_steps = int(teacher_bc_steps)
        self.teacher_bc_weight = float(teacher_bc_weight)
        self.teacher_bc_batch_size = int(teacher_bc_batch_size)

    def act(self, obs):
        if self.teacher is None:
            raise RuntimeError("Attach a frozen teacher before collecting PPO experience")
        obs.set(TEACHER_ACTION, teacher_action_targets(self.teacher, self.actor, obs))
        return super().act(obs)

    def update(self):
        # RolloutStorage.clear resets only its cursor; the tensor data remain
        # available until the next collection, after these supervised steps.
        observations = self.storage.observations.flatten(0, 1)
        result = super().update()
        total = 0.0
        for _ in range(self.teacher_bc_steps):
            ids = torch.randint(
                len(observations), (self.teacher_bc_batch_size,), device=self.device
            )
            batch = observations[ids]
            prediction = self.actor(batch)
            mse = (prediction - batch[TEACHER_ACTION].detach()).square().mean()
            if not torch.isfinite(mse):
                raise RuntimeError("Nonfinite teacher regularization loss")
            self.optimizer.zero_grad(set_to_none=True)
            (self.teacher_bc_weight * mse).backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.actor.parameters() if p.requires_grad], self.max_grad_norm
            )
            self.optimizer.step()
            total += float(mse.detach())
        result["teacher_action_mse"] = total / max(self.teacher_bc_steps, 1)
        return result
