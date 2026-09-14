"""Hierarchical-memory objective; inherited transition/nominal losses are unchanged."""

import torch
from intact_tracking.forward_predictor_objective import (
    ForwardPredictorObjective, DEFAULT_RECURSIVE_WEIGHT, _normalized_state_error,
    _state_loss, _foot_loss, _contact_loss, _counterfactual_representation_loss,
    _masked_mean, _masked_nmse, _masked_rms,
)


class Memory350Objective(ForwardPredictorObjective):
    def _encode_views(self, batch):
        combined = self.model.encode_context(
            torch.cat((batch["history_state"], batch["positive_history_state"]), dim=0),
            torch.cat((batch["history_action"], batch["positive_history_action"]), dim=0),
            torch.cat((batch["state"][:, 0], batch["positive_current_state"]), dim=0),
            torch.cat((batch["history_valid"], batch["positive_history_valid"]), dim=0),
            history_next_state=torch.cat((batch["history_next_state"], batch["positive_history_next_state"]), dim=0),
            memory_interactions=torch.cat((batch["memory_interactions"], batch["positive_memory_interactions"]), dim=0),
            memory_valid=torch.cat((batch["memory_valid"], batch["positive_memory_valid"]), dim=0),
        )
        return combined.split(batch["state"].size(0), dim=0)

    def _extra_representation_loss(self, batch, views):
        return None, {}

    def _representation_response(self, batch, target):
        return _normalized_state_error(
            target, batch["nominal_state"], batch["state_mean"],
            batch["state_std"], batch["delta_std"]), None

    @staticmethod
    def _validate_batch(batch):
        ForwardPredictorObjective._validate_batch(batch)
        required = {prefix + name for prefix in ("", "positive_")
                    for name in ("history_next_state", "memory_interactions", "memory_valid")}
        if not required.issubset(batch):
            raise KeyError(f"Missing hierarchical memory fields: {sorted(required - set(batch))}")

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        recursive_weight: float = DEFAULT_RECURSIVE_WEIGHT,
        *,
        compute_metrics: bool = True,
        validate_batch: bool = True,
    ) -> dict[str, torch.Tensor]:
        if validate_batch:
            self._validate_batch(batch)
        if recursive_weight < 0.0:
            raise ValueError("recursive_weight must be non-negative")
        state = batch["state"]
        action = batch["action"]
        batch_size = state.size(0)
        if tuple(state.shape[1:]) != (6, 71):
            raise ValueError(f"State batch must be [batch,6,71], got {tuple(state.shape)}")
        if tuple(action.shape) != (batch_size, 5, 29):
            raise ValueError(f"Action batch must be [batch,5,29], got {tuple(action.shape)}")
        context_steps = self.model.config.context_history_steps
        expected = {
            "nominal_state": (batch_size, 5, 71),
            "foot": (batch_size, 6, 8),
            "contact_force": (batch_size, 6, 6),
            "contact_binary": (batch_size, 6, 2),
            "history_state": (batch_size, context_steps, 71),
            "history_action": (batch_size, context_steps, 29),
            "history_valid": (batch_size, context_steps),
            "positive_current_state": (batch_size, 71),
            "positive_history_state": (batch_size, context_steps, 71),
            "positive_history_action": (batch_size, context_steps, 29),
            "positive_history_valid": (batch_size, context_steps),
            "positive_pair_valid": (batch_size,),
            "is_nominal": (batch_size,),
            "world_id": (batch_size,),
        }
        invalid = {
            name: (tuple(batch[name].shape), shape)
            for name, shape in expected.items()
            if tuple(batch[name].shape) != shape
        }
        if invalid:
            raise ValueError(f"Invalid Forward Predictor batch shapes: {invalid}")

        views = self._encode_views(batch)
        latent, positive_latent = views[:2]
        history_arguments = {
            "history_state": batch["history_state"],
            "history_action": batch["history_action"],
            "history_foot": batch["history_foot"],
            "history_contact_force": batch["history_contact_force"],
            "history_contact_binary": batch["history_contact_binary"],
            "history_valid": batch["history_valid"],
        }
        normalization_arguments = {
            "state_mean": batch["state_mean"],
            "state_std": batch["state_std"],
            "delta_mean": batch["delta_mean"],
            "delta_std": batch["delta_std"],
        }
        teacher_prediction, _, teacher_foot, teacher_force, teacher_binary = (
            self.model.teacher_forced(
                state,
                action,
                batch["foot"],
                batch["contact_force"],
                batch["contact_binary"],
                **history_arguments,
                **normalization_arguments,
                dynamics_latent=latent,
            )
        )
        recursive_prediction, _, recursive_foot, recursive_force, recursive_binary = (
            self.model.rollout(
                state[:, 0],
                action,
                batch["foot"][:, 0],
                batch["contact_force"][:, 0],
                batch["contact_binary"][:, 0],
                **history_arguments,
                **normalization_arguments,
                dynamics_latent=latent,
            )
        )
        target = state[:, 1:]
        target_foot = batch["foot"][:, 1:]
        target_force = batch["contact_force"][:, 1:]
        target_binary = batch["contact_binary"][:, 1:]
        teacher_error = _normalized_state_error(
            teacher_prediction,
            target,
            batch["state_mean"],
            batch["state_std"],
            batch["delta_std"],
        )
        recursive_error = _normalized_state_error(
            recursive_prediction,
            target,
            batch["state_mean"],
            batch["state_std"],
            batch["delta_std"],
        )

        def prediction_branch_loss(
            error: torch.Tensor,
            foot_prediction: torch.Tensor,
            force_prediction: torch.Tensor,
            binary_prediction: torch.Tensor,
        ) -> torch.Tensor:
            return (
                _state_loss(error, self.loss_config, huber=True)
                + self.loss_config.foot_weight
                * _foot_loss(foot_prediction, target_foot, self.loss_config, huber=True)
                + _contact_loss(
                    force_prediction,
                    binary_prediction,
                    target_force,
                    target_binary,
                    self.loss_config,
                    huber=True,
                )
            )

        teacher_loss = prediction_branch_loss(
            teacher_error, teacher_foot, teacher_force, teacher_binary
        )
        recursive_loss = prediction_branch_loss(
            recursive_error, recursive_foot, recursive_force, recursive_binary
        )
        prediction_loss = teacher_loss + float(recursive_weight) * recursive_loss
        counterfactual_response, response_valid = self._representation_response(batch, target)
        representation_loss, representation_metrics, partner = _counterfactual_representation_loss(
            latent,
            positive_latent,
            counterfactual_response,
            batch["positive_pair_valid"].bool(),
            (batch["history_valid"].any(dim=1) | batch["memory_valid"].any(dim=1)),
            (batch["positive_history_valid"].any(dim=1) | batch["positive_memory_valid"].any(dim=1)),
            batch["world_id"],
            response_distance_scale=self.loss_config.response_distance_scale,
            representation_relation_weight=self.loss_config.representation_relation_weight,
            compute_metrics=compute_metrics,
            response_valid=response_valid,
        )
        total_loss = prediction_loss + self.loss_config.representation_weight * representation_loss
        extra_loss, extra_metrics = self._extra_representation_loss(batch, views)
        if extra_loss is not None:
            total_loss = total_loss + extra_loss
        if not compute_metrics:
            return {
                "loss": total_loss,
                "prediction_loss": prediction_loss.detach(),
                "representation_loss": representation_loss.detach(),
                **extra_metrics,
            }

        with torch.no_grad():
            unchanged = state[:, :1].expand_as(target)
            unchanged_error = _normalized_state_error(
                unchanged,
                target,
                batch["state_mean"],
                batch["state_std"],
                batch["delta_std"],
            )
            shuffled_prediction = self.model.rollout(
                state[:, 0],
                action,
                batch["foot"][:, 0],
                batch["contact_force"][:, 0],
                batch["contact_binary"][:, 0],
                **history_arguments,
                **normalization_arguments,
                dynamics_latent=latent.index_select(0, partner),
            )[0]
            shuffled_error = _normalized_state_error(
                shuffled_prediction,
                target,
                batch["state_mean"],
                batch["state_std"],
                batch["delta_std"],
            )
            nominal = batch["is_nominal"].bool()
            dr = ~nominal
            shuffle_valid = dr & (batch["world_id"] != batch["world_id"].index_select(0, partner))
            true_dr_mse = _masked_mean(recursive_error[:, -1].square(), shuffle_valid)
            shuffled_dr_mse = _masked_mean(shuffled_error[:, -1].square(), shuffle_valid)
            latent_shuffle_ratio = shuffled_dr_mse / true_dr_mse.clamp_min(1.0e-8)

        return {
            "loss": total_loss,
            "prediction_loss": prediction_loss.detach(),
            "representation_loss": representation_loss.detach(),
            "one_step_nmse": _masked_nmse(recursive_error[:, 0], unchanged_error[:, 0]),
            "nominal_five_step_nmse": _masked_nmse(
                recursive_error[:, -1], unchanged_error[:, -1], nominal
            ),
            "dr_five_step_nmse": _masked_nmse(recursive_error[:, -1], unchanged_error[:, -1], dr),
            "latent_shuffle_dr_error_ratio": latent_shuffle_ratio,
            "nominal_counterfactual_rms": _masked_rms(
                counterfactual_response, nominal if response_valid is None else nominal & response_valid),
            "dr_counterfactual_rms": _masked_rms(
                counterfactual_response, dr if response_valid is None else dr & response_valid),
            **representation_metrics,
            **extra_metrics,
        }
