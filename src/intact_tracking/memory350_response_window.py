"""Ten-step A-minus-B labels with unchanged five-step prediction and replay cadence."""

import torch

from intact_tracking.forward_predictor import physical_state_delta
from intact_tracking.memory350_weak_pairs import WeakPairObjective, WeakPairReplayBuffer


RESPONSE_FIELDS = frozenset({"label_response", "label_response_valid"})


def valid_response_window(batches):
    """A single continuous motion/physics trajectory is required for a response."""
    first = batches[0]
    valid = torch.ones_like(first["reset_boundary"], dtype=torch.bool)
    for offset, batch in enumerate(batches):
        valid &= (~batch["reset_boundary"].bool()
                  & (batch["episode_id"] == first["episode_id"])
                  & (batch["episode_step"] == first["episode_step"] + offset)
                  & (batch["motion_id"] == first["motion_id"])
                  & (batch["motion_step"] == first["motion_step"] + offset))
        if "parameters_changed" in batch:
            valid &= ~batch["parameters_changed"].bool()
    return valid


class ResponseWindowCollector:
    """One five-step lookahead block; nominal B always starts at the anchor A state.

    After the initial lookahead, every call advances A by five steps and admits
    five transitions to the original replay. Adjacent labels overlap but each
    nominal rollout is independently restored; two reset B segments are never
    concatenated. Only the admitted block is visible to encoder history/archive.
    """

    def __init__(self):
        self.pending = None

    @torch.no_grad()
    def __call__(self, rollout, nominal_rollout, replay):
        if self.pending is None:
            self.pending = [rollout.step(predictor_only=True) for _ in range(5)]
        following = [rollout.step(predictor_only=True) for _ in range(5)]
        batches = self.pending + following
        actions = torch.stack([batch["joint_target"] for batch in batches], dim=1)
        with torch.inference_mode():
            nominal, diagnostics = nominal_rollout.rollout_joint_targets(
                batches[0]["robot_state"], actions,
                motion_ids=batches[0]["motion_id"], motion_steps=batches[0]["motion_step"],
                motion_files=rollout.motion_files)
        actual = torch.stack([batch["next_robot_state"] for batch in batches], dim=1)
        valid = valid_response_window(batches)
        response = physical_state_delta(nominal, actual).masked_fill(~valid[:, None, None], 0)
        self.pending[-1]["label_response"] = response
        self.pending[-1]["label_response_valid"] = valid
        for index, batch in enumerate(self.pending):
            batch["nominal_next_robot_state"] = nominal[:, index]
            replay.add_step(batch)
        self.pending = following
        rollout.response_label_horizon = 10
        return diagnostics


class ResponseWindowReplay(WeakPairReplayBuffer):
    def _allocate(self):
        initialized = bool(self._history)
        super()._allocate()
        if not initialized:
            self._samples["label_response"] = torch.empty(
                self.capacity, 10, 70, device=self.device)
            self._samples["label_response_valid"] = torch.empty(
                self.capacity, dtype=torch.bool, device=self.device)

    @property
    def estimated_storage_bytes(self):
        return super().estimated_storage_bytes + self.capacity * (10 * 70 * 4 + 1)

    def add_step(self, batch):
        self._response_batch = batch
        try:
            return super().add_step(batch)
        finally:
            self._response_batch = None

    def _append_samples(self, samples, count):
        if not RESPONSE_FIELDS.issubset(self._response_batch):
            raise ValueError("Every emitted five-step sample needs its aligned ten-step response")
        for key in RESPONSE_FIELDS:
            samples[key] = self._response_batch[key][samples["env_id"]]
        super()._append_samples(samples, count)

    def _extra_sample_fields(self, selected, context, normalization):
        result = super()._extra_sample_fields(selected, context, normalization)
        delta_std = self._normalization_tensors(normalization, self.device)[-1]
        result.update(label_response=selected["label_response"] / delta_std,
                      label_response_valid=selected["label_response_valid"])
        return result


class ResponseWindowObjective(WeakPairObjective):
    def _representation_response(self, batch, target):
        # Historical fixed validation batches intentionally retain five-step
        # labels; their prediction and representation diagnostics stay matched.
        if "label_response" not in batch:
            return super()._representation_response(batch, target)
        if batch["label_response"].shape != (target.size(0), 10, 70):
            raise ValueError("Response-window training requires [batch,10,70]")
        return batch["label_response"], batch["label_response_valid"]
