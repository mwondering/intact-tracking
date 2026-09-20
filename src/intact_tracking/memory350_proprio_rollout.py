"""Native DR collection using exactly the policy's cached noisy observations."""

from copy import deepcopy

from intact_tracking.memory350_native_dr import NativeDRTrackerRollout
from intact_tracking.memory350_proprio_inputs import (
    INPUT_CONTRACT, current_proprio, control_action, validate_proprio_observations,
)


class ProprioNativeDRTrackerRollout(NativeDRTrackerRollout):
    def __init__(self, config):
        super().__init__(config)
        try:
            self.proprio_observation_audit = validate_proprio_observations(self.env)
        except BaseException:
            self.close()
            raise

    def step(self, **kwargs):
        before = current_proprio(self.observations).clone()
        batch = super().step(**kwargs)
        batch.update(encoder_state=before,
                     encoder_next_state=current_proprio(self.observations).clone(),
                     encoder_action=control_action(self.env))
        return batch

    @property
    def metadata(self):
        return {**super().metadata, "context_input_contract": deepcopy(INPUT_CONTRACT),
                "proprio_observation_audit": getattr(self, "proprio_observation_audit", None)}
