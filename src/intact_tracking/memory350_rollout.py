"""New rollout version with a 1000-control-step cap; baseline rollout is unchanged."""

from intact_tracking.rollout.online import FixedDRTrackerRollout, _randomize_initial_episode_phases


class Memory350TrackerRollout(FixedDRTrackerRollout):
    def __init__(self, config):
        if not config.tracker_dr_plus_limb_payload:
            raise ValueError("Memory350 uses frozen-tracker DR plus independent four-limb payloads")
        super().__init__(config)
        # No step has been collected yet. MJLab reads timeout length from cfg;
        # the wrapper also stores a cached integer which must match it.
        self.env.cfg.episode_length_s = 1000 * self.env.step_dt
        self.wrapped.max_episode_length = 1000
        if self.env.max_episode_length != 1000:
            raise RuntimeError("The live environment did not adopt the 1000-step episode cap")
        if config.randomize_initial_episode_phase:
            self.initial_episode_phase_summary = _randomize_initial_episode_phases(self.env, config.seed, 1)

    @property
    def metadata(self):
        return {**super().metadata, "episode_length_control_steps": self.env.max_episode_length,
                "episode_length_seconds": self.env.cfg.episode_length_s,
                "memory_protocol": "short50_disjoint_raw_chunk10_long30_v1"}
