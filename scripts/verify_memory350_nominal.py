"""Verify real nominal physics, preserved DR, reset persistence and A/B responses."""

import argparse
import json
from pathlib import Path

import torch

from run_limb_context_experiment import TRACKER, SMOKE_MOTION
from intact_tracking.memory350_nominal_rollout import (
    NominalMemory350RolloutConfig, NominalMemory350TrackerRollout,
)
from intact_tracking.rollout.nominal import NominalPairRollout, NominalPairRolloutConfig
from intact_tracking.forward_predictor import physical_state_delta
from intact_tracking.limb_context_protocol import PAYLOAD_EVENT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = NominalMemory350RolloutConfig(
        checkpoint_file=TRACKER, motion_file=SMOKE_MOTION, num_envs=128, device='cuda:0',
        seed=717, tracker_dr_plus_limb_payload=True)
    errors = {'nominal': [], 'dr': []}
    force_peak = {'nominal': 0., 'dr': 0.}
    with NominalMemory350TrackerRollout(config) as rollout, NominalPairRollout(
        NominalPairRolloutConfig(checkpoint_file=TRACKER, motion_file=SMOKE_MOTION,
                                num_envs=128, device='cuda:0', seed=100717)
    ) as nominal:
        assert rollout.is_nominal[:96].sum() == 48
        assert rollout.is_nominal[96:].sum() == 16
        assert rollout.metadata['nominal_world_local_ids'] == list(range(0, 128, 2))
        rollout.force_pulse.time_to_next_pulse_s.zero_()
        with torch.inference_mode():
            for block in range(80):
                rows = []
                for _ in range(5):
                    rows.append(rollout.step(predictor_only=True))
                    rollout.assert_nominal_clean()
                    force = rollout.env.sim.data.xfrc_applied.clone()
                    for label, mask in [('nominal', rollout.is_nominal), ('dr', ~rollout.is_nominal)]:
                        force_peak[label] = max(force_peak[label], float(force[mask].abs().max()))
                targets = torch.stack([row['joint_target'] for row in rows], 1)
                b, _ = nominal.rollout_joint_targets(rows[0]['robot_state'], targets)
                a = torch.stack([row['next_robot_state'] for row in rows], 1)
                valid = ~torch.stack([row['reset_boundary'] for row in rows], 1).any(1)
                error = physical_state_delta(b, a)
                for label, mask in [('nominal', rollout.is_nominal), ('dr', ~rollout.is_nominal)]:
                    errors[label].append(error[mask & valid].cpu())
                if block % 10 == 0:
                    print(json.dumps({'control_steps': (block + 1) * 5, 'force_peak': force_peak,
                                      'resets': rollout.environments_reset}), flush=True)
        rollout._assert_fixed_dr()
        assert force_peak['nominal'] == 0
        assert force_peak['dr'] > 0
        assert rollout.environments_reset > 0
        payload = rollout.env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
        payload_audit = payload.audit()
        assert payload_audit['nominal_worlds'] == 64
        result = {'passed': True, 'steps': 400, 'num_envs': 128, 'seed': 717,
                  'force_peak_absolute': force_peak, 'reset_slots': rollout.environments_reset,
                  'rollout_metadata': rollout.metadata, 'payload_after_resets': payload_audit,
                  'counterfactual': {}}
        for label, chunks in errors.items():
            error = torch.cat(chunks).double()
            assert torch.isfinite(error).all()
            joint = error[..., 12:41]
            root = error[..., :3]
            result['counterfactual'][label] = {
                'five_step_windows': len(error),
                'joint_position_rms_rad': float(joint.square().mean().sqrt()),
                'root_position_rms_m': float(root.square().mean().sqrt()),
                'joint_position_abs_p99_rad': float(torch.quantile(joint.abs().flatten(), .99)),
            }
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ('rollout_metadata', 'payload_after_resets')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
