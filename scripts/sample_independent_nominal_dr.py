"""Initialize GPU environments and audit independent nominal mixtures, without training."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from intact_tracking.independent_nominal_dr import EVENT
from intact_tracking.memory350_nominal_rollout import (
    NominalMemory350RolloutConfig, NominalMemory350TrackerRollout,
)
from intact_tracking.memory350_rollout import Memory350TrackerRollout
from intact_tracking.rollout.mjlab_adapter import _sha256
from intact_tracking.rollout.online import FixedDRRolloutConfig, _capture_privileged_dynamics_targets


SOURCE = Path("runs/limb_context_20260916_dr16384")
OUTPUT = Path("runs/limb_context_20260917_all_fixed_dr_half_nominal")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True, choices=range(8))
    parser.add_argument("--probability", type=float, default=.5)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = SOURCE / f"shard_{args.shard:02d}"
    metadata = json.loads((source / "metadata.json").read_text())
    is_mixed_population = metadata["nominal_worlds"] > 0
    config_type = NominalMemory350RolloutConfig if is_mixed_population else FixedDRRolloutConfig
    rollout_type = NominalMemory350TrackerRollout if is_mixed_population else Memory350TrackerRollout
    config = config_type(**{**metadata["rollout_config"], "dr_nominal_probability": args.probability})
    out = args.output / f"shard_{args.shard:02d}"
    out.mkdir(parents=True, exist_ok=False)
    with rollout_type(config) as rollout:
        env = rollout.env
        mixture = env.event_manager.get_term_cfg(EVENT).func
        with np.load(source / "physics.npz") as saved:
            original, world, nominal, names = saved["values"], saved["world"], saved["nominal"], saved["names"]
        physics = _capture_privileged_dynamics_targets(env).values.cpu().numpy()
        np.testing.assert_array_equal(names, rollout.privileged_dynamics_names)
        np.testing.assert_array_equal(nominal, rollout.is_nominal.cpu().numpy())
        mask = torch.cat([mixture.masks[k].reshape(config.num_envs, -1) for k in
                          ("com_xyz", "torso_mass", "foot_friction", "armature", "limb_payload")], dim=1).cpu().numpy()
        nominal_value = np.zeros(38, dtype=original.dtype)
        nominal_value[4] = .6
        nominal_value[5:34] = 1
        expected = np.where(mask, nominal_value, original)
        expected[nominal] = nominal_value
        np.testing.assert_array_equal(physics, expected)
        bias = env.scene["robot"].data.encoder_bias.clone()
        expected_bias = torch.where(mixture.masks["encoder_bias"], 0., mixture.original["encoder_bias"])
        expected_bias[torch.from_numpy(nominal).to(env.device)] = 0
        torch.testing.assert_close(bias, expected_bias, rtol=0, atol=0)
        masks_before = {name: value.clone() for name, value in mixture.masks.items()}
        for _ in range(3):
            env.reset()
            rollout._assert_fixed_dr()
            np.testing.assert_array_equal(_capture_privileged_dynamics_targets(env).values.cpu().numpy(), physics)
            torch.testing.assert_close(env.scene["robot"].data.encoder_bias, bias, rtol=0, atol=0)
            for name, before in masks_before.items():
                torch.testing.assert_close(mixture.masks[name], before, rtol=0, atol=0)
        np.savez_compressed(out / "parameters.npz", world=world, names=names, nominal=nominal,
                            original=original, mixed=physics, nominal_mask=mask,
                            encoder_bias=bias.cpu().numpy(),
                            encoder_bias_original=mixture.original["encoder_bias"].cpu().numpy(),
                            encoder_bias_mask=mixture.masks["encoder_bias"].cpu().numpy())
        result = {
            "complete": True, "shard": args.shard, "config": asdict(config),
            "rollout_metadata": rollout.metadata, "mixture_audit": mixture.audit(),
            "parameters_sha256": _sha256(out / "parameters.npz"),
            "source_metadata_sha256": _sha256(source / "metadata.json"),
            "script_sha256": _sha256(Path(__file__)),
            "checks": {"all_38_physics_coordinates_match_original_or_nominal_exactly": True,
                       "all_encoder_bias_coordinates_match_original_or_zero_exactly": True,
                       "masks_and_physics_fixed_across_three_full_resets": True,
                       "no_policy_steps_or_training": True},
            "dr_worlds": int((~nominal).sum()), "nominal_controls": int(nominal.sum()),
        }
        (out / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"shard": args.shard, "complete": True, "checks": result["checks"]}), flush=True)


if __name__ == "__main__":
    main()
