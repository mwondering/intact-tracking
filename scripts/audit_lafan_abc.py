"""Independently verify completed A/B/C budgets, initialization and weight changes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--evaluation", action="store_true")
    args = parser.parse_args()
    results, configs = {}, []
    for arm in "ABC":
        directory = args.root / arm
        config = json.loads((directory / "run_config.json").read_text())
        completed = json.loads((directory / "completion_audit.json").read_text())
        configs.append(config)
        assert config["condition"] == arm
        assert config["initial_actor_and_critic_bitwise_restored"]
        assert completed["completed_updates"] == config["arguments"]["iterations"] == 1000
        assert completed["global_envs"] == 4096 and completed["motion_count"] == 40
        assert completed["all_weights_finite"] and len(set(completed["rank_network_sha256"])) == 1
        initial = torch.load(directory / "checkpoint_initial.pt", map_location="cpu", weights_only=False)
        final = torch.load(directory / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        assert final["completed_updates"] == 1000
        norms, digest = {}, hashlib.sha256()
        for key in ("actor_state_dict", "critic_state_dict"):
            squared_change = 0.0
            for name, value in final[key].items():
                assert torch.isfinite(value).all()
                digest.update(value.numpy().tobytes())
                if name.startswith("mlp."):
                    squared_change += float((value.double() - initial[key][name].double()).square().sum())
                elif key == "actor_state_dict" and not name.startswith("distribution."):
                    assert torch.equal(value, initial[key][name]), name
            assert squared_change > 0, f"{arm} {key} did not learn"
            norms[key] = squared_change ** .5
        assert digest.hexdigest() == completed["rank_network_sha256"][0]
        results[arm] = {"completed_updates": 1000, "global_envs": 4096, "motions": 40,
                        "actor_critic_MLP_change_L2": norms, "final_tensor_sha256": digest.hexdigest(),
                        "frozen_actor_unchanged": True, "rank_states_equal": True}
        del initial, final
    for config in configs[1:]:
        for field in ("tracker_sha256", "motion_sha256", "motion_files", "reward_contract",
                      "research_source_sha256", "input_audit", "trainable_actor", "frozen_actor", "critic"):
            assert config[field] == configs[0][field], field
        for field in ("num_envs", "iterations", "seed", "actor_lr", "critic_lr", "critic_warmup"):
            assert config["arguments"][field] == configs[0]["arguments"][field], field
    for path, digest in configs[0]["research_source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    results["matched_training_protocol"] = True
    (args.root / "independent_training_audit.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.evaluation:
        hashes = {}
        for arm in ("A", "B", "C", "frozen"):
            checkpoint = (Path(configs[0]["tracker_checkpoint"]) if arm == "frozen"
                          else args.root / arm / "checkpoint_final.pt")
            with checkpoint.open("rb") as source:
                hashes[arm] = hashlib.file_digest(source, "sha256").hexdigest()
            for endpoint in ("nominal", "hardest"):
                for seed in (20001, 20002, 20003):
                    row = json.loads((args.root / "eval" / f"{arm}_{endpoint}_{seed}.json").read_text())
                    assert row["checkpoint_sha256"] == hashes[arm]
                    assert row["episodes"] == 320 and row["motions"] == 40
                    assert row["repeats_per_motion"] == 8 and row["max_steps"] == 500
                    assert row["seed"] == seed
                    assert row["reference_timeline_audited"]
                    assert row["partial_reset_survivor_state_audited"]
                    assert row["partial_reset_survivor_history_audited"]
                    for field in ("simulator_preview", "oracle_tracking_features",
                                  "privileged_bias_compensation", "privileged_payload_gravity_compensation",
                                  "action_filter_strength"):
                        assert not row[field], field
                    assert row["orientation_filter_weight"] == 1
        evaluation = {"checkpoint_sha256": hashes, "validated_evaluations": 24,
                      "episodes_per_policy_endpoint": 960, "only_final_trained_checkpoints": True,
                      "no_preview_or_oracle_features_or_action_filters": True,
                      "reference_and_survivor_history_audits_passed": True}
        (args.root / "independent_evaluation_audit.json").write_text(json.dumps(evaluation, indent=2) + "\n")
        results["evaluation"] = evaluation
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
