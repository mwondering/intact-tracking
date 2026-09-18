"""Check saved distributed state, sampling and six-loss accounting."""
import argparse
import json
import math
from pathlib import Path
import torch


def verify(root, mode):
    contract = json.loads((root / f"{mode}_launch_contract.json").read_text())
    world_size = len(contract["physical_gpus"])
    directory = root / (f"smoke_{world_size}gpu" if mode == "smoke" else "stage1_8192")
    config = json.loads((directory / "run_config.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    state = torch.load(directory / "last.pt", map_location="cpu", weights_only=False, mmap=True)
    anchor = json.loads((directory / "nominal_anchor.json").read_text())
    weights = config["objective_weights"]
    assert weights == {
        "teacher_forced_weight": 1., "recursive_weight": .5, "local_positive_weight": .01,
        "dr_nominal_response_weight": .04, "weak_positive_weight": .008,
        "weak_negative_weight": 0., "nominal_anchor_weight": .01}
    assert state["loss_config"]["weak_negative_weight"] == 0
    assert state["representation_supervision"] == "fixed_nominal_direction_response10_v1"
    assert state["supervision_horizons"] == {"predictor": 5, "response_label": 10}
    assert state["nominal_direction_anchor"] == anchor
    torch.testing.assert_close(torch.tensor(anchor["direction"]).norm(),
                               torch.tensor(1.), atol=1e-6, rtol=0)
    assert anchor["validation_used"] is False and anchor["nominal_samples_total"] > 0
    assert len(anchor["sha256_by_rank"]) == world_size and len(set(anchor["sha256_by_rank"])) == 1
    agreement = state["distributed_parameter_agreement"]
    assert agreement["passed"] and len(agreement["sha256_by_rank"]) == world_size
    assert len(set(agreement["sha256_by_rank"])) == 1
    assert config["distributed"]["world_size"] == world_size
    assert len(list(directory.glob("validation_rank_*.pt"))) == world_size
    assert len(list(directory.glob("validation_broad_rank_*.pt"))) == world_size
    assert state["nominal_a_fraction"] == config["arguments"]["nominal_fraction"]
    sampling = []
    for audit in config["dataset"]["runtime_audits_by_rank"]:
        nominal_count = audit["nominal_training_worlds"] + audit["nominal_validation_worlds"]
        assert nominal_count == round(audit["num_envs"] * state["nominal_a_fraction"])
        assert audit["physics"]["nominal_mixture"]["nominal_payload_max_abs_kg"] == 0
        assert audit["physics"]["nominal_mixture"]["nominal_pulses_disabled"]
        detail = {"rank": audit["rank"], "nominal_worlds": nominal_count,
                  "dr_worlds": audit["num_envs"] - nominal_count}
        if config["arguments"]["dr_nominal_probability"]:
            mixture = audit["physics"]["independent_nominal_mixture"]
            assert mixture["probability"] == config["arguments"]["dr_nominal_probability"]
            fractions = [v for values in mixture["dr_only_nominal_mask_fractions"].values() for v in values]
            detail.update(dr_nominal_probability=mixture["probability"],
                          dr_coordinate_nominal_fraction_min=min(fractions),
                          dr_coordinate_nominal_fraction_max=max(fractions))
            if mode == "train":
                assert all(abs(v - mixture["probability"]) < .04 for v in fractions)
        sampling.append(detail)
    first = history[0]["optimization_train"]
    expected = (first["prediction_loss"] + .01*first["representation_positive_loss"]
        + .04*first["dr_nominal_relation_loss"] + .008*first["weak_positive_loss"]
        + .01*first["nominal_anchor_loss"])
    assert math.isclose(first["loss"], expected, rel_tol=2e-5, abs_tol=2e-6)
    assert all(math.isfinite(v) for v in first.values())
    assert first["latent_relation_pairs"] > 0 and first["nominal_anchor_samples"] > 0
    assert not any("weak_negative" in name for name in first)
    assert all(int(value["step"]) == state["optimizer_steps"]
               for value in state["optimizer"]["state"].values())
    details = {}
    if mode == "smoke":
        completion = json.loads((directory / "completion.json").read_text())
        assert completion["completed_updates"] == 2 and state["update"] == 2
    else:
        assert config["arguments"]["num_envs"] == 8192
        assert config["arguments"]["validation_worlds"] == 128
        assert config["arguments"]["batch_size"] * world_size == 4096
        assert config["distributed"]["effective_batch_size_global"] == 4096
        assert config["arguments"]["until_user_stop"] and config["arguments"]["stop_after_updates"] is None
        assert state["optimizer_steps"] == state["update"] * 4
        assert state["scheduler"]["T_max"] == 32000
        assert state["scheduler"]["last_epoch"] == state["optimizer_steps"]
        if config["arguments"]["resume"]:
            source = torch.load(config["arguments"]["resume"], map_location="cpu", weights_only=False, mmap=True)
            assert source["update"] == 15000 and history[0]["update"] == 15001
            assert state["model_config"] == source["model_config"]
            assert state["normalization"] == source["normalization"]
            assert state["scheduler"]["T_max"] == source["scheduler"]["T_max"]
            assert state["optimizer"]["param_groups"][0]["lr"] == 1e-5
            for component, is_encoder in (("encoder", True), ("predictor", False)):
                names = [n for n in state["model"] if n.startswith("context_encoder.") == is_encoder]
                assert any(not torch.equal(state["model"][n], source["model"][n]) for n in names)
                details[component + "_updated"] = True
            details["initialization"] = "resume"
        else:
            assert history[0]["update"] == 1 and history[0]["optimizer_steps"] == 4
            assert anchor["source_update"] == 0
            assert not config["arguments"]["resume_new_stage"]
            assert config["arguments"]["comparison_reference_dir"] is None
            assert config["arguments"]["model_learning_rate"] == 3e-4
            assert state["scheduler"]["base_lrs"] == [3e-4]
            if state["optimizer_steps"] <= 32000:
                expected_lr = max(1e-5, 3e-4 * (1 + math.cos(math.pi * state["optimizer_steps"] / 32000)) / 2)
                assert math.isclose(state["optimizer"]["param_groups"][0]["lr"], expected_lr, rel_tol=1e-8)
            details.update(initialization="random", starts_at_update_one=True,
                           optimizer_and_schedule_fresh=True, anchor_calibrated_at_update_zero=True)
    result = {
        "passed": True, "mode": mode, "checkpoint_update": state["update"],
        "optimizer_steps": state["optimizer_steps"], "loss_weights": weights,
        "nominal_anchor_sha256": anchor["sha256"],
        "nominal_calibration_samples": anchor["nominal_samples_total"],
        "first_update_loss": first["loss"], "reconstructed_loss": expected,
        "first_update_dr_samples_per_microbatch": first["latent_relation_pairs"],
        "first_update_nominal_samples_per_microbatch": first["nominal_anchor_samples"],
        "first_update_cross_motion_pairs_per_microbatch": first["weak_positive_pairs"],
        "world_size": world_size, "sampling_by_rank": sampling,
        "all_model_and_anchor_hashes_agree": True, **details}
    (root / f"{mode}_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_root, args.mode)))
