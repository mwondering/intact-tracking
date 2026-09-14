"""Verify finished budgets, matched inputs/loads, frozen tensors and actual learning."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from intact_tracking.limb_context_protocol import VERSION, TRACKER_SHA256
from intact_tracking.limb_context_dr import validate_context_dr, resolve_dr_profile
from run_limb_context_experiment import ppo_directory, PPO_INITIALIZATION, stage2_authorized, experiment_layout, formal_policies
from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking.limb_context_checkpoint_eval import audit_saved_evaluations
from intact_tracking.limb_context_sampling import audit_sampling_curriculum
from intact_tracking.limb_context_terminations import checkpoint_contract, experiment_profile


def compare_initial_normalization(reference, candidate):
    """Allow float32 simulation rounding across runs, with identical sample counts.

    Compare moments rather than sums: the absolute tolerance must not shrink as
    the number of worlds grows. Within a distributed run hashes remain exact.
    """
    if set(reference) != {"sum", "ssq", "count"} or set(candidate) != set(reference):
        raise ValueError("Initial critic normalization buffers differ")
    for key in reference:
        left, right = reference[key], candidate[key]
        if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError(f"Invalid initial critic normalization {key}")
    if not torch.equal(reference["count"], candidate["count"]) or not (reference["count"] > 0).all():
        raise ValueError("Paired initial critic normalization sample counts differ")
    differences = {}
    count = reference["count"].double()
    for key in ("sum", "ssq"):
        left, right = reference[key], candidate[key]
        if not torch.allclose(left.double() / count, right.double() / count, rtol=1e-6, atol=1e-8):
            raise ValueError(f"Paired initial critic normalization {key} differs beyond float32 tolerance")
        delta = (left.double() - right.double()).abs()
        differences[key] = {"max_absolute_difference": float(delta.max()),
                            "max_moment_difference": float((delta / count).max())}
    return {"passed": True, "sample_count_equal": True, "rtol": 1e-6, "atol": 1e-8,
            "comparison_units": "first and second moments (sum/count and ssq/count)",
            "exact_equal": all(torch.equal(reference[k], candidate[k]) for k in reference),
            "differences": differences}


def run(root):
    layout = experiment_layout(root)
    world_size = layout["gpus_per_policy"]
    global_envs = world_size * layout["num_envs_per_gpu"]
    names = formal_policies(root)
    results, metadata, initial_normalizations = {}, {}, {}
    for name in names:
        directory = ppo_directory(root) / name
        initial = torch.load(directory / "checkpoint_initial.pt", map_location="cpu", weights_only=False)
        final = torch.load(directory / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        meta = final["residual_policy"]
        termination_contract = checkpoint_contract(final)
        if (checkpoint_contract(initial) != termination_contract
                or termination_contract["profile"] != experiment_profile(root)):
            raise ValueError(f"Training termination protocol changed in {name}")
        meta = {**meta, "training_terminations": termination_contract}
        # A resumed run rebuilds its simulator. Compare the original initial
        # checkpoint and its own audit, rather than a later startup's moments.
        initial_audit = initial["residual_policy"]["input_audit"]
        completion = json.loads((directory / "completion.json").read_text())
        if (meta.get("initialization_protocol") != PPO_INITIALIZATION
                or completion.get("initialization_protocol") != PPO_INITIALIZATION
                or initial["completed_updates"] != 0):
            raise ValueError(f"Wrong initialization/restart protocol in {name}")
        if final["cfg"].agent.critic.initial_checkpoint is not None:
            raise ValueError(f"Scratch critic loaded a checkpoint in {name}")
        # Check actual saved tensors, not only the initialization description.
        actor_prefix = "residual_mlp." if meta["fusion"] == "baseline" else "residual_mlp.base."
        critic_prefix = "mlp." if meta["fusion"] == "baseline" else "mlp.base."
        for group, prefix, key in (("actor_state_dict", actor_prefix, "actor_common_trunk_sha256"),
                                  ("critic_state_dict", critic_prefix, "critic_common_trunk_sha256")):
            tensors = [(k[len(prefix):], v) for k, v in initial[group].items() if k.startswith(prefix)]
            if tensor_digest(tensors) != initial_audit[key]:
                raise ValueError(f"Initial common trunk audit differs from saved tensors in {name}")
        if float(initial["critic_state_dict"]["obs_normalizer.count"]) != global_envs:
            raise ValueError(f"Scratch critic statistics did not start from one current DR batch in {name}")
        normalization = {key.removeprefix("obs_normalizer."): value
                         for key, value in initial["critic_state_dict"].items()
                         if key.startswith("obs_normalizer.")}
        if tensor_digest(normalization.items()) != initial_audit["critic_initial_normalizer_sha256"]:
            raise ValueError(f"Initial normalization audit differs from saved tensors in {name}")
        initial_normalizations[name] = normalization
        std = initial["actor_state_dict"]["distribution.std_param"]
        if not torch.equal(std, torch.full_like(std, .25)):
            raise ValueError(f"Scratch actor did not start with fresh action std in {name}")
        scale = meta.get("distributed", {})
        agreement = completion.get("distributed_parameter_agreement", {})
        if (scale.get("world_size") != world_size or scale.get("num_envs_per_rank") != 8192
                or scale.get("global_num_envs") != global_envs
                or completion.get("distributed") != scale
                or not agreement.get("passed") or agreement.get("world_size") != world_size
                or len(agreement.get("ranks", [])) != world_size):
            raise ValueError(f"Distributed training/synchronization contract failed in {name}")
        if not completion["complete"] or completion["completed_updates"] != final["completed_updates"]:
            raise ValueError(f"Inconsistent completion checkpoint in {name}")
        if (meta["dataset"]["loaded_motion_count"] != 129827
                or sum(meta["dataset"]["loaded_motion_counts_per_rank"]) != 129827):
            raise ValueError(f"Distributed motion shards did not cover all files in {name}")
        if meta["version"] != VERSION or meta["tracker_sha256"] != TRACKER_SHA256:
            raise ValueError(f"Wrong experiment/source in {name}")
        if final["completed_updates"] < 5000 or meta["motion_count"] != 129827:
            raise ValueError(f"Incomplete training/data coverage in {name}")
        if meta["input_audit"]["actor_original_features"] != 1645 or meta["input_audit"]["critic_original_features"] != 6330:
            raise ValueError("Original observation dimensions changed")
        if meta["reward_changes"] or meta["physics"]["observation_noise"]:
            raise ValueError("Common reward/noise contract changed")
        changes = {}
        for group in ("actor_state_dict", "critic_state_dict"):
            difference = 0.
            for key, value in final[group].items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite {name}/{group}/{key}")
                if group == "actor_state_dict" and key.startswith("tracker."):
                    if not torch.equal(value, initial[group][key]):
                        raise ValueError("Frozen tracker/normalization changed")
                elif key.startswith(("residual_mlp.", "mlp.")):
                    difference += float((value.double() - initial[group][key].double()).square().sum())
            if difference <= 0:
                raise ValueError(f"{name}/{group} did not learn")
            changes[group] = difference ** .5
        metadata[name] = {**meta, "input_audit": initial_audit}
        results[name] = {"completed_updates": final["completed_updates"], "motions": meta["motion_count"],
                         "original_input_dimensions": [1645, 6330], "network_change_L2": changes,
                         "distributed": scale, "distributed_parameter_agreement": agreement,
                         "initialization_protocol": PPO_INITIALIZATION,
                         "training_terminations": termination_contract,
                         "frozen_tracker_unchanged": True, "all_tensors_finite": True}
        periodic = meta["periodic_evaluation"]
        results[name]["periodic_endpoint_evaluation"] = audit_saved_evaluations(
            directory, final["completed_updates"], periodic["protocol_file"], periodic["enabled_after_update"])
        if (root / "sampling_curriculum.json").exists():
            results[name]["sampling_curriculum"] = audit_sampling_curriculum(directory, final)
        del initial, final
    for seed in (121, 122, 123):
        baseline, film = metadata[f"baseline_{seed}"], metadata[f"film_{seed}"]
        for key in ("dataset", "reward_contract", "physics", "training_starts", "distributed", "training_terminations"):
            if baseline[key] != film[key]:
                raise ValueError(f"Paired seed {seed} differs in {key}")
        if baseline.get("motion_sampling") != film.get("motion_sampling"):
            raise ValueError(f"Paired seed {seed} uses different sampling curricula")
        for key in ("actor_common_trunk_sha256", "critic_common_trunk_sha256"):
            if baseline["input_audit"][key] != film["input_audit"][key]:
                raise ValueError(f"Paired seed {seed} did not start with identical {key}")
        results[f"film_{seed}"]["initial_normalization_vs_baseline"] = compare_initial_normalization(
            initial_normalizations[f"baseline_{seed}"], initial_normalizations[f"film_{seed}"])
        for key in ("seed", "num_envs", "rollout_steps", "actor_lr", "critic_lr", "epochs",
                    "mini_batches", "entropy_coef", "residual_scale", "iterations"):
            if baseline["arguments"][key] != film["arguments"][key]:
                raise ValueError(f"Paired training hyperparameter changed: {key}")
    for name in (f"{fusion}_121" for fusion in layout["controls"]):
        for key in ("dataset", "reward_contract", "physics", "training_starts", "training_terminations"):
            if metadata[name][key] != metadata["baseline_121"][key]:
                raise ValueError(f"Control {name} differs in {key}")
        for key in ("actor_common_trunk_sha256", "critic_common_trunk_sha256"):
            if metadata[name]["input_audit"][key] != metadata["baseline_121"]["input_audit"][key]:
                raise ValueError(f"Control {name} did not start with identical {key}")
        results[name]["initial_normalization_vs_baseline"] = compare_initial_normalization(
            initial_normalizations["baseline_121"], initial_normalizations[name])
    with (root / "stage1/best.pt").open("rb") as handle:
        context_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    for name, meta in metadata.items():
        if name.startswith(("film_", "concat_")) and meta["context_sha256"] != context_hash:
            raise ValueError("Policies used different/further modified context checkpoints")
    stage1 = torch.load(root / "stage1/best.pt", map_location="cpu", weights_only=False)
    validate_context_dr(stage1, layout["dr_profile"])
    for meta in metadata.values():
        resolve_dr_profile(layout["dr_profile"], meta)
    stage1_config = json.loads((root / "stage1/run_config.json").read_text())
    stage1_completion = json.loads((root / "stage1/completion.json").read_text())
    release = json.loads((root / "stage2_release.json").read_text())
    if (not stage1_config["arguments"].get("until_user_stop")
            or stage1_completion.get("target_updates") is not None
            or stage1_completion.get("stopping_mode") != "until_user_stop"
            or not stage1_completion.get("stopped") or stage1_completion.get("hit_cap")
            or not stage2_authorized(root)
            or release.get("stage1_completed_updates") != stage1_completion["completed_updates"]):
        raise ValueError("Stage 1 was not ended and released by the user's convergence decision")
    target_updates = None
    args = stage1_config["arguments"]
    train_worlds = args["num_envs"] - args["validation_worlds"]
    expected = tuple(rank * args["num_envs"] + index
                     for rank in range(stage1_config["distributed"]["world_size"])
                     for index in range(train_worlds))
    if tuple(stage1["normalization"]["world_ids"]) != expected:
        raise ValueError("Normalization includes non-training worlds")
    results["stage1"] = {**stage1_completion, "target_updates": target_updates, "selected_update": stage1["update"],
                         "user_confirmed_converged": True, "user_release": release,
                         "resume_history": stage1_config.get("resume_history", []),
                         "normalization_excludes_validation_worlds": True, "context_sha256": context_hash}
    path = root / "independent_training_audit.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    run(parser.parse_args().run_root.resolve())
