"""Fixed-checkpoint, same-DR comparison; not nominal-goal acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from compare_adaptation_evals import compare

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_legacy_reference(reference_path, candidate_path):
    """Evaluation-only configuration audit, never retroactively sign a model."""
    from types import SimpleNamespace

    import torch
    from omegaconf import OmegaConf

    from intact_tracking.adaptation_reward_contract import assert_original_reward_arguments

    reference, candidate = [torch.load(path, map_location="cpu", weights_only=False)
                            for path in (reference_path, candidate_path)]
    ref, cand = [item["residual_policy"] for item in (reference, candidate)]
    if ref.get("reward_contract"):
        return None
    assert_original_reward_arguments(SimpleNamespace(**ref["arguments"]))
    if ref.get("reward_changes") != {} or ref.get("initialization_sha256") is not None:
        raise ValueError("Legacy reference has reward changes or unaudited initialization")
    if ref["arguments"].get("privileged") or ref.get("imitation_teacher_sha256") is not None:
        raise ValueError("Legacy exception is only for a plain frozen-tracker residual baseline")
    tracker_sha = "fd7bd90d5552e573bbbce1417e9b415c64bb487a76b683ba3c20503b5ec77635"
    if ref["tracker_sha256"] != tracker_sha or cand["tracker_sha256"] != tracker_sha:
        raise ValueError("Unexpected common tracker")
    if not cand.get("reward_contract") or cand.get("reward_changes") != {}:
        raise ValueError("Comparison anchor lacks original-reward signature")
    configurations = [OmegaConf.to_container(item["cfg"].task, resolve=True)
                      for item in (reference, candidate)]
    for key in ("reward", "decimation", "sim"):
        if configurations[0][key] != configurations[1][key]:
            raise ValueError(f"Legacy reward/time-step config differs: {key}")
    source = "src/intact_tracking/adaptation_rewards.py"
    source_sha = digest(ROOT / source)
    if any(meta["research_source_sha256"][source] != source_sha for meta in (ref, cand)):
        raise ValueError("Recorded reward-source versions differ")
    return {
        "reference_checkpoint_sha256": digest(reference_path),
        "status": "Legacy reference configuration audited against signed candidate; not retroactively signed",
        "equal_saved_reward_decimation_sim_config": True,
        "original_reward_arguments_checked": True,
        "recorded_reward_changes": {}, "no_adaptation_initialization_or_teacher": True,
        "reward_source_sha256": source_sha,
        "eligible_for_training_initialization": False,
        "limitation": "Historical run predates runtime reward-signature capture. Configuration and source records support the comparison, but are not a retroactive runtime signature.",
    }


def validate_pair(reference, candidate, identities, legacy_reference_audit=None):
    for key in (
        "protocol", "seed", "motion_ids", "start_frames", "max_steps",
        "metric_names", "motion_files", "physics_world_fingerprints",
    ):
        if reference[key] != candidate[key]:
            raise ValueError(f"Unmatched same-DR evaluation: {key}")
    if not reference["physics_world_fingerprints"]:
        raise ValueError("Missing physical world identity")
    for role, record in (("reference", reference), ("candidate", candidate)):
        if record["checkpoint_sha256"] != identities[role]["sha256"]:
            raise ValueError("Checkpoint identity changed")
        if record["physics"]["physics"] != "dr":
            raise ValueError("Both test environments must use DR")
        if record["protocol"] != "balanced_fixed_starts_v2_isolated_resets" or not all(
            record.get(key) is True for key in (
                "reference_timeline_audited", "partial_reset_survivor_state_audited",
                "partial_reset_survivor_history_audited",
            )
        ):
            raise ValueError("Missing isolated-reset/timeline audit")
        if not record.get("training_reward_audit", {}).get("eligible_fixed_reward_candidate"):
            if not (role == "reference" and legacy_reference_audit and
                    legacy_reference_audit["reference_checkpoint_sha256"] == identities[role]["sha256"]):
                raise ValueError("Original-reward provenance missing")


def audit_actor_input_pair(reference_path, candidate_path):
    import torch

    initial = [torch.load(path.parent / "checkpoint_initial.pt", map_location="cpu", weights_only=False)
               for path in (reference_path, candidate_path)]
    counts = {}
    for key in ("actor_state_dict", "critic_state_dict"):
        a, b = [record[key] for record in initial]
        if a.keys() != b.keys() or any(not torch.equal(a[name], b[name]) for name in a):
            raise ValueError(f"Actor-input experiment did not share initial {key}")
        counts[key] = len(a)
    a, b = [dict(record["residual_policy"]["arguments"]) for record in initial]
    for arguments in (a, b):
        for key, default in {"actor_physics_input": "normal", "latent_low_rank": False,
                             "physics_critic": False, "frozen_nominal_prior": None, "shared_low_rank": 0,
                             "frozen_nominal_prior_deployable_base": False,
                             "add_teacher_refinement": False, "refinement_scale": 1.0}.items():
            arguments.setdefault(key, default)
    differences = {key: [a.get(key), b.get(key)] for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
    if set(differences) != {"actor_physics_input", "output_dir"}:
        raise ValueError(f"Actor-input-only experiment changes other training settings: {differences}")
    if a["actor_physics_input"] not in ("zero", "constant") or b["actor_physics_input"] != "normal":
        raise ValueError("Reference must remove actor physics; candidate must use real physics")
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False)
                   for path in (reference_path, candidate_path)]
    if checkpoints[0]["iter"] != checkpoints[1]["iter"]:
        raise ValueError("Actor-input-only comparison has unequal update counts")
    return {"initial_tensors_bitwise_equal": counts, "arguments_differ": differences,
            "checkpoint_iteration": checkpoints[0]["iter"],
            "scope": "Only actor physical input differs; fixed architecture, critic, initialization, reward, optimizer and training budget"}


def audit_context_replacement(reference_path, candidate_path):
    import torch

    teacher, student = [torch.load(path, map_location="cpu", weights_only=False)
                        for path in (reference_path, candidate_path)]
    teacher_args, student_args = [item["cfg"].agent.actor for item in (teacher, student)]
    if not teacher_args["class_name"].endswith(":PrivilegedAdaptationActor") or not student_args["class_name"].endswith(":ContextAdaptationActor"):
        raise ValueError("Context replacement requires physical teacher and deployable student")
    if teacher_args.get("privilege_schema") != "compact_physics" or teacher_args.get("actor_physics_input", "normal") != "normal":
        raise ValueError("This replacement audit requires true seven-coordinate teacher inputs")
    if teacher_args.get("oracle_tracking_features") or teacher_args.get("oracle_clean_proprio"):
        raise ValueError("This audit isolates static physical input replacement, not oracle frontends")
    if student_args.get("context_feature_adapter") or student_args.get("context_latent_mean"):
        raise ValueError("This replacement audit expects the direct instantaneous-context architecture")
    if student["residual_policy"]["teacher_sha256"] != digest(reference_path):
        raise ValueError("Student was distilled from a different teacher")
    a, b = teacher["actor_state_dict"], student["actor_state_dict"]
    controller = [key for key in b if not key.startswith("context_encoder.")]
    if not controller or any(key not in a or not torch.equal(a[key], b[key]) for key in controller):
        raise ValueError("Context replacement changed controller/preprocessing tensors")
    return {"controller_and_preprocessing_tensors_bitwise_equal": len(controller),
            "teacher_checkpoint_sha256": digest(reference_path),
            "scope": "Only direct true7 physics code is replaced by causal deployable-context estimates; controller and preprocessing are bitwise preserved. Teacher itself remains unqualified against fixed nominal target."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(93001, 93002, 93003))
    parser.add_argument("--gpus", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--actor-input-only", action="store_true")
    parser.add_argument("--context-replacement", action="store_true")
    parser.add_argument("--wait-for-checkpoints", action="store_true")
    args = parser.parse_args()
    if any(gpu not in range(4) for gpu in args.gpus):
        parser.error("Only physical GPUs 0–3 are authorized")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Repeated evaluation seed")
    if args.actor_input_only and args.context_replacement:
        parser.error("Choose an input-only training comparison or a context-replacement comparison")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.wait_for_checkpoints:
        predeclared = {"reference": str(args.reference.resolve()), "candidate": str(args.candidate.resolve()),
                       "seeds": list(args.seeds), "actor_input_only": args.actor_input_only}
        path = output / "predeclared_paths.json"
        if path.exists() and json.loads(path.read_text()) != predeclared:
            raise ValueError("Predeclared checkpoint paths changed")
        if not path.exists():
            path.write_text(json.dumps(predeclared, indent=2) + "\n")
        print(json.dumps({"event": "waiting_for_fixed_checkpoints", **predeclared}), flush=True)
        while not all(path.exists() and time.time() - path.stat().st_mtime > 15
                      for path in (args.reference, args.candidate)):
            time.sleep(15)
    identities = {
        role: {"path": str(path.resolve()), "sha256": digest(path)}
        for role, path in (("reference", args.reference), ("candidate", args.candidate))
    }
    plan = {
        "identities": identities, "evaluation_seeds": list(args.seeds),
        "scope": "One training seed, fixed checkpoints selected on development seed10001. Fresh evaluation worlds/starts only; NOT independent training replications. Architecture and physical input differ together. Both train/test DR, original reward. Not final nominal-goal confirmation.",
    }
    if args.actor_input_only:
        plan["scope"] = "One training seed, predeclared equal-update checkpoints. Same architecture/initialization/critic/reward/optimization; only actor physics input differs. New evaluation worlds are not independent training replications. Not nominal-goal acceptance."
    if args.context_replacement:
        plan["scope"] = "Frozen true7-physics teacher and development-selected student, identical controller/preprocessing. New worlds/starts verify causal context estimates replacing simulator parameters. Teacher remains unqualified versus fixed nominal target; this is intermediate replacement evidence, not full stage-two acceptance."
    manifest = output / "frozen_plan.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != plan:
            raise ValueError("Refusing to replace the frozen comparison plan")
    else:
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
    legacy_reference_audit = audit_legacy_reference(args.reference, args.candidate)
    actor_input_audit = audit_actor_input_pair(args.reference, args.candidate) if args.actor_input_only else None
    context_replacement_audit = audit_context_replacement(args.reference, args.candidate) if args.context_replacement else None

    def evaluate(role, gpu):
        identity = identities[role]
        for seed in args.seeds:
            destination = output / f"{role}_seed{seed}.json"
            if destination.exists():
                continue
            if digest(Path(identity["path"])) != identity["sha256"]:
                raise ValueError("Frozen checkpoint changed before evaluation")
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            environment.pop("PYTHONPATH", None)
            command = [
                sys.executable, "-m", "intact_tracking.cli.adaptation_eval",
                "--physics", "dr", "--checkpoint", identity["path"],
                "--seed", str(seed), "--output", str(destination),
            ]
            print(json.dumps({"event": "started", "role": role, "seed": seed}), flush=True)
            with destination.with_suffix(".log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(json.dumps({"event": "finished", "role": role, "seed": seed}), flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate, role, gpu) for role, gpu in zip(identities, args.gpus, strict=True)]
        for future in futures:
            future.result()
    references, candidates, failures = [], [], []
    for seed in args.seeds:
        paths = [output / f"{role}_seed{seed}.json" for role in identities]
        reference, candidate = [json.loads(path.read_text()) for path in paths]
        validate_pair(reference, candidate, identities, legacy_reference_audit)
        if args.context_replacement and candidate.get("actor_input_contract") != "deployable groups only":
            raise ValueError("Context replacement evaluator did not remove privileged inputs")
        references.append(paths[0])
        candidates.append(paths[1])
        rf, cf = reference["failed"], candidate["failed"]
        failures.append({
            "seed": seed, "reference": sum(rf), "candidate": sum(cf), "episodes": len(rf),
            "new": sum(c and not r for r, c in zip(rf, cf, strict=True)),
            "recovered": sum(r and not c for r, c in zip(rf, cf, strict=True)),
        })
    statistics = compare(references, candidates, margin=1.02 if args.context_replacement else 1.0)
    result = {"plan": plan, "same_physics_worlds_audited": True,
              "legacy_reference_configuration_audit": legacy_reference_audit,
              "actor_input_only_training_audit": actor_input_audit,
              "context_replacement_audit": context_replacement_audit,
              "statistics": statistics, "paired_failures": failures,
              "nominal_goal_acceptance": False,
              "warning": "Bootstrap reflects evaluation seed/motion uncertainty only, not training-seed uncertainty. Actor-input causal scope requires the explicit matching audit; otherwise architecture and input differ together."}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
