"""Check actual paired initialization tensors and training configuration."""

import argparse
import json
import time
from pathlib import Path

import torch

from audit_limb_context_training import compare_initial_normalization
from run_limb_context_experiment import experiment_layout, ppo_directory, ROOT
from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking.limb_context_terminations import checkpoint_contract


def verify(root, seed=121, smoke=False):
    layout = experiment_layout(root)
    count = layout["gpus_per_policy"]
    directory = root / f"smoke_{count}gpu8192_scratch" if smoke else ppo_directory(root)
    checkpoints = {fusion: torch.load(directory / (fusion if smoke else f"{fusion}_{seed}") /
        "checkpoint_initial.pt", map_location="cpu", weights_only=False) for fusion in ("baseline", "film")}
    baseline, film = (checkpoints[fusion]["residual_policy"] for fusion in ("baseline", "film"))
    for key in ("dataset", "physics", "reward_contract", "training_starts", "distributed", "motion_sampling", "training_terminations"):
        if baseline[key] != film[key]:
            raise ValueError(f"Paired startup differs in {key}")
    trunks, normalizers, records = {}, {}, {}
    for fusion, checkpoint in checkpoints.items():
        meta = checkpoint["residual_policy"]
        if checkpoint["completed_updates"] != 0 or meta.get("resume_history"):
            raise ValueError("New experiment did not start from scratch")
        if meta["distributed"]["world_size"] != count or meta["distributed"]["global_num_envs"] != count * 8192:
            raise ValueError("Unexpected training scale")
        if checkpoint_contract(checkpoint)["profile"] != "no_ee_body_pos":
            raise ValueError("EE termination remains enabled")
        audits = meta["training_termination_runtime_audits"]
        if len(audits) != count or any(not row["passed"] or row["ee_body_pos_enabled"] for row in audits):
            raise ValueError("A training rank retained EE termination")
        if not smoke and meta["dataset"]["loaded_motion_count"] != 129827:
            raise ValueError("Incomplete full training catalog")
        trunks[fusion] = {}
        for group, prefix in (("actor_state_dict", "residual_mlp."), ("critic_state_dict", "mlp.")):
            if fusion == "film":
                prefix += "base."
            trunks[fusion][group] = tensor_digest((key[len(prefix):], value)
                for key, value in checkpoint[group].items() if key.startswith(prefix))
        normalizers[fusion] = {key.removeprefix("obs_normalizer."): value
            for key, value in checkpoint["critic_state_dict"].items() if key.startswith("obs_normalizer.")}
        if float(normalizers[fusion]["count"]) != count * 8192:
            raise ValueError("Critic inherited prior normalization statistics")
        std = checkpoint["actor_state_dict"]["distribution.std_param"]
        if not torch.equal(std, torch.full_like(std, .25)):
            raise ValueError("Actor did not start with the configured fresh action std")
        records[fusion] = {"loaded_motion_count": meta["dataset"]["loaded_motion_count"],
            "motion_counts_per_rank": meta["dataset"]["loaded_motion_counts_per_rank"],
            "world_size": count, "global_num_envs": count * 8192,
            "active_terminations": audits[0]["active_terms"], "context_sha256": meta["context_sha256"],
            "initial_checkpoint_update": checkpoint["completed_updates"]}
        if smoke:
            policy_dir = directory / fusion
            completion = json.loads((policy_dir / "completion.json").read_text())
            if not completion["complete"] or not completion["distributed_parameter_agreement"]["passed"]:
                raise ValueError("Distributed smoke test did not complete")
            final = torch.load(policy_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
            for key, value in final["actor_state_dict"].items():
                if key.startswith("tracker.") and not torch.equal(value, checkpoint["actor_state_dict"][key]):
                    raise ValueError("Frozen tracker changed during smoke training")
            for group in ("actor_state_dict", "critic_state_dict"):
                if any(not torch.isfinite(value).all() for value in final[group].values()):
                    raise ValueError("Nonfinite smoke model tensors")
            records[fusion]["completed_updates"] = completion["completed_updates"]
            records[fusion]["distributed_parameter_agreement"] = completion["distributed_parameter_agreement"]
            del final
    if trunks["baseline"] != trunks["film"]:
        raise ValueError("Baseline and FiLM initialized different common trunks")
    release = json.loads((root / "stage2_release.json").read_text())
    if film["context_sha256"] != release["context_sha256"]:
        raise ValueError("Wrong frozen stage-1 encoder")
    result = {"passed": True, "smoke": smoke, "seed": seed, "policies": records,
              "common_trunks_identical": True,
              "critic_initial_normalization": compare_initial_normalization(normalizers["baseline"], normalizers["film"]),
              "unix_time": time.time()}
    path = directory / ("audit.json" if smoke else f"startup_verification_seed_{seed}.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "smoke": smoke, "records": records, "path": str(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Verification output must remain inside this project")
    verify(root, args.seed, args.smoke)
