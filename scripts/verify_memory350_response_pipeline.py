"""Audit real four-rank smoke runs and arm startup before enabling automatic downstream PPO."""

import argparse
import ast
import json
from pathlib import Path
import shutil
import time

from run_limb_context_experiment import ROOT
from resume_memory350_weak_pairs import sha256
from run_memory350_scale_nominal_stage1 import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    startup = read_json(root / "startup_verification.json")
    uploads = read_json(root / "wandb_upload_verification.json")
    initial = read_json(root / "smoke_initialization_verification.json")
    assert startup["passed"] and uploads["passed"] and initial["passed"]
    assert initial["initial_weights_bitwise_equal"] and initial["world_size"] == 4
    evidence = {}
    from intact_tracking.memory350_policy_checkpoint_eval import load_protocol, validate_result
    from run_memory350_compressed_ppo import ASSIGNMENTS, command_for
    from intact_tracking.cli.memory350_compressed_policy_train import build_parser
    protocol_path = root / "protocols/periodic_smoke.json"
    protocol, files = load_protocol(protocol_path)
    for fusion in ASSIGNMENTS:
        folder = root / "smoke" / f"{fusion}_121"
        completion = read_json(folder / "completion.json")
        unbounded = read_json(root / "smoke" / f"{fusion}_unbounded_completion.json")
        config = read_json(folder / "run_config.json")
        assert completion["complete"] and completion["distributed_parameter_agreement"]["passed"]
        assert completion["distributed"]["world_size"] == 4
        assert unbounded["target_updates"] is None and unbounded["completed_updates"] >= 5 and unbounded["stopped"]
        assert unbounded["distributed_parameter_agreement"]["passed"]
        assert config["resume_state_audit"]["passed"]
        assert config["encoder_frozen"] and not config["predictor_executed_in_ppo"]
        assert config["input_audit"]["actor_compression"] == [1645, 512, 256, 128]
        history = [json.loads(line) for line in (folder / "endpoint_eval_metrics.jsonl").read_text().splitlines()]
        assert len(history) == 1
        endpoint = history[0]
        assert endpoint["training_state_preserved"] and endpoint["protocol_sha256"] == sha256(protocol_path)
        update = endpoint["completed_updates"]
        assert endpoint["checkpoint_sha256"] == sha256(folder / f"checkpoint_update_{update:06d}.pt")
        for case in protocol["cases"]:
            path = folder / "endpoint_eval" / f"update_{update:06d}" / f"{case}.json"
            validate_result(read_json(path), protocol, files, endpoint["checkpoint_sha256"], update, case)
            assert path.with_suffix(".traces.npz").exists()
        command = command_for(root, fusion, "/selected/u15000.pt")
        index = command.index("intact_tracking.cli.memory350_compressed_policy_train")
        formal = build_parser().parse_args(command[index + 1:])
        assert formal.until_user_stop and formal.training_ranks == 4 and formal.num_envs == 8192
        assert formal.motion_sampling == "adaptive" and formal.adaptive_after_update == 1000
        evidence[fusion] = {"unbounded_completion": unbounded, "endpoint_completion": completion,
                            "initial_checkpoint_sha256": sha256(folder / "checkpoint_initial.pt"),
                            "endpoint_checkpoint_sha256": endpoint["checkpoint_sha256"],
                            "training_state_preserved": True, "formal_command": command}
    # Pin the implementation and all local policy dependencies for the delayed launch.
    sources = sorted((ROOT / "src/intact_tracking").rglob("*.py")) + [
        ROOT / "scripts/run_memory350_compressed_ppo.py", Path(__file__).resolve(),
        ROOT / "scripts/run_limb_context_experiment.py", ROOT / "scripts/resume_memory350_weak_pairs.py",
        ROOT / "scripts/run_memory350_scale_nominal_stage1.py"]
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    for path in sources:
        target = root / "ppo_source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    result = {"passed": True, "source_sha256": hashes, "smoke": evidence,
              "formal_num_envs_per_rank": 8192, "smoke_num_envs_per_rank": 128,
              "formal_maximum_updates": None,
              "formal_launch_requires_both_u15000_endpoints_and_selection": True,
              "actor_compression": [1645, 512, 256, 128], "latent_dim": 64,
              "verification_scope": "four-rank update/resume/stop and cold/warm evaluation; full-scale formal run starts after selection",
              "verified_at": time.time()}
    write_json(root / "PPO_READY.json", result)
    print(json.dumps({"passed": True, "automatic_ppo_handoff_enabled": True,
                      "source_files": len(hashes), "readiness": str(root / "PPO_READY.json")}), flush=True)


if __name__ == "__main__":
    main()
