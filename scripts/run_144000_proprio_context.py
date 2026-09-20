"""Start the approved scratch proprio122 Memory350 stage of 144000_exp."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

from run_limb_context_experiment import process_environment
from run_memory350_scale_nominal_stage1 import gpu_status
from monitor_memory350_nominal_direction import process_identity

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/144000_exp"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    source = json.loads((RUN / "source_identity.json").read_text())
    verification = json.loads((RUN / "proprio122_checks/verification.json").read_text())
    assert verification["passed"], "The new input checks must pass before launch"
    for name, expected in verification["source_sha256"].items():
        assert sha256(ROOT / name) == expected, f"Verified source changed: {name}"
    assert sha256(Path(source["tracker_checkpoint"])) == source["tracker_sha256"]
    gpus = list(range(8))
    cards = gpu_status(gpus)
    assert all(not card["processes"] and card["free_mib"] > 60000 for card in cards), cards

    output = RUN / "stage1_proprio122_8192"
    record_path = RUN / "proprio122_train_process.json"
    assert not record_path.exists(), "Inspect and archive the previous launch before relaunching"
    assert not output.exists(), "Use a fresh output directory"
    command = [
        str(ROOT / ".venv/bin/python"), "-B", "-u", "-m", "torch.distributed.run",
        "--standalone", "--nproc-per-node=8", "--max-restarts=0", "-m",
        "intact_tracking.cli.forward_memory_proprio_native_train",
        "--checkpoint-file", source["tracker_checkpoint"],
        "--motion-path", source["root"], "--output-dir", str(output),
        "--num-envs", "8192", "--validation-worlds", "128", "--seed", "717",
        "--nominal-fraction", "0.1", "--no-payload", "--dr-nominal-probability", "0",
        "--warmup-steps", "1000", "--max-warmup-steps", "10000",
        "--rollout-steps-per-update", "5", "--gradient-steps-per-update", "4",
        "--batch-size", "1024", "--micro-batch-size", "256",
        "--fixed-probe-batch-size", "512", "--replay-capacity", "262144",
        "--replay-sampling", "motion_balanced", "--amp-dtype", "bfloat16",
        "--model-learning-rate", "0.0003", "--weight-decay", "0.001",
        "--continuation-min-learning-rate", "0.00001", "--context-history-steps", "50",
        "--chunk-depth", "2", "--memory-depth", "4", "--context-depth", "4",
        "--dynamics-latent-dim", "64", "--recursive-weight", "0.5",
        "--representation-weight", "0.01", "--representation-relation-weight", "10",
        "--nominal-anchor-weight", "0.08", "--weak-positive-weight", "0.008",
        "--dr-soft-weight", "0.02", "--dr-soft-h", "0.15", "--dr-soft-temperature", "0.1",
        "--response-distance-scale", "0.6", "--weak-archive-slots", "4",
        "--weak-archive-interval", "200", "--anchor-calibration-batches", "8",
        "--anchor-calibration-batch-size", "256", "--checkpoint-interval", "250",
        "--validation-interval", "100", "--log-interval", "10", "--warmup-log-interval", "100",
        "--until-user-stop", "--updates", "8000", "--wandb",
        "--wandb-project", "intact-forward-predictor", "--wandb-entity", "2486344338-zhejiang-university",
        "--wandb-group", "144000_exp", "--wandb-name", "144000_exp-proprio122-scratch",
        "--wandb-tag", "scratch", "--wandb-tag", "native-dr-flat",
        "--wandb-tag", "memory350", "--wandb-tag", "noisy-proprio122-control-command",
    ]
    env = process_environment()
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    cache = RUN / "runtime"
    cache.mkdir(exist_ok=True)
    env["SP_TRACKING_MULTIMOTION_MANIFEST_DIR"] = str(cache)
    key = ROOT / ".runtime/limb_context/wandb_api_key"
    if key.exists():
        env["WANDB_API_KEY"] = key.read_text().strip()
    env["WANDB_RUN_ID"] = "144000proprio122-" + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    env["WANDB_RESUME"] = "never"
    log = RUN / "proprio122_train.log"
    with log.open("xb") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(child.pid)
    record = {
        **identity, "process_start_ticks": identity["start_ticks"], "command": command,
        "physical_gpus": gpus, "started_at": time.time(), "output": str(output), "log": str(log),
        "initialization": "scratch", "wandb_id": env["WANDB_RUN_ID"], "maximum_updates": None,
        "input_contract": verification["input_contract"], "verified_source_sha256": verification["source_sha256"],
        "tracker_sha256": source["tracker_sha256"],
        "user_authorization": "好,就这么训. 现在长期记忆和短期记忆是怎么组织的",
    }
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"pid": child.pid, "gpus": gpus, "output": str(output), "log": str(log)}))


if __name__ == "__main__":
    main()
