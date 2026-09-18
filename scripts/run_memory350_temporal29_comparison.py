"""Launch the authorized original-observation MLP / temporal Transformer comparison."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/limb_context_20260916_temporal29_transformer"
OLD = ROOT / "runs/limb_context_20260916_payload256_top5_moe"
CONTEXT = ROOT / "runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/update_035857.pt"
MOTIONS = "/data_zcy/wxy/motion_data_correct/motion_data_full"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("both", "baseline", "latent"), default="both")
    parser.add_argument("--run-suffix", default="", help="Unique W&B suffix after archiving a failed launch")
    args = parser.parse_args()
    selected = ("baseline", "latent") if args.arm == "both" else (args.arm,)
    validation = json.loads((ROOT / "runs/limb_context_20260916_temporal29_implementation_check/validation_summary.json").read_text())
    stopped = {}
    for arm in selected:
        check = validation["training"][arm]
        if not all(check[key] for key in ("parameter_agreement", "resume_model_optimizer_normalizer_exact", "all_losses_finite")):
            raise RuntimeError(f"{arm} implementation validation failed")
        completion = json.loads((OLD / arm / "completion.json").read_text())
        checkpoint = OLD / arm / "checkpoint_interrupted.pt"
        if not completion["stopped"] or not completion["distributed_parameter_agreement"]["passed"] or not checkpoint.is_file():
            raise RuntimeError(f"{arm} prior training has not saved and stopped")
        stopped[arm] = {"completed_updates": completion["completed_updates"], "checkpoint": str(checkpoint)}
        for target in (RUN / arm, RUN / f"{arm}.log", RUN / f"{arm}_process.json"):
            if target.exists():
                raise FileExistsError(target)
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = (process / "cmdline").read_bytes().decode().split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if "intact_tracking.cli.payload_prototype_train" in argv and any(str(OLD / arm) in argv for arm in stopped):
            raise RuntimeError(f"Old training process {process.name} is still alive")
    if not CONTEXT.is_file() or not Path(MOTIONS).is_dir():
        raise FileNotFoundError("Context checkpoint or motion dataset is missing")

    env_base = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                    PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl", WANDB_RESUME="allow")
    for key, name in {"TMPDIR": "tmp", "MPLCONFIGDIR": "matplotlib", "XDG_CACHE_HOME": "cache",
                      "WARP_CACHE_PATH": "warp", "TORCHINDUCTOR_CACHE_DIR": "inductor",
                      "WANDB_CACHE_DIR": "wandb_cache", "WANDB_DIR": "wandb", "CUDA_CACHE_PATH": "cuda",
                      "WANDB_CONFIG_DIR": "wandb_config", "WANDB_DATA_DIR": "wandb_data",
                      "SP_TRACKING_MULTIMOTION_MANIFEST_DIR": "manifests",
                      "MJLAB_BOOTSTRAP_DEBUG_DIR": "bootstrap"}.items():
        directory = ROOT / ".runtime/limb_context" / name
        directory.mkdir(parents=True, exist_ok=True)
        env_base[key] = str(directory)
    env_base["WANDB_API_KEY"] = (ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip()
    sources = [Path(__file__), ROOT / "src/intact_tracking/payload_prototype_physics.py",
               *sorted((ROOT / "src/intact_tracking").glob("memory350_token_*.py")),
               *sorted((ROOT / "src/intact_tracking/cli").glob("memory350_token_policy_*.py"))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    RUN.mkdir(parents=True, exist_ok=True)
    for arm, architecture, gpus in (("baseline", "mlp", [0, 1, 2, 3]),
                                     ("latent", "transformer", [4, 5, 6, 7])):
        if arm not in selected:
            continue
        output = RUN / arm
        run_id = "temporal29-" + hashlib.sha256(str(output).encode()).hexdigest()[:12]
        if args.run_suffix:
            run_id += "-" + args.run_suffix
        command = [str(ROOT / ".venv/bin/python"), "-B", "-u", "-m", "torch.distributed.run",
                   "--standalone", "--nproc-per-node=4", "--max-restarts=0", "-m",
                   "intact_tracking.cli.memory350_token_policy_train", "--architecture", architecture,
                   "--output-dir", str(output), "--training-ranks", "4", "--num-envs", "8192",
                   "--rollout-steps", "24", "--epochs", "5", "--mini-batches", "4",
                   "--actor-lr", "0.0001", "--critic-lr", "0.0005", "--entropy-coef", "0.0002",
                   "--seed", "121", "--save-interval", "250", "--policy-precision", "fp32",
                   "--motion-path", MOTIONS, "--dr-profile", "load_only", "--anchor-fraction", "0",
                   "--dr-sampling", "independent_uniform", "--motion-sampling", "uniform",
                   "--training-terminations", "original", "--until-user-stop",
                   "--wandb-mode", "online", "--wandb-group", RUN.name,
                   "--wandb-name", f"temporal29-{arm}-4x8192-all-uniform"]
        if arm == "latent":
            command.extend(["--context-checkpoint", str(CONTEXT), "--critic-observation", "privileged"])
        env = dict(env_base, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
                   WANDB_RUN_ID=run_id, WANDB_MODE="online")
        log_path = RUN / f"{arm}.log"
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"pid": process.pid, "command": command, "gpus": gpus, "wandb_run_id": run_id,
                  "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "output": str(output), "log": str(log_path), "unbounded": True,
                  "source_sha256": hashes, "replaced_training": stopped,
                  "sampling": "Every world has independent continuous uniform limb loads; no fixed load anchors; uniform motion",
                  "limb_max_masses_kg": [2.5, 2.5, 4.0, 4.0], "num_envs_per_gpu": 8192}
        (RUN / f"{arm}_process.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({key: record[key] for key in ("pid", "gpus", "output", "wandb_run_id")}), flush=True)


if __name__ == "__main__":
    main()
