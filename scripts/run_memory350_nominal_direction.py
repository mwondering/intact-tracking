"""Prepare or launch Memory350 nominal-direction training with explicit sampling."""

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
import subprocess
import time

from run_limb_context_experiment import ROOT, PYTHON, process_environment

ENTRY = "intact_tracking.cli.forward_memory_nominal_direction_train"
SOURCE = ROOT / "runs/limb_context_20260912_memory350_response_window_ablation/response10/stage1_8192"
DEFAULT_ROOT = ROOT / "runs/limb_context_20260917_memory350_nominal_direction_ab004_anchor001"
SCRATCH_ROOT = ROOT / "runs/limb_context_20260917_memory350_nominal_direction_scratch_ab004_anchor001"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def arguments_to_command(parser, values, *, num_gpus=8):
    result = [PYTHON, "-B", "-m", "torch.distributed.run", "--standalone",
              f"--nproc-per-node={num_gpus}", "-m", ENTRY]
    for action in parser._actions:
        if action.dest == "help":
            continue
        value = values.get(action.dest)
        if value is None:
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            result.append(flag if value else next(x for x in action.option_strings if x.startswith("--no-")))
        elif action.nargs == 0:
            if value:
                result.append(flag)
        elif isinstance(action, argparse._AppendAction):
            for item in value:
                result.extend((flag, str(item)))
        elif isinstance(value, (list, tuple)):
            result.extend((flag, *map(str, value)))
        else:
            result.extend((flag, str(value)))
    return result


def prepare(run_root, mode, anchor_weight, *, from_scratch=False, num_gpus=8,
            nominal_fraction=.5, dr_nominal_probability=0., limb_max_masses_kg=None):
    from intact_tracking.cli.forward_memory_nominal_direction_train import build_parser
    from intact_tracking.cli.forward_memory_scale_nominal_train import _validate_arguments
    source = json.loads((SOURCE / "run_config.json").read_text())
    parser = build_parser()
    output = run_root / (f"smoke_{num_gpus}gpu" if mode == "smoke" else "stage1_8192")
    values = vars(parser.parse_args([
        "--checkpoint-file", source["arguments"]["checkpoint_file"],
        "--motion-path", source["arguments"]["motion_path"], "--output-dir", str(output)]))
    for name, value in source["arguments"].items():
        if name in values:
            values[name] = value
    values.update(
        output_dir=str(output), resume=str(SOURCE / "update_015000.pt"), resume_new_stage=True,
        comparison_reference_dir=None, stop_after_updates=None, updates=8000, until_user_stop=True,
        batch_size=4096 // num_gpus, micro_batch_size=256, num_envs=8192, validation_worlds=128,
        nominal_fraction=nominal_fraction, dr_nominal_probability=dr_nominal_probability,
        limb_max_masses_kg=limb_max_masses_kg,
        representation_weight=.01, representation_relation_weight=4., response_distance_scale=.3,
        weak_positive_weight=.008, nominal_anchor_weight=anchor_weight,
        wandb_group=run_root.name, wandb_name=f"memory350-response10-nominal{100*nominal_fraction:g}-direction-{num_gpus}x8192",
        wandb=True, wandb_mode="online", anchor_calibration_batches=8,
        anchor_calibration_batch_size=256, bounded_smoke=False)
    if from_scratch:
        values.update(resume=None, resume_new_stage=False, model_learning_rate=3e-4,
                      wandb_name=f"memory350-response10-nominal{100*nominal_fraction:g}-direction-scratch-{num_gpus}x8192")
    if mode == "smoke":
        motions = run_root / "smoke_motion_files"
        motions.mkdir(exist_ok=True)
        dataset = Path(source["arguments"]["motion_path"])
        selected = sorted(itertools.islice(dataset.rglob("*.motion.npz"), 32))
        if len(selected) < 8:
            raise ValueError("Need at least eight real motions for an eight-rank smoke")
        for index, path in enumerate(selected):
            target = motions / f"{index:03d}_{path.name}"
            if not target.exists():
                shutil.copy2(path, target)
        values.update(resume=None, resume_new_stage=False, motion_path=str(motions),
            motion_file=None, num_envs=128, validation_worlds=16,
            batch_size=16, micro_batch_size=16, fixed_probe_batch_size=16,
            replay_capacity=4096, updates=2, stop_after_updates=2, until_user_stop=False,
            gradient_steps_per_update=1, warmup_steps=1000, max_warmup_steps=2000,
            checkpoint_interval=1, validation_interval=1, log_interval=1,
            warmup_log_interval=100, bounded_smoke=True, wandb=False,
            anchor_calibration_batches=4, anchor_calibration_batch_size=64)
    command = arguments_to_command(parser, values, num_gpus=num_gpus)
    actual = parser.parse_args(command[command.index(ENTRY) + 1:])
    _validate_arguments(actual, require_comparison_reference=False, allow_multimotion_smoke=True,
                        allow_nominal_fraction=True)
    if vars(actual) != values:
        raise ValueError("Serialized training arguments changed their meaning")
    if mode == "train":
        assert actual.num_envs == 8192 and actual.batch_size * num_gpus == 4096
        assert actual.representation_weight * actual.representation_relation_weight == .04
    return command, values, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--from-scratch", action="store_true",
                        help="Initialize predictor, encoder, optimizer, schedule and normalization afresh")
    parser.add_argument("--nominal-anchor-weight", type=float, default=.01)
    parser.add_argument("--num-gpus", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--nominal-fraction", type=float, default=.5)
    parser.add_argument("--dr-nominal-probability", type=float, default=0.)
    parser.add_argument("--limb-max-masses-kg", type=float, nargs=4)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    root = (args.run_root or (SCRATCH_ROOT if args.from_scratch else DEFAULT_ROOT)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command, values, output = prepare(root, args.mode, args.nominal_anchor_weight,
            from_scratch=args.from_scratch, num_gpus=args.num_gpus,
            nominal_fraction=args.nominal_fraction, dr_nominal_probability=args.dr_nominal_probability,
            limb_max_masses_kg=args.limb_max_masses_kg)
        process_file = root / f"{args.mode}_process.json"
        if process_file.exists():
            old = json.loads(process_file.read_text())
            if Path(f"/proc/{old['pid']}").exists():
                raise RuntimeError("The recorded process still exists; inspect it before relaunching")
            raise RuntimeError("A previous launch is recorded; preserve it and choose a new run root")
        contract = {
            "mode": args.mode, "formal_command": command, "arguments": values,
            "physical_gpus": list(range(args.num_gpus)), "num_envs_per_rank": values["num_envs"],
            "effective_global_batch": args.num_gpus * values["batch_size"],
            "source_checkpoint": values["resume"], "created_at": time.time(),
            "initialization": "random" if values["resume"] is None else "resume",
            "configuration_template": str(SOURCE / "run_config.json"),
            "total_loss_weights": {"teacher": 1., "recursive": .5, "local_positive": .01,
                "dr_nominal_response": .04, "cross_motion_positive": .008,
                "nominal_anchor": args.nominal_anchor_weight, "weak_negative": 0.},
        }
        write_json(root / f"{args.mode}_launch_contract.json", contract)
        if not args.launch:
            print(json.dumps({"prepared": True, "mode": args.mode, "output": str(output)}))
            return
        apps = subprocess.check_output([
            "nvidia-smi", "--id=" + ",".join(map(str, range(args.num_gpus))),
            "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if apps:
            raise RuntimeError("GPUs have active compute processes; no existing jobs will be interrupted")
        env = process_environment()
        env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, range(args.num_gpus))), PYTHONUNBUFFERED="1")
        if values["wandb"]:
            key = ROOT / ".runtime/limb_context/wandb_api_key"
            if key.exists():
                env["WANDB_API_KEY"] = key.read_text().strip()
            env["WANDB_RUN_ID"] = "m350nomdir-" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
            env["WANDB_RESUME"] = "allow"
        log_path = root / f"{args.mode}.log"
        with log_path.open("x") as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        record = {"pid": child.pid, "command": command, "log": str(log_path),
                  "physical_gpus": list(range(args.num_gpus)), "started_at": time.time(),
                  "process_start_ticks": int(Path(f"/proc/{child.pid}/stat").read_text().split()[21])}
        write_json(process_file, record)
        print(json.dumps({"started": True, "mode": args.mode, "pid": child.pid, "log": str(log_path)}))


if __name__ == "__main__":
    main()
