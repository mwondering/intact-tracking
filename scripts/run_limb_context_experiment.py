"""Local persistent scheduler; starts jobs only on currently unused GPUs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")
TRACKER = ("/data_zcy/wxy/SP_Tracking/logs/rsl_rl/g1_tracking/"
           "2026-09-02_04-48-42_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_"
           "8gpu_12288env_motion_data_correct/checkpoint_72000.pt")
DATASET = "/data_zcy/wxy/motion_data_correct/motion_data_full"
SMOKE_MOTION = DATASET + "/AMASS_LAFAN_Qingtong/lafan_qingtong/dance1_subject2.motion.npz"
PPO_SUBDIRECTORY = "ppo_2gpu8192_scratch"
PPO_SMOKE_SUBDIRECTORY = "smoke_2gpu8192_scratch"
PPO_INITIALIZATION = "residual_actor_and_critic_from_scratch_v3"
PPO_GPUS = 2
PPO_ENVS_PER_GPU = 8192
ALLOWED_GPUS = (0, 1, 2, 3)
DEFAULT_STAGE1_LOSS = {
    "representation_weight": 0.01,
    "representation_relation_weight": 1.0,
    "response_distance_scale": 1.0,
}


def experiment_layout(run_root):
    path = Path(run_root) / "experiment_layout.json"
    result = {"gpus_per_policy": 2, "num_envs_per_gpu": 8192, "allowed_gpus": list(ALLOWED_GPUS),
              "paired_seeds": [121, 122, 123], "controls": ["concat", "constant"], "reuse_stage1": False,
              "stage1_only": False, "context_history_steps": 100, "encoder_gpu_pool": None,
              "stage1_loss": dict(DEFAULT_STAGE1_LOSS), "dr_profile": "load_only"}
    if path.exists():
        result.update(json.loads(path.read_text()))
    count = result["gpus_per_policy"]
    allowed = result["allowed_gpus"]
    if (count not in (2, 4) or result["num_envs_per_gpu"] != 8192
            or len(allowed) != 2 * count or len(set(allowed)) != len(allowed)
            or not set(allowed).issubset(range(8))
            or result["paired_seeds"] != [121, 122, 123]
            or not set(result["controls"]).issubset({"concat", "constant"})):
        raise ValueError("Invalid matched residual experiment layout")
    if (not isinstance(result["context_history_steps"], int) or result["context_history_steps"] < 10
            or not isinstance(result["stage1_only"], bool)
            or (result["stage1_only"] and result["reuse_stage1"])):
        raise ValueError("Invalid fresh context encoder layout")
    pool = result["encoder_gpu_pool"]
    if pool is not None and (len(pool) != 4 or len(set(pool)) != 4 or not set(pool).issubset(allowed)):
        raise ValueError("The encoder GPU pool must contain four allowed GPUs")
    loss = result["stage1_loss"]
    if not isinstance(loss, dict) or set(loss) - set(DEFAULT_STAGE1_LOSS):
        raise ValueError("Unknown stage1 loss overrides")
    loss = {**DEFAULT_STAGE1_LOSS, **loss}
    for name, value in loss.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (name == "response_distance_scale" and value == 0)):
            raise ValueError(f"Invalid stage1 loss override: {name}")
    result["stage1_loss"] = loss
    if result["dr_profile"] not in ("load_only", "tracker_dr_plus_limb_payload"):
        raise ValueError("Invalid limb context DR profile")
    return result


def formal_policies(run_root):
    layout = experiment_layout(run_root)
    if layout["stage1_only"]:
        return []
    return [f"{fusion}_{seed}" for seed in layout["paired_seeds"] for fusion in ("baseline", "film")] + [
        f"{fusion}_121" for fusion in layout["controls"]]


def select_job_gpus(job, available):
    allowed = job.get("allowed_gpus", ALLOWED_GPUS)
    pool = job.get("gpu_pool", allowed)
    selected = [gpu for gpu in available if gpu in pool and gpu in allowed]
    return selected[:job["gpus_needed"]] if len(selected) >= job["gpus_needed"] else []


def stage2_authorized(run_root):
    """Only an explicit user convergence decision may release stage 2."""
    path = run_root / "stage2_release.json"
    if not path.exists():
        return False
    decision = json.loads(path.read_text())
    if not decision.get("approved") or decision.get("decision_source") != "explicit_user_convergence_decision":
        return False
    checkpoint = run_root / "stage1/best.pt"
    if not checkpoint.exists():
        return False
    with checkpoint.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return decision.get("context_sha256") == digest


def ppo_directory(run_root):
    return run_root / f"ppo_{experiment_layout(run_root)['gpus_per_policy']}gpu8192_scratch"


def job_log_path(run_root, job):
    directory = run_root / "logs"
    if job["phase"] == "ppo":
        directory = directory / ppo_directory(run_root).name
    return directory / f"{job['name']}.log"


def process_environment():
    runtime = ROOT / ".runtime/limb_context"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MUJOCO_GL="egl")
    for key, name in {"TMPDIR": "tmp", "MPLCONFIGDIR": "matplotlib", "XDG_CACHE_HOME": "cache",
                      "WARP_CACHE_PATH": "warp", "TORCHINDUCTOR_CACHE_DIR": "inductor",
                      "WANDB_CACHE_DIR": "wandb_cache", "WANDB_DIR": "wandb", "CUDA_CACHE_PATH": "cuda",
                      "WANDB_CONFIG_DIR": "wandb_config", "WANDB_DATA_DIR": "wandb_data",
                      "SP_TRACKING_MULTIMOTION_MANIFEST_DIR": "manifests", "MJLAB_BOOTSTRAP_DEBUG_DIR": "bootstrap"}.items():
        path = runtime / name
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def training_termination_profile(run_root):
    # Keep the scheduler usable with system Python, without importing torch.
    path = ROOT / "src/intact_tracking/limb_context_terminations.py"
    spec = importlib.util.spec_from_file_location("limb_termination_profiles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.experiment_profile(run_root)


def jobs(run_root):
    layout = experiment_layout(run_root)
    count = layout["gpus_per_policy"]
    fusions = [fusion for fusion in ("baseline", "constant", "film", "concat")
               if fusion in ("baseline", "film") or fusion in layout["controls"]]
    stage1 = run_root / "stage1"
    smoke_root = run_root / "smoke"
    nominal_output = smoke_root / "nominal_audit.json"
    result = [{"name": "nominal_audit", "phase": "audit", "gpus_needed": 1,
               "command": ["-m", "intact_tracking.cli.limb_context_nominal_audit",
                           "--motion-file", SMOKE_MOTION, "--output", str(nominal_output)],
               "output": str(smoke_root), "completion_file": str(nominal_output)}]
    predictor = ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
        "-m", "intact_tracking.cli.forward_predictor_train", "--checkpoint-file", TRACKER,
        "--motion-path", DATASET, "--output-dir", str(stage1), "--limb-payload-only",
        "--no-payload", "--nominal-fraction", "0", "--no-wandb", "--seed", "717",
        "--num-envs", "1152", "--validation-worlds", "128", "--updates", "8000",
        "--until-user-stop", "--continuation-min-learning-rate", "0.00001",
        "--warmup-steps", "500",
        "--batch-size", "1024", "--micro-batch-size", "256", "--fixed-probe-batch-size", "512",
        "--checkpoint-interval", "250", "--validation-interval", "100"]
    if layout["context_history_steps"] != 100:
        predictor += ["--context-history-steps", str(layout["context_history_steps"])]
    if layout["dr_profile"] == "tracker_dr_plus_limb_payload":
        predictor[predictor.index("--limb-payload-only")] = "--tracker-dr-plus-limb-payload"
    for name, default in DEFAULT_STAGE1_LOSS.items():
        if layout["stage1_loss"][name] != default:
            predictor += ["--" + name.replace("_", "-"), str(layout["stage1_loss"][name])]
    # Exercise the actual four-rank model, replay and microbatch shapes first.
    smoke_predictor = list(predictor)
    smoke_predictor.remove("--until-user-stop")
    smoke_predictor[smoke_predictor.index("--motion-path")] = "--motion-file"
    smoke_predictor[smoke_predictor.index("--motion-file") + 1] = SMOKE_MOTION
    smoke_predictor[smoke_predictor.index("--output-dir") + 1] = str(smoke_root / "stage1")
    smoke_predictor[smoke_predictor.index("--updates") + 1] = "2"
    result.append({"name": "smoke_stage1", "phase": "encoder", "gpus_needed": 4,
                   "command": smoke_predictor, "output": str(smoke_root / "stage1"),
                   "dependencies": ["nominal_audit"]})
    for fusion in fusions:
        output = run_root / f"smoke_{count}gpu8192_scratch" / fusion
        command = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={count}",
                   "-m", "intact_tracking.cli.limb_context_train", "--fusion", fusion,
                   "--output-dir", str(output), "--iterations", "3", "--num-envs", str(PPO_ENVS_PER_GPU),
                   "--motion-file", SMOKE_MOTION]
        dependencies = ["stage1"]
        if fusion in ("film", "concat"):
            command += ["--context-checkpoint", str(stage1 / "best.pt")]
        result.append({"name": f"smoke_{fusion}", "phase": "ppo", "gpus_needed": count,
                       "command": command, "output": str(output), "dependencies": dependencies})
    result.append({"name": "stage1", "phase": "encoder", "gpus_needed": 4, "command": predictor,
                   "output": str(stage1), "dependencies": ["smoke_stage1"]})
    pairs = [(fusion, (seed,)) for seed in layout["paired_seeds"] for fusion in ("baseline", "film")]
    for fusion, seeds in [*pairs, *((fusion, (121,)) for fusion in layout["controls"])]:
        for seed in seeds:
            output = ppo_directory(run_root) / f"{fusion}_{seed}"
            command = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={count}",
                       "-m", "intact_tracking.cli.limb_context_train", "--fusion", fusion,
                       "--seed", str(seed), "--output-dir", str(output), "--iterations", "5000",
                       "--motion-path", DATASET, "--num-envs", str(PPO_ENVS_PER_GPU),
                       "--save-interval", "100",
                       "--endpoint-eval-protocol", str(run_root / "periodic_endpoints.json")]
            if fusion in ("film", "concat"):
                command += ["--context-checkpoint", str(stage1 / "best.pt")]
            dependencies = [f"smoke_{f}" for f in fusions]
            dependencies += ["stage1"]
            result.append({"name": f"{fusion}_{seed}", "phase": "ppo", "gpus_needed": count, "command": command,
                           "output": str(output), "dependencies": dependencies})
    for job in result:
        if job["phase"] == "ppo":
            if layout["dr_profile"] != "load_only":
                job["command"] += ["--dr-profile", layout["dr_profile"]]
            fusion = job["command"][job["command"].index("--fusion") + 1]
            job["gpu_pool"] = layout["allowed_gpus"][:count] if fusion in ("baseline", "constant") else layout["allowed_gpus"][count:]
        else:
            job["gpu_pool"] = layout["encoder_gpu_pool"] or layout["allowed_gpus"]
        job["allowed_gpus"] = layout["allowed_gpus"]
    if layout["reuse_stage1"]:
        result = [job for job in result if job["name"] not in ("nominal_audit", "smoke_stage1")]
        for job in result:
            if job["name"] == "stage1":
                job.update(dependencies=[], reused=True)
    curriculum_path = run_root / "sampling_curriculum.json"
    if curriculum_path.exists():
        curriculum = json.loads(curriculum_path.read_text())
        if not curriculum.get("approved") or curriculum["adaptive_after_update"] != 1000:
            raise ValueError("The adaptive continuation requires the approved 1000-update boundary")
        for job in result:
            if job["phase"] == "ppo":
                job["command"] += ["--motion-sampling", "adaptive", "--adaptive-after-update",
                                   "0" if job["name"].startswith("smoke_") else "1000"]
    termination_profile = training_termination_profile(run_root)
    if termination_profile != "original":
        for job in result:
            if job["phase"] == "ppo":
                job["command"] += ["--training-terminations", termination_profile]
    return [job for job in result if job["phase"] != "ppo"] if layout["stage1_only"] else result


def pending_job_hold(run_root, job):
    """Gate future launches without signalling or restarting current workers."""
    path = run_root / "launch_holds.json"
    if not path.exists():
        return None
    rule = json.loads(path.read_text()).get("jobs", {}).get(job["name"])
    if rule is None:
        return None
    completion_path = (ROOT / rule["completion_file"]).resolve()
    if not completion_path.is_relative_to(ROOT):
        raise ValueError("Launch dependencies must remain inside the project")
    expected = rule["required_values"]
    if not expected or not isinstance(expected, dict):
        raise ValueError("Launch dependencies require explicit completion fields")
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    return (None if all(key in completion and completion[key] == value for key, value in expected.items())
            else rule["reason"])


def job_completion(job):
    path = Path(job.get("completion_file", str(Path(job["output"]) / "completion.json")))
    result = json.loads(path.read_text()) if path.exists() else {}
    if job["phase"] == "audit":
        done = result.get("passed", False)
    elif job["phase"] == "encoder":
        if "--until-user-stop" in job["command"]:
            root = Path(job["output"]).parent
            path = root / "stage2_release.json"
            decision = json.loads(path.read_text()) if path.exists() else {}
            done = (result.get("stopping_mode") == "until_user_stop" and result.get("stopped")
                    and stage2_authorized(root)
                    and decision.get("stage1_completed_updates") == result.get("completed_updates"))
        else:
            target = int(job["command"][job["command"].index("--updates") + 1])
            done = (result.get("converged") or (result.get("hit_cap") and
                    result.get("completed_updates", 0) >= target)) and not result.get("stopped")
    else:
        scale = result.get("distributed", {})
        agreement = result.get("distributed_parameter_agreement", {})
        target = int(job["command"][job["command"].index("--iterations") + 1])
        count = job.get("gpus_needed", PPO_GPUS)
        done = (result.get("complete", False) and result.get("completed_updates", 0) >= target
                and scale.get("world_size") == count
                and scale.get("num_envs_per_rank") == PPO_ENVS_PER_GPU
                and scale.get("global_num_envs") == count * PPO_ENVS_PER_GPU
                and agreement.get("passed") and agreement.get("world_size") == count
                and result.get("initialization_protocol") == PPO_INITIALIZATION)
    return bool(done), result


def available_gpus(reserved, allowed_gpus=ALLOWED_GPUS):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True)
    return [int(index) for line in output.splitlines() for index, memory, utilization in [line.split(",")]
            if int(index) in allowed_gpus and int(index) not in reserved and int(memory) < 512 and int(utilization) < 15]


def write_state(path, state):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def ensure_wandb_sync(run_root, env):
    """The durable uploader owns W&B for all jobs, including native-logging-off workers."""
    script = ROOT / "scripts/sync_limb_context_wandb.py"
    directory = run_root / ".wandb_sync"
    directory.mkdir(exist_ok=True)
    destination_file = run_root / "wandb_destination.json"
    destination = json.loads(destination_file.read_text()) if destination_file.exists() else {}
    env = dict(env)
    # A detached uploader must own its service, even when this helper is called
    # from a short-lived process that queried the W&B API first.
    for key in ("_WANDB_SERVICE", "WANDB_SERVICE"):
        env.pop(key, None)
    if "credential_file" in destination:
        credential = (ROOT / destination["credential_file"]).resolve()
        if not credential.is_relative_to(ROOT):
            raise ValueError("W&B credentials must remain within the project")
        env["WANDB_API_KEY"] = credential.read_text().strip()
    process_file = directory / "process.json"
    if process_file.exists():
        previous = json.loads(process_file.read_text())
        process = Path(f"/proc/{previous['pid']}")
        if process.exists() and process.joinpath("stat").read_text().split()[2] != "Z":
            if str(script).encode() in process.joinpath("cmdline").read_bytes():
                return
    command = [PYTHON, "-u", str(script), "--root", str(run_root)]
    for key in ("entity", "project", "stage1_project"):
        if key in destination:
            command += ["--" + key.replace("_", "-"), destination[key]]
    with (run_root / "logs/wandb_sync.log").open("a") as handle:
        process = subprocess.Popen(command,
                                   cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    write_state(process_file, {"pid": process.pid, "started_at": time.time()})


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in ("Z", "X") else fields[19]
    except FileNotFoundError:
        return None


def saved_worker_is_running(previous):
    """A reused PID must never be mistaken for an old training worker."""
    if previous.get("status") != "running" or process_identity(previous["pid"]) is None:
        return False
    try:
        actual = Path(f"/proc/{previous['pid']}/cmdline").read_bytes().split(b"\0")[:-1]
    except FileNotFoundError:
        return False
    expected = [str(value).encode() for value in (PYTHON, "-u", *previous["command"])]
    return actual == expected


class AdoptedProcess:
    """Observe a detached worker without restarting it or signalling its PID."""

    def __init__(self, previous, job):
        self.pid, self.job = previous["pid"], job
        self.identity = process_identity(self.pid)
        actual = Path(f"/proc/{self.pid}/cmdline").read_bytes().split(b"\0")[:-1]
        expected = [str(value).encode() for value in (PYTHON, "-u", *previous["command"])]
        if self.identity is None or actual != expected:
            raise RuntimeError("Previous worker PID no longer matches its saved command")

    def poll(self):
        if process_identity(self.pid) == self.identity:
            return None
        done, completion = job_completion(self.job)
        clean_manual_stop = (self.job["phase"] == "encoder"
                             and "--until-user-stop" in self.job["command"]
                             and completion.get("stopping_mode") == "until_user_stop"
                             and completion.get("stopped"))
        return 0 if done or clean_manual_stop or completion.get("planned_sampling_transition") else 1


def without_resume(command):
    command = list(command)
    if "--resume" in command:
        index = command.index("--resume")
        del command[index:index + 2]
    return command


def advance_sampling_phase(job, code, completion):
    if code != 0 or not completion.get("planned_sampling_transition"):
        return False
    command = job["command"]
    if ("--motion-sampling" not in command or command[command.index("--motion-sampling") + 1] != "adaptive"
            or completion["completed_updates"] != int(command[command.index("--adaptive-after-update") + 1])
            or completion.get("stopped") or completion["motion_sampling"]["active_mode"] != "uniform"):
        raise ValueError("Unexpected planned sampling transition")
    checkpoint = Path(job["output"]) / f"checkpoint_update_{completion['completed_updates']:06d}.pt"
    if not checkpoint.exists():
        raise ValueError("The uniform phase did not save its exact transition checkpoint")
    job["command"] = without_resume(command) + ["--resume", str(checkpoint)]
    job.update(status="pending", sampling_phase="adaptive", sampling_transition_update=completion["completed_updates"])
    return True


def run(run_root, adopt_running=False):
    if not run_root.is_relative_to(ROOT):
        raise ValueError("Experiment directory must remain inside project")
    run_root.mkdir(parents=True, exist_ok=True)
    if (run_root / "archive/closed.json").exists():
        raise ValueError("This experiment was closed by the user; start a new run root")
    layout = experiment_layout(run_root)
    lock = (run_root / "scheduler.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = run_root / "logs"
    logs.mkdir(exist_ok=True)
    queue = jobs(run_root)
    processes = {}
    env = process_environment()
    state_path = run_root / "state.json"
    previous_jobs = {j["name"]: j for j in json.loads(state_path.read_text())["jobs"]} if state_path.exists() else {}
    for job in queue:
        job["status"] = "pending"
        previous = previous_jobs.get(job["name"], {})
        if saved_worker_is_running(previous):
            if not adopt_running:
                raise RuntimeError(f"Previous child {job['name']} PID {previous['pid']} is still running; use --adopt-running")
            if without_resume(previous["command"]) != job["command"]:
                raise RuntimeError(f"Running {job['name']} has different arguments; checkpoint and stop it before changing its configuration")
            processes[job["name"]] = AdoptedProcess(previous, job)
            # Keep the current dependency/GPU restrictions when adopting a live
            # worker whose immutable command has already been checked above.
            job.update({key: previous[key] for key in ("status", "pid", "gpus", "started_at", "command") if key in previous})
            job.update(adopted=True, adopted_at=time.time())
            continue
        output = Path(job["output"])
        done, completion = job_completion(job)
        if done:
            job.update(status="complete", completion=completion)
        elif job["phase"] != "audit":
            checkpoint = output / ("last.pt" if job["phase"] == "encoder" else "checkpoint_final.pt")
            if job["phase"] != "encoder":
                # An evaluation may fail after saving a newer exact-update
                # checkpoint while an earlier interrupted final file still exists.
                checkpoints = sorted(output.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime) if output.exists() else []
                checkpoint = checkpoints[-1] if checkpoints else checkpoint
            if checkpoint.exists():
                job["command"] += ["--resume", str(checkpoint)]
    write_state(run_root / "protocol_jobs.json", queue)
    while True:
        for job in queue:
            if job["status"] == "running":
                process = processes[job["name"]]
                code = process.poll()
                if code is None:
                    continue
                job["exit_code"], job["finished_at"] = code, time.time()
                done, complete = job_completion(job)
                if advance_sampling_phase(job, code, complete):
                    job["uniform_phase_completion"] = complete
                    print(json.dumps({"event": "resume_with_adaptive_sampling", **job}), flush=True)
                    continue
                waiting = (job["phase"] == "encoder" and "--until-user-stop" in job["command"]
                           and code == 0 and complete.get("stopped"))
                job["status"] = "complete" if code == 0 and done else ("waiting_for_user" if waiting else "failed")
                job["completion"] = complete
                print(json.dumps(job), flush=True)
        authorized = stage2_authorized(run_root)
        for job in queue:
            if job["phase"] == "encoder" and job["status"] == "waiting_for_user":
                done, completion = job_completion(job)
                if done:
                    job.update(status="complete", completion=completion)
            if job["phase"] == "ppo" and job["status"] in ("pending", "waiting_for_user"):
                job["status"] = "pending" if authorized else "waiting_for_user"
        reserved = {gpu for job in queue if job["status"] == "running" for gpu in job["gpus"]}
        available = available_gpus(reserved, layout["allowed_gpus"])
        completed = {job["name"] for job in queue if job["status"] == "complete"}
        for job in queue:
            if job["status"] != "pending" or not set(job.get("dependencies", ())).issubset(completed):
                continue
            hold = pending_job_hold(run_root, job)
            if hold is not None:
                job["launch_hold"] = hold
                continue
            job.pop("launch_hold", None)
            selected = select_job_gpus(job, available)
            if not selected:
                continue
            available = [gpu for gpu in available if gpu not in selected]
            job_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(map(str, selected)))
            command = [PYTHON, "-u", *job["command"]]
            log_path = job_log_path(run_root, job)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as handle:
                process = subprocess.Popen(command, cwd=ROOT, env=job_env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            processes[job["name"]] = process
            job.update(status="running", pid=process.pid, gpus=selected, started_at=time.time())
            print(json.dumps(job), flush=True)
        write_state(run_root / "state.json", {"updated_at": time.time(), "scheduler_pid": os.getpid(),
                    "allowed_gpus": layout["allowed_gpus"], "stage2_authorized": authorized, "jobs": queue})
        ensure_wandb_sync(run_root, env)
        if all(job["status"] == "complete" for job in queue):
            break
        if not any(job["status"] == "running" for job in queue) and any(job["status"] == "failed" for job in queue):
            raise RuntimeError("A training job failed; see state.json and its log before restarting")
        time.sleep(20)
    if layout["stage1_only"]:
        return
    subprocess.run([PYTHON, "-u", str(ROOT / "scripts/audit_limb_context_training.py"), str(run_root)],
                   cwd=ROOT, env=env, check=True)
    subprocess.run([PYTHON, "-u", str(ROOT / "scripts/evaluate_limb_context_experiment.py"),
                    "--run-root", str(run_root)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--adopt-running", action="store_true")
    args = parser.parse_args()
    run(Path(args.run_root).resolve(), args.adopt_running)
