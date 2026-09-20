"""Observe the authorized run; write health/trends without controlling training."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


EXPECTED_WEIGHTS = {
    "teacher_forced_weight": 1., "recursive_weight": .5,
    "local_positive_weight": .01, "dr_nominal_response_weight": .04,
    "weak_positive_weight": .008, "weak_negative_weight": 0., "nominal_anchor_weight": .01,
}
PROBE_KEYS = (
    "dr_five_step_nmse", "nominal_five_step_nmse", "dr_nominal_relation_loss",
    "nominal_anchor_loss", "nominal_anchor_cosine", "latent_response_correlation",
    "latent_distance_mean", "latent_target_distance_mean", "latent_shuffle_dr_error_ratio",
    "dr_center_rank_loss", "dr_center_rank_weighted_loss", "dr_center_rank_comparisons",
    "dr_center_rank_accuracy", "dr_center_rank_worlds",
    "dr_soft_loss", "dr_soft_weighted_loss", "dr_soft_worlds",
    "dr_soft_target_self_mass", "dr_soft_target_entropy", "dr_soft_effective_neighbors",
    "dr_soft_predicted_self_mass",
)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def json_file(path):
    # The trainer's progress writer truncates before writing; retry a racing read.
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.03)


def complete_rows(path):
    if not path.exists():
        return []
    # An unfinished last append is not evidence of a failed trainer.
    return [json.loads(line) for line in path.read_text().split("\n")[:-1] if line.strip()]


def process_identity(pid, expected_start=None):
    root = Path("/proc") / str(pid)
    try:
        text = (root / "stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        start, state = int(fields[19]), fields[0]
        command = (root / "cmdline").read_bytes().split(b"\0")
        return {"pid": pid, "state": state, "start_ticks": start,
                "identity_matches": expected_start is None or start == expected_start,
                "live": state not in ("Z", "X", "x") and (expected_start is None or start == expected_start),
                "command": [part.decode(errors="replace") for part in command if part]}
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "state": "missing", "identity_matches": False, "live": False}


def worker_processes(leader):
    if not leader["live"]:
        return []
    children = Path(f"/proc/{leader['pid']}/task/{leader['pid']}/children").read_text().split()
    workers = []
    for child in children:
        identity = process_identity(int(child))
        if any(entry in identity.get("command", []) for entry in (
                "intact_tracking.cli.forward_memory_nominal_direction_train",
                "intact_tracking.cli.forward_memory_nominal_dr_rank_train",
                "intact_tracking.cli.forward_memory_native_dr_train",
                "intact_tracking.cli.forward_memory_nominal_dr_soft_train")):
            workers.append({k: v for k, v in identity.items() if k != "command"})
    return workers


def check_loss(values, weights=None):
    weights = EXPECTED_WEIGHTS if weights is None else weights
    expected = (values["prediction_loss"]
                + weights["local_positive_weight"] * values["representation_positive_loss"]
                + weights["dr_nominal_response_weight"] * values["dr_nominal_relation_loss"]
                + weights["weak_positive_weight"] * values["weak_positive_loss"]
                + weights["nominal_anchor_weight"] * values["nominal_anchor_loss"])
    rank = "dr_center_rank_weight" in weights
    if rank:
        expected += weights["dr_center_rank_weight"] * values["dr_center_rank_loss"]
    soft = "dr_soft_weight" in weights
    if soft:
        expected += weights["dr_soft_weight"] * values["dr_soft_loss"]
    extra = rank or soft
    matches = math.isclose(values["loss"], expected, rel_tol=2e-5, abs_tol=2e-6)
    return {"finite": all(math.isfinite(v) for v in values.values()),
            "term_count": 7 if extra else 6, "term_sum_matches": matches,
            ("seven_term_sum_matches" if extra else "six_term_sum_matches"): matches,
            "negative_term_absent": not any("weak_negative" in key for key in values),
            "reconstructed_loss": expected}


def probe_trend(rows):
    if not rows:
        return {}
    first, last = rows[0], rows[-1]
    result = {}
    for key in PROBE_KEYS:
        if key not in first["fixed_probe"] or key not in last["fixed_probe"]:
            continue
        before, after = first["fixed_probe"][key], last["fixed_probe"][key]
        result[key] = {"first": before, "latest": after, "delta": after - before,
                       "relative_change_percent": 100 * (after / before - 1) if before else None}
    window = rows[-6:]
    seconds_per_update = None
    if len(window) > 1:
        seconds_per_update = ((window[-1]["unix_time"] - window[0]["unix_time"])
                              / (window[-1]["update"] - window[0]["update"]))
    return {"first_update": first["update"], "latest_update": last["update"],
            "reports": len(rows), "seconds_per_update_recent": seconds_per_update,
            "metrics": result,
            "fixed_probe_cross_motion_pairs": last["fixed_probe"]["weak_positive_pairs"],
            "fixed_probe_cross_motion_available": last["fixed_probe"]["weak_positive_pairs"] > 0}


class Monitor:
    def __init__(self, root):
        self.root, self.run = root, root / "stage1_8192"
        self.initial_anchor = json_file(self.run / "nominal_anchor.json")
        config_path = self.run / "run_config.json"
        self.initial_config = json_file(config_path) if config_path.exists() else None
        self.expected_weights = (self.initial_config["objective_weights"] if self.initial_config
                                 else dict(EXPECTED_WEIGHTS))
        self.world_size = (self.initial_config["distributed"]["world_size"] if self.initial_config
                           else len(self.initial_anchor["sha256_by_rank"]))
        self.checkpoint_signature = None
        self.checkpoint = None

    def checkpoint_audit(self):
        path = self.run / "last.pt"
        stat = path.stat()
        signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if signature == self.checkpoint_signature:
            return self.checkpoint
        import torch
        torch.set_num_threads(1)
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        anchor = json_file(self.run / "nominal_anchor.json")
        vector = torch.tensor(state["nominal_direction_anchor"]["direction"], dtype=torch.float32)
        digest = hashlib.sha256(vector.numpy().tobytes()).hexdigest()
        agreement = state["distributed_parameter_agreement"]
        loss = state["loss_config"]
        weights = self.expected_weights
        gradient_steps = (self.initial_config["arguments"]["gradient_steps_per_update"]
                          if self.initial_config else 4)
        checks = {
            "all_rank_model_agreement": (agreement["passed"]
                and len(agreement["sha256_by_rank"]) == self.world_size
                and len(set(agreement["sha256_by_rank"])) == 1),
            "fixed_anchor_preserved": (state["nominal_direction_anchor"] == anchor == self.initial_anchor
                                       and digest == self.initial_anchor["sha256"]),
            "all_rank_anchor_agreement": len(anchor["sha256_by_rank"]) == self.world_size and set(anchor["sha256_by_rank"]) == {digest},
            "optimizer_steps_match": state["optimizer_steps"] == state["update"] * gradient_steps,
            "supervision_matches": state["supervision_horizons"] == {"predictor": 5, "response_label": 10},
            "weights_match": (loss["representation_weight"] == weights["local_positive_weight"]
                and math.isclose(loss["representation_weight"] * loss["representation_relation_weight"],
                                 weights["dr_nominal_response_weight"], rel_tol=1e-12)
                and loss["nominal_anchor_weight"] == weights["nominal_anchor_weight"]
                and loss["weak_positive_weight"] == weights["weak_positive_weight"]
                and loss["weak_negative_weight"] == weights["weak_negative_weight"] == 0
                and loss.get("dr_center_rank_weight", 0.) == weights.get("dr_center_rank_weight", 0.)
                and loss.get("dr_soft_weight", 0.) == weights.get("dr_soft_weight", 0.)),
        }
        if self.initial_config:
            args = self.initial_config["arguments"]
            probability = state["training_physics"].get("independent_nominal_mixture", {}).get("probability", 0.)
            checks["sampling_preserved"] = (state["nominal_a_fraction"] == args["nominal_fraction"]
                and probability == args["dr_nominal_probability"]
                and state["training_physics"].get("limb_max_masses_kg") == args.get("limb_max_masses_kg"))
            checks["full_loss_configuration_preserved"] = loss == self.initial_config["loss"]
            if "dr_center_rank_weight" in weights:
                checks["dr_ranking_schema_preserved"] = (
                    loss.get("dr_center_rank_version") == 1
                    and state.get("dr_metric_schema") == self.initial_config["dr_center_rank_contract"]["schema"])
            if "dr_soft_weight" in weights:
                checks["dr_soft_schema_preserved"] = (
                    loss.get("dr_soft_version") == 1
                    and state.get("dr_metric_schema") == self.initial_config["dr_soft_contract"]["schema"])
        result = {"update": state["update"], "optimizer_steps": state["optimizer_steps"],
                  "mtime": stat.st_mtime, "bytes": stat.st_size,
                  "anchor_sha256": digest, "checks": checks, "passed": all(checks.values())}
        self.checkpoint_signature, self.checkpoint = signature, result
        return result

    def snapshot(self):
        now = time.time()
        record = json_file(self.root / "train_process.json")
        leader = process_identity(record["pid"], record["process_start_ticks"])
        workers = worker_processes(leader)
        progress = json_file(self.run / "progress.json")
        config = json_file(self.run / "run_config.json")
        rows = complete_rows(self.run / "metrics.jsonl")
        light = complete_rows(self.run / "training_metrics.jsonl")
        latest = rows[-1] if rows else None
        alerts = []
        if not leader["live"]:
            alerts.append("recorded_training_process_missing_terminal_or_replaced")
        if len([w for w in workers if w["live"]]) != self.world_size:
            alerts.append(f"expected_{self.world_size}_live_workers")
        age = now - progress["unix_time"]
        if age > 300:
            alerts.append("progress_observation_stale_requires_inspection")
        if config["objective_weights"] != self.expected_weights:
            alerts.append("loss_configuration_changed")
        accounting = check_loss(latest["optimization_train"], self.expected_weights) if latest else None
        if accounting and not all(accounting[k] for k in ("finite", "term_sum_matches", "negative_term_absent")):
            alerts.append("reported_training_loss_invalid")
        if latest and not all(math.isfinite(v) for v in latest["fixed_probe"].values()):
            alerts.append("validation_metric_nonfinite")
        rank0 = None
        if light:
            last = light[-1]
            values = {k.removeprefix("optimization_rank0/"): v for k, v in last.items()
                      if k.startswith("optimization_rank0/")}
            rank0 = {"update": last["update"], "values": values,
                     "accounting": check_loss(values, self.expected_weights)}
            if not all(rank0["accounting"][k] for k in ("finite", "term_sum_matches", "negative_term_absent")):
                alerts.append("rank0_training_loss_invalid")
        checkpoint = self.checkpoint_audit()
        if not checkpoint["passed"]:
            alerts.append("checkpoint_contract_failed")
        if progress["completed_updates"] - checkpoint["update"] > 500:
            alerts.append("checkpoint_more_than_two_save_intervals_behind")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True, timeout=10)
        return {"checked_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "unix_time": now, "status": "attention" if alerts else "running",
                "alerts": alerts, "leader": {k: v for k, v in leader.items() if k != "command"},
                "workers": workers, "progress": progress, "progress_age_seconds": age,
                "latest_report_update": latest["update"] if latest else None,
                "latest_global_train": latest["optimization_train"] if latest else None,
                "latest_rank0_train": rank0, "loss_accounting": accounting,
                "validation_trend": probe_trend(rows), "checkpoint": checkpoint,
                "gpu_csv_index_memory_mib_util_percent": gpu.splitlines(),
                "control_actions": "none; observations never stop or restart training"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        raise ValueError("Monitoring interval must be between 1 and 60 seconds")
    output = args.run_root / "monitor"
    output.mkdir(exist_ok=True)
    with (output / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor = Monitor(args.run_root)
        identity = process_identity(os.getpid())
        atomic_json(output / "process.json", {"pid": os.getpid(), "start_ticks": identity["start_ticks"],
                    "interval_seconds": args.interval, "watch": args.watch})
        while True:
            try:
                result = monitor.snapshot()
                atomic_json(output / "status.json", result)
                with (output / "history.jsonl").open("a") as handle:
                    handle.write(json.dumps(json_safe(result), allow_nan=False) + "\n")
                print(json.dumps({"update": result["progress"]["completed_updates"],
                                  "status": result["status"], "alerts": result["alerts"]}), flush=True)
            except Exception as error:
                # Observation failures are recorded, never interpreted as a stopped job.
                result = {"status": "observation_error", "unix_time": time.time(),
                          "error": f"{type(error).__name__}: {error}"}
                atomic_json(output / "observation_error.json", result)
                atomic_json(output / "status.json", result)
                print(json.dumps(result), flush=True)
            if not args.watch:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
