"""Tail only preview-v2 TensorBoard scalar logs to W&B without touching training.

No checkpoints, code, datasets, environment variables or console logs uploaded.
The independent CPU sidecar backfills existing events and discovers new v2 runs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import wandb
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util

VERSION = "simulator_preview_original_inputs_v2"
ABC_VERSION = "lafan_abc_tracker_head_finetune_warmcritic_v2"


def scalar_value(value):
    kind = value.WhichOneof("value")
    if kind == "simple_value":
        return float(value.simple_value)
    if kind == "tensor":
        array = tensor_util.make_ndarray(value.tensor)
        if array.size == 1 and array.dtype.kind in "biuf":
            return float(array.item())
    return None


def metric_axis(tag):
    # RSL-RL uses elapsed seconds, not iterations, for these duplicate curves.
    return "elapsed_seconds" if tag.endswith("/time") else "iteration"


def run_config(metadata, directory):
    arguments = metadata["arguments"]
    if metadata.get("version") == ABC_VERSION:
        return {
            "experiment_version": ABC_VERSION, "condition": metadata["condition"],
            "physics": metadata["physics_mode"], "physics_details": metadata["physics"]["details"],
            "global_num_envs": metadata["global_num_envs"], "world_size": metadata["world_size"],
            "num_envs_per_rank": arguments["num_envs"], "motion_count": metadata["motion_count"],
            "seed": arguments["seed"], "requested_updates": arguments["iterations"],
            "real_transitions_per_update": metadata["real_transitions_per_update"],
            "actor_lr": arguments["actor_lr"], "critic_lr": arguments["critic_lr"],
            "critic_warmup": arguments["critic_warmup"], "input_audit": metadata["input_audit"],
            "trainable_actor": metadata["trainable_actor"], "frozen_actor": metadata["frozen_actor"],
            "critic": metadata.get("critic", "not recorded"),
            "initial_actor_and_critic_bitwise_restored": metadata.get(
                "initial_actor_and_critic_bitwise_restored", False),
            "optimizer_initialization": "fresh in all arms",
            "reward_sha256": metadata["reward_contract"]["sha256"], "reward_changes": {},
            "extra_input_dim": 0, "source_log_directory": directory.name,
            "metric_warning": "training metrics; endpoint results require fixed-start evaluation",
        }
    return {
        "experiment_version": VERSION, "variant": metadata["variant"],
        "physics": metadata["physics_mode"], "num_envs": arguments["num_envs"],
        "motion": Path(metadata.get("motion_file") or metadata.get("motion_path") or "unknown").name,
        "motion_count": metadata.get("motion_count", 1),
        "dataset": metadata.get("dataset", {}),
        "dr_profile": metadata.get("dr_profile", "right-hand-1-3kg"),
        "payload_configuration": metadata.get("physics", {}).get("details", {}),
        "seed": arguments["seed"],
        "rollout_steps": arguments["rollout_steps"],
        "requested_iterations": arguments["iterations"],
        "actor_lr": arguments["actor_lr"], "critic_lr": arguments["critic_lr"],
        "critic_warmup_updates": arguments["critic_warmup_updates"],
        "epochs": arguments["epochs"], "mini_batches": arguments["mini_batches"],
        "extra_input_dim": metadata["extra_input_dim"],
        "input_audit": metadata["input_audit"],
        "reward_sha256": metadata["reward_contract"]["sha256"],
        "reward_changes": metadata["reward_changes"],
        "source_log_directory": directory.name,
        "logging_method": "independent TensorBoard scalar sidecar; training is not restarted",
        "metric_warning": "Training rollout metrics, not matched fixed-start evaluation results",
    }


class Tail:
    def __init__(self, directory, metadata, state, args):
        self.directory, self.state = directory, state
        self.loaders = {}
        self.pending = {}
        self.defined = set()
        self.steps_per_update = metadata.get("real_transitions_per_update") or (
            metadata["arguments"]["num_envs"] * metadata["arguments"]["rollout_steps"])
        events = sorted(directory.glob("events.out.tfevents.*"))
        if "id" not in state:
            identity = str(directory.resolve()) + ":" + events[0].name
            state["id"] = "pv2-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
            state["cursors"] = {}
        self.run = wandb.init(
            entity=args.entity, project=args.project, id=state["id"],
            name=directory.name,
            group=("lafan-abc-finetune" if metadata.get("version") == ABC_VERSION else
                   "full-data-preview-v2" if metadata.get("motion_count", 1) > 1
                   else "single-motion-preview-v2"), job_type="training-log-mirror",
            tags=["abc-finetune" if metadata.get("version") == ABC_VERSION else "preview-v2",
                  metadata["variant"], metadata["physics_mode"], "tensorboard-mirror"],
            config=run_config(metadata, directory), resume="allow", reinit="create_new",
            mode="online", dir=str(args.state_dir), save_code=False,
            settings=wandb.Settings(console="off", disable_code=True, disable_git=True,
                                    x_disable_stats=True, x_disable_meta=True),
        )
        self.run.define_metric("iteration")
        self.run.define_metric("elapsed_seconds")
        self.run.define_metric("*", step_metric="iteration")
        state["url"] = self.run.url
        print(json.dumps({"run": directory.name, "url": self.run.url}), flush=True)

    def read(self):
        now = time.time()
        for path in sorted(self.directory.glob("events.out.tfevents.*")):
            if str(path) not in self.loaders:
                self.loaders[str(path)] = EventFileLoader(str(path))
            loader = self.loaders[str(path)]
            for event in loader.Load():
                for value in event.summary.value:
                    scalar = scalar_value(value)
                    if scalar is None or not math.isfinite(scalar):
                        continue
                    cursor = self.state["cursors"].get(value.tag, [-1, -1.0])
                    if event.wall_time <= cursor[1]:
                        continue
                    axis = metric_axis(value.tag)
                    key = (axis, int(event.step))
                    row = self.pending.setdefault(key, {"values": {}, "cursors": {}, "updated": now})
                    row["values"][value.tag] = scalar
                    row["cursors"][value.tag] = [int(event.step), event.wall_time]
                    row["updated"] = now

    def flush(self, *, force=False):
        now = time.time()
        latest = {}
        for axis, step in self.pending:
            latest[axis] = max(latest.get(axis, -1), step)
        uploaded = 0
        for (axis, step), row in sorted(tuple(self.pending.items()), key=lambda item: (item[0][0], item[0][1])):
            # Hold the live tail briefly so an async writer can finish its row.
            if not force and step == latest[axis] and now - row["updated"] < 2:
                continue
            for tag in row["values"]:
                if tag not in self.defined:
                    self.run.define_metric(tag, step_metric=axis)
                    self.defined.add(tag)
            payload = {**row["values"], axis: step}
            if axis == "iteration":
                payload.update(completed_updates=step + 1, real_transitions=(step + 1) * self.steps_per_update)
                self.run.summary["sync/last_iteration"] = step
                self.state["last_iteration"] = step
            self.run.log(payload)
            self.state["cursors"].update(row["cursors"])
            del self.pending[(axis, step)]
            uploaded += 1
        if uploaded:
            self.run.summary["sync/last_upload_utc_seconds"] = now
            print(json.dumps({"run": self.directory.name, "uploaded_rows": uploaded,
                              "last_iteration": self.state.get("last_iteration")}), flush=True)
        return uploaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/preview_v2"))
    parser.add_argument("--project", default="intact-preview-v2")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll interval must be positive")
    args.root = args.root.resolve()
    args.state_dir = args.root / ".wandb_sync"
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "sidecar.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.state_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and (state["entity"] != args.entity or state["project"] != args.project):
        raise ValueError("Existing sync state belongs to another W&B destination")
    state.update(entity=args.entity, project=args.project, pid=os.getpid())
    state.setdefault("runs", {})
    tails = {}
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def save_state():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(temporary, state_path)

    try:
        while not stopping:
            for directory in sorted(args.root.iterdir()):
                if not directory.is_dir() or directory.name in tails:
                    continue
                path = directory / "run_config.json"
                if not path.is_file() or not list(directory.glob("events.out.tfevents.*")):
                    continue
                metadata = json.loads(path.read_text())
                if metadata.get("version") not in (VERSION, ABC_VERSION):
                    continue
                entry = state["runs"].setdefault(directory.name, {})
                tails[directory.name] = Tail(directory, metadata, entry, args)
                save_state()
            for tail in tails.values():
                tail.read()
                tail.flush(force=args.once)
            save_state()
            if args.once:
                break
            time.sleep(args.poll_seconds)
    finally:
        for tail in tails.values():
            tail.read()
            tail.flush(force=True)
            tail.run.finish()
        save_state()
        lock.close()


if __name__ == "__main__":
    main()
