"""Explicit motion-sampling phases and rank-local adaptive statistics on resume."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import torch

VERSION = "limb_motion_sampling_curriculum_v1"
STATE_FIELDS = ("bin_visit_count", "bin_failure_count", "_adaptive_pending_visit_count",
                "_adaptive_pending_failure_count")
PARAMETERS = ("adaptive_uniform_ratio", "adaptive_bin_width_s", "adaptive_bin_width_steps",
              "adaptive_prior_visit_count", "adaptive_prior_failure_count",
              "adaptive_failure_rate_ema_iterations", "adaptive_failure_rate_window_iterations",
              "adaptive_probability_max_over_mean", "adaptive_sequence_length_agnostic",
              "adaptive_max_prob_per_bin", "adaptive_max_prob_per_motion",
              "adaptive_pre_failure_sample_window_steps")


def active_mode(requested, after_update, completed_updates):
    if requested not in ("uniform", "adaptive") or after_update < 0 or completed_updates < 0:
        raise ValueError("Invalid motion-sampling curriculum")
    return "adaptive" if requested == "adaptive" and completed_updates >= after_update else "uniform"


def phase_budget(requested, after_update, completed_updates, target):
    if target is None:
        if requested == "adaptive" and completed_updates < after_update:
            return after_update - completed_updates
        return None
    remaining = max(0, target - completed_updates)
    if requested == "adaptive" and completed_updates < after_update:
        remaining = min(remaining, after_update - completed_updates)
    return remaining


def configure_motion_sampling(env_cfg, requested, after_update, completed_updates):
    command = env_cfg.commands["motion"]
    command.sampling_mode = active_mode(requested, after_update, completed_updates)
    command.rewind.enabled = False
    command.adaptive_bin_snapshot_interval_iterations = 0
    result = {"version": VERSION, "requested_mode": requested, "adaptive_after_update": after_update,
              "active_mode": command.sampling_mode, "failure_rewind_enabled": False,
              "adaptive_bin_snapshot_interval_iterations": 0,
              "adaptive_sampling": dataclasses.asdict(command.adaptive_sampling),
              "parameters": {key: getattr(command, key) for key in PARAMETERS},
              "scope": "motion/bin sampling only; per-world independent limb payloads remain U(0,4)",
              "statistics": "rank-local motion shards; adaptive statistics begin from priors at the phase boundary"}
    return json.loads(json.dumps(result))


def tracker_sampling_alignment(previous, desired):
    """Permit the audited legacy-native transition that restores tracker rewind.

    Probability parameters, statistics semantics, shard layout and curriculum
    must remain identical. Only the former forced rewind override and its new
    reference metadata can change.
    """
    if previous == desired:
        return True
    if (desired.get("sampling_contract") != "tracker_checkpoint_adaptive_v1"
            or not desired.get("reference_tracker_sha256")
            or previous.get("active_mode") != "adaptive"
            or previous.get("failure_rewind_enabled") is not False
            or desired.get("failure_rewind_enabled") is not True
            or desired.get("rewind", {}).get("enabled") is not True):
        return False
    expected = dict(previous)
    rewind = dict(previous.get("rewind", desired["rewind"]))
    rewind["enabled"] = True
    expected.update(failure_rewind_enabled=True, rewind=rewind,
                    sampling_contract=desired["sampling_contract"],
                    reference_tracker_sha256=desired["reference_tracker_sha256"])
    return expected == desired


def validate_sampling_resume(previous, desired, update, *, align_to_tracker=False, switch_to_uniform=False):
    old = previous.get("motion_sampling")
    old_mode = old["active_mode"] if old else "uniform"
    if switch_to_uniform:
        if (align_to_tracker or desired['requested_mode'] != 'uniform'
                or desired['active_mode'] != 'uniform' or desired['failure_rewind_enabled']
                or desired.get('rewind', {}).get('enabled', False)):
            raise ValueError('Uniform resume must disable adaptive sampling and failure rewind')
        if old:
            for key in ('adaptive_after_update', 'adaptive_sampling', 'parameters'):
                if old[key] != desired[key]:
                    raise ValueError(f'Uniform resume changed unrelated sampler setting: {key}')
            if 'rewind' in old:
                expected = dict(old['rewind'], enabled=False)
                if desired.get('rewind') != expected:
                    raise ValueError('Uniform resume changed unrelated rewind settings')
        return old_mode
    if old_mode != desired["active_mode"]:
        if not (old_mode == "uniform" and desired["active_mode"] == "adaptive"
                and update == desired["adaptive_after_update"]):
            raise ValueError("Change to adaptive sampling only from the exact curriculum checkpoint")
    if old and old_mode == "adaptive":
        if align_to_tracker:
            if not tracker_sampling_alignment(old, desired):
                raise ValueError("Tracker sampling alignment may only restore the source rewind contract")
            return old_mode
        for key in ("requested_mode", "adaptive_after_update", "adaptive_sampling", "parameters",
                    "failure_rewind_enabled"):
            if old[key] != desired[key]:
                raise ValueError(f"Adaptive resume changed {key}")
        if old.get("rewind") != desired.get("rewind"):
            raise ValueError("Adaptive resume changed rewind")
    return old_mode


def checkpoint_configuration(source, train, metadata):
    from intact_tracking.cli.residual_policy_train import _checkpoint_configuration
    from omegaconf import OmegaConf
    cfg = _checkpoint_configuration(source, train, metadata)
    sampling = metadata["motion_sampling"]
    base = "task.command.command."
    updates = {"sampling_mode": sampling["active_mode"],
               "rewind.enabled": sampling["failure_rewind_enabled"],
               "adaptive_bin_snapshot_interval_iterations": 0,
               "adaptive_sampling": sampling["adaptive_sampling"], **sampling["parameters"]}
    if "rewind" in sampling:
        updates.pop("rewind.enabled")
        updates["rewind"] = sampling["rewind"]
    for key, value in updates.items():
        OmegaConf.update(cfg, base + key, value, merge=False)
    if "training_terminations" in metadata:
        OmegaConf.update(cfg, "task.terminations", metadata["training_terminations"]["active_terms"], merge=False)
    return cfg


def state_digest(value):
    """Canonical model/optimizer digest independent of tensor device and pickle IDs."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            h.update(f"tensor/{tensor.dtype}/{tuple(tensor.shape)}".encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            h.update(b"mapping")
            for key in sorted(item, key=repr):
                h.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(f"sequence/{len(item)}".encode())
            for child in item: visit(child)
        else:
            h.update(f"{type(item).__name__}/{item!r}".encode())
    visit(value)
    return h.hexdigest()


class SamplingCheckpoint:
    def __init__(self, directory, distributed, configuration):
        self.directory, self.distributed, self.configuration = Path(directory), distributed, configuration

    def command(self, runner):
        return runner.env.unwrapped.command_manager.get_term("motion")

    def shard_hash(self, command):
        return hashlib.sha256("\n".join(command.motion_files).encode()).hexdigest()

    def prepare(self, runner):
        if self.configuration["active_mode"] != "adaptive":
            runner.motion_sampling_state = None
            return
        command, dist = self.command(runner), self.distributed
        assert command.cfg.sampling_mode == "adaptive"
        assert command.cfg.rewind.enabled == self.configuration["failure_rewind_enabled"]
        directory = self.directory / "sampling_state"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"update_{runner.completed_learning_updates:06d}_rank_{dist.rank}.pt"
        payload = {"version": VERSION, "rank": dist.rank, "world_size": dist.world_size,
                   "completed_updates": runner.completed_learning_updates,
                   "motion_shard_sha256": self.shard_hash(command), "configuration": self.configuration,
                   "ema_last_iteration": command._adaptive_ema_last_iteration,
                   "statistics": {key: getattr(command, key).detach().cpu().clone() for key in STATE_FIELDS},
                   "episode_state": "Open visits belong to simulator episodes and are rebuilt on resume"}
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=False)
            if state_digest(existing) != state_digest(payload):
                raise ValueError("An adaptive statistics checkpoint would be overwritten with different data")
        else:
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
        with path.open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        header = {"rank": dist.rank, "path": str(path.resolve()), "sha256": sha,
                  "motion_shard_sha256": payload["motion_shard_sha256"]}
        headers = dist.all_gather_object(header)
        runner.motion_sampling_state = {"version": VERSION, "completed_updates": runner.completed_learning_updates,
                                        "world_size": dist.world_size, "ranks": headers}

    def restore(self, runner, previous_mode, *, reset=False, align_to_tracker=False):
        if reset:
            if self.configuration["active_mode"] != "adaptive":
                raise ValueError("Resetting adaptive statistics requires active adaptive sampling")
            command = self.command(runner)
            # The simulator and its open episode visits are newly constructed.
            # Reset only the aggregate sampler state, without reading any of
            # the previous checkpoint's per-rank statistics files.
            valid = command.bin_valid_mask
            for key in STATE_FIELDS:
                value = getattr(command, key)
                if not isinstance(value, torch.Tensor) or value.shape != valid.shape:
                    raise ValueError(f"Unsupported adaptive reset buffer: {key}")
            command.bin_visit_count.fill_(command.adaptive_prior_visit_count)
            command.bin_failure_count.fill_(command.adaptive_prior_failure_count)
            command.bin_visit_count.masked_fill_(~valid, 0.)
            command.bin_failure_count.masked_fill_(~valid, 0.)
            command._adaptive_pending_visit_count.zero_()
            command._adaptive_pending_failure_count.zero_()
            command._adaptive_ema_last_iteration = None
            runner.loaded_motion_sampling_state = None
            runner.motion_sampling_state = None
            return {
                "restored": False, "reset_to_priors": True,
                "reason": "explicit_reset_adaptive_sampling_on_resume",
                "rank": self.distributed.rank,
                "completed_updates": runner.completed_learning_updates,
                "prior_visit_count": float(command.adaptive_prior_visit_count),
                "prior_failure_count": float(command.adaptive_prior_failure_count),
                "valid_bins": int(valid.sum()),
                "visit_prior_max_error": float((command.bin_visit_count -
                    valid.to(command.bin_visit_count.dtype) * command.adaptive_prior_visit_count).abs().max()),
                "failure_prior_max_error": float((command.bin_failure_count -
                    valid.to(command.bin_failure_count.dtype) * command.adaptive_prior_failure_count).abs().max()),
                "pending_visit_count": float(command._adaptive_pending_visit_count.sum()),
                "pending_failure_count": float(command._adaptive_pending_failure_count.sum()),
                "ema_last_iteration": None, "open_episode_visits_rebuilt": True,
            }
        if self.configuration['active_mode'] == 'uniform':
            command = self.command(runner)
            if command.cfg.sampling_mode != 'uniform' or command.cfg.rewind.enabled:
                raise ValueError('Uniform resume runtime must disable adaptive sampling and rewind')
            runner.loaded_motion_sampling_state = None
            runner.motion_sampling_state = None
            return {'restored': False, 'adaptive_statistics_discarded': True,
                    'reason': 'uniform_sampling_does_not_restore_adaptive_statistics',
                    'rank': self.distributed.rank, 'previous_mode': previous_mode,
                    'completed_updates': runner.completed_learning_updates}
        if previous_mode != "adaptive":
            return {"restored": False, "reason": "Uniform prefix had no adaptive statistics; initialize from priors"}
        from intact_tracking.limb_context_protocol import PROJECT_ROOT
        record = getattr(runner, "loaded_motion_sampling_state", None)
        if (not record or record["version"] != VERSION
                or record["completed_updates"] != runner.completed_learning_updates
                or record["world_size"] != self.distributed.world_size):
            raise ValueError("Adaptive resume is missing matching per-rank statistics")
        headers = record["ranks"]
        if sorted(row["rank"] for row in headers) != list(range(self.distributed.world_size)):
            raise ValueError("Adaptive checkpoint rank coverage changed")
        header = next(row for row in headers if row["rank"] == self.distributed.rank)
        path = Path(header["path"]).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError("Sampler checkpoint must remain inside the project")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != header["sha256"]:
                raise ValueError("Adaptive statistics checkpoint changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        command = self.command(runner)
        configuration_matches = payload["configuration"] == self.configuration
        if not configuration_matches and align_to_tracker:
            configuration_matches = tracker_sampling_alignment(payload["configuration"], self.configuration)
        if (payload["motion_shard_sha256"] != self.shard_hash(command)
                or payload["rank"] != self.distributed.rank
                or payload["world_size"] != self.distributed.world_size
                or payload["completed_updates"] != runner.completed_learning_updates
                or not configuration_matches):
            raise ValueError("Adaptive statistics belong to a different shard or sampling configuration")
        if set(payload["statistics"]) != set(STATE_FIELDS):
            raise ValueError("Adaptive statistics are incomplete")
        for key, value in payload["statistics"].items():
            target = getattr(command, key)
            if target.shape != value.shape or target.dtype != value.dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid adaptive sampling statistics: {key}")
            target.copy_(value.to(target.device))
        command._adaptive_ema_last_iteration = payload["ema_last_iteration"]
        return {"restored": True, "sha256": header["sha256"], "rank": self.distributed.rank,
                "statistics_sha256": state_digest(payload["statistics"]),
                "tracker_sampling_aligned": payload["configuration"] != self.configuration,
                "pending_completed_visits_restored": True, "open_episode_visits_rebuilt": True}


def audit_sampling_curriculum(directory, final, boundary=1000):
    """Check the actual update history, transition source and saved sampler tensors."""
    from intact_tracking.limb_context_checkpoint_eval import digest
    from intact_tracking.limb_context_protocol import PROJECT_ROOT

    directory = Path(directory)
    sampling = final["residual_policy"]["motion_sampling"]
    if (sampling["version"] != VERSION or sampling["requested_mode"] != "adaptive"
            or sampling["active_mode"] != "adaptive" or sampling["adaptive_after_update"] != boundary
            or sampling["failure_rewind_enabled"]
            or final["cfg"].task.command.command.sampling_mode != "adaptive"
            or final["cfg"].task.command.command.rewind.enabled):
        raise ValueError("Final checkpoint changed the requested sampling curriculum")
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    if [row["completed_updates"] for row in rows] != list(range(1, final["completed_updates"] + 1)):
        raise ValueError("Training history has missing, duplicated or extra updates")
    for row in rows:
        actual = row.get("motion_sampling", {})
        expected = row["completed_updates"] > boundary
        if (actual.get("adaptive_enabled", False) != expected
                or actual.get("failure_rewind_enabled", False)
                or (expected and actual.get("adaptive_after_update") != boundary)):
            raise ValueError("Actual training update used the wrong sampling phase")
    transitions = [row for row in final["residual_policy"].get("resume_history", [])
                   if row.get("motion_sampling_from") == "uniform" and row.get("motion_sampling_to") == "adaptive"]
    if (len(transitions) != 1 or transitions[0]["completed_updates"] != boundary
            or not transitions[0].get("restoration_audit", {}).get("passed")):
        raise ValueError("Missing exact model/optimizer restoration audit at the phase boundary")
    checkpoint = directory / f"checkpoint_update_{boundary:06d}.pt"
    if digest(checkpoint) != transitions[0]["checkpoint_sha256"]:
        raise ValueError("The selected uniform checkpoint changed")
    record = final.get("motion_sampling_state")
    if (not record or record["version"] != VERSION or record["world_size"] != 2
            or record["completed_updates"] != final["completed_updates"]
            or sorted(row["rank"] for row in record["ranks"]) != [0, 1]):
        raise ValueError("Final adaptive checkpoint lacks both ranks' sampling statistics")
    ranks = []
    for header in record["ranks"]:
        path = Path(header["path"]).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or digest(path) != header["sha256"]:
            raise ValueError("Final sampler checkpoint moved or changed")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (state["configuration"] != sampling or state["rank"] != header["rank"]
                or state["world_size"] != 2 or state["completed_updates"] != final["completed_updates"]
                or state["motion_shard_sha256"] != header["motion_shard_sha256"]
                or set(state["statistics"]) != set(STATE_FIELDS)):
            raise ValueError("Final adaptive statistics have the wrong shard/configuration")
        statistics = state["statistics"]
        if any(not torch.isfinite(value).all() or (value < 0).any() for value in statistics.values()):
            raise ValueError("Nonfinite or negative adaptive sampling counts")
        failures = statistics["bin_failure_count"].sum().item()
        if failures <= 0:
            raise ValueError("Adaptive sampler accumulated no failure statistics")
        ranks.append({"rank": header["rank"], "sha256": header["sha256"], "decayed_failure_count": failures})
    return {"passed": True, "uniform_updates": boundary, "adaptive_updates": final["completed_updates"] - boundary,
            "exact_transition_restoration": transitions[0]["restoration_audit"], "sampler_ranks": ranks}
