"""Training-only termination profiles, with explicit legacy resume compatibility."""

from __future__ import annotations

import copy
import json
from pathlib import Path

VERSION = "limb_training_terminations_v1"
ORIGINAL = "original"
NO_EE_BODY_POS = "no_ee_body_pos"
PROFILES = (ORIGINAL, NO_EE_BODY_POS)
PROFILE_FILE = "training_terminations.json"


def _profile(value):
    if value not in PROFILES:
        raise ValueError(f"Unknown training termination profile: {value}")
    return value


def experiment_profile(root):
    """New experiments omit EE termination; existing matrices keep their contract."""
    root = Path(root)
    path = root / PROFILE_FILE
    if path.exists():
        return _profile(json.loads(path.read_text())["profile"])
    profiles = set()
    for directory in ("ppo_2gpu8192_scratch", "smoke_2gpu8192_scratch", "ppo_4gpu8192_scratch", "smoke_4gpu8192_scratch"):
        for path in (root / directory).glob("*/run_config.json"):
            metadata = json.loads(path.read_text())
            profiles.add(_profile(metadata.get("training_terminations", {}).get("profile", ORIGINAL)))
    if len(profiles) > 1:
        raise ValueError("Experiment mixes training termination profiles")
    return next(iter(profiles), NO_EE_BODY_POS)


def resolve_profile(requested, output, project_root, previous=None):
    """Honor a pinned experiment, or the checkpoint when resuming standalone PPO."""
    output, project_root = Path(output).resolve(), Path(project_root).resolve()
    if not output.is_relative_to(project_root):
        raise ValueError("Training output must remain in the project")
    pinned = None
    for directory in (output, *output.parents):
        path = directory / PROFILE_FILE
        if path.exists():
            pinned = _profile(json.loads(path.read_text())["profile"])
            break
        if directory == project_root:
            break
    if requested is not None:
        requested = _profile(requested)
        if pinned is not None and pinned != requested:
            raise ValueError("Training termination profile differs from this experiment; use a new run root")
        return requested
    if pinned is not None:
        return pinned
    if previous is not None:
        return _profile(previous["residual_policy"].get("training_terminations", {}).get("profile", ORIGINAL))
    return NO_EE_BODY_POS


def _contract(original_terms, profile):
    profile = _profile(profile)
    original_terms = copy.deepcopy(original_terms)
    names = [item["name"] for item in original_terms]
    if names.count("ee_body_pos") != 1 or len(names) != len(set(names)):
        raise ValueError("Expected exactly one ee_body_pos termination in the source tracker")
    disabled = ["ee_body_pos"] if profile == NO_EE_BODY_POS else []
    return {"version": VERSION, "profile": profile, "disabled_terms": disabled,
            "original_terms": original_terms,
            "active_terms": [item for item in original_terms if item["name"] not in disabled],
            "evaluation_profile": ORIGINAL}


def configure_training_terminations(env_cfg, source, profile):
    from omegaconf import OmegaConf
    contract = _contract(OmegaConf.to_container(source.task.terminations, resolve=True), profile)
    if list(env_cfg.terminations) != [item["name"] for item in contract["original_terms"]]:
        raise ValueError("Prepared training terminations differ from the source tracker")
    # Keep the shared rollout/stage-1/evaluation configuration untouched.
    env_cfg.terminations = {name: term for name, term in env_cfg.terminations.items()
                            if name not in contract["disabled_terms"]}
    return contract


def checkpoint_contract(checkpoint):
    """Verify the saved runtime config, including checkpoints predating profiles."""
    from omegaconf import OmegaConf
    actual = OmegaConf.to_container(checkpoint["cfg"].task.terminations, resolve=True)
    recorded = checkpoint["residual_policy"].get("training_terminations")
    if recorded is None:
        return _contract(actual, ORIGINAL)
    expected = _contract(recorded["original_terms"], recorded["profile"])
    if recorded != expected or actual != expected["active_terms"]:
        raise ValueError("Saved training termination configuration differs from its contract")
    return expected


def validate_termination_resume(previous, desired):
    if checkpoint_contract(previous) != desired:
        raise ValueError("Resume changed training terminations; use a separate experiment for a changed task")


def audit_runtime_terminations(manager, contract):
    names = [item["name"] for item in contract["active_terms"]]
    if list(manager.active_terms) != names:
        raise ValueError("Runtime termination manager differs from the training contract")
    return {"passed": True, "active_terms": names,
            "ee_body_pos_enabled": "ee_body_pos" in names}
