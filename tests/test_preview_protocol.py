import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from intact_tracking.cli.adaptation_eval import configure_physics
from intact_tracking.cli.simulator_preview_train import build_parser, validate_resume
from intact_tracking.environment.mdp.shared_motion import share_motion_arrays, shared_motion_arrays
from intact_tracking.preview_protocol import (
    LIMBS,
    add_limb_payloads,
    audit_limb_payloads,
    dataset_identity,
)


def test_payloads_are_four_distinct_shins_hands_and_do_not_stack():
    existing = object()
    cfg = SimpleNamespace(events={"existing": SimpleNamespace(func=existing)})
    details = add_limb_payloads(cfg)
    assert cfg.events["existing"].func is existing
    assert details["total_added_mass_range_kg"] == [8, 16]
    assert len(cfg.events) == 5
    for name, (body, position, _) in LIMBS.items():
        event = cfg.events["preview_payload_" + name]
        assert event.mode == "startup" and event.params["mass_range_kg"] == (2, 4)
        assert event.params["body_name"] == body
        if "shin" in name:
            assert body.endswith("knee_link") and position[2] < 0
    with pytest.raises(ValueError, match="stack"):
        add_limb_payloads(cfg)


def test_actual_payload_audit_checks_all_limbs():
    values = {"preview_payload_" + name: torch.linspace(2, 4, 10) for name in LIMBS}
    env = SimpleNamespace(num_envs=10, event_manager=SimpleNamespace(
        get_term_cfg=lambda name: SimpleNamespace(func=SimpleNamespace(observe=lambda: values[name]))))
    result = audit_limb_payloads(env)
    assert result["total_min_kg"] == 8 and result["total_max_kg"] == 16
    values["preview_payload_right_shin"][0] = 5
    with pytest.raises(RuntimeError, match="right_shin"):
        audit_limb_payloads(env)


def test_full_dataset_identity_includes_every_npz_and_detects_edits(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    np.savez(tmp_path / "a.npz", joint_pos=np.zeros((2, 29)))
    np.savez(nested / "b.npz", joint_pos=np.zeros((3, 29)))
    files, identity = dataset_identity(motion_path=str(tmp_path))
    assert len(files) == identity["motion_count"] == 2
    assert identity["groups"] == {"(root)": 1, "nested": 1}
    np.savez(nested / "b.npz", joint_pos=np.zeros((4, 29)))
    assert dataset_identity(motion_path=str(tmp_path))[1]["manifest_sha256"] != identity["manifest_sha256"]


def test_sharing_is_scoped_checks_order_and_rejects_mutable_subsets():
    cfg = SimpleNamespace(motion_type="isaaclab", fk_from_joint_pos=False,
                          recompute_joint_vel_from_joint_pos=False, load_compact_qpos=False,
                          reference_storage_mode="qpos_only_actor_fk", actor_reference_fps=50,
                          body_names=["pelvis"])
    source = SimpleNamespace(cfg=cfg, motion_files=("a", "b"), body_indexes=torch.tensor([0]),
                             device="cpu", motion=object())
    assert shared_motion_arrays(cfg, ["a", "b"], source.body_indexes, "cpu") is None
    with share_motion_arrays(source):
        assert shared_motion_arrays(cfg, ["a", "b"], source.body_indexes, "cpu") is source.motion
        with pytest.raises(ValueError, match="catalog"):
            shared_motion_arrays(cfg, ["b", "a"], source.body_indexes, "cpu")
        changed = copy.copy(cfg)
        changed.actor_reference_fps = 60
        with pytest.raises(ValueError, match="fps"):
            shared_motion_arrays(changed, ["a", "b"], source.body_indexes, "cpu")
    assert shared_motion_arrays(cfg, ["a", "b"], source.body_indexes, "cpu") is None
    source.motion_store = object()
    with pytest.raises(ValueError, match="full-catalog"), share_motion_arrays(source):
        pass


def test_motion_inputs_are_mutually_exclusive():
    prefix = ["--variant", "preview", "--output-dir", "unused", "--iterations", "1"]
    args = build_parser().parse_args(prefix + ["--motion-path", "data", "--dr-profile", "hands-shins-2-4kg"])
    assert args.motion_file is None and args.motion_path == "data"
    with pytest.raises(SystemExit):
        build_parser().parse_args(prefix + ["--motion-path", "data", "--motion-file", "a.npz"])


def test_resume_rejects_payload_and_dataset_changes():
    from omegaconf import OmegaConf

    from intact_tracking.simulator_preview_experiment import EXPERIMENT_VERSION

    train = {key: {} for key in ("actor", "critic", "obs_groups", "algorithm")}
    meta = {"version": EXPERIMENT_VERSION, "variant": "preview", "tracker_sha256": "t",
            "motion_sha256": "m", "physics_mode": "dr", "motion_count": 2,
            "dr_profile": "hands-shins-2-4kg", "dataset": {"manifest_sha256": "m"},
            "arguments": {"num_envs": 4096, "seed": 121, "rollout_steps": 24, "initial_action_std": .1}}
    checkpoint = {"residual_policy": meta, "cfg": OmegaConf.create({"agent": train})}
    validate_resume(checkpoint, meta, train)
    with pytest.raises(ValueError, match="DR profile"):
        validate_resume(checkpoint, {**meta, "dr_profile": "right-hand-1-3kg"}, train)
    with pytest.raises(ValueError, match="dataset"):
        validate_resume(checkpoint, {**meta, "dataset": {"manifest_sha256": "other"}}, train)


def test_physics_profile_keeps_other_events_and_rewards(monkeypatch):
    import intact_tracking.cli.adaptation_eval as module

    monkeypatch.setattr(module, "_filter_disturbance_events", lambda cfg: [])
    monkeypatch.setattr(module, "_clear_missing_motion_exclusions", lambda cfg: [])
    reward = object()
    event = SimpleNamespace(func=object())
    cfg = SimpleNamespace(events={"armature": event}, commands={"motion": SimpleNamespace()},
                          rewards={"original": reward})
    configure_physics(cfg, "dr", "hands-shins-2-4kg")
    assert cfg.events["armature"] is event and cfg.rewards["original"] is reward
    assert len(cfg.events) == 5
