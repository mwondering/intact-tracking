import copy
import re
from types import SimpleNamespace

import pytest
import torch

from intact_tracking.environment.assets.robots.safety import get_safe_g1_articulation
from intact_tracking.limb_context_protocol import sample_limb_masses
from intact_tracking.residual_uniform_protocol import (
    MAX_MASSES, WRIST_JOINTS, sample_masses, wrist_articulation,
    representative_profiles,
)
from intact_tracking.cli.residual_uniform_train import build_parser

JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def test_uniform_loads_have_requested_marginals_and_preserve_old_experiments():
    before = torch.get_rng_state().clone()
    mass = sample_masses(20000, 121)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(mass, sample_masses(20000, 121))
    torch.testing.assert_close(mass.mean(0), torch.tensor(MAX_MASSES) / 2, atol=.03, rtol=0)
    assert (mass >= 0).all() and (mass < torch.tensor(MAX_MASSES)).all()
    legacy = sample_limb_masses(20000, 121)
    assert (legacy[:, :2] > 2.5).any()
    torch.testing.assert_close(legacy[:, 2:], mass[:, 2:], atol=0, rtol=0)


@pytest.mark.parametrize("mass", [[4, 0, 0, 0], [0, 2.51, 0, 0], [0, 0, 4.1, 0], [0, 0, 0, float("nan")]])
def test_invalid_fixed_payload_is_rejected(mass):
    with pytest.raises(ValueError, match="Payload"):
        sample_masses(8, 1, mass)


def test_fixed_profiles_use_the_new_asymmetric_caps():
    profiles = representative_profiles()["profiles"]
    assert len(profiles) == 8
    assert profiles[3]["masses_kg"] == [2.5, 2.5, 4.0, 4.0]
    for row in profiles:
        torch.testing.assert_close(sample_masses(13, 3, row["masses_kg"]), torch.tensor(row["masses_kg"]).float().expand(13, 4))


def test_wrist_force_edit_preserves_all_other_joint_properties():
    old = get_safe_g1_articulation()
    snapshot = copy.deepcopy(old)
    new = wrist_articulation(old)
    assert old == snapshot
    for before, after in zip(old.actuators, new.actuators, strict=True):
        wrist = any(re.fullmatch(p, j) for p in before.target_names_expr for j in WRIST_JOINTS)
        expected = copy.deepcopy(before)
        if wrist:
            assert before.effort_limit == 5
            expected.effort_limit = 10.0
        assert after == expected


def test_new_cli_cannot_silently_switch_to_adaptive_or_raw_scalar_bounds(tmp_path):
    parser = build_parser()
    argv = ["--fusion", "baseline", "--output-dir", str(tmp_path)]
    args = parser.parse_args(argv)
    assert args.motion_sampling == "uniform" and args.adaptive_after_update == 0
    for flags in (["--motion-sampling", "adaptive"], ["--residual-scale", ".25"], ["--residual-torque-limit-nm", "20"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv + flags)


def test_capped_payload_composite_inertia_is_idempotent():
    from intact_tracking.residual_uniform_protocol import CappedLimbPayload
    n = 16
    model = SimpleNamespace(body_mass=torch.full((1, 4), 2.), body_ipos=torch.zeros(1, 4, 3),
                            body_inertia=torch.full((1, 4, 3), .1), body_iquat=torch.zeros(1, 4, 4))
    model.body_iquat[..., 0] = 1
    sim = SimpleNamespace(model=model, expanded_fields=set())

    def expand(fields):
        for name, value in vars(model).items():
            setattr(model, name, value.expand(n, *value.shape[1:]).clone())
        sim.expanded_fields.update(fields)
    sim.expand_model_fields = expand
    robot = SimpleNamespace(find_bodies=lambda names, **kwargs: (list(range(4)), names),
                            indexing=SimpleNamespace(body_ids=torch.arange(4)))
    env = SimpleNamespace(sim=sim, scene={"robot": robot}, num_envs=n, device="cpu")
    payload = CappedLimbPayload(SimpleNamespace(params={"seed": 121}), env)
    payload(env, None)
    torch.testing.assert_close(payload.observe(), payload.mass, atol=1e-6, rtol=1e-6)
    first = {name: value.clone() for name, value in vars(model).items()}
    payload(env, None)
    for name, expected in first.items():
        torch.testing.assert_close(getattr(model, name), expected, atol=0, rtol=0)
    assert (model.body_inertia > 0).all()
