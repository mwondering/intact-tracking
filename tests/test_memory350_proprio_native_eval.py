from types import SimpleNamespace
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from intact_tracking.cli import memory350_proprio_native_policy_eval as evaluation


def test_native_eval_has_no_extra_payload_and_fixed_start_sampling(monkeypatch):
    cfg = SimpleNamespace(commands={"motion": SimpleNamespace(
        sampling_mode="adaptive", rewind=SimpleNamespace(enabled=True))})
    monkeypatch.setattr(evaluation, "configure_proprio_physics", lambda *a, **kw: {"extra_payload": False})
    result = evaluation.configure_evaluation_physics(cfg, 1, profile=evaluation.native.PROFILE)
    assert not result["extra_payload"]
    assert cfg.commands["motion"].sampling_mode == "uniform"
    assert not cfg.commands["motion"].rewind.enabled
    with pytest.raises(ValueError, match="no additional"):
        evaluation.configure_evaluation_physics(cfg, 1, profile=evaluation.native.PROFILE, fixed_masses=[0]*4)


def test_query_inactive_resets_do_not_change_other_worlds_force_random_stream(monkeypatch):
    resets = []
    source = SimpleNamespace(reset=lambda env_ids=None: resets.append(env_ids))
    env = SimpleNamespace(event_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(
        func=SimpleNamespace(source=source))))
    monkeypatch.setattr(evaluation.native, "environment_factory", lambda *a, **kw: env)
    assert evaluation.paired_environment(None) is env
    source.reset([1])
    assert resets == [[1]]
    env.native_paired_query = True
    source.reset([2])
    assert resets == [[1]]


def test_physics_fingerprint_includes_controller_and_nominal_ids(monkeypatch):
    action = SimpleNamespace(delay=torch.zeros(2, 1), alpha=torch.ones(2, 1),
        joint_offset=torch.zeros(2, 29), boot_delay=torch.zeros(2, 1))
    env = SimpleNamespace(num_envs=2, device="cpu", native_policy_nominal_ids=torch.tensor([0]),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(encoder_bias=torch.zeros(2, 29)))},
        action_manager=SimpleNamespace(get_term=lambda _: action))
    monkeypatch.setattr(evaluation, "_capture_privileged_dynamics_targets",
                        lambda _: SimpleNamespace(values=torch.zeros(2, 5)))
    first = evaluation.evaluation_world_metadata(env)
    assert first["is_nominal"] == [True, False] and not first["extra_payload"]
    assert first["physics_world_fingerprints"][0] == first["physics_world_fingerprints"][1]
    action.delay[1] = 2
    second = evaluation.evaluation_world_metadata(env)
    assert first["physics_world_fingerprints"][0] == second["physics_world_fingerprints"][0]
    assert first["physics_world_fingerprints"][1] != second["physics_world_fingerprints"][1]


def test_partial_reset_does_not_redraw_surviving_proprio_or_advance_noise_rng(monkeypatch):
    calls = []
    env = SimpleNamespace(common_step_counter=5, observation_manager=SimpleNamespace(
        compute=lambda: calls.append(1) or {"estimator_history": torch.randn(2, 6100)}))
    monkeypatch.setattr(evaluation.EvaluationWrapper, "unwrapped", property(lambda _: env))
    wrapper = object.__new__(evaluation.EvaluationWrapper)
    wrapper._sensor_step, wrapper._sensor_frame = None, None
    first = wrapper._context_state().clone()
    rng = torch.random.get_rng_state().clone()
    # No control step has elapsed: a cleared observation cache must not cause
    # an extra compute/noise draw before the next action is simulated.
    torch.testing.assert_close(wrapper._context_state(), first, atol=0, rtol=0)
    assert calls == [1] and torch.equal(rng, torch.random.get_rng_state())
    env.common_step_counter += 1
    assert not torch.equal(wrapper._context_state(), first)
    assert calls == [1, 1]


def test_force_audit_accepts_early_failure_but_rejects_changed_common_forces():
    spec = importlib.util.spec_from_file_location(
        "compare_native_tracking", Path(__file__).resolve().parents[1] / "scripts/compare_native_tracking.py")
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)

    def diagnostics(forces):
        wrapper = object.__new__(evaluation.EvaluationWrapper)
        wrapper.query_forces = list(forces)
        return wrapper.evaluation_diagnostics

    forces = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3)
    full, short = diagnostics(forces), diagnostics(forces[:2])
    assert full["query_force_sha256"] == hashlib.sha256(forces.numpy().tobytes()).hexdigest()
    assert comparator.audit_query_forces(full, short, 2) == (2, False)
    changed = forces[:2].clone()
    changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="Unmatched forces"):
        comparator.audit_query_forces(full, diagnostics(changed), 2)
    with pytest.raises(ValueError, match="does not cover"):
        comparator.audit_query_forces(full, short, 3)
    legacy_full = {k: v for k, v in full.items() if k != "query_force_prefix_sha256"}
    legacy_short = {k: v for k, v in short.items() if k != "query_force_prefix_sha256"}
    assert comparator.audit_query_forces(legacy_full, legacy_full, 4) == (4, True)
    with pytest.raises(ValueError, match="require force prefix"):
        comparator.audit_query_forces(legacy_full, legacy_short, 2)
