from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from intact_tracking.limb_context_grouped_dr import (
    GROUPS, STATIC_EVENTS, Grid256LimbPayload, GroupedStaticRandomization,
    audit_grouped_ranks, configure_grid256_dr, group_ids, mass_table, sample_grid_masses,
    world_parameter_view,
)
from intact_tracking.limb_context_protocol import PAYLOAD_EVENT


def test_all_256_load_combinations_have_exact_per_gpu_and_global_replication():
    masses = sample_grid_masses(8192)
    unique, counts = torch.unique(masses, dim=0, return_counts=True)
    assert unique.shape == (256, 4)
    assert torch.equal(counts, torch.full_like(counts, 32))
    assert torch.equal(unique, mass_table())
    assert torch.equal(masses[:256], masses[256:512])
    assert (masses == 0).all(-1).sum() == (masses == 4).all(-1).sum() == 32
    assert (counts * 4 == 128).all()
    for limb in range(4):
        assert torch.unique(masses[:, limb], return_counts=True)[1].tolist() == [2048] * 4


def test_read_only_audit_handles_shared_and_expanded_mujoco_storage():
    shared = torch.arange(29, dtype=torch.float32)
    for value in (shared, shared[None]):
        view = world_parameter_view(value, 512, (29,))
        assert view.shape == (512, 29) and view.stride(0) == 0
        assert view.data_ptr() == shared.data_ptr()
    per_world = shared[None].repeat(512, 1)
    per_world[1, 0] = 7
    assert world_parameter_view(per_world, 512, (29,), expanded=True) is per_world
    with pytest.raises(ValueError, match="Unrecognized"):
        world_parameter_view(shared, 512, (29,), expanded=True)


@pytest.mark.parametrize("count", [0, 128, 255, 257, 8191])
def test_incomplete_load_grid_is_rejected(count):
    with pytest.raises(ValueError, match="multiple of 256"):
        group_ids(count)


def test_grouped_static_sampling_restores_rng_and_is_independent_of_rank_rng(monkeypatch):
    def original(env, ids, **kwargs):
        env.sim.model.body_mass[ids] = torch.rand(len(ids), 3)

    monkeypatch.setitem(STATIC_EVENTS, "base_mass", (original, ("body_mass",)))
    results = []
    for rank_seed in (121, 1000124):
        torch.manual_seed(rank_seed)
        model = SimpleNamespace(body_mass=torch.ones(512, 3))
        sim = SimpleNamespace(model=model, expanded_fields=GroupedStaticRandomization.model_fields)
        env = SimpleNamespace(device="cpu", num_envs=512, sim=sim)
        cfg = EventTermCfg(func=GroupedStaticRandomization, mode="startup",
                           params={"grouped_event": "base_mass", "grouped_seed": 301131})
        event = GroupedStaticRandomization(cfg, env)
        rng = torch.get_rng_state().clone()
        event(env, None)
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(model.body_mass[:256], model.body_mass[256:])
        assert torch.unique(model.body_mass, dim=0).shape[0] == GROUPS
        results.append(model.body_mass)
    assert torch.equal(*results)


def test_encoder_bias_is_replicated_as_well_as_physical_model_fields(monkeypatch):
    def original(env, ids, **kwargs):
        env.scene["robot"].data.encoder_bias[ids] = .02 * torch.rand(len(ids), 29) - .01

    monkeypatch.setitem(STATIC_EVENTS, "encoder_bias", (original, ()))
    env = SimpleNamespace(device="cpu", num_envs=768,
                          sim=SimpleNamespace(expanded_fields=GroupedStaticRandomization.model_fields),
                          scene={"robot": SimpleNamespace(data=SimpleNamespace(encoder_bias=torch.zeros(768, 29)))})
    cfg = EventTermCfg(func=GroupedStaticRandomization, mode="startup",
                       params={"grouped_event": "encoder_bias", "grouped_seed": 302140})
    event = GroupedStaticRandomization(cfg, env)
    event(env, None)
    bias = env.scene["robot"].data.encoder_bias
    assert torch.equal(bias[:256], bias[256:512]) and torch.equal(bias[:256], bias[512:])
    before = bias.clone()
    event.reset(torch.tensor([0, 300, 700]))
    assert torch.equal(before, bias)
    with pytest.raises(ValueError, match="startup only"):
        event(env, torch.tensor([0]))


def test_grid_configuration_preserves_motion_noise_pushes_and_physical_ranges():
    events = {name: EventTermCfg(func=function, mode="startup") for name, (function, _) in STATIC_EVENTS.items()}
    for name in ("base_com", "base_mass"):
        events[name].params = {"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                               "ranges": (-1, 1), "operation": "add"}
    def push(env, env_ids, **kwargs):
        pass
    events["push_robot"] = EventTermCfg(func=push, mode="step", params={"force_abs_range_n": (0, 10)})
    cfg = SimpleNamespace(events=events,
        observations={"policy": SimpleNamespace(enable_corruption=True)},
        commands={"motion": SimpleNamespace(rewind=SimpleNamespace(enabled=True))})
    metadata = configure_grid256_dr(cfg, 121)
    assert cfg.events["push_robot"].func is push
    assert cfg.events["push_robot"].params == {"force_abs_range_n": (0, 10)}
    assert cfg.observations["policy"].enable_corruption
    assert cfg.events[PAYLOAD_EVENT].func is Grid256LimbPayload
    assert metadata["grouped_dr"]["masses_by_profile_kg"] == mass_table().tolist()
    for name in STATIC_EVENTS:
        assert cfg.events[name].func is GroupedStaticRandomization
        assert cfg.events[name].mode == "startup"
    assert cfg.events["base_mass"].params["ranges"] == (-1, 1)


def test_cross_rank_audit_rejects_different_banks_and_counts():
    def row(bank="same", replicas=32):
        return {"physics": {"grouped_dr": {"static_bank_sha256": bank,
            "counts_per_profile": [replicas] * GROUPS, "replicas_per_profile_per_rank": replicas}}}
    result = audit_grouped_ranks([row() for _ in range(4)])
    assert result["replicas_per_profile_all_ranks"] == 128
    with pytest.raises(RuntimeError, match="different"):
        audit_grouped_ranks([row(), row("different")])
    with pytest.raises(RuntimeError, match="unequal"):
        audit_grouped_ranks([row(), row(replicas=16)])


def test_diagnostics_donors_separate_same_dr_from_cross_dr_without_adding_policy_inputs():
    from intact_tracking.memory350_compressed_policy import CompressedContextActor
    actor = torch.nn.Module()
    actor.residual_mlp = torch.nn.Linear(1, 1)
    CompressedContextActor.configure_grouped_diagnostics(actor, 8192)
    ids = group_ids(8192)
    assert torch.equal(ids, ids[actor._same_dr_donors])
    assert (ids != ids[actor._cross_dr_donors]).all()
    assert (actor._same_dr_donors != torch.arange(8192)).all()
    assert len(torch.unique(actor._cross_dr_donors)) == 8192
    assert not any("donors" in key for key in actor.state_dict())
