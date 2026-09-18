import numpy as np

from intact_tracking.moe_routing_diagnostics import (
    transition_masks, temporal_statistics, weighted_physics_statistics, geometry_statistics,
)


def example(routes):
    r = np.asarray(routes, dtype=np.uint8).reshape(-1, 1)
    return {"routes": r, "episode_ids": np.zeros_like(r, dtype=np.int32),
            "motion_ids": np.zeros_like(r, dtype=np.int32),
            "motion_steps": np.arange(len(r))[:, None],
            "short_count": np.full_like(r, 50), "long_count": np.full_like(r, 30)}


def test_actual_switches_are_counted_without_any_center_updates():
    trace = example([0, 1, 0, 1, 0])
    result, _, _ = temporal_statistics(trace, .02, start=0, k=2)
    assert result['rates']['steady_full_memory_within_motion']['switch_fraction'] == 1
    assert result['rates']['steady_full_memory_within_motion']['switches_per_second'] == 50


def test_reset_switches_are_excluded_from_within_motion_rate():
    trace = example([0, 0, 1, 1, 1])
    trace['episode_ids'][2:] = 1
    trace['motion_ids'][2:] = 8
    trace['motion_steps'][2:] -= 2
    changed, masks, _ = transition_masks(trace, start=0)
    assert (changed & masks['steady_boundary']).sum() == 1
    assert (changed & masks['steady_full_memory_within_motion']).sum() == 0
    trace['short_count'][3] = 10
    _, masks, _ = transition_masks(trace, start=0)
    assert masks['steady_full_memory_within_motion'].sum() == 1


def test_DR_separation_uses_range_units_and_exposure_weights():
    dr = np.asarray([[0., 4.], [0., 8.], [2., 4.], [2., 8.]])
    schema = {'lower': [0., 4.], 'upper': [2., 8.], 'groups': {'split': [0], 'mixed': [1]}}
    memberships = np.asarray([[10, 0], [10, 0], [0, 10], [0, 10]])
    result = weighted_physics_statistics(dr, memberships, schema)
    assert result['groups']['split']['variance_explained'] == 1
    assert result['groups']['mixed']['variance_explained'] == 0
    assert result['groups']['split']['within_std_over_global_std'] == 0
    assert result['groups']['mixed']['within_std_over_global_std'] == 1


def test_pair_geometry_excludes_self_world_and_preserves_physical_units():
    z = np.repeat(np.eye(2, 64), 4, axis=0)
    samples = {'latent': z, 'routes': np.repeat([0,1],4), 'worlds': np.repeat(np.arange(4),2),
               'raw_norm': np.full(8,2.), 'margin': np.ones(8), 'motion_ids': np.tile([1,2],4)}
    dr = np.zeros((4,9)); dr[2:,:] = 2.
    schema = {'lower': [0.]*9, 'upper': [2.]*9, 'groups': {'all': list(range(9))},
              'names': [str(i) for i in range(9)]}
    result = geometry_statistics(samples,dr,schema,np.eye(2,64),pair_count=10000)
    assert result['latent']['within_expert_different_worlds']['mean'] == 0
    np.testing.assert_allclose(result['latent']['between_experts_different_worlds']['mean'],np.sqrt(2))
    assert result['dr']['all']['within_expert_different_worlds']['mean'] == 0
    assert result['dr']['all']['between_experts_different_worlds']['mean'] == 1
    assert result['absolute_DR_differences']['between_experts_different_worlds']['0']['mean'] == 2
    assert result['same_world_different_motion']['different_expert_fraction'] == 0


def test_router_minimum_audit_accepts_equal_centers_but_rejects_wrong_route():
    import pytest
    import torch
    from intact_tracking.cli.moe_routing_diagnostic import audit_routed_minimum
    latent = torch.eye(1,64)
    centers = torch.cat((latent, latent, -latent))
    # Both indices 0 and 1 are valid minima. topk's tie order is irrelevant.
    for route in (0,1):
        _, margin, audit = audit_routed_minimum(latent,centers,torch.tensor([route]))
        assert margin.item() == 0
        assert audit['maximum_selected_distance_excess'] == 0
    with pytest.raises(RuntimeError,match='non-minimum'):
        audit_routed_minimum(latent,centers,torch.tensor([2]))
