from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from intact_tracking.memory350_heavy_dr import (
    EVENT, INERTIAL_FIELDS, MASS_LIMITS, StratifiedLimbPayload,
    composite_inertial_fields, stratified_payload_samples,
)
from intact_tracking.memory350_native_dr import native_metric_schema, native_nominal_ids
from intact_tracking.preview_protocol import LIMBS


def test_mass_strata_are_balanced_across_all_eight_ranks_and_continuous():
    global_counts = torch.zeros(256, dtype=torch.long)
    for rank in range(8):
        mass, offsets, groups = stratified_payload_samples(16384, .1, rank, 717 + rank)
        nominal = native_nominal_ids(16384, .1)
        assert len(nominal) == 1639
        assert (groups[nominal] == -1).all()
        assert not mass[nominal].any() and not offsets[nominal].any()
        dr = groups >= 0
        counts = torch.bincount(groups[dr], minlength=256)
        assert counts.min() == 57 and counts.max() == 58
        global_counts += counts
        lower_bin = (groups[dr, None] // torch.tensor([64, 16, 4, 1])) % 4
        fraction = mass[dr] / torch.tensor(MASS_LIMITS) * 4 - lower_bin
        assert ((fraction >= 0) & (fraction < 1)).all()
        assert torch.unique(mass[dr], dim=0).shape[0] == 14745
        assert (fraction.mean(0) - .5).abs().max() < .02
        assert offsets[dr].abs().max() <= .05
        # Final xyz are independent cube draws, not ball rejection samples.
        assert offsets[dr].norm(dim=-1).max() > .08
        xyz = offsets[dr].reshape(-1, 3)
        correlation = torch.corrcoef(xyz.T) - torch.eye(3)
        assert correlation.abs().max() < .02
    assert int(global_counts.sum()) == 117960
    assert int((global_counts == 460).sum()) == 56
    assert int((global_counts == 461).sum()) == 200


def make_base(n):
    quat = torch.tensor([.7, .2, -.1, .3]).expand(n, 4, 4).clone()
    quat /= quat.norm(dim=-1, keepdim=True)
    return {'body_mass': torch.full((n, 4), 1.4),
            'body_ipos': torch.tensor([.02, -.01, .04]).expand(n, 4, 3).clone(),
            'body_inertia': torch.tensor([.02, .03, .04]).expand(n, 4, 3).clone(),
            'body_iquat': quat}


def parallel_axis(mass, offset):
    return mass[..., None, None] * (offset.square().sum(-1)[..., None, None] * torch.eye(3)
                                    - offset[..., :, None] * offset[..., None, :])


def test_composite_fields_match_mass_weighted_com_and_full_parallel_axis_tensor():
    base = make_base(512)
    mass, offset, group = stratified_payload_samples(512, .1, 0, 717)
    combined = composite_inertial_fields(base, mass, offset)
    pos = torch.tensor([row[1] for row in LIMBS.values()]) + offset
    total = base['body_mass'] + mass
    center = (base['body_mass'][..., None] * base['body_ipos'] + mass[..., None] * pos) / total[..., None]
    torch.testing.assert_close(combined['body_mass'], total)
    torch.testing.assert_close(combined['body_ipos'], center)
    sizes = torch.tensor([row[2] for row in LIMBS.values()])
    cuboid = mass[..., None] * (sizes.square().sum(-1, keepdim=True) - sizes.square()) / 12
    rotation = matrix_from_quat(base['body_iquat'])
    expected = (rotation @ torch.diag_embed(base['body_inertia']) @ rotation.mT
                + torch.diag_embed(cuboid)
                + parallel_axis(base['body_mass'], base['body_ipos'] - center)
                + parallel_axis(mass, pos - center))
    rotation = matrix_from_quat(combined['body_iquat'])
    actual = rotation @ torch.diag_embed(combined['body_inertia']) @ rotation.mT
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    assert (torch.linalg.eigvalsh(actual) > 0).all()
    for name in INERTIAL_FIELDS:
        assert torch.equal(combined[name][group < 0], base[name][group < 0])


def test_event_does_not_accumulate_payload_and_labels_match_audited_physics():
    n = 512
    model = SimpleNamespace(**make_base(n))
    asset = SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.arange(4)),
                            find_bodies=lambda *a, **k: (list(range(4)), list(LIMBS)))
    env = SimpleNamespace(num_envs=n, device='cpu', scene={'robot': asset},
        sim=SimpleNamespace(model=model, expanded_fields=StratifiedLimbPayload.model_fields))
    cfg = SimpleNamespace(params={'nominal_fraction': .1, 'rank': 0, 'seed': 717,
                                 'max_masses_kg': MASS_LIMITS, 'com_half_width_m': .05})
    event = StratifiedLimbPayload(cfg, env)
    event(env, None)
    first = {name: getattr(model, name).clone() for name in INERTIAL_FIELDS}
    event(env, torch.arange(0, n, 2))
    event(env, slice(1, None, 2))
    for name in INERTIAL_FIELDS:
        assert torch.equal(first[name], getattr(model, name))
    audit = event.audit()
    assert audit['nominal_inertial_fields_exact']
    labels, values = event.privileged_dynamics_targets()
    assert len(labels) == 16 and values.shape == (n, 16)
    torch.testing.assert_close(values[:, :4], model.body_mass - event.base['body_mass'])
    model.body_ipos[1, 0, 0] += .01
    with pytest.raises(AssertionError):
        event.audit()


def test_heavy_metric_is_108_coordinates_and_16_balanced_factors():
    names = [f'base_com/com_offset/torso_link/{a}' for a in 'xyz']
    names += ['base_mass/relative_mass/torso_link', 'foot_friction/friction/shared/0']
    names += [f'motor_params_implicit/{kind}/joint_{j}'
              for kind in ('kp_scale', 'kd_scale', 'armature_scale') for j in range(29)]
    names += [f'{EVENT}/added_mass_kg/{limb}' for limb in LIMBS]
    names += [f'{EVENT}/payload_com_offset/{limb}/{a}' for limb in LIMBS for a in 'xyz']
    params = {'base_com': {'ranges': {i: [-.075, .075] for i in range(3)}},
              'base_mass': {'ranges': [-1., 1.]}, 'foot_friction': {'ranges': [.3, 2.]},
              'motor_params_implicit': {k: {'.*': [.8, 1.2]} for k in
                  ('stiffness_range', 'damping_range', 'armature_range')},
              EVENT: {'max_masses_kg': MASS_LIMITS, 'com_half_width_m': .05}}
    schema = native_metric_schema(names, params, {'torso_link': 10.})
    assert len(schema['names']) == 108 and len(schema['groups']) == 16
    for columns in schema['groups'].values():
        assert sum(schema['coordinate_weights'][i] for i in columns) == pytest.approx(1 / 16)
    assert schema['lower'][-12:] == [-.05] * 12
    assert schema['upper'][-16:-12] == list(MASS_LIMITS)
