from __future__ import annotations

import pytest
import torch

from intact_tracking.adaptation_modulation import (
    LatentModulatedResidual,
    PhysicsBilinearResidual,
    PhysicsLowRankResidual,
)
from intact_tracking.residual_policy import _last_linear


def test_modulation_starts_at_zero_residual_and_identity_hidden_conditioning():
    model = LatentModulatedResidual(9, 3, (8, 6), 2)
    value = torch.randn(7, 9)
    assert torch.count_nonzero(model(value)) == 0
    assert _last_linear(model) is model.output
    layer = model.layers[0]
    expected = torch.nn.functional.elu(layer.linear(value))
    torch.testing.assert_close(layer(value, value[:, -2:]), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="width"):
        model(torch.randn(7, 10))


def test_modulation_learns_and_scripts_without_any_extra_input():
    model = LatentModulatedResidual(9, 3, (8, 6), 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    value = torch.randn(7, 9)
    target = value[:, -1:].expand(-1, 3)
    for _ in range(4):
        optimizer.zero_grad()
        (model(value) - target).square().mean().backward()
        optimizer.step()
    assert model.layers[0].condition.weight.count_nonzero() > 0
    altered = value.clone()
    altered[:, -2:] = altered[:, -2:].roll(1, 0)
    assert not torch.equal(model(value), model(altered))
    scripted = torch.jit.script(model)
    torch.testing.assert_close(scripted(value), model(value), atol=0, rtol=0)


def test_bilinear_zero_nominal_identity_survives_arbitrary_training_weights():
    model = PhysicsBilinearResidual(12, 3, (8, 6), 7)
    value = torch.randn(9, 12)
    assert model(value).count_nonzero() == 0
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    value[:, -7:] = 0
    assert model(value).count_nonzero() == 0
    value[:, -7:] = torch.randn(9, 7)
    assert model(value).count_nonzero() > 0
    double_code = value.clone()
    double_code[:, -7:] *= 2
    torch.testing.assert_close(model(double_code), 2 * model(value))
    torch.testing.assert_close(torch.jit.script(model)(value), model(value), atol=0, rtol=0)


def test_compact_physics_code_is_nominal_zero_and_fresh_per_world():
    from types import SimpleNamespace

    from intact_tracking.adaptation_policy import compact_physics_observation

    default = {
        "body_mass": torch.tensor([0.5, 5.0]), "body_ipos": torch.zeros(2, 3),
        "geom_friction": torch.tensor([[1.0, 0.1, 0.1], [1.0, 0.1, 0.1]]),
        "dof_armature": torch.tensor([0.0, 0.1, 0.2]),
    }
    actual = SimpleNamespace(**{name: value.unsqueeze(0).repeat(3, *([1] * value.ndim)) for name, value in default.items()})
    model = SimpleNamespace(
        nbody=2, ngeom=2,
        body=lambda i: SimpleNamespace(name=("right_wrist_yaw_link", "torso_link")[i]),
        geom=lambda i: SimpleNamespace(name=("left_foot", "right_foot")[i]),
    )
    env = SimpleNamespace(num_envs=3, sim=SimpleNamespace(model=actual, mj_model=model, get_default_field=lambda name: default[name]))
    assert compact_physics_observation(env).count_nonzero() == 0
    actual.body_mass[0, 0] += 1.5
    actual.body_ipos[1, 1, 1] = 0.0375
    actual.dof_armature[2, 1:] *= 1.1
    code = compact_physics_observation(env)
    expected = torch.zeros(3, 7)
    expected[0, 0] = expected[1, 3] = expected[2, 6] = 0.5
    torch.testing.assert_close(code, expected)


def test_low_rank_modulation_preserves_source_and_nominal_identity_after_learning():
    from rsl_rl.modules import MLP

    # RSL-RL reuses the SAME activation object across hidden layers. This is
    # intentionally not a handcrafted Sequential with distinct ELU objects.
    base = MLP(5, 3, (8, 6), "elu")
    model = PhysicsLowRankResidual(base, 7)
    value = torch.randn(9, 12)
    assert model(value).count_nonzero() == 0
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(4):
        optimizer.zero_grad()
        (model(value) - 1).square().mean().backward()
        optimizer.step()
    changed = [k for k, v in before.items() if not torch.equal(v, model.state_dict()[k])]
    assert changed and all(".down." in k or ".up." in k for k in changed)
    value[:, -7:] = 0
    assert model(value).count_nonzero() == 0
    value[:, -7:] = torch.randn(9, 7)
    assert model(value).count_nonzero() > 0
    torch.testing.assert_close(torch.jit.script(model)(value), model(value), atol=0, rtol=0)
    model.requires_grad_(True)
    model.freeze_reference()
    assert not any(p.requires_grad for p in model.reference.parameters())
    assert not any(p.requires_grad for layer in model.layers for p in layer.base.parameters())


def test_low_rank_reference_preserves_feature_adapter_input_gradients():
    from rsl_rl.modules import MLP

    model = PhysicsLowRankResidual(MLP(5, 3, (8, 6), "elu"), 7)
    features = torch.randn(4, 5, requires_grad=True)
    code = torch.randn(4, 7)
    value = model(torch.cat((features, code), -1))
    assert value.count_nonzero() == 0
    gradient, = torch.autograd.grad(value.sum(), features)
    # Equal branches have not just equal values but exactly canceling input
    # derivatives. A detached reference would incorrectly train the frontend.
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.reference.parameters())


def test_actuator_physics_code_preserves_individual_joints_without_current_state(monkeypatch):
    from types import SimpleNamespace

    import intact_tracking.adaptation_policy as policy

    monkeypatch.setattr(policy, "compact_physics_observation", lambda env: torch.zeros(env.num_envs, 7))
    default = torch.cat((torch.zeros(6), torch.linspace(0.1, 0.4, 29)))
    actual = default[None].repeat(3, 1)
    bias = torch.zeros(3, 29)
    robot = SimpleNamespace(indexing=SimpleNamespace(joint_v_adr=torch.arange(6, 35)),
                            data=SimpleNamespace(encoder_bias=bias))
    env = SimpleNamespace(num_envs=3, scene={"robot": robot},
                          sim=SimpleNamespace(model=SimpleNamespace(dof_armature=actual),
                                              get_default_field=lambda name: default))
    assert policy.actuator_physics_observation(env).count_nonzero() == 0
    actual[1, 10] *= 1.1
    bias[2, 5] = 0.005
    code = policy.actuator_physics_observation(env)
    expected = torch.zeros(3, 64)
    expected[1, 10] = expected[2, 40] = 0.5
    torch.testing.assert_close(code, expected)
    bias[2, 5] = -0.005
    assert policy.actuator_physics_observation(env)[2, 40] == -0.5


def test_shared_low_rank_can_learn_without_any_physical_information():
    from rsl_rl.modules import MLP

    model = PhysicsLowRankResidual(MLP(5, 3, (8, 6), "elu"), 7, 2, shared_rank=4)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    value = torch.cat((torch.randn(9, 5), torch.zeros(9, 7)), -1)
    assert model(value).count_nonzero() == 0
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(4):
        optimizer.zero_grad()
        (model(value) - 1).square().mean().backward()
        optimizer.step()
    changed = [key for key, val in model.state_dict().items() if not torch.equal(val, before[key])]
    assert changed and all(".shared_down." in key or ".shared_up." in key for key in changed)
    assert model(value).count_nonzero() > 0
    torch.testing.assert_close(torch.jit.script(model)(value), model(value), atol=0, rtol=0)
