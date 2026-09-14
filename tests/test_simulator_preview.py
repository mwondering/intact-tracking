from types import SimpleNamespace

import torch
from tensordict import TensorDict

from intact_tracking.adaptation_policy import PRIVILEGE
from intact_tracking.residual_policy import DecayVecNorm, WarmStartedHeftCritic, _make_heft_mlp
from intact_tracking.simulator_preview import (
    HORIZON,
    OUTCOME_DIM,
    REFERENCE_DIM,
    PreviewAwareHeftCritic,
    copy_runtime,
    preview_features,
)


def example():
    generator = torch.Generator().manual_seed(10)
    target = torch.randn(3, HORIZON, 71, generator=generator)
    target[..., 3:7] /= target[..., 3:7].norm(dim=-1, keepdim=True)
    state = target.clone()
    initial = target[:, 0].clone()
    body = torch.zeros(3, HORIZON)
    return state, target, initial, body


def test_five_step_features_have_fixed_layout_and_zero_matching_error():
    states, targets, initial, body = example()
    reference, outcomes = preview_features(states, targets, initial, body)
    assert reference.shape == (3, REFERENCE_DIM)
    assert outcomes.shape == (3, OUTCOME_DIM)
    torch.testing.assert_close(outcomes.reshape(3, 5, 144)[..., 71:], torch.zeros(3, 5, 73))


def test_features_are_translation_invariant_and_do_not_modify_inputs():
    inputs = example()
    saved = tuple(value.clone() for value in inputs)
    expected = preview_features(*inputs)
    for value, before in zip(inputs, saved, strict=True):
        assert torch.equal(value, before)
    for value in inputs[:3]:
        value[..., :2] += torch.tensor([12.0, -9.0])
    actual = preview_features(*inputs)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


def test_quaternion_sign_is_not_reported_as_tracking_error():
    states, targets, initial, body = example()
    states[..., 3:7] *= -1
    _, outcomes = preview_features(states, targets, initial, body)
    assert outcomes.reshape(3, 5, 144)[..., 71:142].abs().max() == 0


def test_copy_runtime_preserves_object_ownership_and_copies_history_cursor():
    source = SimpleNamespace(buffer=torch.randn(3, 5, 7), pointer=4, _env=object(), uninitialized=None)
    target = SimpleNamespace(buffer=torch.zeros(3, 5, 7), pointer=0, _env=object(), uninitialized=torch.ones(1))
    owned_env = target._env
    copy_runtime(source, target)
    assert target._env is owned_env and target._env is not source._env
    assert target.pointer == 4 and target.uninitialized is None
    assert target.buffer.data_ptr() != source.buffer.data_ptr()
    assert torch.equal(target.buffer, source.buffer)
    target.buffer.add_(1)
    assert not torch.equal(target.buffer, source.buffer)


def test_preview_critic_initial_identity_and_learning_from_same_state_pairs(tmp_path):
    torch.manual_seed(13)
    source = torch.nn.Module()
    source.obs_normalizer = DecayVecNorm(5)
    source.mlp = _make_heft_mlp(5, (8, 6), 1)
    path = tmp_path / "value.pt"
    torch.save({"critic_state_dict": source.state_dict()}, path)
    # Identical current state; only the shared privileged observation differs.
    obs = TensorDict({"state": torch.zeros(16, 5), PRIVILEGE: torch.randn(16, 7)}, [16])
    groups = {"actor": ["state"], "critic": ["state", PRIVILEGE]}
    critic = PreviewAwareHeftCritic(obs, groups, "critic", 1,
                                   initial_checkpoint=str(path), hidden_dims=(8, 6))
    baseline = WarmStartedHeftCritic(obs, {"critic": ["state"]}, "critic", 1,
                                    initial_checkpoint=str(path), hidden_dims=(8, 6))
    torch.testing.assert_close(critic(obs), baseline(obs), atol=0, rtol=0)
    assert critic.preview_value[0].in_features == 12
    assert groups["critic"] == ["state", PRIVILEGE]
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.003)
    target = obs[PRIVILEGE][:, :1].clone()
    initial_loss = float((critic(obs) - target).square().mean().detach())
    for _ in range(12):
        optimizer.zero_grad()
        (critic(obs) - target).square().mean().backward()
        optimizer.step()
    assert float((critic(obs) - target).square().mean().detach()) < initial_loss
    shuffled = obs.clone(recurse=True)
    shuffled[PRIVILEGE] = obs[PRIVILEGE].roll(1, 0)
    assert not torch.equal(critic(obs), critic(shuffled))
    differentiated = obs.clone(recurse=True)
    differentiated[PRIVILEGE].requires_grad_(True)
    gradient = torch.autograd.grad(critic(differentiated).sum(), differentiated[PRIVILEGE])[0]
    assert gradient.abs().sum() > 0
