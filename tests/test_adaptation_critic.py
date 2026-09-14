from __future__ import annotations

from types import SimpleNamespace

import torch
from tensordict import TensorDict

import intact_tracking.adaptation_critic as module
from intact_tracking.residual_policy import DecayVecNorm, WarmStartedHeftCritic, _make_heft_mlp


def test_physics_value_starts_at_exact_source_then_uses_physics(tmp_path):
    source = torch.nn.Module()
    source.obs_normalizer = DecayVecNorm(5)
    source.obs_normalizer.update(torch.randn(20, 5))
    source.mlp = _make_heft_mlp(5, (8, 6), 1)
    path = tmp_path / "critic.pt"
    torch.save({"critic_state_dict": source.state_dict()}, path)
    obs = TensorDict({"state": torch.randn(10, 5), module.CRITIC_PHYSICS: torch.randn(10, 7)}, [10])
    groups = {"actor": ["state"], "critic": ["state", module.CRITIC_PHYSICS]}
    value = module.PhysicsAwareHeftCritic(obs, groups, "critic", 1, initial_checkpoint=str(path), hidden_dims=(8, 6))
    baseline = WarmStartedHeftCritic(obs, {"critic": ["state"]}, "critic", 1,
                                    initial_checkpoint=str(path), hidden_dims=(8, 6))
    torch.testing.assert_close(value(obs), baseline(obs), atol=0, rtol=0)
    assert groups["actor"] == ["state"]
    assert groups["critic"] == ["state", module.CRITIC_PHYSICS]
    optimizer = torch.optim.Adam(value.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        (value(obs) - obs[module.CRITIC_PHYSICS][:, :1]).square().mean().backward()
        optimizer.step()
    shuffled = obs.clone(recurse=True)
    shuffled[module.CRITIC_PHYSICS] = obs[module.CRITIC_PHYSICS].roll(1, 0)
    assert not torch.equal(value(obs), value(shuffled))


def test_critic_wrapper_keeps_reward_done_and_action_objects_unchanged(monkeypatch):
    actions, reward, done, extras = object(), object(), object(), object()
    env = object()
    seen = []

    def step(value):
        assert value is actions
        return TensorDict({"actor": torch.ones(2, 3)}, [2]), reward, done, extras

    def physics(environment):
        assert environment is env
        seen.append(environment)
        return torch.randn(2, 7)

    monkeypatch.setattr(module, "physics_observation", physics)
    wrapper = module.PhysicsCriticWrapper(SimpleNamespace(unwrapped=env, step=step))
    obs, r, d, e = wrapper.step(actions)
    assert r is reward and d is done and e is extras
    assert module.CRITIC_PHYSICS in obs
    torch.testing.assert_close(obs["actor"], torch.ones(2, 3), atol=0, rtol=0)
    wrapper.step(actions)
    assert len(seen) == 2
