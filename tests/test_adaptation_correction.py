from types import SimpleNamespace

import pytest
import torch
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict
from torch import nn

import intact_tracking.residual_policy as residual_module
from intact_tracking.adaptation_correction import (
    CORRECTION_PHYSICS,
    ContextPhysicsCorrectionActor,
    StaticPhysicsCorrectionWrapper,
    correction_config_requires_privilege,
    correction_initialization_state,
)
from intact_tracking.adaptation_policy import ContextAdaptationActor
from intact_tracking.context_export import ContextInferenceModule


class FakeTracker(nn.Module):
    def __init__(self, obs, groups, obs_set, output_dim, **kwargs):
        super().__init__()
        self.policy_input_dim = 3
        self.mlp = nn.Linear(3, output_dim)
        self.distribution = GaussianDistribution(output_dim, init_std=0.3)
        self.history_normalizer = nn.Identity()

    def get_latent(self, obs):
        return obs["sensor"]

    def populate_policy_context_cache(self, obs):
        pass


def test_static_correction_preserves_initial_source_freezes_it_and_has_no_student_privileges(tmp_path, monkeypatch):
    monkeypatch.setattr(residual_module, "SPV52HeightContactEstimatorActor", FakeTracker)
    path = tmp_path / "tracker.pt"
    torch.save({"actor_state_dict": FakeTracker(None, None, None, 2).state_dict()}, path)
    obs = TensorDict({"sensor": torch.randn(4, 3), "estimator_history": torch.randn(4, 6100),
                      CORRECTION_PHYSICS: torch.randn(4, 3)}, [4])
    groups = {"actor": ["sensor", "estimator_history"]}
    kwargs = dict(tracker_checkpoint=str(path), tracker_actor_kwargs={}, tracker_obs_groups=groups,
                  use_dynamics_latent=False, residual_hidden_dims=(16, 8), adaptation_latent_dim=4,
                  context_auxiliary_dim=10, train_base_policy=False, train_context_encoder=False,
                  train_residual_policy=False)
    source = ContextAdaptationActor(obs, groups, "actor", 2, **kwargs).eval()
    teacher = ContextPhysicsCorrectionActor(obs, groups, "actor", 2, **kwargs).eval()
    teacher.load_state_dict(correction_initialization_state(teacher, source.state_dict()), strict=True)
    torch.testing.assert_close(teacher(obs), source(obs), atol=0, rtol=0)
    for key, value in source.context_auxiliary_head.state_dict().items():
        torch.testing.assert_close(value, teacher.correction_physics_decoder.state_dict()[key], atol=0, rtol=0)
    before = {key: value.clone() for key, value in teacher.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in teacher.parameters() if p.requires_grad], lr=0.001)
    optimizer.zero_grad()
    (teacher(obs) - 1).square().mean().backward()
    optimizer.step()
    changed = [key for key, value in teacher.state_dict().items() if not torch.equal(value, before[key])]
    assert changed and all(key.startswith("correction_mlp.") for key in changed)
    with pytest.raises(KeyError):
        teacher(obs.select("sensor", "estimator_history"))
    with pytest.raises(ValueError, match="true-physics"):
        ContextInferenceModule(teacher)
    student = ContextPhysicsCorrectionActor(obs, groups, "actor", 2, correction_use_privilege=False, **kwargs).eval()
    student.load_state_dict(teacher.state_dict(), strict=True)
    expected = student(obs.select("sensor", "estimator_history"))
    poisoned = obs.clone()
    poisoned[CORRECTION_PHYSICS].fill_(torch.nan)
    torch.testing.assert_close(student(poisoned), expected, atol=0, rtol=0)
    latent = student.instant_context_latent(obs)
    torch.testing.assert_close(student.estimated_correction_physics(latent),
                               student.context_auxiliary_head(latent)[..., [3, 5, 6]].clamp(-1, 1))
    cfg = dict(class_name="intact_tracking.adaptation_correction:ContextPhysicsCorrectionActor")
    assert correction_config_requires_privilege(cfg)
    assert not correction_config_requires_privilege(dict(cfg, correction_use_privilege=False))


def test_static_wrapper_freshly_reads_physics_without_state_labels():
    defaults = dict(body_mass=torch.tensor([0.5, 2.0]), body_ipos=torch.zeros(2, 3))
    model = SimpleNamespace(body_mass=torch.tensor([[1.5, 2.0], [3.5, 2.0]]),
                            body_ipos=torch.tensor([[[0.0, 0, 0], [0.075, -0.075, 0]],
                                                   [[0.0, 0, 0], [0.0, 0.0, 0]]]))
    env = SimpleNamespace(num_envs=2, sim=SimpleNamespace(model=model, get_default_field=defaults.__getitem__))
    wrapper = object.__new__(StaticPhysicsCorrectionWrapper)
    wrapper.wrapped = SimpleNamespace(unwrapped=env)
    wrapper.body_ids = [0, 1]
    obs = TensorDict({}, [2])
    first = wrapper.attach(obs)[CORRECTION_PHYSICS].clone()
    torch.testing.assert_close(first, torch.tensor([[-1., 1., -1.], [1., 0., 0.]]))
    model.body_mass[:, 0] += 0.25
    current = wrapper.attach(obs)[CORRECTION_PHYSICS]
    torch.testing.assert_close(current[:, 0], first[:, 0] + 0.25)


def test_failure_cost_excludes_timeouts():
    from intact_tracking.adaptation_rewards import failure_event

    env = SimpleNamespace(termination_manager=SimpleNamespace(
        terminated=torch.tensor([True, False, False]), time_outs=torch.tensor([False, True, False])))
    torch.testing.assert_close(failure_event(env), torch.tensor([1., 0., 0.]))
