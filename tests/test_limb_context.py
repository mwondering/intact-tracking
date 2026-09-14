import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tensordict import TensorDict

from intact_tracking.limb_context_policy import ConditionedMLP, LimbContextCritic, LimbContextResidualActor
from intact_tracking.limb_context_protocol import UniformLimbPayload, sample_limb_masses
from intact_tracking.residual_policy import DecayVecNorm, _make_heft_mlp
from test_residual_policy import _FakeTracker, _fake_actor_checkpoint


def test_independent_uniform_masses_and_rng_isolation():
    before = torch.get_rng_state().clone()
    masses = sample_limb_masses(20000, 121)
    assert torch.equal(before, torch.get_rng_state())
    assert (masses >= 0).all() and (masses < 4).all()
    assert torch.max((masses.mean(0) - 2).abs()) < .04
    assert torch.max((torch.corrcoef(masses.T) - torch.eye(4)).abs()) < .025
    assert torch.equal(masses, sample_limb_masses(20000, 121))
    assert not torch.equal(masses, sample_limb_masses(20000, 122))
    assert torch.equal(sample_limb_masses(5, 1, (0, 1, 2, 4)), torch.tensor([[0., 1, 2, 4]]).expand(5, 4))


@pytest.mark.parametrize("fixed", [None, (0, 0, 0, 0), (4, 4, 4, 4), (4, 0, 2, 0)])
def test_composite_payload_is_exact_at_zero_and_idempotent(fixed):
    n = 16
    model = SimpleNamespace(body_mass=torch.full((1, 4), 2.), body_ipos=torch.zeros(1, 4, 3),
                            body_inertia=torch.full((1, 4, 3), .1), body_iquat=torch.zeros(1, 4, 4))
    model.body_iquat[..., 0] = 1
    sim = SimpleNamespace(model=model, expanded_fields=set())

    def expand(names):
        for name, value in vars(model).items():
            setattr(model, name, value.expand(n, *value.shape[1:]).clone())
        sim.expanded_fields.update(names)

    sim.expand_model_fields = expand
    asset = SimpleNamespace(find_bodies=lambda names, **kw: (list(range(4)), names),
                            indexing=SimpleNamespace(body_ids=torch.arange(4)))
    env = SimpleNamespace(sim=sim, scene={"robot": asset}, num_envs=n, device="cpu")
    payload = UniformLimbPayload(SimpleNamespace(params={"seed": 121, "fixed_masses": fixed}), env)
    payload(env, None)
    torch.testing.assert_close(payload.observe(), payload.mass, atol=1e-6, rtol=1e-6)
    once = {name: value.clone() for name, value in vars(model).items()}
    payload(env, None)
    for name, value in vars(model).items():
        torch.testing.assert_close(value, once[name], atol=0, rtol=0)
        torch.testing.assert_close(value[payload.mass == 0], payload.base[name][payload.mass == 0], atol=0, rtol=0)
    assert (model.body_inertia > 0).all()


@pytest.mark.parametrize("fusion", ["film", "concat", "constant"])
def test_conditioning_identity_and_learnability(fusion):
    torch.manual_seed(7)
    original = _make_heft_mlp(7, (12, 8), 1)
    module = ConditionedMLP(copy.deepcopy(original), 5, fusion, width=10)
    x, z = torch.randn(32, 7), torch.randn(32, 5, requires_grad=True)
    torch.testing.assert_close(module(torch.cat((x, z), -1)), original(x), atol=1e-7, rtol=1e-6)
    optimizer = torch.optim.Adam(module.parameters(), lr=.01)
    target = z.detach()[:, :1] + .3
    for _ in range(5):
        optimizer.zero_grad()
        (module(torch.cat((x, z), -1)) - target).square().mean().backward()
        optimizer.step()
    assert z.grad is None or torch.count_nonzero(z.grad) == 0
    real = module(torch.cat((x, z), -1))
    shuffled = module(torch.cat((x, z.roll(1, 0)), -1))
    if fusion == "constant":
        torch.testing.assert_close(real, shuffled, atol=0, rtol=0)
    else:
        assert (real - shuffled).abs().mean() > .001


def test_matched_actors_and_critics_preserve_source_initial_functions(tmp_path, monkeypatch):
    import intact_tracking.residual_policy as policy
    monkeypatch.setattr(policy, "SPV52HeightContactEstimatorActor", _FakeTracker)
    checkpoint = tmp_path / "tracker.pt"
    _fake_actor_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    original = nn.Module()
    original.mlp = _make_heft_mlp(3, (8, 4), 1)
    original.obs_normalizer = DecayVecNorm(3)
    original.obs_normalizer.update(torch.randn(100, 3))
    saved["critic_state_dict"] = original.state_dict()
    torch.save(saved, checkpoint)
    obs = TensorDict({"features": torch.randn(8, 3), "dynamics_latent": torch.randn(8, 5)}, [8])
    groups = {"actor": ["features"], "critic": ["features"]}
    actions, values, trunks = [], [], []
    for fusion in ("baseline", "film", "concat", "constant"):
        torch.manual_seed(11)
        actor = LimbContextResidualActor(obs, groups, "actor", 2, tracker_checkpoint=str(checkpoint),
            tracker_actor_kwargs={}, tracker_obs_groups=groups, fusion_mode=fusion,
            dynamics_latent_dim=5, residual_hidden_dims=(8, 4))
        critic = LimbContextCritic(obs, groups, "critic", 1, initial_checkpoint=str(checkpoint),
                                  fusion_mode=fusion, dynamics_latent_dim=5, hidden_dims=(8, 4))
        actions.append(actor(obs))
        values.append(critic(obs))
        trunk = actor.residual_mlp if fusion == "baseline" else actor.residual_mlp.base
        trunks.append(trunk[0].weight[:, :3])
        assert not any(p.requires_grad for p in actor.tracker.parameters())
        for name, value in original.obs_normalizer.state_dict().items():
            torch.testing.assert_close(critic.obs_normalizer.state_dict()[name], value, atol=0, rtol=0)
    for action, value, trunk in zip(actions, values, trunks):
        torch.testing.assert_close(action, actions[0], atol=0, rtol=0)
        torch.testing.assert_close(value, values[0], atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(trunk, trunks[0], atol=0, rtol=0)


def test_scratch_models_match_across_fusions_without_loading_source_critic(tmp_path, monkeypatch):
    import intact_tracking.residual_policy as policy
    from intact_tracking.limb_context_distributed import tensor_digest
    monkeypatch.setattr(policy, "SPV52HeightContactEstimatorActor", _FakeTracker)
    checkpoint = tmp_path / "tracker.pt"
    _fake_actor_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    # A source value network is deliberately unusable. Only the frozen actor
    # may load its own source tensors; scratch critics need no checkpoint.
    saved["critic_state_dict"] = {"invalid": torch.tensor(float('nan'))}
    torch.save(saved, checkpoint)
    obs = TensorDict({"features": torch.randn(16, 3), "dynamics_latent": torch.randn(16, 5)}, [16])
    groups = {"actor": ["features"], "critic": ["features"]}
    hashes, values = [], []
    for fusion in ("baseline", "film", "concat", "constant"):
        torch.manual_seed(11)
        actor = LimbContextResidualActor(obs, groups, "actor", 2, tracker_checkpoint=str(checkpoint),
            tracker_actor_kwargs={}, tracker_obs_groups=groups, fusion_mode=fusion,
            dynamics_latent_dim=5, residual_hidden_dims=(8, 4), initialization_seed=10128,
            initial_action_std=.25)
        # FiLM consumed extra random draws before critic construction.
        critic = LimbContextCritic(obs, groups, "critic", 1, initial_checkpoint=None,
            fusion_mode=fusion, dynamics_latent_dim=5, hidden_dims=(8, 4), initialization_seed=20124)
        assert float(critic.obs_normalizer.count)==0
        critic.update_normalization(obs)
        assert float(critic.obs_normalizer.count)==16
        action=actor(obs)
        actor.distribution.update(action)
        torch.testing.assert_close(action, actor.tracker.mlp(obs['features']), atol=0, rtol=0)
        torch.testing.assert_close(actor.output_std, torch.full_like(actor.output_std,.25),atol=0,rtol=0)
        assert not torch.equal(actor.distribution.std_param, actor.tracker.distribution.std_param)
        trunk=actor.residual_mlp if fusion=='baseline' else actor.residual_mlp.base
        critic_trunk=critic.mlp if fusion=='baseline' else critic.mlp.base
        hashes.append((tensor_digest(trunk.named_parameters()), tensor_digest(critic_trunk.named_parameters()),
                       tensor_digest(critic.obs_normalizer.state_dict().items())))
        values.append(critic(obs))
        assert all(torch.isfinite(p).all() for p in critic.parameters())
        assert any(torch.count_nonzero(p)>0 for p in critic_trunk.parameters())
        optimizer=torch.optim.Adam([*trunk.parameters(),*critic.parameters()],lr=.01)
        before=critic_trunk[0].weight.detach().clone()
        ((critic(obs)-3).square().mean()+(actor(obs)-1).square().mean()).backward()
        optimizer.step()
        assert not torch.equal(before,critic_trunk[0].weight)
        assert all(p.grad is None for p in actor.tracker.parameters())
    assert len(set(hashes))==1
    for value in values[1:]:
        torch.testing.assert_close(value,values[0],atol=1e-6,rtol=1e-6)
    with pytest.raises(ValueError, match='must not load'):
        LimbContextCritic(obs,groups,'critic',1,initial_checkpoint=str(checkpoint),
                         initialization_seed=20124,hidden_dims=(8,4))


def test_critic_initialization_seed_is_independent_and_preserves_global_rng():
    obs=TensorDict({'features':torch.randn(8,3)},[8])
    groups={'critic':['features']}
    torch.manual_seed(7)
    before=torch.get_rng_state().clone()
    first=LimbContextCritic(obs,groups,'critic',1,initial_checkpoint=None,
                           initialization_seed=91,hidden_dims=(8,4))
    assert torch.equal(torch.get_rng_state(),before)
    torch.rand(1000)
    second=LimbContextCritic(obs,groups,'critic',1,initial_checkpoint=None,
                            initialization_seed=91,hidden_dims=(8,4))
    other=LimbContextCritic(obs,groups,'critic',1,initial_checkpoint=None,
                           initialization_seed=92,hidden_dims=(8,4))
    assert torch.equal(first.mlp[0].weight,second.mlp[0].weight)
    assert not torch.equal(first.mlp[0].weight,other.mlp[0].weight)


def test_validation_worlds_do_not_enter_training_replay():
    from intact_tracking.limb_context_validation import SplitWorldReplay
    training_rows, validation_rows = [], []
    train = SimpleNamespace(num_worlds=3, add_step=training_rows.append)
    validation = SimpleNamespace(add_step=validation_rows.append)
    split = SplitWorldReplay(train, validation)
    split.add_step({"world_id": torch.arange(5), "robot_state": torch.arange(10).reshape(5, 2)})
    assert training_rows[0]["world_id"].tolist() == [0, 1, 2]
    assert validation_rows[0]["world_id"].tolist() == [3, 4]
    split.collect_validation = False
    split.add_step({"world_id": torch.arange(5)})
    assert len(training_rows) == 2 and len(validation_rows) == 1


def test_concat_affine_is_equivalent_to_literal_input_concatenation():
    import torch.nn.functional as functional
    module = ConditionedMLP(_make_heft_mlp(7, (12, 8), 1), 5, "concat")
    nn.init.normal_(module.latent_input.weight, std=.1)
    x, z = torch.randn(13, 7), torch.randn(13, 5)
    joined = torch.cat((x, z), -1)
    expected = functional.linear(joined, torch.cat((module.base[0].weight, module.latent_input.weight), -1), module.base[0].bias)
    for layer in list(module.base)[1:]:
        expected = layer(expected)
    torch.testing.assert_close(module(joined), expected)


def test_common_prefix_metric_exposes_failure_truncation_bias():
    import numpy as np
    from intact_tracking.limb_context_results import paired_rows
    reference = {"protocol": "test", "seed": 1, "motion_ids": [0], "start_frames": [0],
        "horizons": [3], "metric_names": ["error_body_pos", "error_joint_pos"], "max_steps": 3,
        "motion_files": ["a"], "physics_world_fingerprints": ["same"], "reward_contract": {},
        "failed": [False], "episode_lengths": [3], "per_episode_metrics": [[67, 67]],
        "episode_returns": [3], "motions": 1}
    candidate = {**reference, "failed": [True], "episode_lengths": [1],
                 "per_episode_metrics": [[1, 1]], "episode_returns": [1]}
    rtrace = {"body_joint": np.array([[[1, 1], [100, 100], [100, 100]]]), "lengths": np.array([3])}
    ctrace = {"body_joint": np.array([[[1, 1], [0, 0], [0, 0]]]), "lengths": np.array([1])}
    rows, new, rescued = paired_rows(reference, candidate, rtrace, ctrace)
    assert rows[0, 2] < rows[0, 0]
    np.testing.assert_array_equal(rows[0, 10:12], rows[0, 12:14])
    assert new == 1 and rescued == 0
    with pytest.raises(ValueError, match="physics_world_fingerprints"):
        paired_rows(reference, {**candidate, "physics_world_fingerprints": ["different"]}, rtrace, ctrace)
