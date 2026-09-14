from types import SimpleNamespace

import torch

from intact_tracking.memory350_online_kmeans import OnlineKMeansRouter, fit_kmeans, squared_distances, normalized_latent
from intact_tracking.memory350_moe_policy import HardRoutedMLP, configure_moe_models
from intact_tracking.memory350_moe_training import OnlineMoEPPO
from intact_tracking.limb_context_distributed import DistributedResidualPPO


def population(k=16, n=12):
    g = torch.Generator().manual_seed(123)
    return torch.eye(k, 64).repeat_interleave(n, 0) + .01 * torch.randn(k * n, 64, generator=g)


def test_kmeans_separates_known_groups_and_matches_nearest_center():
    x = population()
    centers, inertia = fit_kmeans(x, 16, restarts=2)
    ids = squared_distances(normalized_latent(x), centers).argmin(-1).reshape(16, -1)
    assert all(len(row.unique()) == 1 for row in ids)
    assert len(ids[:, 0].unique()) == 16
    assert inertia < .02


def test_online_updates_are_explicit_bounded_and_checkpointed():
    x = population()
    r = OnlineKMeansRouter(center_rate=.1, max_switch_fraction=.02)
    r.initialize(x)
    before = {k: v.clone() for k, v in r.state_dict().items()}
    r.eval()
    ids = r(x)
    assert all(torch.equal(v, r.state_dict()[k]) for k, v in before.items())
    shifted = x + .02
    result = r.update_centers(shifted)
    assert result['router_center_updates'] == 1
    assert result['router_center_shift_rms'] > 0
    assert result['router_center_update_switch_fraction'] <= .02
    assert not list(r.parameters())
    restored = OnlineKMeansRouter(center_rate=.1)
    restored.load_state_dict(r.state_dict(), strict=True)
    torch.testing.assert_close(restored(shifted), r(shifted), atol=0, rtol=0)
    torch.testing.assert_close(restored.assignment_count, r.assignment_count, atol=0, rtol=0)


def test_empty_centers_retain_their_expert_identity():
    r = OnlineKMeansRouter()
    r.initialize(population())
    point = r.centers[0:1].clone().repeat(20, 1)
    old = r.centers.clone()
    r.update_centers(point)
    torch.testing.assert_close(r.centers[1:], old[1:], atol=0, rtol=0)


def test_hard_dispatch_only_trains_selected_heads_but_all_grads_exist():
    m = HardRoutedMLP(12, 3, compression_dims=(32, 128), fusion='concat', seed=5)
    m.router.centers.copy_(torch.eye(16, 64))
    x = torch.randn(9, 12)
    z = torch.eye(16, 64)[0].repeat(9, 1).requires_grad_()
    output = m(torch.cat((x, z), -1))
    expected = m.heads[0](m.observation(x))
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    output.square().mean().backward()
    assert z.grad is None or z.grad.count_nonzero() == 0
    assert all(p.grad is not None for p in m.parameters())
    assert m.heads[0][-1].weight.grad.abs().sum() > 0
    assert m.observation[0][0].weight.grad.abs().sum() > 0
    for head in m.heads[1:]:
        assert all(p.grad.count_nonzero() == 0 for p in head.parameters())


def test_dispatch_preserves_batch_permutation_and_original_order():
    m = HardRoutedMLP(12, 3, compression_dims=(32, 128), fusion='concat', seed=5)
    m.router.centers.copy_(torch.eye(16, 64))
    with torch.no_grad():
        for i, head in enumerate(m.heads):
            head[-1].weight.zero_()
            head[-1].bias.fill_(i)
    permutation = torch.tensor([15, 2, 7, 0, 2, 14, 7])
    value = torch.cat((torch.randn(len(permutation), 12), torch.eye(16, 64)[permutation]), -1)
    expected = permutation[:, None].expand(-1, 3).float()
    torch.testing.assert_close(m(value), expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.cat([m(row[None]) for row in value]), expected, atol=0, rtol=0)


def test_single_head_baseline_matches_initial_moe_value_and_has_no_router():
    options = dict(feature_dim=12, output_dim=1, compression_dims=(32, 128), seed=9)
    base = HardRoutedMLP(**options, fusion='baseline')
    moe = HardRoutedMLP(**options, fusion='concat')
    moe.router.centers.copy_(torch.eye(16, 64))
    x, z = torch.randn(16, 12), torch.eye(16, 64)
    assert base.router is None and len(base.heads) == 1 and len(moe.heads) == 16
    torch.testing.assert_close(base(x), moe(torch.cat((x, z), -1)), atol=1e-6, rtol=1e-6)
    for key, value in base.observation.state_dict().items():
        torch.testing.assert_close(value, moe.observation.state_dict()[key], atol=0, rtol=0)
    assert not ({id(p) for p in moe.heads[0].parameters()} & {id(p) for p in moe.heads[1].parameters()})


def test_center_updates_happen_after_ppo_and_copy_to_independent_critic(monkeypatch):
    r, c = OnlineKMeansRouter(), OnlineKMeansRouter()
    r.initialize(population())
    c.load_state_dict(r.state_dict())
    initial = r.centers.clone()
    instance = OnlineMoEPPO.__new__(OnlineMoEPPO)
    instance.actor = SimpleNamespace(residual_mlp=SimpleNamespace(router=r))
    instance.critic = SimpleNamespace(mlp=SimpleNamespace(router=c))
    instance.storage = SimpleNamespace(observations={'dynamics_latent': population() + .02})
    def ppo_update(self):
        torch.testing.assert_close(r.centers, initial, atol=0, rtol=0)
        return {'value': 1.0}
    monkeypatch.setattr(DistributedResidualPPO, 'update', ppo_update)
    result = instance.update()
    assert result['value'] == 1 and result['router_center_updates'] == 1
    assert not torch.equal(r.centers, initial)
    assert r.centers.data_ptr() != c.centers.data_ptr()
    for key, value in r.state_dict().items():
        torch.testing.assert_close(value, c.state_dict()[key], atol=0, rtol=0)


def test_configuration_preserves_separate_actor_and_critic_encoders():
    cfg = configure_moe_models({'actor': {}, 'critic': {}}, 'concat', scratch_seed=121)
    assert cfg['actor']['compression_dims'] == [512, 256, 128]
    assert cfg['critic']['compression_dims'] == [1024, 512, 256, 128]
    assert cfg['actor']['num_experts'] == cfg['critic']['num_experts'] == 16
    assert cfg['actor']['tracker_action_input'] is True
