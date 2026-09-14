from types import SimpleNamespace

import pytest
import torch

from intact_tracking.tracker_finetune import (
    FourLimbExperimentPayload,
    SynchronizedScratchCritic,
    SynchronizedWarmStartedCritic,
    lafan_files,
    payload_samples,
)


def test_abc_load_distribution_and_isolated_rng():
    state = torch.get_rng_state().clone()
    assert payload_samples("A", 2048, 121).count_nonzero() == 0
    assert torch.equal(payload_samples("B", 2048, 121), torch.full((2048, 4), 4.0))
    c = payload_samples("C", 2048, 121)
    assert (c == 0).all(-1).sum() == 512
    assert (c == 4).all(-1).sum() == 512
    assert ((c > 0).all(-1) & (c < 4).all(-1)).sum() == 1024
    assert torch.equal(state, torch.get_rng_state())
    assert not torch.equal(c, payload_samples("C", 2048, 122))
    with pytest.raises(ValueError):
        payload_samples("C", 3, 121)


def test_lafan_does_not_double_weight_crops(tmp_path):
    for name in ("dance1_subject2.motion.npz", "fight1_subject3.motion.npz", "fight1_subject3_612_680.motion.npz"):
        (tmp_path / name).touch()
    (tmp_path / "singlejump").mkdir()
    (tmp_path / "singlejump/jumps1_subject1.motion.npz").touch()
    assert len(lafan_files(tmp_path)) == 2


def test_payload_expands_before_snapshot_and_is_idempotent():
    n = 16
    model = SimpleNamespace(body_mass=torch.full((1, 4), 2.), body_ipos=torch.zeros(1, 4, 3),
                            body_inertia=torch.full((1, 4, 3), .1), body_iquat=torch.zeros(1, 4, 4))
    model.body_iquat[..., 0] = 1
    sim = SimpleNamespace(model=model, expanded_fields=set())

    def expand(names):
        for name, tensor in vars(model).items():
            setattr(model, name, tensor.expand(n, *tensor.shape[1:]).clone())
        sim.expanded_fields.update(names)

    sim.expand_model_fields = expand
    asset = SimpleNamespace(find_bodies=lambda names, **kwargs: (list(range(4)), names),
                            indexing=SimpleNamespace(body_ids=torch.arange(4)))
    env = SimpleNamespace(sim=sim, scene={"robot": asset}, num_envs=n, device="cpu")
    payload = FourLimbExperimentPayload(SimpleNamespace(params={"condition": "C", "seed": 2}), env)
    payload(env, None)
    once = {name: tensor.clone() for name, tensor in vars(model).items()}
    payload(env, None)
    for name, tensor in vars(model).items():
        torch.testing.assert_close(tensor, once[name], atol=0, rtol=0)
        zero = (payload.mass == 0).all(-1)
        torch.testing.assert_close(tensor[zero], payload.base[name][zero], atol=0, rtol=0)
    assert payload.audit()["nominal_worlds"] == n // 4
    assert payload.audit()["hardest_worlds"] == n // 4


def test_distributed_normalization_matches_pooled_samples(monkeypatch):
    from tensordict import TensorDict

    groups = {"critic": ["x"]}
    a, b = torch.randn(7, 3), torch.randn(7, 3) + 10
    critic = SynchronizedScratchCritic(TensorDict({"x": a}, [7]), groups, "critic", 1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def reduce(value):
        other = b.double()
        value.add_(torch.cat((other.sum(0), other.square().sum(0), other.new_tensor([len(b)]))))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    critic.update_normalization({"x": a})
    all_x = torch.cat((a, b))
    torch.testing.assert_close(critic.obs_normalizer.mean, all_x.mean(0))
    torch.testing.assert_close(critic.obs_normalizer._var[0], all_x.var(0, unbiased=False))
    assert critic.obs_normalizer.count == 14


def test_warm_critic_restores_and_pools_decay_statistics(tmp_path, monkeypatch):
    from tensordict import TensorDict

    from intact_tracking.residual_policy import DecayVecNorm, _make_heft_mlp

    original = torch.nn.Module()
    original.obs_normalizer = DecayVecNorm(3)
    original.mlp = _make_heft_mlp(3, (8, 4), 1)
    original.obs_normalizer.update(torch.randn(100, 3))
    checkpoint = tmp_path / "source.pt"
    torch.save({"critic_state_dict": original.state_dict()}, checkpoint)
    a, b = torch.randn(7, 3), torch.randn(7, 3) + 10
    critic = SynchronizedWarmStartedCritic(
        TensorDict({"x": a}, [7]), {"critic": ["x"]}, "critic", 1,
        initial_checkpoint=str(checkpoint), hidden_dims=(8, 4))
    for name, value in original.state_dict().items():
        torch.testing.assert_close(critic.state_dict()[name], value, atol=0, rtol=0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def reduce(value):
        other = b.double()
        value.add_(torch.cat((other.sum(0), other.square().sum(0), other.new_tensor([len(b)]))))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    for _ in range(2):
        critic.update_normalization({"x": a})
        original.obs_normalizer.update(torch.cat((a, b)))
        for name, value in original.obs_normalizer.state_dict().items():
            torch.testing.assert_close(critic.obs_normalizer.state_dict()[name], value)
    frozen = {name: value.clone() for name, value in critic.state_dict().items()}
    critic.eval()
    critic.update_normalization({"x": a})
    for name, value in frozen.items():
        torch.testing.assert_close(critic.state_dict()[name], value, atol=0, rtol=0)
