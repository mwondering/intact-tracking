import copy

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.modules.distribution import GaussianDistribution

from intact_tracking.fixed_five_ppo import (
    FixedPartition, FixedFivePPO, dispatch_indices, randomize_paired_episode_phase, split_episode_endings,
)
from intact_tracking.residual_policy import DecayVecNorm


class SmallActor(nn.Module):
    fusion_mode = "baseline"
    use_dynamics_latent = False
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.tracker = nn.Linear(3, 29).requires_grad_(False)
        self.head = nn.Linear(3, 29)
        self.distribution = GaussianDistribution(29, init_std=.25)

    def populate_tracker_cache(self, obs):
        pass

    update_normalization = populate_tracker_cache

    def _base_features_and_action(self, obs):
        return obs["x"], self.tracker(obs["x"]).detach()

    def forward(self, obs, stochastic_output=False, **kwargs):
        x, base = self._base_features_and_action(obs)
        self.last_base_action = base
        mean = base + self.head(x)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def get_output_log_prob(self, actions):
        return self.distribution.log_prob(actions)

    def get_hidden_state(self):
        return None

    def reset(self, *args, **kwargs):
        pass

    @property
    def output_mean(self):
        return self.distribution.mean

    @property
    def output_distribution_params(self):
        return self.distribution.params

    @property
    def output_entropy(self):
        return self.distribution.entropy


class SmallCritic(nn.Module):
    fusion_mode = "baseline"
    is_recurrent = False
    obs_dim = 3

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 1)
        self.obs_normalizer = DecayVecNorm(3)

    def forward(self, obs, **kwargs):
        return self.head(self.obs_normalizer(obs["x"]))

    def update_normalization(self, obs):
        self.obs_normalizer.update(obs["x"])

    def get_hidden_state(self):
        return None

    def reset(self, *args, **kwargs):
        pass


def example():
    torch.manual_seed(42)
    labels = torch.tensor([0, 1, 2, 3, 0, 2])
    obs = TensorDict({"x": torch.randn(12, 3), "source_slot": torch.arange(12)[:, None]}, [12])
    options = dict(num_learning_epochs=1, num_mini_batches=2, schedule="fixed",
                   actor_learning_rate=1e-3, critic_learning_rate=1e-3)
    system = FixedFivePPO(SmallActor(), SmallCritic(), obs, labels,
                         rollout_steps=2, algorithm_cfg=options, device="cpu")
    return system, obs


def fill(system, obs):
    obs = system.prepare_observations(obs)
    for step in range(2):
        action, means = system.act(obs)
        assert action.shape == means.shape == (12, 29)
        for ids, alg in zip(system.indices, system.algorithms):
            if not len(ids):
                assert alg.transition.actions is None
                continue
            torch.testing.assert_close(alg.transition.actions, action[ids])
            torch.testing.assert_close(alg.transition.actions_log_prob, alg.actor.get_output_log_prob(action[ids]))
        obs = system.prepare_observations(obs.clone())
        reward = torch.arange(12).float() + step
        system.process_step(obs, reward, torch.zeros(12), {"time_outs": torch.zeros(12)})
    return obs


def test_partition_is_mean_of_unit_latents_not_renormalized_center():
    sums = torch.tensor([[1., 1.], [0., 0.], [-2., 0.]], dtype=torch.float64)
    p = FixedPartition.from_sums(sums, torch.tensor([2, 2, 2]), torch.tensor([1., 0.]),
                                 torch.tensor([.5, 1., 1.5], dtype=torch.float64))
    torch.testing.assert_close(p.centers[0], torch.tensor([.5, .5], dtype=torch.float64))
    assert p.labels.tolist() == [1, 2, 3]  # Boundary goes up; far outliers stay in last class.
    restored = FixedPartition.from_state_dict(p.state_dict(), "cpu")
    torch.testing.assert_close(restored.labels, p.labels)
    with pytest.raises(ValueError, match="mature"):
        FixedPartition.from_sums(sums, torch.tensor([0, 1, 1]), p.reference, p.boundaries)


def test_baseline_owns_all_class_replicas_and_no_expert_transitions():
    system, obs = example()
    obs = fill(system, obs)
    for ids, alg in zip(system.indices, system.algorithms):
        torch.testing.assert_close(alg.storage.observations["source_slot"][0, :, 0], ids)
        torch.testing.assert_close(alg.storage.rewards[0, :, 0], ids.float())
    assert system.indices[-1].tolist() == list(range(6, 12))
    assert sum(len(ids) for ids in system.indices[:4]) == len(system.indices[-1])
    assert len({id(a.optimizer) for a in system.algorithms}) == 5


def test_updating_one_ppo_leaves_all_other_models_and_normalizers_unchanged():
    system, obs = example()
    obs = fill(system, obs)
    before = [copy.deepcopy((a.actor.state_dict(), a.critic.state_dict())) for a in system.algorithms]
    alg = system.algorithms[0]
    alg.compute_returns(obs[system.indices[0]])
    alg.update()
    assert not torch.equal(alg.actor.head.weight, before[0][0]["head.weight"])
    for i in range(1, 5):
        for model, state in zip((system.algorithms[i].actor, system.algorithms[i].critic), before[i]):
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)
        assert not system.algorithms[i].optimizer.state


def test_five_updates_save_and_restore_all_optimizers_and_fixed_assignment():
    system, obs = example()
    obs = fill(system, obs)
    report = system.update(obs)
    assert len(report) == 5 and system.role_updates == [1]*5
    assert system.transitions == [4, 2, 4, 2, 12]
    assert all(a.optimizer.state for a in system.algorithms)
    saved = copy.deepcopy(system.state_dict())
    restored, _ = example()
    restored.load_state_dict(saved)
    assert restored.transitions == system.transitions
    for a, b in zip(system.algorithms, restored.algorithms):
        for p, q in zip(a.actor.parameters(), b.actor.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        assert len(a.optimizer.state) == len(b.optimizer.state)
    bad = copy.deepcopy(saved)
    bad["labels"][0] = 1
    with pytest.raises(AssertionError):
        restored.load_state_dict(bad)


def test_empty_class_does_not_reassign_other_worlds():
    indices = dispatch_indices(torch.tensor([0, 0, 3]))
    assert [x.tolist() for x in indices] == [[0, 1], [], [], [2], [3, 4, 5]]


def test_empty_classes_skip_updates_without_affecting_baseline():
    labels = torch.tensor([0, 0, 3, 3, 0, 3])
    obs = TensorDict({"x": torch.randn(12, 3)}, [12])
    system = FixedFivePPO(SmallActor(), SmallCritic(), obs, labels,
                         rollout_steps=2, algorithm_cfg=dict(num_learning_epochs=1, num_mini_batches=2,
                                                            schedule="fixed"),
                         device="cpu")
    result = system.update(fill(system, obs))
    assert system.role_updates == [1, 0, 0, 1, 1]
    assert system.transitions == [6, 0, 0, 6, 12]
    for i in (1, 2):
        assert result[f"expert_{i}"]["skipped_empty_class"]
        assert not system.algorithms[i].optimizer.state


def test_final_bootstrap_uses_own_next_state_tracker_action():
    system, obs = example()
    obs = fill(system, obs)
    obs["x"] = obs["x"] + 10
    alg = system.algorithms[0]
    selected = obs[system.indices[0]]
    expected = alg.actor.tracker(selected["x"]).detach()
    captured = []
    handle = alg.critic.register_forward_pre_hook(
        lambda _, args: captured.append(args[0]["frozen_tracker_action"].clone()))
    try:
        alg.compute_returns(selected)
    finally:
        handle.remove()
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected, atol=0, rtol=0)


def test_latent_is_rejected_at_actor_critic_boundary():
    system, obs = example()
    obs["dynamics_latent"] = torch.randn(12, 64)
    with pytest.raises(ValueError, match="Routing"):
        system.prepare_observations(obs)


def _distributed_worker(rank, rendezvous):
    from datetime import timedelta
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                                         world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(42 + rank)
        labels = torch.tensor([0, 0, 1, 2, 3] if rank == 0 else [0, 1, 1, 2, 2, 3, 3])
        n = 2*len(labels)
        obs = TensorDict({"x": torch.randn(n, 3)}, [n])
        system = FixedFivePPO(SmallActor(), SmallCritic(), obs, labels, rollout_steps=2,
            algorithm_cfg=dict(num_learning_epochs=1, num_mini_batches=2, schedule="fixed"), device="cpu")
        assert system.global_counts == [3, 3, 3, 3, 12]
        assert system.audit_rank_agreement()["passed"]
        other_counts = [1, 2, 2, 2, 7] if rank == 0 else [2, 1, 1, 1, 5]
        for i, alg in enumerate(system.algorithms):
            own = len(system.indices[i])
            expected = ((2 + 4*rank)*own + (6 - 4*rank)*other_counts[i]) / system.global_counts[i]
            for model in (alg.actor, alg.critic):
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.grad = torch.full_like(parameter, 2 + 4*rank)
            alg.reduce_parameters()
            for model in (alg.actor, alg.critic):
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, expected))
            alg.optimizer.zero_grad(set_to_none=True)
        for _ in range(2):
            system.act(obs)
            obs = system.prepare_observations(obs.clone())
            system.process_step(obs, torch.arange(n).float(), torch.zeros(n), {})
        system.update(obs)
        assert system.audit_rank_agreement()["passed"]
        assert system.role_updates == [1]*5
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_unequal_class_counts_weight_gradients_and_keep_models_identical(tmp_path):
    torch.multiprocessing.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_randomized_episode_phases_spread_timeouts_and_preserve_paired_worlds():
    from types import SimpleNamespace
    env = SimpleNamespace(num_envs=32768, max_episode_length=1000,
                          episode_length_buf=torch.zeros(32768, dtype=torch.long))
    rng = torch.random.get_rng_state().clone()
    phase = randomize_paired_episode_phase(env, seed=123)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(env.episode_length_buf[:16384], env.episode_length_buf[16384:])
    assert int(phase.min()) == 0 and int(phase.max()) == 999
    previous = env.episode_length_buf.clone()
    randomize_paired_episode_phase(env, seed=123)
    torch.testing.assert_close(env.episode_length_buf, previous)
    randomize_paired_episode_phase(env, seed=124)
    assert not torch.equal(env.episode_length_buf, previous)
    # More than two full episode cycles: no synchronized 16k-world timeout wave.
    counts = []
    ages = phase.clone()
    for _ in range(100):
        endings = 0
        for _ in range(24):
            ages += 1
            done = ages >= 1000
            endings += int(done.sum())
            ages[done] = 0
        counts.append(endings)
    assert min(counts) > 250 and max(counts) < 550


def test_episode_endings_distinguish_timeout_from_failure_and_handle_overlap():
    done = torch.tensor([0, 1, 1, 1])
    timeout = torch.tensor([0, 1, 0, 1])
    failure = torch.tensor([0, 0, 1, 1])
    normal, failed = split_episode_endings(done, timeout, failure)
    assert normal.tolist() == [False, True, False, False]
    assert failed.tolist() == [False, False, True, True]
    assert torch.equal(normal | failed, done.bool())
    assert not bool((normal & failed).any())
    with pytest.raises(ValueError, match="no timeout or failure"):
        split_episode_endings(torch.ones(1), torch.zeros(1), torch.zeros(1))
