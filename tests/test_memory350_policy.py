from types import SimpleNamespace

import pytest
import torch

from intact_tracking.memory350_inference import Memory350Inference
from intact_tracking.memory350_model import HierarchicalContextEncoder, Memory350Config
from intact_tracking.memory350_policy_inference import CachedMemory350Inference


def checkpoint():
    torch.manual_seed(502)
    config = Memory350Config(context_dim=16, context_depth=1, chunk_depth=1, memory_depth=1)
    encoder = HierarchicalContextEncoder(config).eval().requires_grad_(False)
    return SimpleNamespace(config=config, encoder=encoder, state_mean=torch.linspace(-1, 1, 71),
                           state_std=torch.linspace(.5, 2, 71), action_mean=torch.linspace(-.5, .5, 29),
                           action_std=torch.linspace(.5, 1.5, 29))


def interaction(step, episodes, local, motion):
    state = torch.randn(len(episodes), 71)
    state[:, 3:7] = torch.nn.functional.normalize(state[:, 3:7], dim=-1)
    return {"robot_state": state, "joint_target": torch.randn(len(episodes), 29),
            "next_robot_state": state + .1 * torch.randn_like(state),
            "episode_id": episodes.clone(), "episode_step": local.clone(),
            "motion_id": motion.clone(), "motion_step": local.clone(),
            "reset_boundary": torch.tensor([step % 37 == 36, step % 81 == 80])}


def test_cache_matches_original_encoder_through_resets_overwrites_and_parameter_changes():
    c = checkpoint()
    cached = CachedMemory350Inference(c, 2, batch_size=2, chunk_batch_size=7, use_bfloat16=False)
    original = Memory350Inference(c, 2, batch_size=2, use_bfloat16=False)
    episodes, local, motion = torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), torch.tensor([101, 102])
    torch.testing.assert_close(cached.encode(), original.encode(), atol=2e-6, rtol=2e-6)
    frozen = {k: v.clone() for k, v in c.encoder.state_dict().items()}
    for step in range(501):
        batch = interaction(step, episodes, local, motion)
        if step == 220:
            batch["parameters_changed"] = torch.tensor([True, False])
        cached.append(batch)
        original.append(batch)
        if step % 13 == 0 or batch["reset_boundary"].any() or step in (219, 220, 500):
            torch.testing.assert_close(cached.encode(), original.encode(), atol=3e-6, rtol=3e-6)
            assert torch.equal(cached.history_valid, original.memory.ordered_short()[1].T)
        boundary = batch["reset_boundary"]
        episodes += boundary.long()
        local = torch.where(boundary, 0, local + 1)
        motion += boundary.long() * 17
    for name, value in c.encoder.state_dict().items():
        torch.testing.assert_close(value, frozen[name], atol=0, rtol=0)
    assert all(p.grad is None for p in c.encoder.parameters())
    cached.encode()
    assert cached.last_chunks_encoded == cached.last_memories_encoded == 0


def test_explicit_episode_reset_preserves_long_and_parameter_change_clears_it():
    c = checkpoint()
    cached = CachedMemory350Inference(c, 2, use_bfloat16=False)
    ids, steps, motion = torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), torch.tensor([1, 2])
    for step in range(100):
        batch = interaction(step, ids, steps + step, motion)
        batch["reset_boundary"].zero_()
        cached.append(batch)
    cached.encode()
    old_total = cached.memory.total_chunks.clone()
    cached.episode_reset(torch.tensor([0]))
    assert cached.memory.short_count.tolist() == [0, 50]
    assert cached.memory.total_chunks[0] > old_total[0]
    assert cached.memory.total_chunks[1] == old_total[1]
    cached.encode()
    preserved = cached.memory_features[1].clone()
    cached.invalidate_parameters(torch.tensor([True, False]))
    cached.encode()
    assert not cached.memory.read_chunks()[1][0].any()
    assert torch.count_nonzero(cached.memory_features[0]) == 0
    torch.testing.assert_close(cached.memory_features[1], preserved, atol=0, rtol=0)


def test_frozen_cache_rejects_trainable_or_training_encoder():
    c = checkpoint()
    c.encoder.train()
    with pytest.raises(ValueError, match="frozen"):
        CachedMemory350Inference(c, 2)
    c.encoder.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        CachedMemory350Inference(c, 2)
