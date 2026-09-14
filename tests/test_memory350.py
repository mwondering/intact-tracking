import copy
from dataclasses import asdict

import pytest
import torch

from intact_tracking.data.predictor_online import ForwardPredictorNormalizationStats
from intact_tracking.forward_predictor_objective import ForwardPredictorLossConfig
from intact_tracking.memory350_bank import InteractionMemory
from intact_tracking.memory350_model import HierarchicalContextEncoder, Memory350Config, Memory350Predictor
from intact_tracking.memory350_objective import Memory350Objective
from intact_tracking.memory350_replay import Memory350ReplayBuffer


def _step(step, episode=0, episode_step=None, reset=False, world=0):
    local = step if episode_step is None else episode_step
    state = torch.zeros(1, 71); state[:, 3] = 1; state[:, 0] = step
    following = state.clone(); following[:, 0] += 1
    return {
        "robot_state": state, "next_robot_state": following,
        "nominal_next_robot_state": following.clone(), "joint_target": torch.full((1, 29), float(step)),
        "foot": torch.zeros(1, 8), "next_foot": torch.zeros(1, 8),
        "contact_force": torch.zeros(1, 6), "next_contact_force": torch.zeros(1, 6),
        "contact_binary": torch.zeros(1, 2, dtype=torch.bool), "next_contact_binary": torch.zeros(1, 2, dtype=torch.bool),
        "reset_boundary": torch.tensor([reset]), "is_nominal": torch.tensor([False]),
        "world_id": torch.tensor([world]), "episode_id": torch.tensor([episode]), "episode_step": torch.tensor([local]),
        "motion_id": torch.tensor([episode + 100]), "motion_step": torch.tensor([local]),
    }


def _push(bank, batch):
    bank.begin_step(batch["episode_id"], batch["episode_step"], batch["motion_id"], batch["motion_step"])
    bank.finish_step(torch.cat((batch["robot_state"], batch["joint_target"], batch["next_robot_state"]), -1), batch["reset_boundary"])


def _visible(bank):
    short, mask = bank.ordered_short(); long, valid = bank.read_chunks()
    return short[0, mask[0], 0].tolist(), long[0, valid[0], :, 0].flatten().tolist()


def test_latest350_are_disjoint_with_short_priority_and_partial_chunk_staging():
    memory = InteractionMemory(1)
    for step in range(377):
        _push(memory, _step(step))
    short, long = _visible(memory)
    assert short == list(range(327, 377))
    assert long == list(range(20, 320))
    assert memory.pending_count.item() == 7
    assert not set(short) & set(long)
    assert len(short) + len(long) == 350


def test_reset_flushes_complete_chunks_discards_tail_and_invalid_transition():
    memory = InteractionMemory(1)
    for step in range(67):
        _push(memory, _step(step))
    _push(memory, _step(67, reset=True))
    short, long = _visible(memory)
    assert short == [] and long == list(range(60))
    assert memory.discarded_tail_transitions == 7
    assert memory.pending_count.item() == 0
    for step in range(68, 80):
        _push(memory, _step(step, episode=1, episode_step=step - 68))
    short, long = _visible(memory)
    assert short == list(range(68, 80)) and long == list(range(60))


def test_short_trials_do_not_form_fake_cross_reset_chunks():
    memory = InteractionMemory(1)
    for step in range(130):
        local = step % 13
        _push(memory, _step(step, episode=step // 13, episode_step=local, reset=local == 12))
    chunks, valid = memory.read_chunks()
    assert valid.sum() == 10
    for chunk in chunks[0, valid[0], :, 0]:
        assert torch.equal(chunk.diff(), torch.ones(9))
        assert int(chunk[0]) // 13 == int(chunk[-1]) // 13
    assert memory.discarded_tail_transitions == 20


def test_parameter_change_clears_memory_only_for_affected_world():
    memory = InteractionMemory(2)
    for step in range(100):
        one, two = _step(step), _step(step + 1000, episode_step=step, world=1)
        _push(memory, {name: torch.cat((one[name], two[name])) for name in one})
    before, valid = memory.read_chunks()
    memory.invalidate(torch.tensor([True, False]))
    after, after_valid = memory.read_chunks()
    assert not after_valid[0].any() and memory.short_count[0] == 0
    torch.testing.assert_close(after[1], before[1])
    torch.testing.assert_close(after_valid[1], valid[1])


def test_replay_reads_query_time_memory_without_future_or_short_overlap():
    replay = Memory350ReplayBuffer(num_worlds=1, capacity=64, sampling_mode="uniform")
    for step in range(100):
        replay.add_step(_step(step))
    index = torch.nonzero(replay._samples["collector_step"][:replay._size] == 64).flatten()
    selected = {name: value[index] for name, value in replay._samples.items()}
    context = replay._materialize_context(selected)
    # Query at t=60, even though the current collector has already reached 100.
    assert context["state"][0, context["valid"][0], 0].tolist() == list(range(10, 60))
    assert context["memory"][0, context["memory_valid"][0], :, 0].flatten().tolist() == list(range(10))
    assert context["next_state"][0, context["valid"][0], 0].tolist() == list(range(11, 61))


def _normalization():
    return ForwardPredictorNormalizationStats(
        state_mean=(0.,) * 71, state_std=(1.,) * 71,
        action_mean=(0.,) * 29, action_std=(1.,) * 29,
        foot_mean=(0.,) * 8, foot_std=(1.,) * 8,
        contact_force_mean=(0.,) * 6, contact_force_std=(1.,) * 6,
        delta_mean=(0.,) * 70, delta_std=(1.,) * 70, world_ids=(0, 1))


def _replay_batch():
    replay = Memory350ReplayBuffer(num_worlds=2, capacity=256)
    for step in range(140):
        local = step % 35
        one = _step(step, episode=step // 35, episode_step=local, reset=local == 34)
        two = _step(step + 1000, episode=step // 35, episode_step=local, reset=local == 34, world=1)
        two["nominal_next_robot_state"][:, 0] -= .5
        replay.add_step({name: torch.cat((one[name], two[name])) for name in one})
    return replay.sample_batch(8, _normalization(), positive_ready_only=True)


def test_padded_short_context_with_long_memory_still_learns_representation():
    torch.manual_seed(13)
    config = Memory350Config(transformer_dim=32, transformer_depth=1, transformer_heads=4,
                             context_dim=16, context_heads=4, memory_depth=1)
    model = Memory350Predictor(config)
    batch = _replay_batch()
    assert not batch["history_valid"].all(1).any()
    assert batch["memory_valid"].any(1).any() and batch["positive_pair_valid"].all()
    objective = Memory350Objective(model, ForwardPredictorLossConfig(
        representation_relation_weight=2., response_distance_scale=.75))
    out = objective(batch)
    assert out["latent_relation_pairs"] > 0
    out["loss"].backward()
    for prefix in ("context_encoder.chunk_encoder", "context_encoder.memory_encoder", "context_encoder.transformer"):
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_masked_garbage_is_ignored_and_final_encoder_gets_one_memory_token():
    torch.manual_seed(7)
    encoder = HierarchicalContextEncoder(Memory350Config(context_dim=16, context_heads=4, memory_depth=1)).eval()
    batch = _replay_batch()
    args = [batch[k] for k in ("history_state", "history_action", "history_next_state", "history_valid", "memory_interactions", "memory_valid")]
    lengths = []
    hook = encoder.transformer.register_forward_pre_hook(lambda module, inputs: lengths.append(inputs[0].shape[1]))
    expected = encoder(*args)
    changed = [x.clone() for x in args]
    for index in (0, 1, 2):
        changed[index][~changed[3]] = float("nan")
    changed[4][~changed[5]] = float("nan")
    actual = encoder(*changed)
    hook.remove()
    torch.testing.assert_close(actual, expected)
    assert lengths == [52, 52]
    empty = [torch.zeros_like(x) for x in args]
    assert torch.isfinite(encoder(*empty)).all()


def test_architecture_round_trip_is_distinct_from_original_baseline():
    from intact_tracking.forward_predictor import ForwardPredictorConfig
    assert ForwardPredictorConfig().context_history_steps == 100
    config = Memory350Config()
    assert Memory350Config(**asdict(config)) == config
    assert config.architecture_version != ForwardPredictorConfig().architecture_version


def test_checkpoint_online_inference_matches_replay_query_after_later_collection(tmp_path):
    from intact_tracking.memory350_inference import load_memory350_checkpoint, Memory350Inference

    config = Memory350Config(transformer_dim=32, transformer_depth=1, transformer_heads=4,
                             context_dim=16, context_heads=4, memory_depth=1)
    model = Memory350Predictor(config).eval()
    path = tmp_path / "context.pt"
    torch.save({"model_config": asdict(config), "architecture_version": config.architecture_version,
                "model": model.state_dict(), "normalization": asdict(_normalization()),
                "tracker": {"checkpoint_sha256": "test-tracker"}}, path)
    frozen = load_memory350_checkpoint(path, device="cpu", expected_tracker_sha256="test-tracker")
    assert not any(p.requires_grad for p in frozen.encoder.parameters())
    online = Memory350Inference(frozen, 1)
    replay = Memory350ReplayBuffer(num_worlds=1, capacity=64)
    for step in range(100):
        if step == 60:
            expected = online.encode()
        batch = _step(step)
        replay.add_step(batch); online.append(batch)
    index = torch.nonzero(replay._samples["collector_step"][:replay._size] == 64).flatten()
    context = replay._materialize_context({name: value[index] for name, value in replay._samples.items()})
    actual = frozen.encoder(context["state"], context["action"], context["next_state"], context["valid"],
                             context["memory"], context["memory_valid"])
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="tracker checkpoints differ"):
        load_memory350_checkpoint(path, device="cpu", expected_tracker_sha256="different-tracker")


def test_new_cli_is_unbounded_and_microbatches_slice_memory():
    from intact_tracking.cli.forward_memory_train import build_parser, _validate_arguments, _slice_predictor_batch

    args = build_parser().parse_args(["--checkpoint-file", "tracker.pt", "--motion-path", "motions", "--output-dir", "new-run"])
    _validate_arguments(args)
    assert args.until_user_stop and args.num_envs == 8192 and args.context_history_steps == 50
    assert args.representation_relation_weight == 2 and args.response_distance_scale == .75
    batch = _replay_batch()
    sliced = _slice_predictor_batch(batch, 2, 5)
    for name in ("memory_interactions", "memory_valid", "history_next_state", "positive_memory_interactions"):
        torch.testing.assert_close(sliced[name], batch[name][2:5])
    assert sliced["state_mean"] is batch["state_mean"]


def test_replay_ring_wrap_preserves_chunk_continuity_and_query_causality():
    replay = Memory350ReplayBuffer(num_worlds=1, capacity=32, sampling_mode="uniform", seed=17)
    episode, local, duration = 0, 0, 23
    trial_for_step = {}
    for step in range(1800):
        boundary = local == duration - 1
        trial_for_step[step] = episode
        replay.add_step(_step(step, episode=episode, episode_step=local, reset=boundary))
        if boundary:
            episode += 1; local = 0; duration = (23, 78, 137, 9, 51)[episode % 5]
        else:
            local += 1
        if step > 400 and step % 100 == 99:
            indices = replay._active_sample_indices()
            selected = {name: value[indices] for name, value in replay._samples.items()}
            context = replay._materialize_context(selected)
            for row in range(len(indices)):
                query = int(selected["state"][row, 0, 0])
                short = context["state"][row, context["valid"][row], 0].tolist()
                chunks = context["memory"][row, context["memory_valid"][row], :, 0]
                long = chunks.flatten().tolist()
                assert not set(short) & set(long)
                assert all(value < query for value in short + long)
                for chunk in chunks:
                    assert torch.equal(chunk.diff(), torch.ones(9))
                    assert len({trial_for_step[int(value)] for value in chunk}) == 1
