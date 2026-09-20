from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from intact_tracking.memory350_inference import load_memory350_checkpoint, Memory350Inference
from intact_tracking.memory350_model import Memory350Predictor
from intact_tracking.memory350_nominal_dr_soft import NominalDRSoftLossConfig, NominalDRSoftObjective
from intact_tracking.memory350_policy_inference import CachedMemory350Inference
from intact_tracking.memory350_proprio_inputs import (
    INPUT_CONTRACT, PROPRIO_WIDTHS, ProprioMemory350Config, current_proprio, control_action,
    validate_input_checkpoint,
)
from intact_tracking.memory350_proprio_policy import ProprioNativePolicyWrapper
from intact_tracking.memory350_proprio_replay import ProprioMemory350Replay
from test_memory350 import _step


def test_latest_proprio_is_term_major_and_reuses_noise_without_drawing_again():
    frames = [torch.randn(3, 50, width) for width in PROPRIO_WIDTHS]
    history = torch.cat([x.flatten(1) for x in frames], -1)
    rng = torch.random.get_rng_state().clone()
    frame = current_proprio({"estimator_history": history})
    torch.testing.assert_close(frame, torch.cat([x[:, -1] for x in frames], -1), rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert frame.shape == (3, 122)
    assert not torch.equal(frame, history[:, -122:])
    with pytest.raises(ValueError, match="term-major"):
        current_proprio({"estimator_history": torch.zeros(3, 122)})


def test_command_is_independent_of_substep_applied_action_and_physical_targets(monkeypatch):
    command = torch.full((2, 29), .7)
    term = SimpleNamespace(raw_action=command, applied_action=torch.full_like(command, .5),
                           physical_target_trace=torch.full((2, 4, 29), 8.))
    env = SimpleNamespace(num_envs=2, action_manager=SimpleNamespace(get_term=lambda _: term))
    value = control_action(env)
    torch.testing.assert_close(value, command, rtol=0, atol=0)
    term.applied_action.fill_(-100)
    term.physical_target_trace.fill_(100)
    torch.testing.assert_close(control_action(env), value, rtol=0, atol=0)
    command.fill_(0)
    assert (value == .7).all()  # Stored transitions cannot alias the next command.
    monkeypatch.setattr(ProprioNativePolicyWrapper, "unwrapped", property(lambda _: env))
    wrapper = object.__new__(ProprioNativePolicyWrapper)
    torch.testing.assert_close(wrapper._context_action(), command)


def test_sp_command_is_constant_while_original_substep_filter_changes():
    from intact_tracking.environment.mdp.actions import SpTrackingJointPositionAction
    action = object.__new__(SpTrackingJointPositionAction)
    action.cfg = SimpleNamespace(clip=None, raw_action_clip=None)
    action._env = SimpleNamespace(num_envs=3, device="cpu")
    action._raw_actions = torch.zeros(3, 29)
    action._decimation, action._history_len = 4, 8
    action.delay = torch.tensor([[0], [0], [2]])
    action.alpha = torch.tensor([[1.], [.8], [.8]])
    action._action_history = torch.zeros(3, 8, 29)
    action.applied_action = torch.zeros(3, 29)
    action._scale = .5
    action._default_offset = torch.full((3, 29), .2)
    action.joint_offset = torch.full((3, 29), .1)
    action.boot_delay = torch.zeros(3, 1, dtype=torch.long)
    action.boot_target = torch.zeros(3, 29)
    action.process_actions(torch.ones(3, 29))
    applied, targets = [], []
    for substep in range(4):
        action._update_processed_actions(substep)
        applied.append(action.applied_action[:, 0].clone())
        targets.append(action._processed_actions[:, 0].clone())
        assert action.raw_action.eq(1).all()
    expected = torch.tensor([[1., 1., 1., 1.], [.8, .96, .992, .9984], [0., 0., .8, .96]])
    torch.testing.assert_close(torch.stack(applied, 1), expected)
    torch.testing.assert_close(torch.stack(targets, 1), expected * .5 + .3)


def _proprio_step(step, world):
    local = step % 500
    b = _step(step + 1000 * world, episode=step // 500, episode_step=local, reset=local == 499, world=world)
    b["is_nominal"][:] = world == 0
    b["dr_metric"] = torch.tensor([[world / 4., world / 8.]])
    b["label_response"] = torch.ones(1, 10, 70) * (world / 10.)
    b["label_response_valid"] = torch.tensor([local < 490])
    b["encoder_state"] = torch.sin(torch.arange(122)[None] + step / 10. + world)
    b["encoder_next_state"] = torch.sin(torch.arange(122)[None] + (step + 1) / 10. + world)
    b["encoder_action"] = torch.cos(torch.arange(29)[None] + step / 10. + world)
    return b


@pytest.fixture(scope="module")
def replay_data():
    torch.set_num_threads(2)
    replay = ProprioMemory350Replay(num_worlds=4, capacity=512, dr_metric_dim=2,
                                    sampling_mode="uniform", require_rank_probe=False)
    for step in range(1250):
        worlds = [_proprio_step(step, i) for i in range(4)]
        replay.add_step({k: torch.cat([b[k] for b in worlds]) for k in worlds[0]})
    normalization = replay.normalizer.snapshot_from_packed(replay.normalizer.packed_statistics(), tuple(range(4)))
    replay.normalizer.freeze()
    return replay, normalization


def _small_model():
    return Memory350Predictor(ProprioMemory350Config(
        context_dim=16, context_heads=4, context_depth=1, chunk_depth=1, memory_depth=1,
        transformer_dim=32, transformer_heads=4, transformer_depth=1))


def test_replay_preserves_sensor_history_and_separate_privileged_predictor_data(replay_data):
    replay, normalization = replay_data
    batch = replay.sample_batch(32, normalization)
    assert batch["history_state"].shape == (32, 50, 122)
    assert batch["history_next_state"].shape == (32, 50, 122)
    assert batch["memory_interactions"].shape == (32, 30, 10, 273)
    assert batch["weak_memory_interactions"].shape == (32, 30, 10, 273)
    assert batch["predictor_history_state"].shape == (32, 50, 71)
    assert batch["predictor_history_action"].shape == (32, 50, 29)
    assert batch["state"].shape == (32, 6, 71)
    assert batch["label_response"].shape == (32, 10, 70)
    assert batch["weak_pair_valid"].any()
    assert replay.memory.short.shape[-1] == replay.weak_archive["raw"].shape[-1] == 273
    assert len(normalization.context_state_mean) == 122
    assert max(abs(x) for x in normalization.context_action_mean) < 1
    assert max(normalization.action_mean) > 100  # Different physical-target statistics.
    # A distributed sum of identical moment batches must preserve the estimate.
    doubled = replay.normalizer.snapshot_from_packed(2 * replay.normalizer.packed_statistics(), tuple(range(4)))
    assert normalization == doubled


def test_context_ignores_truth_targets_and_labels_but_learns_all_prediction_branches(replay_data):
    replay, normalization = replay_data
    batch = replay.sample_batch(12, normalization)
    model = _small_model().eval()
    objective = NominalDRSoftObjective(model, NominalDRSoftLossConfig())
    anchor = torch.zeros(64)
    anchor[0] = 1
    objective.set_anchor(anchor)
    before = objective._encode_views(batch)
    changed = deepcopy(batch)
    for key in ("state", "positive_current_state", "nominal_state", "label_response", "dr_metric",
                "foot", "contact_force", "contact_binary", "action", "predictor_history_state", "predictor_history_action"):
        changed[key] = torch.zeros_like(changed[key])
    after = objective._encode_views(changed)
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    out = objective(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.context_encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_checkpoint_direct_cached_and_training_normalization_agree(tmp_path, replay_data):
    replay, normalization = replay_data
    model = _small_model().eval()
    state = {"model": model.state_dict(), "model_config": asdict(model.config),
             "architecture_version": model.config.architecture_version,
             "normalization": asdict(normalization), "context_input_contract": deepcopy(INPUT_CONTRACT),
             "tracker": {"checkpoint_sha256": "test_tracker"}}
    path = tmp_path / "encoder.pt"
    torch.save(state, path)
    loaded = load_memory350_checkpoint(path, device="cpu", expected_tracker_sha256="test_tracker")
    direct = Memory350Inference(loaded, 1, use_bfloat16=False)
    cached = CachedMemory350Inference(loaded, 1, use_bfloat16=False)
    for step in range(600):
        b = _proprio_step(step, 0)
        interaction = {**b, "robot_state": b["encoder_state"], "joint_target": b["encoder_action"],
                       "next_robot_state": b["encoder_next_state"]}
        direct.append(interaction)
        cached.append(interaction)
    with torch.no_grad():
        torch.testing.assert_close(direct.encode(), cached.encode(), atol=3e-6, rtol=1e-5)
        short, valid = direct.memory.ordered_short()
        long, long_valid = direct.memory.read_chunks()
        normalized = cached._normalize(short).masked_fill(~valid[..., None], 0)
        expected = model.context_encoder(normalized[..., :122], normalized[..., 122:151], normalized[..., 151:],
                                         valid, cached._normalize(long), long_valid)
        torch.testing.assert_close(cached.encode(), expected, atol=3e-6, rtol=1e-5)
    assert loaded.state_mean.numel() == 122
    bad = deepcopy(state)
    bad["context_input_contract"]["action_source"] = "joint_pos action term.applied_action"
    with pytest.raises(ValueError, match="truth71"):
        validate_input_checkpoint(bad)
    torch.save(bad, path)
    with pytest.raises(ValueError):
        load_memory350_checkpoint(path, device="cpu")


def test_proprio_ppo_requires_explicit_new_checkpoint():
    from intact_tracking.cli.memory350_proprio_native_policy_train import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--fusion", "concat", "--output-dir", "unused"])
    args = parser.parse_args(["--fusion", "concat", "--output-dir", "unused", "--context-checkpoint", "new_encoder.pt"])
    assert args.num_envs == 8192 and args.training_ranks == 8
    assert args.motion_sampling == "adaptive" and args.training_terminations == "original"


def test_native_resume_validates_action_extension_without_rejecting_its_own_metadata():
    from intact_tracking.cli.forward_memory_native_dr_train import validate_native_response_contract, NATIVE_PHYSICAL_ACTION
    from intact_tracking.cli.forward_memory_nominal_direction_train import validate_response_contract
    actual = {"version": 1, "relation": "fixed anchor", "response_scale": .6}
    previous = dict(actual, physical_action=NATIVE_PHYSICAL_ACTION)
    validate_native_response_contract(previous, actual, base=validate_response_contract)
    with pytest.raises(ValueError, match="physical action"):
        validate_native_response_contract(dict(previous, physical_action="last target"), actual, base=validate_response_contract)
    with pytest.raises(ValueError, match="frozen-anchor"):
        validate_native_response_contract(previous, dict(actual, response_scale=.3), base=validate_response_contract)
