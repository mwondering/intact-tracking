import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from intact_tracking.latent_history import LatentHistory
from intact_tracking.memory350_bank import InteractionMemory
from intact_tracking.memory350_deploy import (
    INPUT_NAME, INPUT_WIDTH, InteractionHistory, Memory350Policy, current_proprio, input_layout,
)
from intact_tracking.memory350_proprio_inputs import current_proprio as torch_proprio


def test_deploy_input_layout_is_contiguous_and_preserves_tracker_prefix():
    layout = input_layout()
    assert layout[0] == {"name": "tracker_observation", "shape": [8199], "offset": 0, "width": 8199}
    assert layout[-1]["offset"] + layout[-1]["width"] == INPUT_WIDTH == 104085
    assert all(b["offset"] == a["offset"] + a["width"] for a, b in zip(layout, layout[1:]))


def test_numpy_proprio_uses_term_major_latest_frames_exactly():
    value = np.arange(8199, dtype=np.float32)
    expected = torch_proprio({"estimator_history": torch.from_numpy(value[4:6104][None])})
    np.testing.assert_array_equal(current_proprio(value), expected[0].numpy())
    with pytest.raises(ValueError):
        current_proprio(np.zeros(6100))
    value[0] = np.nan
    with pytest.raises(ValueError):
        current_proprio(value)


def test_numpy_memory_matches_training_at_evictions_resets_wraparound_and_physical_changes():
    torch.set_num_threads(1)
    reference = InteractionMemory(1, token_dim=273)
    deployed = InteractionHistory()
    rng = np.random.default_rng(98)
    episode, episode_step, motion, motion_step = 0, 0, 0, 0
    for step in range(1600):
        discontinuity = step in (125, 431, 1219)
        boundary = step in (59, 400, 1057, 1374)
        changed = step == 800
        if discontinuity:
            motion += 1
            motion_step = 0
        if changed:
            deployed.reset(keep_long_memory=False)
        reference.begin_step(*(torch.tensor([x]) for x in (episode, episode_step, motion, motion_step)),
                             parameters_changed=torch.tensor([changed]))
        interaction = rng.standard_normal(273).astype(np.float32)
        reference.finish_step(torch.from_numpy(interaction[None]), torch.tensor([boundary]))
        deployed.append(interaction[:122], interaction[122:151], interaction[151:],
                        reset_boundary=boundary, discontinuity=discontinuity)
        expected_short, expected_short_valid = reference.ordered_short()
        expected_long, expected_long_valid = reference.read_chunks()
        for actual, expected in zip(deployed.tensors(), (expected_short, expected_short_valid,
                                                         expected_long, expected_long_valid), strict=True):
            np.testing.assert_array_equal(actual, expected[0].numpy())
        if boundary:
            episode += 1
            episode_step = -1
        episode_step += 1
        motion_step += 1


def test_runtime_uses_actual_previous_command_and_clears_only_expected_history():
    client = object.__new__(Memory350Policy)
    client.history = InteractionHistory()
    client.previous_latents = np.zeros((4, 64), np.float32)
    client.previous_proprio = client.previous_action = None
    captured = []

    def infer(names, feed):
        captured.append(feed[INPUT_NAME].copy())
        return [np.full((1, 29), len(captured), np.float32),
                np.full((1, 64), len(captured), np.float32)]

    client.session = SimpleNamespace(run=infer)
    reference_latents = LatentHistory(1)
    for step in range(80):
        reset = step == 65
        value = np.full(8199, step, np.float32)
        client.step(value, command_applied=np.full(29, .7, np.float32), reset_boundary=reset)
        actual_history = np.concatenate((captured[-1][0, -256:], np.full(64, step + 1)))
        expected_history = reference_latents.append(torch.full((1, 64), step + 1),
                                                    reset=torch.tensor([reset]))
        np.testing.assert_array_equal(actual_history, expected_history[0].numpy())
    assert len(client.history.chunks) > 0
    for interaction in client.history.short:
        np.testing.assert_array_equal(interaction[122:151], np.full(29, .7, np.float32))
    client.step(np.ones(8199), parameters_changed=True)
    assert not client.history.short and not client.history.chunks
    assert np.count_nonzero(client.previous_latents[:-1]) == 0


def test_client_rejects_mismatched_model_and_metadata(tmp_path):
    pytest.importorskip("onnxruntime")
    (tmp_path / "policy.onnx").write_bytes(b"wrong model")
    (tmp_path / "policy.json").write_text(json.dumps({
        "deployment_contract": "memory350_proprio122_history5_onnx_v1", "onnx_sha256": "wrong"}))
    with pytest.raises(ValueError, match="do not match"):
        Memory350Policy(tmp_path)


def test_latest_aliases_switch_as_one_generation_and_preserve_foreign_files(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from watch_memory350_onnx import publish_latest

    for generation in ("checkpoint_100", "checkpoint_200"):
        directory = tmp_path / "deploy" / generation
        directory.mkdir(parents=True)
        for name in ("policy.onnx", "policy.json", "deploy_metadata.json", "policy_runtime.py"):
            (directory / name).write_text(generation)
        publish_latest(tmp_path, directory)
        assert (tmp_path / "policy.onnx").resolve().parent == directory
        assert (tmp_path / "policy.json").read_text() == generation
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    (unrelated / "policy.onnx").write_text("owned by another model")
    with pytest.raises(FileExistsError):
        publish_latest(unrelated, directory)
    assert (unrelated / "policy.onnx").read_text() == "owned by another model"
