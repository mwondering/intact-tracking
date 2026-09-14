import pytest
import torch

from intact_tracking.forward_predictor_objective import _counterfactual_representation_loss
from intact_tracking.memory350_response_window import (
    ResponseWindowCollector, ResponseWindowReplay, ResponseWindowObjective, valid_response_window,
)
from intact_tracking.memory350_weak_pairs import WeakPairReplayBuffer, WeakPairLossConfig
from intact_tracking.memory350_model import Memory350Predictor, Memory350Config
from test_memory350 import _step, _normalization


class FakeA:
    def __init__(self, boundary=None):
        self.collector_step = 0
        self.motion_files = ()
        self.boundary = boundary

    def step(self, predictor_only):
        step = self.collector_step
        self.collector_step += 1
        batch = _step(step, reset=step == self.boundary)
        batch["next_robot_state"][:, 0] += .1 * (step + 1)
        return batch


class FakeB:
    def __init__(self):
        self.calls = []

    def rollout_joint_targets(self, state, actions, **kwargs):
        self.calls.append((state.clone(), actions.clone()))
        output = state[:, None].repeat(1, 10, 1)
        output[:, :, 0] += torch.arange(1, 11)
        return output, {}


def test_overlapping_labels_restore_same_start_keep_five_step_cadence_and_causal_context():
    replay = ResponseWindowReplay(num_worlds=1, capacity=64, sampling_mode="uniform")
    control = WeakPairReplayBuffer(num_worlds=1, capacity=64, sampling_mode="uniform")
    a, b, collector = FakeA(), FakeB(), ResponseWindowCollector()
    for _ in range(20):
        collector(a, b, replay)
    for step in range(100):
        batch = _step(step)
        batch["next_robot_state"][:, 0] += .1 * (step + 1)
        control.add_step(batch)
    assert a.collector_step == 105 and replay.collector_step == 100
    assert replay.total_samples_generated == control.total_samples_generated == 20
    for start, (state, actions) in enumerate(b.calls):
        assert state[0, 0] == start * 5
        assert actions.shape == (1, 10, 29)
        torch.testing.assert_close(actions[0, :, 0], torch.arange(start * 5, start * 5 + 10).float())
    selected = {key: value[:20] for key, value in replay._samples.items()}
    assert selected["state"].shape == (20, 6, 71)
    assert selected["action"].shape == (20, 5, 29)
    assert selected["label_response"].shape == (20, 10, 70)
    for name in ("state", "action", "collector_step", "memory_total", "memory_start", "motion_step"):
        torch.testing.assert_close(selected[name], control._samples[name][:20])
    ctx = replay._materialize_context(selected)
    for row in range(20):
        if ctx["valid"][row].any():
            assert ctx["state"][row, ctx["valid"][row], 0].max() < row * 5
    # Nominal positions start + [1..10]; actual-minus-nominal is .1*(absolute step+1).
    expected = .1 * (torch.arange(20)[:, None] * 5 + torch.arange(1, 11)[None])
    torch.testing.assert_close(selected["label_response"][:, :, 0], expected, atol=1e-5, rtol=1e-5)


def test_boundary_in_lookahead_masks_only_response_and_retains_prediction_sample():
    replay = ResponseWindowReplay(num_worlds=1, capacity=16)
    collector = ResponseWindowCollector()
    collector(FakeA(boundary=7), FakeB(), replay)
    assert replay.total_samples_generated == 1
    assert not replay._samples["label_response_valid"][0]
    assert replay._samples["state"][0, -1, 0] > 0
    assert not replay._samples["label_response"][0].any()


@pytest.mark.parametrize("kind", ["episode", "motion", "phase", "physics"])
def test_entire_label_window_rejects_discontinuities(kind):
    batches = [_step(i) for i in range(10)]
    assert valid_response_window(batches).all()
    if kind == "episode":
        batches[8]["episode_id"] += 1
    elif kind == "motion":
        batches[8]["motion_id"] += 1
    elif kind == "phase":
        batches[8]["motion_step"] += 1
    else:
        batches[8]["parameters_changed"] = torch.tensor([True])
    assert not valid_response_window(batches).any()


def test_response_validity_does_not_disable_local_positive_gradients():
    latent = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    loss, metrics, _ = _counterfactual_representation_loss(
        latent, latent.flip(0), torch.zeros(2, 10, 70),
        *([torch.ones(2, dtype=torch.bool)] * 3), torch.arange(2),
        response_distance_scale=.3, response_valid=torch.zeros(2, dtype=torch.bool))
    assert metrics["latent_relation_pairs"] == 0 and metrics["representation_positive_loss"] > 0
    loss.backward()
    assert torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0


def test_prediction_loss_ignores_last_five_response_steps_but_relation_changes():
    torch.set_num_threads(2)
    replay = ResponseWindowReplay(num_worlds=1, capacity=32, sampling_mode="uniform")
    a, b, collector = FakeA(), FakeB(), ResponseWindowCollector()
    for _ in range(15):
        collector(a, b, replay)
    batch = replay.sample_batch(4, _normalization())
    batch["world_id"] = torch.arange(4)
    batch["label_response"].zero_()
    model = Memory350Predictor(Memory350Config(
        transformer_dim=32, transformer_depth=1, transformer_heads=4,
        context_dim=16, context_heads=4, memory_depth=1))
    objective = ResponseWindowObjective(model, WeakPairLossConfig())
    before = objective(batch, compute_metrics=False)
    batch["label_response"][2:, 5:] = 5.
    after = objective(batch, compute_metrics=False)
    torch.testing.assert_close(before["prediction_loss"], after["prediction_loss"], atol=0, rtol=0)
    assert before["representation_loss"] != after["representation_loss"]
    after["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
