import torch
from torch import nn
from types import SimpleNamespace

from intact_tracking.memory350_compressed_policy import (
    CompressedConcatMLP, CompressedContextActor, configure_compressed_models,
)
from intact_tracking.limb_context_sampling import phase_budget


def test_actor_compression_shape_and_identical_zero_latent_baseline():
    baseline = CompressedConcatMLP(1645, 29, compression_dims=(512, 256, 128), fusion="baseline", seed=44)
    latent = CompressedConcatMLP(1645, 29, compression_dims=(512, 256, 128), fusion="concat", seed=44)
    assert [(m.in_features, m.out_features) for m in latent.observation.modules() if isinstance(m, nn.Linear)] == [
        (1645, 512), (512, 256), (256, 128)]
    assert latent.head[0].in_features == 192
    x, z = torch.randn(8, 1645), torch.randn(8, 64)
    torch.testing.assert_close(baseline(x), latent(torch.cat((x, z), -1)), atol=0, rtol=0)
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in latent.parameters())
    # After latent columns learn, baseline must still ignore arbitrary z.
    with torch.no_grad():
        baseline.head[0].weight[:, 128:].normal_()
        latent.head[0].weight[:, 128:].copy_(baseline.head[0].weight[:, 128:])
    torch.testing.assert_close(baseline(x), baseline(torch.cat((x, z), -1)), atol=0, rtol=0)
    assert not torch.allclose(latent(torch.cat((x, z), -1)), latent(torch.cat((x, z * 0), -1)))


def test_trainable_compressor_and_latent_columns_but_detached_encoder_output():
    model = CompressedConcatMLP(1645, 1, compression_dims=(512, 256, 128), seed=55)
    x, z = torch.randn(5, 1645), torch.randn(5, 64, requires_grad=True)
    model(torch.cat((x, z), -1)).square().mean().backward()
    assert z.grad is None or torch.count_nonzero(z.grad) == 0
    assert model.observation[0][0].weight.grad.abs().sum() > 0
    assert model.head[0].weight.grad[:, 128:].abs().sum() > 0


def test_actor_starts_at_zero_residual():
    model = CompressedConcatMLP(1645, 29, compression_dims=(512, 256, 128), zero_output=True)
    assert torch.count_nonzero(model(torch.randn(7, 1709))) == 0


def test_unbounded_curriculum_has_one_finite_prefix_then_no_cap():
    assert phase_budget("adaptive", 1000, 0, None) == 1000
    assert phase_budget("adaptive", 1000, 999, None) == 1
    assert phase_budget("adaptive", 1000, 1000, None) is None
    assert phase_budget("adaptive", 1000, 1000000, None) is None
    assert phase_budget("uniform", 1000, 0, None) is None


def test_tracker_action_extension_preserves_original_initialization_and_learns():
    old = CompressedConcatMLP(1645, 29, compression_dims=(512, 256, 128), seed=91)
    new = CompressedConcatMLP(1645, 29, compression_dims=(512, 256, 128), seed=91,
                             tracker_action_dim=29)
    assert new.head[0].in_features == 221
    for key, value in old.observation.state_dict().items():
        torch.testing.assert_close(value, new.observation.state_dict()[key], atol=0, rtol=0)
    torch.testing.assert_close(old.head[0].weight[:, :128], new.head[0].weight[:, :128], atol=0, rtol=0)
    torch.testing.assert_close(old.head[0].weight[:, 128:], new.head[0].weight[:, 157:], atol=0, rtol=0)
    for key, value in old.head.state_dict().items():
        if key != "0.weight":
            torch.testing.assert_close(value, new.head.state_dict()[key], atol=0, rtol=0)
    assert torch.count_nonzero(new.head[0].weight[:, 128:157]) == 0
    x = torch.randn(8, 1645)
    action = torch.randn(8, 29, requires_grad=True)
    latent = torch.randn(8, 64, requires_grad=True)
    value = torch.cat((x, action, latent), -1)
    torch.testing.assert_close(new(value), old(torch.cat((x, latent), -1)))
    new(value).square().mean().backward()
    assert action.grad is None or torch.count_nonzero(action.grad) == 0
    assert latent.grad is None or torch.count_nonzero(latent.grad) == 0
    assert new.head[0].weight.grad[:, 128:157].abs().sum() > 0
    assert new.head[0].weight.grad[:, 157:].abs().sum() > 0
    assert new.observation[0][0].weight.grad.abs().sum() > 0


def test_current_tracker_mean_reaches_action_slots_with_one_tracker_forward():
    class Tracker(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(1645, 29)
            self.distribution = SimpleNamespace(deterministic_output=lambda x: x)

        def get_latent(self, obs):
            return obs["features"]

    for fusion in ("baseline", "concat"):
        actor = CompressedContextActor.__new__(CompressedContextActor)
        nn.Module.__init__(actor)
        actor.tracker = Tracker().requires_grad_(False)
        actor.tracker_action_input = True
        actor.use_dynamics_latent = fusion == "concat"
        actor.residual_input_mode = "tracker_features"
        actor.dynamics_latent_group = "dynamics_latent"
        actor.residual_scale = 0.25
        actor.residual_mlp = CompressedConcatMLP(
            1645, 29, compression_dims=(512, 256, 128), fusion=fusion, tracker_action_dim=29)
        # Emulate learned action and context columns so a missing/permuted input
        # cannot pass merely because every initial residual is zero.
        with torch.no_grad():
            actor.residual_mlp.head[0].weight[:, 128:].normal_(std=.03)
            actor.tracker.mlp.bias.fill_(4.0)
        obs = {"features": torch.randn(4, 1645), "dynamics_latent": torch.randn(4, 64)}
        tracker_outputs, head_inputs = [], []
        handle = actor.tracker.mlp.register_forward_hook(lambda m, i, out: tracker_outputs.append(out.detach()))
        head_handle = actor.residual_mlp.head.register_forward_pre_hook(lambda m, i: head_inputs.append(i[0].detach()))
        mean = actor(obs)
        handle.remove()
        head_handle.remove()
        assert len(tracker_outputs) == len(head_inputs) == 1
        raw_action = tracker_outputs[0]
        torch.testing.assert_close(head_inputs[0][:, 128:157], raw_action, atol=0, rtol=0)
        expected_latent = obs["dynamics_latent"] if fusion == "concat" else torch.zeros_like(obs["dynamics_latent"])
        torch.testing.assert_close(head_inputs[0][:, 157:], expected_latent, atol=0, rtol=0)
        torch.testing.assert_close(mean, raw_action + actor.last_residual_mean, atol=0, rtol=0)
        swapped = actor._residual_input(obs, obs["features"], raw_action,
                                        latent_override=obs["dynamics_latent"].roll(1, 0))
        torch.testing.assert_close(swapped[:, 1645:1674], raw_action, atol=0, rtol=0)
        if fusion == "concat":
            torch.testing.assert_close(swapped[:, 1674:], obs["dynamics_latent"].roll(1, 0), atol=0, rtol=0)


def test_tracker_action_configuration_keeps_critic_and_old_contract_unchanged():
    source = {"actor": {}, "critic": {}}
    old = configure_compressed_models(source, "concat", scratch_seed=121)
    new = configure_compressed_models(source, "concat", scratch_seed=121, tracker_action_input=True)
    assert "tracker_action_input" not in old["actor"]
    assert new["actor"].pop("tracker_action_input") is True
    assert old == new
    assert source == {"actor": {}, "critic": {}}
    assert phase_budget("adaptive", 1000, 250, 500) == 250
