from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from intact_tracking.memory350_bank import InteractionMemory
from intact_tracking.memory350_native_dr import native_metric_schema
from intact_tracking.memory350_tracker_action_policy import configure_tracker_action_models
from intact_tracking.residual_dr_aux import (
    DR_TARGET_GROUP, DR_HISTORY_WEIGHT_GROUP, DRAuxiliaryObjective, dr_aux_layout,
    dr_history_weight, capture_dr_aux_targets,
)
from test_memory350_tracker_action_policy import models, observation, _algorithm


def schema_and_params():
    names = [f"base_com/com_offset/torso_link/{axis}" for axis in "xyz"]
    names += ["base_mass/relative_mass/torso_link", "foot_friction/friction/shared/0"]
    names += [f"motor_params_implicit/{kind}_scale/joint_{j}" for kind in ("kp", "kd", "armature") for j in range(29)]
    params = {"base_com": {"ranges": {0: [-.075, .075], 1: [-.05, .05], 2: [-.05, .05]}},
              "base_mass": {"ranges": [-1., 1.], "asset_cfg": "torso"},
              "foot_friction": {"ranges": [.3, 2.]}, "motor_params_implicit": {
                  "stiffness_range": {".*": [.8, 1.2]}, "damping_range": {".*": [.8, 1.2]},
                  "armature_range": {".*": [.8, 1.2]}}}
    return native_metric_schema(names, params, {"torso_link": 10.}), params


def attach_labels(obs, weight=1.):
    # Encoding a label and its history weight in a known observation column
    # lets us check alignment after randomized rollout minibatching.
    obs[DR_TARGET_GROUP] = obs["features"][:, :1].sigmoid().expand(-1, 92).clone()
    obs[DR_HISTORY_WEIGHT_GROUP] = obs["features"][:, 1:2].sigmoid() * weight
    return obs


@pytest.mark.parametrize("motor_weight", [0., .1])
def test_group_balance_zero_weight_exclusion_and_name_based_mapping(motor_weight):
    schema, _ = schema_and_params()
    objective = DRAuxiliaryObjective(schema, motor_weight)
    pred = torch.ones(3, 92, requires_grad=True)
    target = torch.zeros_like(pred)
    target[:, 3] = float("nan")
    if motor_weight == 0:
        target[:, 5:] = float("nan")
    loss, stats = objective(pred, target, torch.ones(3, 1))
    assert loss.item() == pytest.approx(1.)
    loss.backward()
    assert pred.grad[:, 3].eq(0).all()
    assert pred.grad[:, 0].sum().item() == pytest.approx(2 / (4 + 3 * motor_weight))
    assert pred.grad[:, 5:34].sum().item() == pytest.approx(2 * motor_weight / (4 + 3 * motor_weight))
    if motor_weight == 0:
        assert pred.grad[:, 5:].eq(0).all()
    assert stats[7].item() == pytest.approx(3 * .15)  # Physical COM x MAE sum, metres.
    permutation = torch.randperm(92)
    reordered = {key: [schema[key][i] for i in permutation] for key in ("names", "lower", "upper")}
    other, _ = DRAuxiliaryObjective(reordered, motor_weight)(pred.detach()[:, permutation], target[:, permutation], torch.ones(3, 1))
    torch.testing.assert_close(other, loss)
    partial, _ = objective(pred, target, torch.full((3, 1), .2))
    torch.testing.assert_close(partial, loss * .2)  # Early histories really lower the loss.
    empty, _ = objective(pred, target, torch.zeros(3, 1))
    assert empty.item() == 0 and torch.isfinite(empty)
    empty.backward()


def test_history_snapshot_tracks_retained_memory_and_parameter_invalidation():
    memory = InteractionMemory(3, token_dim=1)
    memory.short_count.copy_(torch.tensor([0, 25, 50]))
    memory.total_chunks.copy_(torch.tensor([0, 10, 50]))
    memory.session_start.copy_(torch.tensor([0, 0, 15]))
    snapshot = dr_history_weight(memory)
    torch.testing.assert_close(snapshot[:, 0], torch.tensor([0, 125 / 350, 1.]))
    memory._flush(torch.tensor([2]))
    assert dr_history_weight(memory)[2, 0] == pytest.approx(300 / 350)
    memory.invalidate(torch.tensor([False, True, True]))
    assert dr_history_weight(memory).eq(0).all()
    torch.testing.assert_close(snapshot[:, 0], torch.tensor([0, 125 / 350, 1.]))


def test_targets_use_actual_physics_and_range_normalization(monkeypatch):
    import intact_tracking.rollout.online as online
    schema, params = schema_and_params()
    actual = torch.tensor([schema["lower"], schema["upper"]])
    monkeypatch.setattr(online, "_capture_privileged_dynamics_targets",
                        lambda env: SimpleNamespace(names=schema["names"], values=actual))
    monkeypatch.setattr(online, "_entity_indices_and_names", lambda *args: (torch.tensor([0]), ["torso_link"]))
    monkeypatch.setattr(online, "_expanded_and_default_field", lambda *args: (None, torch.tensor([10.])))
    env = SimpleNamespace(event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(params=params[name])))
    result = capture_dr_aux_targets(env, schema)
    torch.testing.assert_close(result, torch.stack((torch.zeros(92), torch.ones(92))))
    actual.fill_(999.)
    torch.testing.assert_close(result, torch.stack((torch.zeros(92), torch.ones(92))))
    with pytest.raises(ValueError, match="outside"):
        capture_dr_aux_targets(env, schema)
    changed = deepcopy(schema)
    changed["lower"][0] -= .1
    with pytest.raises(ValueError, match="ranges differ"):
        capture_dr_aux_targets(env, changed)


def test_wrapper_attaches_supervision_to_each_observation_with_its_history_snapshot():
    from intact_tracking.memory350_proprio_policy import ProprioNativePolicyWrapper
    wrapper = object.__new__(ProprioNativePolicyWrapper)
    wrapper._latent = torch.randn(8, 320)
    wrapper._dr_aux_targets = torch.rand(8, 92)
    memory = InteractionMemory(8, token_dim=1)
    wrapper.context = SimpleNamespace(memory=memory)
    first = wrapper._attach(observation())
    memory.short_count.fill_(20)
    second = wrapper._attach(observation())
    assert first[DR_HISTORY_WEIGHT_GROUP].eq(0).all()
    torch.testing.assert_close(second[DR_HISTORY_WEIGHT_GROUP], torch.full((8, 1), 20 / 350))
    torch.testing.assert_close(second[DR_TARGET_GROUP], wrapper._dr_aux_targets)
    wrapper._dr_aux_targets = None
    evaluation = wrapper._attach(observation())
    assert DR_TARGET_GROUP not in evaluation and DR_HISTORY_WEIGHT_GROUP not in evaluation


def test_native_training_entry_enables_shared_head_and_supports_disabled_ablation(monkeypatch):
    from intact_tracking.cli import memory350_proprio_native_policy_train as entry
    schema, _ = schema_and_params()
    args = entry.build_parser().parse_args([
        "--fusion", "concat", "--context-checkpoint", "new.pt", "--output-dir", "unused"])
    assert args.dr_aux_coef == .05 and args.dr_aux_motor_weight == 0.
    assert args.entropy_coef == .005 and args.initial_action_std == 1.
    fake_base = SimpleNamespace(ManagerBasedRlEnv=lambda *a, **kw: None)
    monkeypatch.setattr(entry, "base", fake_base)
    monkeypatch.setattr(entry.torch, "load", lambda *a, **kw: {"dr_metric_schema": schema})
    monkeypatch.setattr(entry, "validate_proprio_context", lambda *args: None)
    entry.configure(args)
    config = fake_base.configure_context_models({"actor": {}, "critic": {}}, "concat", scratch_seed=121)
    assert config["actor"]["dr_aux_schema"] == schema
    assert config["actor"]["dr_aux_motor_weight"] == 0.
    assert config["algorithm"]["dr_aux_coef"] == .05
    assert fake_base.LimbContextWrapper.keywords["dr_aux_schema"] == schema
    args.dr_aux_coef = 0.
    entry.configure(args)
    config = fake_base.configure_context_models({"actor": {}, "critic": {}}, "concat", scratch_seed=121)
    assert "dr_aux_schema" not in config["actor"]
    assert fake_base.LimbContextWrapper.keywords["dr_aux_schema"] is None


def test_aux_gradient_reaches_shared_actor_and_latent_path_without_privileged_leak(models):
    schema, _ = schema_and_params()
    actor, critic, obs = models(dr_aux_schema=schema, initial_action_std=1., residual_output_mode="unbounded")
    for name in ("features", "dynamics_latent", "frozen_tracker_action"):
        obs[name].requires_grad_()
    attach_labels(obs)
    action = actor(obs)
    torch.testing.assert_close(action, actor.last_base_action, atol=0, rtol=0)
    prediction = actor.dr_aux_head(actor.dr_aux_features)
    torch.testing.assert_close(prediction, actor.predict_dr(obs), atol=0, rtol=0)
    loss, _ = actor.dr_aux_objective(prediction, obs[DR_TARGET_GROUP], obs[DR_HISTORY_WEIGHT_GROUP])
    loss.backward()
    assert actor.residual_mlp.base[0].weight.grad.abs().sum() > 0
    latent_grad = actor.residual_mlp.latent_input.weight.grad
    assert all(latent_grad[:, 64 * i:64 * (i + 1)].abs().sum() > 0 for i in range(5))
    assert actor.residual_mlp.base[-1].weight.grad is None  # Independent output heads.
    assert actor.dr_aux_head.weight.grad[3].eq(0).all()
    assert actor.dr_aux_head.weight.grad[5:].eq(0).all()
    assert all(p.grad is None for p in actor.tracker.parameters())
    assert all(p.grad is None for p in critic.parameters())
    assert all(obs[key].grad is None for key in ("features", "dynamics_latent", "frozen_tracker_action"))
    changed = obs.detach().clone()
    changed[DR_TARGET_GROUP].fill_(12345.)
    changed[DR_HISTORY_WEIGHT_GROUP].zero_()
    changed["priv"].fill_(54321.)
    torch.testing.assert_close(actor(changed), action, atol=0, rtol=0)
    torch.testing.assert_close(actor.predict_dr(changed), prediction, atol=0, rtol=0)


def rollout(algorithm, obs, weight=1.):
    with torch.inference_mode():
        for step in range(2):
            algorithm.act(obs)
            obs = attach_labels(observation(step + 1), weight)
            algorithm.process_env_step(obs, torch.randn(8), torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]),
                                       {"motion_resample_boundary": torch.zeros(8, dtype=torch.bool)})
        algorithm.compute_returns(obs)
    return obs


@pytest.mark.parametrize("weight", [0., 1.])
def test_ppo_aux_update_alignment_save_restore_and_label_free_inference(models, tmp_path, weight):
    schema, _ = schema_and_params()
    options = dict(dr_aux_schema=schema, initial_action_std=1., residual_output_mode="unbounded")
    actor, critic, obs = models(**options)
    attach_labels(obs, weight)
    algo = _algorithm(actor, critic, obs, dr_aux_coef=.05, entropy_coef=.005)
    before = deepcopy(actor.state_dict())
    rollout(algo, obs, weight)
    for batch in algo.storage.mini_batch_generator(2, 1):
        stored = batch.observations
        torch.testing.assert_close(stored[DR_TARGET_GROUP][:, :1], stored["features"][:, :1].sigmoid())
        torch.testing.assert_close(stored[DR_HISTORY_WEIGHT_GROUP], stored["features"][:, 1:2].sigmoid() * weight)
    losses = algo.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert losses["AuxDR/valid_fraction"] == weight
    assert not any("mass" in name for name in losses)
    assert not any(name.startswith(("AuxDR/kp_", "AuxDR/kd_", "AuxDR/armature_")) for name in losses)
    assert losses["AuxDR/weighted_loss"] == pytest.approx(.05 * losses["AuxDR/loss"])
    assert losses["AuxDR/shared_gradient_ratio_valid"] == 0.  # Initial action head is exactly zero.
    assert (losses["AuxDR/shared_aux_gradient_norm"] > 0) == bool(weight)
    assert torch.equal(actor.dr_aux_head.weight, before["dr_aux_head.weight"]) == (weight == 0)
    torch.testing.assert_close(actor.dr_aux_head.weight[3], before["dr_aux_head.weight"][3], atol=0, rtol=0)
    torch.testing.assert_close(actor.dr_aux_head.weight[5:], before["dr_aux_head.weight"][5:], atol=0, rtol=0)
    torch.testing.assert_close(actor.dr_aux_head.bias[5:], before["dr_aux_head.bias"][5:], atol=0, rtol=0)
    for key, value in actor.tracker.state_dict().items():
        torch.testing.assert_close(value, before["tracker." + key], atol=0, rtol=0)
    path = tmp_path / "aux.pt"
    torch.save(algo.save(), path)
    restored_actor, restored_critic, fresh_obs = models(**options)
    restored = _algorithm(restored_actor, restored_critic, attach_labels(fresh_obs), dr_aux_coef=.05, entropy_coef=.005)
    restored.load(torch.load(path, weights_only=False), None, strict=True)
    evaluation = observation(7)
    torch.testing.assert_close(actor(evaluation), restored_actor(evaluation), atol=0, rtol=0)
    torch.testing.assert_close(actor.predict_dr(evaluation), restored_actor.predict_dr(evaluation), atol=0, rtol=0)
    assert len(restored.optimizer.state) == len(algo.optimizer.state)


def test_disabled_aux_preserves_old_weights_rng_and_checkpoint_layout(models):
    schema, _ = schema_and_params()
    torch.manual_seed(7)
    original, critic, obs = models()
    rng = torch.random.get_rng_state()
    torch.manual_seed(7)
    enabled, other_critic, _ = models(dr_aux_schema=schema)
    torch.testing.assert_close(rng, torch.random.get_rng_state(), atol=0, rtol=0)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, enabled.state_dict()[key], atol=0, rtol=0)
    for key, value in critic.state_dict().items():
        torch.testing.assert_close(value, other_critic.state_dict()[key], atol=0, rtol=0)
    assert not any("dr_aux" in key for key in original.state_dict())
    original.load_state_dict(original.state_dict(), strict=True)
    with pytest.raises(ValueError, match="together"):
        _algorithm(original, critic, obs, dr_aux_coef=.05)
    cfg = configure_tracker_action_models({"actor": {}, "critic": {}}, "concat", dr_aux_coef=0., dr_aux_schema=schema)
    assert "dr_aux_schema" not in cfg["actor"] and "dr_aux_coef" not in cfg.get("algorithm", {})
    cfg = configure_tracker_action_models(cfg, "concat", dr_aux_coef=.05, dr_aux_schema=schema)
    assert cfg["algorithm"]["dr_aux_coef"] == .05 and cfg["actor"]["dr_aux_schema"] == schema
    with pytest.raises(ValueError, match="nonnegative"):
        dr_aux_layout(schema, float("nan"))


def _distributed_worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(), rank=rank, world_size=2)
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            rank_dir = directory / str(rank)
            rank_dir.mkdir()
            make = models.__wrapped__(rank_dir, monkeypatch)
            schema, _ = schema_and_params()
            actor, critic, obs = make(dr_aux_schema=schema, residual_output_mode="unbounded")
            attach_labels(obs, rank)  # Rank zero has no valid history at all.
            algo = _algorithm(actor, critic, obs, dr_aux_coef=.05,
                              multi_gpu_cfg={"global_rank": rank, "world_size": 2})
            algo.broadcast_parameters()
            before = actor.dr_aux_head.weight.detach().clone()
            rollout(algo, obs, rank)
            local_weight_sum = algo.storage.observations[DR_HISTORY_WEIGHT_GROUP].sum().item()
            losses = algo.update()
            assert losses["AuxDR/valid_fraction"] == .5
            assert losses["AuxDR/loss"] > 0 and losses["AuxDR/shared_aux_gradient_norm"] > 0
            assert not torch.equal(before, actor.dr_aux_head.weight)
            flat = torch.cat([p.detach().flatten() for module in (actor, critic) for p in module.parameters()])
            gathered = [torch.empty_like(flat) for _ in range(2)]
            dist.all_gather(gathered, flat)
            torch.testing.assert_close(gathered[0], gathered[1], atol=0, rtol=0)
            weight = torch.tensor(local_weight_sum)
            dist.all_reduce(weight)
            assert losses["AuxDR/history_weight_mean"] == pytest.approx(weight.item() / 32)
    finally:
        dist.destroy_process_group()


def test_multi_rank_aux_with_empty_history_rank_stays_synchronized(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)
