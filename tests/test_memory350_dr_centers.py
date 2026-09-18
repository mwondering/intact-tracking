import copy
import json
from dataclasses import asdict
import math
from types import SimpleNamespace

import pytest
import torch

from intact_tracking.cli import forward_memory_scale_nominal_train as trainer
from intact_tracking.cli.forward_memory_dr_center_train import (
    REPRESENTATION_METRICS, build_parser, configure_checkpoint, configure_run_metadata, configure_trainer,
    validate_resume_loss_config,
)
from intact_tracking.limb_context_dr import TRACKER_DR, validate_context_dr
from intact_tracking.memory350_dr_center_rollout import dr_metric_schema, normalize_dr_metric
from intact_tracking.memory350_dr_centers import (
    CENTER_BATCH_FIELDS, DRCenterLossConfig, DRCenterObjective,
    DRCenterReplayBuffer, dr_center_relation_loss, dr_center_pair_valid, dr_cross_motion_positive_loss,
)
from intact_tracking.memory350_model import Memory350Config, Memory350Predictor
from test_memory350 import _normalization, _step


def _loss(anchor, archived, parameters, worlds=None, valid=None, nominal=None, **kwargs):
    n = len(anchor)
    return dr_center_relation_loss(
        anchor, archived, parameters,
        torch.arange(n) if worlds is None else worlds, torch.zeros(n, dtype=torch.long),
        torch.ones(n, dtype=torch.bool) if valid is None else valid,
        torch.zeros(n, dtype=torch.bool) if nominal is None else nominal, **kwargs)


def test_relation_matches_physical_distances_and_identical_dr_is_not_a_negative_class():
    angle = torch.tensor(.5, requires_grad=True)
    z = torch.stack((torch.stack((angle.cos(), angle.sin())), torch.tensor([1., 0.])))
    loss, metrics = _loss(z, z, torch.zeros(2, 3), nominal=torch.ones(2, dtype=torch.bool))
    distance = torch.linalg.vector_norm(z[0] - z[1])
    expected = torch.nn.functional.smooth_l1_loss(distance, torch.tensor(0.), beta=.25)
    torch.testing.assert_close(loss, expected)
    assert metrics["dr_center_pairs"] == metrics["dr_center_nominal_pairs"] == 1
    assert metrics["dr_center_target_distance_mean"] == 0
    loss.backward()
    assert angle.grad > 0  # Same physics pulls the two centers together despite different IDs.

    parameters = torch.tensor([[0., 0.], [.3, .4]], requires_grad=True)
    loss, metrics = _loss(z.detach(), z.detach(), parameters)
    torch.testing.assert_close(metrics["dr_parameter_distance_mean"], torch.tensor(.5))
    torch.testing.assert_close(metrics["dr_center_target_distance_mean"], torch.tensor(1.25))
    assert not loss.requires_grad  # Simulator targets never acquire gradients.


def test_centers_pool_same_world_rows_without_renormalizing_the_mean():
    z = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], requires_grad=True)
    other = torch.tensor([[0., 1.], [1., 0.], [-1., 0.]], requires_grad=True)
    loss, metrics = _loss(z, other, torch.tensor([[0.], [0.], [1.]]), worlds=torch.tensor([3, 3, 7]))
    assert metrics["dr_center_valid_worlds"] == 2 and metrics["dr_center_pairs"] == 1
    torch.testing.assert_close(metrics["dr_center_distance_mean"], torch.tensor(math.sqrt(2.5)))
    loss.backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(other.grad).all()
    assert z.grad.abs().sum() > 0 and other.grad.abs().sum() > 0


def test_no_within_or_cross_motion_positive_loss_even_for_widely_separated_views():
    z = torch.tensor([[1., 0.], [-1., 0.]], requires_grad=True)
    other = -z
    loss, metrics = _loss(z, other, torch.zeros(2, 1), worlds=torch.tensor([3, 3]))
    assert metrics["dr_cross_motion_distance_mean"] == 2
    assert metrics["dr_within_center_distance_mean"] == 1
    assert metrics["dr_center_pairs"] == 0 and loss == 0
    loss.backward()
    assert torch.equal(z.grad, torch.zeros_like(z))


def test_masked_empty_pairs_and_coincident_centers_have_finite_zero_gradients():
    for valid in (torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool)):
        z = torch.zeros(3, 64, requires_grad=True)
        loss, metrics = _loss(z, z, torch.zeros(3, 2), valid=valid)
        loss.backward()
        assert loss == 0 and torch.isfinite(z.grad).all()
        assert all(torch.isfinite(v) and not v.requires_grad for v in metrics.values())


def test_dr_positive_pulls_both_views_without_nominal_or_invalid_gradients():
    # Orthogonal unit views have squared distance 2, independent of latent dim.
    z = torch.zeros(4, 64)
    other = torch.zeros_like(z)
    z[:, 0] = 3.
    other[:, 1] = 4.
    z.requires_grad_(); other.requires_grad_()
    valid = torch.tensor([True, True, False, False])
    nominal = torch.tensor([False, True, False, True])
    loss, metrics = dr_cross_motion_positive_loss(z, other, valid, nominal)
    torch.testing.assert_close(loss, torch.tensor(2.))
    assert metrics["dr_positive_pairs"] == 1
    assert metrics["dr_positive_fraction_of_dr"] == .5
    loss.backward()
    assert z.grad[0].norm() > 0 and other.grad[0].norm() > 0
    assert torch.equal(z.grad[1:], torch.zeros_like(z.grad[1:]))
    assert torch.equal(other.grad[1:], torch.zeros_like(other.grad[1:]))
    after, _ = dr_cross_motion_positive_loss(
        z.detach() - .1 * z.grad, other.detach() - .1 * other.grad, valid, nominal)
    assert after < loss.detach()
    empty, metrics = dr_cross_motion_positive_loss(z, other, ~torch.ones_like(valid), nominal)
    gradients = torch.autograd.grad(empty, (z, other))
    assert empty == 0 and all(torch.equal(g, torch.zeros_like(g)) for g in gradients)
    assert all(torch.isfinite(v) and not v.requires_grad for v in metrics.values())


@pytest.fixture(scope="module")
def replay():
    torch.set_num_threads(2)
    buffer = DRCenterReplayBuffer(num_worlds=4, capacity=64, sampling_mode="uniform", dr_metric_dim=2)
    for step in range(1985):
        local = step % 100
        rows = [_step(step + 10000 * i, episode=step // 100, episode_step=local,
                      reset=local == 99, world=i) for i in range(4)]
        batch = {name: torch.cat([row[name] for row in rows]) for name in rows[0]}
        batch["is_nominal"] = torch.tensor([True, False, True, False])
        batch["dr_metric"] = torch.tensor([[0., 0.], [.3, .4], [0., 0.], [.5, 0.]])
        buffer.add_step(batch)
    return buffer


def test_archive_includes_nominal_and_ready_probes_are_causal_cross_motion_centers(replay):
    assert len(replay.weak_archive["env_ids"]) == 4
    assert replay.can_sample_positive_pairs(16)
    batch = replay.sample_batch(16, _normalization(), positive_ready_only=True)
    assert batch["center_pair_valid"].all()
    assert batch["is_nominal"].any() and (~batch["is_nominal"]).any()
    assert not batch["positive_pair_valid"].any()  # Disabled local pairs.
    assert (batch["center_world_id"] == batch["world_id"]).all()
    assert (batch["center_session"] == batch["physics_session"]).all()
    assert (batch["center_motion_id"] != batch["motion_id"]).all()
    for i in range(len(batch["state"])):
        current = torch.cat((batch["history_state"][i, :, 0],
                             batch["memory_interactions"][i, :, :, 0].flatten()))
        archived = torch.cat((batch["center_history_state"][i, :, 0],
                              batch["center_memory_interactions"][i, :, :, 0].flatten()))
        assert not set(current.tolist()) & set(archived.tolist())
        assert archived.max() < current.min()
    torch.testing.assert_close(batch["dr_metric"], replay._current_dr_metric[batch["world_id"]])
    assert CENTER_BATCH_FIELDS.issubset(batch) and not any(k.startswith("weak_") for k in batch)


@pytest.mark.parametrize("invalid_field", [
    "center_pair_valid", "center_world_id", "center_motion_id", "center_session",
    "history_valid", "memory_valid", "center_history_valid", "center_memory_valid",
])
def test_positive_eligibility_rejects_wrong_physics_motion_and_incomplete_histories(replay, invalid_field):
    batch = replay.sample_batch(16, _normalization(), positive_ready_only=True)
    assert dr_center_pair_valid(batch).all()
    changed = {**batch, invalid_field: batch[invalid_field].clone()}
    if invalid_field == "center_motion_id":
        changed[invalid_field][0] = batch["motion_id"][0]
    elif invalid_field in ("center_world_id", "center_session"):
        changed[invalid_field][0] += 1
    elif changed[invalid_field].ndim == 1:
        changed[invalid_field][0] = False
    else:
        changed[invalid_field][0, 0] = False
    eligible = dr_center_pair_valid(changed)
    assert not eligible[0] and eligible[1:].all()


def test_changed_parameters_invalidate_archive_and_crossing_prediction_windows(replay):
    buffer = copy.deepcopy(replay)
    metrics = buffer._current_dr_metric.clone()
    metrics[1] = torch.tensor([.9, .1])
    for step in range(1985, 2055):
        local = step % 100
        rows = [_step(step + 10000 * i, episode=step // 100, episode_step=local,
                      reset=local == 99, world=i) for i in range(4)]
        batch = {name: torch.cat([row[name] for row in rows]) for name in rows[0]}
        batch.update(dr_metric=metrics, is_nominal=torch.tensor([True, False, True, False]))
        buffer.add_step(batch)  # No external parameters_changed flag; labels detect it.
        if step == 1989:
            active = buffer._active_sample_indices()
            indices = active[(buffer._samples["world_id"][active] == 1)
                             & (buffer._samples["collector_step"][active] == 1989)]
            assert len(indices) == 1
            context = buffer._materialize_context({k: v[indices] for k, v in buffer._samples.items()})
            assert not context["valid"].any() and not context["memory_valid"].any()
    assert buffer.memory.session.tolist() == [0, 1, 0, 0]
    sampled = buffer.sample_batch(32, _normalization())
    changed = sampled["world_id"] == 1
    assert changed.any() and not sampled["center_pair_valid"][changed].any()
    assert (sampled["physics_session"][changed] == 1).all()
    assert not dr_center_pair_valid(sampled)[changed].any()
    torch.testing.assert_close(sampled["dr_metric"][changed], metrics[1].expand(int(changed.sum()), -1))


def test_real_objective_uses_only_center_relation_and_predictor_and_reencodes_archives(replay, monkeypatch):
    torch.manual_seed(11)
    model = Memory350Predictor(Memory350Config(
        transformer_dim=32, transformer_depth=1, transformer_heads=4,
        context_dim=16, context_heads=4, memory_depth=1))
    objective = DRCenterObjective(model, DRCenterLossConfig())
    batch = replay.sample_batch(16, _normalization(), positive_ready_only=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Old local/response/weak representation path was executed")

    monkeypatch.setattr("intact_tracking.memory350_objective._counterfactual_representation_loss", forbidden)
    monkeypatch.setattr("intact_tracking.memory350_weak_pairs.weak_pair_losses", forbidden)
    original_response = objective._representation_response
    monkeypatch.setattr(objective, "_representation_response", forbidden)
    views = objective._encode_views(batch)
    before = views[1].detach().clone()
    representation = objective._representation_terms(
        batch, views, batch["state"][:, 1:], compute_metrics=False)[0]
    context_parameters = list(model.context_encoder.named_parameters())
    representation_gradients = torch.autograd.grad(representation, [p for _, p in context_parameters])
    for level in ("chunk_encoder", "memory_encoder", "transformer"):
        gradients = [g for (name, _), g in zip(context_parameters, representation_gradients, strict=True)
                     if name.startswith(level)]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    result = objective(batch, compute_metrics=False, validate_batch=False)
    expected = result["prediction_loss"] + .02 * result["dr_center_relation_loss"]
    torch.testing.assert_close(result["loss"].detach(), expected)
    assert result["dr_center_pairs"] > 0
    corrupted_unused = {**batch, "nominal_state": torch.full_like(batch["nominal_state"], float("nan")),
                        "positive_history_state": torch.full_like(batch["positive_history_state"], float("nan"))}
    unchanged = objective(corrupted_unused, compute_metrics=False, validate_batch=False)
    torch.testing.assert_close(unchanged["loss"], result["loss"])
    result["loss"].backward()
    for level in ("chunk_encoder", "memory_encoder", "transformer"):
        gradients = [p.grad for name, p in model.context_encoder.named_parameters() if name.startswith(level)]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    torch.optim.SGD(model.parameters(), lr=.01).step()
    assert not torch.equal(objective._encode_views(batch)[1].detach(), before)
    monkeypatch.setattr(objective, "_representation_response", original_response)
    report = objective(batch)
    assert set(REPRESENTATION_METRICS).issubset(report)
    assert all(torch.isfinite(v) for v in report.values())
    old_probe = {k: v for k, v in batch.items() if k not in CENTER_BATCH_FIELDS}
    with pytest.raises(KeyError, match="DR-center batch fields"):
        objective(old_probe, compute_metrics=False, validate_batch=False)


def test_positive_objective_adds_independent_coefficient_and_reuses_current_views(replay, monkeypatch):
    torch.manual_seed(13)
    model = Memory350Predictor(Memory350Config(
        transformer_dim=32, transformer_depth=1, transformer_heads=4,
        context_dim=16, context_heads=4, memory_depth=1))
    config = DRCenterLossConfig(representation_weight=.2, dr_distance_scale=.2, dr_positive_weight=.1)
    objective = DRCenterObjective(model, config)
    batch = replay.sample_batch(16, _normalization(), positive_ready_only=True)
    # The replay fixture uses large step/world IDs to audit causal histories.
    # Bring those tags into an ordinary feature range for the encoder test.
    for name in ("state", "history_state", "history_action", "history_next_state", "memory_interactions",
                 "center_history_state", "center_history_action", "center_history_next_state",
                 "center_memory_interactions"):
        batch[name] = batch[name] * 1e-4
    eligible = dr_center_pair_valid(batch) & ~batch["is_nominal"]
    assert eligible.any()
    views = objective._encode_views(batch)
    for view in views:
        view.retain_grad()
    positive, metrics = objective._extra_representation_loss(batch, views)
    assert metrics["dr_positive_pairs"] == eligible.sum()
    positive.backward()
    assert all(v.grad[eligible].norm() > 0 for v in views)
    assert all(torch.equal(v.grad[~eligible], torch.zeros_like(v.grad[~eligible])) for v in views)
    for level in ("chunk_encoder", "memory_encoder", "transformer"):
        gradients = [p.grad for name, p in model.context_encoder.named_parameters() if name.startswith(level)]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    before = views[1].detach().clone()
    torch.optim.SGD(model.parameters(), lr=.01).step()
    assert not torch.equal(objective._encode_views(batch)[1].detach(), before)

    calls = []
    encode = model.encode_context
    def counted_encode(*args, **kwargs):
        calls.append(len(args[0]))
        return encode(*args, **kwargs)
    monkeypatch.setattr(model, "encode_context", counted_encode)
    result = objective(batch, compute_metrics=False, validate_batch=False)
    assert calls == [2 * len(batch["state"])]
    expected = (result["prediction_loss"] + .2 * result["dr_center_relation_loss"]
                + .1 * result["dr_positive_loss"])
    torch.testing.assert_close(result["loss"].detach(), expected)
    torch.testing.assert_close(result["dr_positive_weighted_loss"], .1 * result["dr_positive_loss"])
    no_positive = DRCenterObjective(model, DRCenterLossConfig(representation_weight=.2, dr_distance_scale=.2))
    reference = no_positive(batch, compute_metrics=False, validate_batch=False)
    for name in ("prediction_loss", "dr_center_relation_loss", "dr_positive_loss"):
        torch.testing.assert_close(result[name], reference[name])


def _schema_inputs():
    names = [f"base_com/com_offset/torso_link/{axis}" for axis in "xyz"]
    names += ["base_mass/relative_mass/torso_link", "foot_friction/friction/shared/0"]
    names += [f"motor/armature_scale/joint_{i}" for i in range(29)]
    names += [f"context_uniform_limb_payload/added_mass_kg/{limb}"
              for limb in ("left_hand", "right_hand", "left_shin", "right_shin")]
    params = {
        "base_com": {"operation": "add", "ranges": {i: (-.075, .075) for i in range(3)}},
        "base_mass": {"operation": "add", "ranges": (-1, 1)},
        "foot_friction": {"operation": "abs", "shared_random": True, "ranges": (.3, 2)},
        "motor": {"armature_range": {".*": (.8, 1.2)}, "mode": "uniform"},
        "context_uniform_limb_payload": {},
    }
    return names, params, {"torso_link": 5.}


def test_dr_metric_uses_fixed_ranges_and_equal_factor_weights_not_batch_variance():
    schema = dr_metric_schema(*_schema_inputs())
    raw = torch.tensor([schema["lower"], schema["upper"]])
    features = normalize_dr_metric(raw, schema)
    torch.testing.assert_close(features[0], torch.zeros(38))
    torch.testing.assert_close((features[1] - features[0]).norm(), torch.tensor(1.))
    low = raw[0].clone()
    one_load = low.clone(); one_load[-1] = raw[1, -1]
    all_armatures = low.clone(); all_armatures[5:34] = raw[1, 5:34]
    differences = normalize_dr_metric(torch.stack((low, one_load, all_armatures)), schema)
    torch.testing.assert_close(differences[1].norm(), torch.tensor(math.sqrt(.1)))
    torch.testing.assert_close(differences[2].norm(), differences[1].norm())
    # The relative torso-mass coordinate spans 2 kg / compiled mass, not 2 kg.
    assert schema["lower"][3] == -.2 and schema["upper"][3] == .2
    torch.testing.assert_close(normalize_dr_metric(raw[:1], schema), features[:1])
    raw[0, 0] -= 1
    with pytest.raises(ValueError, match="outside"):
        normalize_dr_metric(raw, schema)
    names, params, defaults = _schema_inputs()
    with pytest.raises(ValueError, match="No audited"):
        dr_metric_schema([*names, "motor/encoder_bias/joint_0"], params, defaults)


def test_dr_metric_uses_the_same_hand_caps_as_the_simulator():
    names, params, defaults = _schema_inputs()
    params["context_uniform_limb_payload"]["max_masses_kg"] = (2.5, 2.5, 4., 4.)
    schema = dr_metric_schema(names, params, defaults)
    assert schema["upper"][-4:] == [2.5, 2.5, 4., 4.]
    raw = torch.tensor(schema["lower"])[None].repeat(2, 1)
    raw[1, -4] = 2.5
    encoded = normalize_dr_metric(raw, schema)
    torch.testing.assert_close((encoded[1] - encoded[0]).norm(), torch.tensor(math.sqrt(.1)))
    raw[1, -4] = 2.6
    with pytest.raises(ValueError, match="outside"):
        normalize_dr_metric(raw, schema)


def test_cli_slices_center_fields_records_true_objective_and_protects_resume(tmp_path, monkeypatch):
    # Isolate the same module-alias setup used by all existing variant entries.
    for name in list(vars(trainer)):
        if not name.startswith("__"):
            monkeypatch.setattr(trainer, name, getattr(trainer, name))
    args = build_parser().parse_args([
        "--checkpoint-file", "tracker.pt", "--motion-path", "motions", "--output-dir", str(tmp_path)])
    configure_trainer(args)
    trainer._validate_arguments(args)
    assert (args.chunk_depth, args.memory_depth, args.context_depth) == (2, 4, 4)
    assert args.representation_weight == .02 and args.representation_relation_weight == 0
    assert args.replay_sampling == "uniform" and args.stop_after_updates is None
    assert args.until_user_stop
    assert args.limb_max_masses_kg == (2.5, 2.5, 4., 4.)
    smoke = copy.copy(args)
    smoke.bounded_smoke = True
    smoke.updates = 1
    trainer._validate_arguments(smoke)
    assert not smoke.until_user_stop
    batch = {name: torch.arange(8) for name in CENTER_BATCH_FIELDS}
    batch["state_mean"] = torch.zeros(71)
    sliced = trainer._slice_predictor_batch(batch, 2, 4)
    assert all(torch.equal(sliced[k], torch.tensor([2, 3])) for k in CENTER_BATCH_FIELDS)
    assert sliced["state_mean"] is batch["state_mean"]
    config = DRCenterLossConfig()
    assert DRCenterLossConfig(**asdict(config)) == config
    trainer._validate_resume_loss_config(asdict(config), asdict(config))
    with pytest.raises(ValueError, match="changed"):
        trainer._validate_resume_loss_config(asdict(config), asdict(DRCenterLossConfig(dr_distance_scale=.2)))
    schema = dr_metric_schema(*_schema_inputs())
    metadata = {
        "arguments": vars(args), "loss": asdict(config), "batch_a_rollout": {"dr_metric_schema": schema},
        **{key: {} for key in ("architecture", "memory_contract", "replay", "validation", "research_source_sha256")},
    }
    configure_run_metadata(metadata)
    assert not metadata["dr_center_contract"]["within_radius_loss"]
    assert metadata["objective_weights"]["effective_relation_weight"] == 0
    state = {"loss_config": asdict(config), "dr_profile": TRACKER_DR,
             "supervision_horizons": {"predictor": 5}, "privileged_dynamics": {}}
    configure_checkpoint(state, SimpleNamespace(dr_metric_schema=schema))
    assert not state["nominal_counterfactual_representation_supervision"]
    assert state["supervision_horizons"]["response_label"] is None
    validate_context_dr(state, TRACKER_DR)
    broken = {**state, "dr_metric_schema": {}}
    with pytest.raises(ValueError, match="provenance"):
        validate_context_dr(broken, TRACKER_DR)


@pytest.mark.parametrize("kwargs", [
    {"dr_distance_scale": 0}, {"dr_distance_scale": float("nan")},
    {"dr_relation_beta": -1}, {"representation_relation_weight": 2},
    {"representation_weight": float("inf")}, {"dr_center_objective_version": 2},
    {"dr_positive_weight": -1}, {"dr_positive_weight": float("nan")},
    {"dr_positive_weight": float("inf")},
])
def test_invalid_center_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        DRCenterLossConfig(**kwargs)


def test_retuning_only_changes_the_explicit_representation_coefficient():
    previous = asdict(DRCenterLossConfig())
    actual = asdict(DRCenterLossConfig(representation_weight=.04))
    with pytest.raises(ValueError, match="changed"):
        validate_resume_loss_config(previous, actual)
    validate_resume_loss_config(previous, actual, allow_weight_change=True)
    for field, value in (("dr_distance_scale", .2), ("dr_relation_beta", .1),
                         ("joint_position_weight", 2.)):
        with pytest.raises(ValueError, match="changed"):
            validate_resume_loss_config(previous, {**actual, field: value}, allow_weight_change=True)


def test_retuning_weight_and_scale_needs_both_explicit_options():
    previous = asdict(DRCenterLossConfig(representation_weight=.04))
    actual = asdict(DRCenterLossConfig(representation_weight=.2, dr_distance_scale=.2))
    for options in ({}, {"allow_weight_change": True}, {"allow_scale_change": True}):
        with pytest.raises(ValueError, match="changed"):
            validate_resume_loss_config(previous, actual, **options)
    validate_resume_loss_config(previous, actual, allow_weight_change=True, allow_scale_change=True)
    for field, value in (("dr_relation_beta", .1), ("joint_position_weight", 2.)):
        with pytest.raises(ValueError, match="changed"):
            validate_resume_loss_config(previous, {**actual, field: value},
                                        allow_weight_change=True, allow_scale_change=True)


def test_legacy_checkpoint_positive_default_is_zero_and_retuning_needs_explicit_option(monkeypatch):
    for name in list(vars(trainer)):
        if not name.startswith("__"):
            monkeypatch.setattr(trainer, name, getattr(trainer, name))
    previous = asdict(DRCenterLossConfig(representation_weight=.04))
    del previous["dr_positive_weight"]
    unchanged = asdict(DRCenterLossConfig(representation_weight=.04))
    validate_resume_loss_config(previous, unchanged)
    args = build_parser().parse_args([
        "--checkpoint-file", "tracker.pt", "--motion-path", "motions", "--output-dir", "new",
        "--resume", "old/update_010000.pt", "--resume-new-stage", "--representation-weight", ".2",
        "--dr-distance-scale", ".2", "--dr-positive-weight", ".1",
        "--retune-representation-weight", "--retune-dr-distance-scale", "--retune-dr-positive-weight"])
    configure_trainer(args)
    restored = trainer._normalize_resume_loss_config(previous)
    assert restored["dr_positive_weight"] == 0  # Must not inherit new CLI partial's .1.
    actual = asdict(trainer.ForwardPredictorLossConfig(representation_weight=.2))
    assert actual["dr_positive_weight"] == .1
    with pytest.raises(ValueError, match="changed"):
        validate_resume_loss_config(restored, actual, allow_weight_change=True, allow_scale_change=True)
    trainer._validate_resume_loss_config(restored, actual)
    validate_resume_loss_config(actual, actual)  # Ordinary new-checkpoint resume is protected too.
    with pytest.raises(ValueError, match="changed"):
        validate_resume_loss_config(actual, {**actual, "dr_positive_weight": .2})


def test_new_stage_allows_eight_rank_repartition_but_protects_global_batch_and_data():
    old_args = vars(build_parser().parse_args([
        "--checkpoint-file", "tracker.pt", "--motion-path", "motions", "--output-dir", "old"]))
    previous = {"arguments": old_args, "distributed": {"world_size": 4}}
    new_args = {**old_args, "resume": "old/update_001863.pt", "resume_new_stage": True,
                "output_dir": "new", "batch_size": 512}
    actual = {"arguments": new_args, "distributed": {"world_size": 8}}
    trainer._validate_resume_run_config(previous, actual)
    with pytest.raises(ValueError, match="global batch"):
        trainer._validate_resume_run_config(previous, {
            **actual, "arguments": {**new_args, "batch_size": 1024}})
    with pytest.raises(ValueError, match="batch_size"):
        trainer._validate_resume_run_config(previous, {
            **actual, "arguments": {**new_args, "resume_new_stage": False}})
    for field, value in (("seed", 22), ("motion_path", "other"), ("num_envs", 4096),
                         ("gradient_steps_per_update", 8)):
        with pytest.raises(ValueError, match=field):
            trainer._validate_resume_run_config(previous, {
                **actual, "arguments": {**new_args, field: value}})


@pytest.mark.parametrize("tune_scale", [False, True])
@pytest.mark.parametrize("tune_positive", [False, True])
def test_retuned_stage_reads_parent_contract_without_modifying_it(tmp_path, monkeypatch, tune_scale, tune_positive):
    for name in list(vars(trainer)):
        if not name.startswith("__"):
            monkeypatch.setattr(trainer, name, getattr(trainer, name))
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    args = build_parser().parse_args([
        "--checkpoint-file", "tracker.pt", "--motion-path", "motions", "--output-dir", str(source)])
    configure_trainer(args)
    metadata = {"arguments": vars(args), "loss": asdict(DRCenterLossConfig()),
                "batch_a_rollout": {"dr_metric_schema": dr_metric_schema(*_schema_inputs())},
                **{key: {} for key in ("architecture", "memory_contract", "replay", "validation", "research_source_sha256")}}
    configure_run_metadata(metadata)
    # Actual historical contracts predate all three positive-related keys.
    for key in ("dr_positive_weight", "dr_positive_pairing", "dr_positive_loss"):
        metadata["dr_center_contract"].pop(key)
    metadata["loss"].pop("dr_positive_weight")
    original = json.dumps(metadata)
    (source / "run_config.json").write_text(original)
    next_command = [
        "--checkpoint-file", "tracker.pt", "--motion-path", "motions", "--output-dir", str(destination),
        "--resume", str(source / "update_001863.pt"), "--resume-new-stage",
        "--retune-representation-weight", "--representation-weight", ".04"]
    if tune_scale:
        next_command += ["--retune-dr-distance-scale", "--dr-distance-scale", ".2"]
    if tune_positive:
        next_command += ["--retune-dr-positive-weight", "--dr-positive-weight", ".1"]
    next_args = build_parser().parse_args(next_command)
    configure_trainer(next_args)
    candidate = copy.deepcopy(metadata)
    candidate["arguments"] = vars(next_args)
    candidate["loss"] = asdict(DRCenterLossConfig(representation_weight=.04,
                                                dr_distance_scale=.2 if tune_scale else .3,
                                                dr_positive_weight=.1 if tune_positive else 0.))
    configure_run_metadata(candidate)
    assert candidate["representation_weight_tuning"]["factor"] == 2
    assert candidate["dr_center_contract"]["representation_weight"] == .04
    if tune_scale:
        assert candidate["dr_distance_scale_tuning"]["previous_scale"] == .3
        assert candidate["dr_distance_scale_tuning"]["scale"] == .2
        assert candidate["dr_center_contract"]["dr_distance_scale"] == .2
        assert not candidate["representation_weight_tuning"]["other_losses_and_dr_metric_preserved"]
        unapproved_scale = copy.deepcopy(candidate)
        unapproved_scale["arguments"]["retune_dr_distance_scale"] = False
        with pytest.raises(ValueError, match="same DR-center objective"):
            configure_run_metadata(unapproved_scale)
    if tune_positive:
        assert candidate["dr_positive_weight_tuning"]["previous_weight"] == 0
        assert candidate["dr_positive_weight_tuning"]["weight"] == .1
        assert candidate["dr_center_contract"]["weak_positive_loss"]
        assert candidate["objective_weights"]["dr_cross_motion_positive_total_weight"] == .1
        assert candidate["objective_weights"]["weak_negative_weight"] == 0
        unapproved_positive = copy.deepcopy(candidate)
        unapproved_positive["arguments"]["retune_dr_positive_weight"] = False
        with pytest.raises(ValueError, match="same DR-center objective"):
            configure_run_metadata(unapproved_positive)
    assert (source / "run_config.json").read_text() == original
    assert not destination.exists()
    changed = copy.deepcopy(candidate)
    changed["batch_a_rollout"]["dr_metric_schema"]["upper"][-4] = 3.
    with pytest.raises(ValueError, match="metric schema"):
        configure_run_metadata(changed)
    no_opt_in = copy.copy(next_args)
    no_opt_in.resume_new_stage = False
    if tune_positive:
        no_opt_in.retune_representation_weight = no_opt_in.retune_dr_distance_scale = False
    elif tune_scale:
        no_opt_in.retune_representation_weight = False
    with pytest.raises(ValueError, match="resume-new-stage"):
        configure_trainer(no_opt_in)
