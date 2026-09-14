"""Verify the trained short-only control has no long-history information path."""

from dataclasses import asdict
import json

import pytest
import torch

from intact_tracking.memory350_model import Memory350Config, Memory350Predictor
from intact_tracking.memory350_objective import Memory350Objective
from intact_tracking.forward_predictor_objective import ForwardPredictorLossConfig
from intact_tracking.short50_model import Short50Config, Short50Predictor
from intact_tracking.short50_comparison import reference_normalization, reference_probes
from test_memory350 import _replay_batch, _normalization


def configs():
    common = dict(transformer_dim=32, transformer_depth=1, transformer_heads=4,
                  context_dim=16, context_heads=4)
    return Memory350Config(**common), Short50Config(**common)


def test_fresh_common_weights_and_rng_match_memory350_exactly():
    memory_config, short_config = configs()
    torch.manual_seed(717)
    reference = Memory350Predictor(memory_config)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(717)
    short = Short50Predictor(short_config)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert not any('chunk_encoder' in name or 'memory_encoder' in name for name, _ in short.named_parameters())
    source = reference.state_dict()
    for name, value in short.state_dict().items():
        expected = source[name]
        if name == 'context_encoder.position':
            expected = expected[:, [0, *range(2, 52)]]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert Short50Config(**asdict(short_config)) == short_config


def encode(model, batch):
    return model.encode_context(batch['history_state'], batch['history_action'], batch['state'][:,0],
                                batch['history_valid'], history_next_state=batch['history_next_state'],
                                memory_interactions=batch['memory_interactions'], memory_valid=batch['memory_valid'])


def test_model_ignores_long_memory_even_when_values_and_validity_change():
    _, config = configs()
    model = Short50Predictor(config).eval()
    batch = _replay_batch()
    batch['memory_interactions'].requires_grad_(True)
    expected = encode(model, batch)
    expected.square().sum().backward()
    assert batch['memory_interactions'].grad is None
    changed = {**batch, 'memory_interactions': torch.full_like(batch['memory_interactions'], float('nan')),
               'memory_valid': ~batch['memory_valid']}
    torch.testing.assert_close(encode(model, changed), expected, rtol=0, atol=0)


def test_only_51_tokens_and_padding_do_not_leak_invalid_history():
    _, config = configs()
    model = Short50Predictor(config).eval()
    batch = _replay_batch()
    lengths = []
    hook = model.context_encoder.transformer.register_forward_pre_hook(lambda _, args: lengths.append(args[0].shape[1]))
    expected = encode(model, batch)
    changed = dict(batch)
    for key in ('history_state','history_action','history_next_state'):
        changed[key] = batch[key].clone()
        changed[key][~batch['history_valid']] = float('nan')
    torch.testing.assert_close(encode(model, changed), expected, rtol=0, atol=0)
    hook.remove()
    assert lengths == [51, 51]
    empty = {**changed, 'history_valid': torch.zeros_like(batch['history_valid'])}
    assert torch.isfinite(encode(model, empty)).all()


def test_short_objective_trains_all_retained_parameters_with_padded_history():
    _, config = configs()
    model = Short50Predictor(config)
    objective = Memory350Objective(model, ForwardPredictorLossConfig(
        representation_relation_weight=2, response_distance_scale=.75))
    batch = _replay_batch()
    assert not batch['history_valid'].all(1).any()
    result = objective(batch)
    result['loss'].backward()
    assert result['latent_relation_pairs'] > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_reference_normalization_and_validation_world_guard(tmp_path):
    reference, output = tmp_path/'reference', tmp_path/'output'
    reference.mkdir(); output.mkdir()
    stats = _normalization()
    value = asdict(stats); value['world_ids'] = [0]
    (reference/'normalization.json').write_text(json.dumps(value))
    norm = reference_normalization(reference, (0,))
    with pytest.raises(ValueError, match='worlds differ'):
        reference_normalization(reference, (1,))
    batch = _replay_batch()
    batch['world_id'].fill_(1)
    for prefix in ('validation','validation_broad'):
        torch.save(batch, reference/f'{prefix}_rank_0.pt')
    _, _, evidence = reference_probes(reference, output, 0, 2, 1, norm, 'cpu')
    assert len(evidence) == 2
    batch['world_id'].fill_(0)
    torch.save(batch, reference/'validation_rank_0.pt')
    with pytest.raises(ValueError, match='non-validation'):
        reference_probes(reference, output, 0, 2, 1, norm, 'cpu')


def test_cli_defaults_match_reference_training_and_support_separate_probe_source():
    from intact_tracking.cli.forward_memory_train import build_parser as memory_parser
    from intact_tracking.cli.forward_short50_train import build_parser, _validate_arguments
    args = ['--checkpoint-file','tracker.pt','--motion-path','motions','--output-dir','new-run']
    old = vars(memory_parser().parse_args(args))
    new = build_parser().parse_args(args+['--comparison-reference-dir','reference'])
    _validate_arguments(new)
    actual = vars(new).copy(); actual.pop('comparison_reference_dir')
    assert old == actual
