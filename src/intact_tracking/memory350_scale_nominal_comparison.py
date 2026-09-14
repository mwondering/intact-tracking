"""Require nominal50 physics as well as the existing encoder-depth controls."""

import json
from pathlib import Path

from intact_tracking.memory350_scale_comparison import (
    PAIRED_UPDATES, reference_contract as _depth_contract,
    reference_normalization, reference_probes as _reference_probes,
)


def reference_contract(reference, actual):
    source = json.loads((Path(reference) / 'run_config.json').read_text())
    if source['arguments'].get('nominal_fraction') != .5 or actual['arguments'].get('nominal_fraction') != .5:
        raise ValueError('Both encoder-depth controls require 50% nominal A worlds')
    result = _depth_contract(reference, actual)
    for a, b in zip(source['dataset']['runtime_audits_by_rank'],
                    actual['dataset']['runtime_audits_by_rank'], strict=True):
        for key in ('nominal_training_worlds', 'dr_training_worlds',
                    'nominal_validation_worlds', 'dr_validation_worlds'):
            if a[key] != b[key]:
                raise ValueError(f'Nominal50 reference partition mismatch: {key}')
        mixture = b['physics']['nominal_mixture']
        if a['physics']['nominal_mixture'] != mixture:
            raise ValueError('Nominal50 physics audit differs from reference')
        if (mixture['nominal_count'] != b['num_envs'] // 2
                or mixture['dr_count'] != b['num_envs'] // 2
                or mixture['nominal_payload_max_abs_kg'] != 0
                or not mixture['nominal_pulses_disabled']
                or not mixture['dr_physics_unchanged_from_original_sampling']
                or mixture['nominal_restore']['model_field_max_abs_error'] != 0
                or mixture['nominal_restore']['encoder_bias_max_abs_error'] != 0):
            raise ValueError('Nominal50 physics restoration did not pass')
    result['nominal_a_fraction'] = .5
    result['nominal_mixture_matched'] = True
    return result


def reference_probes(reference, output, rank, num_envs, validation_worlds, normalization, device):
    fixed, broad, evidence = _reference_probes(
        reference, output, rank, num_envs, validation_worlds, normalization, device)
    for batch in (fixed, broad):
        expected = batch['world_id'].remainder(2).eq(0)
        if (not bool((batch['is_nominal'] == expected).all())
                or not bool(expected.any()) or not bool((~expected).any())):
            raise ValueError('Reference probe must contain correctly labeled nominal and DR worlds')
    return fixed, broad, evidence
