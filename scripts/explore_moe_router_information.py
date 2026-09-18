"""Compare non-PPO metric transforms and sparse/dense continuous gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from intact_tracking.router_information_probe import (
    STRONG_COORDINATES, affine_predict, fit_projection, fit_euclidean_centers,
    gate_weights, gate_summary, project, readout_amplification, regression_metrics,
    select_readout, squared_distances,
)
from probe_moe_latent_payload import load_case


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda a:a.tolist())+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source, root = Path(args.source), Path(args.output)
    root.mkdir(exist_ok=True)
    (root/'models').mkdir(exist_ok=True)
    (root/'readouts').mkdir(exist_ok=True)
    a, b = (load_case(source, 'sample', seed) for seed in (30402, 30401))
    meta = json.loads((source/'sample_seed30402/result.json').read_text())
    schema = meta['dr_schema']
    lower, width = np.asarray(schema['lower']), np.asarray(schema['upper'])-schema['lower']
    order = np.random.default_rng(887).permutation(len(a['raw_dr']))
    fit = np.isin(a['worlds'], order[:768])
    validation = ~fit
    assert np.all(np.bincount(a['worlds']) == 32) and np.all(np.bincount(b['worlds']) == 32)
    x, val_x, test_x = a['latent'][fit], a['latent'][validation], b['latent']
    all_y = (a['raw_dr'][a['worlds']]-lower)/width
    y, val_y = all_y[fit], all_y[validation]
    test_y = (b['raw_dr'][b['worlds']]-lower)/width
    groups = schema['groups']
    protocol = {
        'policy_checkpoint_sha256': meta['checkpoint_sha256'], 'context_sha256': meta['context_sha256'],
        'source': str(source.resolve()), 'fit_seed': 30402, 'exploratory_test_seed': 30401,
        'fit_world_ids': order[:768].tolist(), 'validation_world_ids': order[768:].tolist(),
        'samples_per_world': 32, 'selection': 'Fit transforms/centers/readouts on 768 worlds; select readout alpha and gate settings only by 256 validation worlds. Seed30401 is an exploratory generalization report, never part of fitting.',
        'strong_coordinates': STRONG_COORDINATES,
        'strong_coordinate_selection': 'Four loads, all three COM offsets, friction; identified as readable in preceding diagnostic, before this candidate sweep. All67 parameter metrics retained.',
        'candidate_selection_rule': 'For each top_k and supervision family, find best validation strong8 R2; within0.02 of it choose smallest decoder prototype span, requiring each mean gate weight>=0.005 and <=0.25.',
        'heldout_seed30403': 'Reserved until selected.json has been written. Its latent/DR outcomes are not read here.',
        'PPO_gradients_to_router': False,
        'hypotheses': ['metric scaling can expose low-amplitude environment directions',
                       'continuous expert weights retain information removed by argmin',
                       'high temperature can hide information in tiny weight changes, so decoder amplification is reported'],
        'source_sha256': {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in
                          ['scripts/explore_moe_router_information.py', 'src/intact_tracking/router_information_probe.py', 'scripts/probe_moe_latent_payload.py']},
    }
    save_json(root/'protocol.json', protocol)
    full_decoder, full_validation, decoder_grid = select_readout(x, y, val_x, val_y, groups)
    full_test = regression_metrics(test_y, affine_predict(test_x, full_decoder), groups)
    np.savez(root/'full_latent_decoder.npz', **full_decoder)
    save_json(root/'full_latent_decoder.json', {'validation': full_validation, 'exploratory_test': full_test, 'alpha_grid': decoder_grid})
    identity = fit_projection(x, a['worlds'][fit], 'identity')
    projections = [('saved', identity, 'existing'), ('raw', identity, 'unsupervised')]
    for floor in (1e-2, 1e-4, 1e-6):
        projections.append((f'whiten_f{floor:g}', fit_projection(x, a['worlds'][fit], 'whiten', floor=floor), 'same_world_only'))
    for dimension in (8, 16, 64):
        projections.append((f'reliability{dimension}', fit_projection(x, a['worlds'][fit], 'reliability', floor=1e-6, dimension=dimension), 'same_world_only'))
    projections.append(('decoded_dr8', {'matrix': full_decoder['matrix'][:, STRONG_COORDINATES],
                                      'offset': full_decoder['offset'][STRONG_COORDINATES],
                                      'kind': 'physical_DR_supervised', 'clip_unit_range': True}, 'physical_DR_supervised'))
    rows = []
    for metric_name, projection, supervision in projections:
        projected, val_projected, test_projected = (project(data, projection) for data in (x, val_x, test_x))
        if metric_name == 'saved':
            with np.load(source/'sample_seed30402/traces.npz') as f:
                centers = f['centers']
            inertia = float(squared_distances(projected, centers).min(-1).mean())
        else:
            centers, inertia = fit_euclidean_centers(projected)
        center_distance = squared_distances(centers, centers)
        np.fill_diagonal(center_distance, np.inf)
        temperature_scale = float(np.median(center_distance.min(-1)))
        assert temperature_scale > 0
        distances = [squared_distances(data, centers) for data in (projected, val_projected, test_projected)]
        np.savez(root/'models'/f'{metric_name}.npz', matrix=projection['matrix'], offset=projection['offset'], centers=centers,
                 clip_unit_range=projection.get('clip_unit_range', False), normalize_after_projection=projection.get('normalize_after_projection', False))
        save_json(root/'models'/f'{metric_name}.json', {'projection': projection, 'supervision': supervision,
                                                       'training_inertia': inertia, 'temperature_scale': temperature_scale})
        for top_k in (1, 2, 4, 8, 16):
            for multiplier in ((1.,) if top_k == 1 else (.1, .3, 1., 3.)):
                name = f'{metric_name}_k{top_k}_t{multiplier:g}'
                temperature = temperature_scale*multiplier
                weights, val_weights, test_weights = [gate_weights(distance, top_k=top_k, temperature=temperature) for distance in distances]
                model, val_metrics, grid = select_readout(weights, y, val_weights, val_y, groups)
                test_metrics = regression_metrics(test_y, affine_predict(test_weights, model), groups)
                row = {'name': name, 'metric': metric_name, 'supervision': supervision, 'top_k': top_k,
                       'temperature_multiplier': multiplier, 'temperature': temperature,
                       'readout_alpha': model['alpha'], 'readout_validation_grid': grid,
                       'validation': val_metrics, 'exploratory_test': test_metrics,
                       'validation_gate': gate_summary(val_weights), 'test_gate': gate_summary(test_weights),
                       'amplification': readout_amplification(model)}
                rows.append(row)
                np.savez(root/'readouts'/f'{name}.npz', **model)
        save_json(root/'sweep.json', rows)
        best = max([row for row in rows if row['metric'] == metric_name], key=lambda row:row['validation']['strong8_r2'])
        print(json.dumps({'metric': metric_name, 'best': best['name'],
                          'validation_strong8_r2': best['validation']['strong8_r2'],
                          'test_strong8_r2': best['exploratory_test']['strong8_r2'],
                          'test_load_r2': best['exploratory_test']['load_r2'],
                          'readout_prototype_span': best['amplification']['strong8_mean_prototype_span_in_DR_ranges']}), flush=True)
    selected = {'source_protocol_sha256': hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(), 'candidates': []}
    for family in ('without_numerical_DR_labels', 'physical_DR_supervised'):
        for top_k in (1, 2, 4, 8, 16):
            candidates = [row for row in rows if row['top_k'] == top_k and
                          ((row['supervision'] == 'physical_DR_supervised') == (family == 'physical_DR_supervised')) and
                          row['validation_gate']['minimum_expert_weight'] >= .005 and
                          row['validation_gate']['maximum_expert_weight'] <= .25]
            best_score = max(row['validation']['strong8_r2'] for row in candidates)
            candidates = [row for row in candidates if row['validation']['strong8_r2'] >= best_score-.02]
            chosen = min(candidates, key=lambda row:row['amplification']['strong8_mean_prototype_span_in_DR_ranges'])
            selected['candidates'].append({'family': family, **{key: chosen[key] for key in ['name','metric','top_k','temperature','temperature_multiplier']}})
    # Keep the current hard router and a softening-only control explicit.
    for name in ['saved_k1_t1', 'saved_k4_t0.3', 'saved_k16_t0.3']:
        row = next(row for row in rows if row['name'] == name)
        selected['candidates'].append({'family':'controls', **{key:row[key] for key in ['name','metric','top_k','temperature','temperature_multiplier']}})
    save_json(root/'selected.json', selected)
    print(json.dumps({'selected': selected['candidates'], 'candidate_count': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
