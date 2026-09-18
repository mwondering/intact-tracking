"""Compare fixed factorized gates: four two-coordinate groups, 16 experts total."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from intact_tracking.router_information_probe import (
    affine_predict, gate_summary, project, readout_amplification,
    regression_metrics, select_readout, sparse_renormalize,
)
from probe_moe_latent_payload import load_case


def factorized_weights(features, mode):
    """Coordinates: two hands, two shins, COM x/y, COM z/friction."""
    features = np.clip(np.asarray(features, np.float32), 0, 1)
    pairs = features.reshape(*features.shape[:-1], 4, 2)
    u, v = pairs[..., 0], pairs[..., 1]
    if mode == 'triangular':
        low = u+v <= 1
        weights = np.stack((np.maximum(1-u-v, 0), np.where(low, u, 1-v),
                            np.where(low, v, 1-u), np.maximum(u+v-1, 0)), -1)
    else:
        weights = np.stack(((1-u)*(1-v), u*(1-v), (1-u)*v, u*v), -1)
        if mode == 'hard':
            weights = np.eye(4, dtype=np.float32)[weights.argmax(-1)]
        elif mode == 'top2':
            weights = sparse_renormalize(weights, 2)
        elif mode != 'bilinear':
            raise ValueError(mode)
    return (weights/4).reshape(*features.shape[:-1], 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    source, root = Path(args.source), Path(args.root)
    a, b = (load_case(source, 'sample', seed) for seed in (30402, 30401))
    with np.load(root/'models/decoded_dr8.npz') as f:
        projection = {key: f[key] for key in f.files}
    meta = json.loads((source/'sample_seed30402/result.json').read_text())
    schema = meta['dr_schema']
    lower, width = np.asarray(schema['lower']), np.asarray(schema['upper'])-schema['lower']
    order = np.random.default_rng(887).permutation(len(a['raw_dr']))
    fit = np.isin(a['worlds'], order[:768])
    y = (a['raw_dr'][a['worlds']]-lower)/width
    test_y = (b['raw_dr'][b['worlds']]-lower)/width
    features, test_features = project(a['latent'], projection), project(b['latent'], projection)
    rows = []
    for mode in ('hard', 'top2', 'triangular', 'bilinear'):
        weights, test_weights = factorized_weights(features, mode), factorized_weights(test_features, mode)
        model, validation, grid = select_readout(weights[fit], y[fit], weights[~fit], y[~fit], schema['groups'])
        row = {'name': f'factorized_{mode}', 'mode': mode, 'metric': 'decoded_dr8',
               'supervision': 'physical_DR_supervised', 'validation': validation,
               'exploratory_test': regression_metrics(test_y, affine_predict(test_weights, model), schema['groups']),
               'test_gate': gate_summary(test_weights), 'amplification': readout_amplification(model), 'readout_grid': grid}
        rows.append(row)
        np.savez(root/'readouts'/f'factorized_{mode}.npz', **model)
        print(json.dumps({'mode':mode,'test_strong8_r2':row['exploratory_test']['strong8_r2'],
                          'test_load_r2':row['exploratory_test']['load_r2'],
                          'active_experts':row['test_gate']['active_experts_per_point'],
                          'prototype_span':row['amplification']['strong8_mean_prototype_span_in_DR_ranges']}), flush=True)
    result = {'protocol': 'Same 768/256 training/validation worlds and independent exploratory seed as the metric sweep. Projection fitted on latent to normalized DR8; at inference features are predicted from latent, never the true test DR.',
              'groups': [['left_hand','right_hand'], ['left_shin','right_shin'], ['COM_x','COM_y'], ['COM_z','friction']],
              'expert_semantics': 'Four groups each with four corner experts; the policy would sum four environment-factor corrections. Experts no longer correspond to a partition into complete DR environments.',
              'PPO_gradients_to_router':False, 'all_gates_nonnegative_sum_one':True,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rows':rows}
    (root/'factorized.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
