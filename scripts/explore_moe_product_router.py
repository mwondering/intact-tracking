"""Four KMeans groups preserve 16 experts while routing several factors at once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from intact_tracking.router_information_probe import (
    affine_predict, fit_euclidean_centers, gate_weights, gate_summary, project,
    readout_amplification, regression_metrics, select_readout, squared_distances,
)
from probe_moe_latent_payload import load_case


def product_weights(features, centers, temperatures, group_top_k):
    return np.concatenate([gate_weights(squared_distances(features[..., 2*g:2*g+2], centers[g]),
                                        top_k=group_top_k, temperature=temperatures[g])/4
                           for g in range(4)], axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source, root = Path(args.source), Path(args.root)
    a, b = (load_case(source, 'sample', seed) for seed in (30402, 30401))
    with np.load(root/'models/decoded_dr8.npz') as f:
        projection = {key: f[key] for key in f.files}
    schema = json.loads((source/'sample_seed30402/result.json').read_text())['dr_schema']
    lower, width = np.asarray(schema['lower']), np.asarray(schema['upper'])-schema['lower']
    order = np.random.default_rng(887).permutation(len(a['raw_dr']))
    fit = np.isin(a['worlds'], order[:768])
    y, test_y = [(case['raw_dr'][case['worlds']]-lower)/width for case in (a,b)]
    features, test_features = project(a['latent'], projection), project(b['latent'], projection)
    centers, scales = [], []
    for g in range(4):
        center, _ = fit_euclidean_centers(features[fit, 2*g:2*g+2], k=4)
        centers.append(center)
        distances = squared_distances(center, center)
        np.fill_diagonal(distances, np.inf)
        scales.append(float(np.median(distances.min(-1))))
    centers, scales = np.asarray(centers), np.asarray(scales)
    np.savez(root/'models/product_dr8.npz', matrix=projection['matrix'], offset=projection['offset'],
             clip_unit_range=True, centers=centers, temperature_scales=scales)
    rows, selected = [], []
    for group_k in (1,2,3,4):
        for multiplier in ((1.,) if group_k == 1 else (.1,.3,1.,3.)):
            name = f'product_dr8_gk{group_k}_t{multiplier:g}'
            temperatures = scales*multiplier
            weights = product_weights(features, centers, temperatures, group_k)
            test_weights = product_weights(test_features, centers, temperatures, group_k)
            model, validation, grid = select_readout(weights[fit], y[fit], weights[~fit], y[~fit], schema['groups'])
            row = {'name':name,'metric':'product_dr8','kind':'product_kmeans','group_top_k':group_k,
                   'temperature_multiplier':multiplier,'temperatures':temperatures.tolist(),
                   'validation':validation,'exploratory_test':regression_metrics(test_y,affine_predict(test_weights,model),schema['groups']),
                   'validation_gate':gate_summary(weights[~fit]),'test_gate':gate_summary(test_weights),
                   'amplification':readout_amplification(model),'readout_grid':grid}
            rows.append(row)
            np.savez(root/'readouts'/f'{name}.npz', **model)
        options=[row for row in rows if row['group_top_k']==group_k]
        best=max(row['validation']['strong8_r2'] for row in options)
        chosen=min([row for row in options if row['validation']['strong8_r2']>=best-.02],
                   key=lambda row:row['amplification']['strong8_mean_prototype_span_in_DR_ranges'])
        selected.append({key:chosen[key] for key in ['name','metric','kind','group_top_k','temperature_multiplier','temperatures']})
        print(json.dumps({'selected':chosen['name'],'test_R2':chosen['exploratory_test']['strong8_r2'],
                          'load_R2':chosen['exploratory_test']['load_r2'],
                          'amplification':chosen['amplification']['strong8_mean_prototype_span_in_DR_ranges']}),flush=True)
    result={'protocol':'Same fit/validation/exploratory worlds as initial sweep. Four 2D KMeans groups with four centers each; each group contributes exactly one quarter of total weight. Centers fitted without PPO gradients; compatible with later groupwise online updates.',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,'selected':selected}
    (root/'product.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
