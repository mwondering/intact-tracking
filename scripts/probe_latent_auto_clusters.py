"""Explore automatic cluster counts on fixed, calibrated environment latents."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.cluster import AffinityPropagation, HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture
import torch
import torch.nn.functional as F


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def describe(model, labels, loads):
    groups, sizes = np.unique(labels[labels >= 0], return_counts=True)
    hands = np.unique(loads[:, :2], axis=0)
    result = {
        'clusters': len(groups), 'cluster_sizes': sizes.tolist(),
        'noise_count': int((labels < 0).sum()), 'noise_fraction': float((labels < 0).mean()),
        'hand_patterns_with_one_common_nonnoise_class_across_all_16_leg_settings': int(sum(
            len(np.unique(labels[np.all(loads[:, :2] == hand, axis=1)])) == 1
            and labels[np.all(loads[:, :2] == hand, axis=1)][0] >= 0 for hand in hands)),
    }
    if isinstance(model, BayesianGaussianMixture):
        result.update(converged=bool(model.converged_), iterations=int(model.n_iter_),
                      weights=model.weights_.tolist(),
                      components_above_one_anchor_expected_weight=int((model.weights_ > 1 / len(labels)).sum()),
                      max_components=model.n_components,
                      all_component_slots_occupied=len(groups) == model.n_components)
    elif isinstance(model, AffinityPropagation):
        result.update(iterations=int(model.n_iter_), exemplar_anchor_ids=model.cluster_centers_indices_.tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source, output = args.calibration.resolve(), args.output.resolve()
    saved = torch.load(source / 'prototypes.pt', map_location='cpu', weights_only=False, mmap=True)
    samples = torch.load(source / 'calibration_samples.pt', map_location='cpu', weights_only=False, mmap=True)
    metadata = json.loads((source / 'report.json').read_text())
    assert saved['version'] == 'payload256_cross_world_prototypes_v1'
    unit = F.normalize(samples['latent'].reshape(-1, 64).float(), dim=-1)
    ids, fold, full = samples['ids'], (samples['worlds'] // 256) % 4, samples['full'].flatten()
    assert torch.equal(ids, samples['worlds'] % 256)
    fitting = (fold < 2) & full
    validation = (fold == 2) & full
    fitting_centers = torch.stack([unit[fitting & (ids == i)].mean(0) for i in range(256)])
    torch.testing.assert_close(fitting_centers, saved['centers'], atol=1e-7, rtol=0)
    validation_centers = torch.stack([unit[validation & (ids == i)].mean(0) for i in range(256)])
    assert torch.isfinite(validation_centers).all()
    x, v = fitting_centers.numpy().astype(np.float64), validation_centers.numpy().astype(np.float64)
    loads = saved['loads_kg'].numpy().astype(np.float64)
    sq_distance = np.maximum((x*x).sum(1)[:, None] + (x*x).sum(1)[None] - 2*x@x.T, 0)
    median_preference = float(np.median(-sq_distance))
    candidates = []
    for size in (8, 16, 32):
        candidates.append((f'hdbscan_min{size}', HDBSCAN, dict(
            min_cluster_size=size, min_samples=5, metric='euclidean',
            cluster_selection_method='eom', allow_single_cluster=True, store_centers='medoid', n_jobs=1)))
    for factor in (.5, 1., 2.):
        candidates.append((f'affinity_preference_x{factor:g}', AffinityPropagation, dict(
            preference=median_preference*factor, damping=.9, max_iter=1000,
            convergence_iter=50, random_state=731)))
    for maximum in (16, 32):
        for prior in (.1, 1.):
            candidates.append((f'bayesian_max{maximum}_prior{prior:g}', BayesianGaussianMixture, dict(
                n_components=maximum, covariance_type='diag', weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=prior, reg_covar=1e-6, max_iter=1000, n_init=3, random_state=731)))
    output.mkdir(parents=True, exist_ok=False)
    artifacts = dict(prototype_latents=x, independent_world_prototype_latents=v, loads_kg=loads)
    summary = {
        'source': str(source), 'source_hashes': {name: digest(source/name) for name in
                                              ('prototypes.pt', 'calibration_samples.pt', 'report.json')},
        'encoder_sha256': metadata['context_sha256'], 'script_sha256': digest(Path(__file__)),
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        'method': 'Fit on 256 equal-weight reference environment prototypes (mean unit 64-D latent). '
                  'No whitening, PCA, DR coordinates or query scores used by clustering. All settings fixed '
                  'before inspecting results. Refit each setting on prototypes from independent validation '
                  'worlds at the same load anchors; ARI compares partitions modulo label permutations.',
        'scope': 'A four-level-per-limb load grid, nominal background, full memory, frozen tracker. '
                 'Counts can reflect grid spacing and chosen resolution. Refit ARI is clustering reproducibility, '
                 'not frozen-router query accuracy, unseen-load generalization, control performance or an optimal '
                 'expert count. Validation prototypes use four worlds per anchor versus eight reference worlds. '
                 'Test fold 3 is not fitted or scored. Noise is not silently assigned to an expert.',
        'results': [],
    }
    for name, constructor, settings in candidates:
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model = constructor(**settings).fit(x)
            other = constructor(**settings).fit(v)
        labels = model.predict(x) if isinstance(model, BayesianGaussianMixture) else model.labels_
        validation_labels = other.predict(v) if isinstance(other, BayesianGaussianMixture) else other.labels_
        common = (labels >= 0) & (validation_labels >= 0)
        row = {
            'name': name, 'settings': settings, 'reference': describe(model, labels, loads),
            'independent_world_refit': describe(other, validation_labels, loads),
            'refit_ari_all_anchors_noise_as_one_label': float(adjusted_rand_score(labels, validation_labels)),
            'refit_common_nonnoise_anchors': int(common.sum()),
            'refit_ari_common_nonnoise': float(adjusted_rand_score(labels[common], validation_labels[common]))
                                      if common.sum() >= 2 else None,
            'warnings': [str(w.message) for w in captured], 'seconds': time.monotonic()-started,
        }
        summary['results'].append(row)
        artifacts[name+'_reference_labels'] = labels
        artifacts[name+'_independent_world_labels'] = validation_labels
        if isinstance(model, HDBSCAN):
            artifacts[name+'_medoids'] = model.medoids_
            artifacts[name+'_membership_strength'] = model.probabilities_
        elif isinstance(model, AffinityPropagation):
            artifacts[name+'_exemplar_anchor_ids'] = model.cluster_centers_indices_
        else:
            artifacts[name+'_mixture_means'] = model.means_
            artifacts[name+'_mixture_weights'] = model.weights_
        print(json.dumps({'name': name, 'clusters': row['reference']['clusters'],
                          'noise': row['reference']['noise_count'], 'sizes': row['reference']['cluster_sizes'],
                          'refit_clusters': row['independent_world_refit']['clusters'],
                          'refit_ari': row['refit_ari_all_anchors_noise_as_one_label'],
                          'warnings': row['warnings']}, ensure_ascii=False), flush=True)
    np.savez_compressed(output/'partitions.npz', **artifacts)
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
