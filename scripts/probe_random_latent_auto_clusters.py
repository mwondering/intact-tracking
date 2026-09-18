"""Automatic clustering of randomly selected individual rollout latents; no prototypes."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import sklearn
from sklearn.cluster import HDBSCAN, AffinityPropagation
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    readout, output = args.readout.resolve(), args.output.resolve()
    provenance = json.loads((readout/'summary.json').read_text())
    source = Path(provenance['source'])
    with np.load(source/'latents.npz', allow_pickle=False) as saved:
        data = {k: saved[k] for k in ('world', 'nominal', 'short_steps', 'long_chunks', 'step', 'motion', 'source_row')}
    with np.load(readout/'predictions.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['all_source_rows'], data['source_row'])
        z = saved['latent_current'].astype(np.float64)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    eligible = (~data['nominal']) & (data['short_steps'] == 50) & (data['long_chunks'] == 30)
    worlds = np.unique(data['world'][eligible])
    with np.load(provenance['physics']['path'], allow_pickle=False) as saved:
        physics, names = saved['values'], saved['names']
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        'encoder': provenance['checkpoint'],
        'source_readout': str(readout), 'source_readout_summary_sha256': digest(readout/'summary.json'),
        'source_latents_sha256': digest(readout/'predictions.npz'),
        'script_sha256': digest(Path(__file__)), 'sklearn_version': sklearn.__version__,
        'eligible_windows': int(eligible.sum()), 'worlds': len(worlds),
        'method': 'For each fixed random seed, uniformly select ONE actual complete-history latent per '
                  'continuous random DR world. No per-world averaging, environment prototypes, PCA, whitening '
                  'or DR labels in clustering. Unit-normalized raw 64-D latents. Three draws use the same worlds '
                  'but independently selected windows; all settings fixed before results.',
        'scope': 'Current u35857 encoder, CPU FP32 re-encoding of saved LaFAN frozen-tracker histories; '
                 'full tracker background DR plus continuous hand [0,2.5] / shin [0,4] loads, not load-only PPO. '
                 'Repeated draws measure sensitivity to selected histories, not independent environments or '
                 'policy training seeds. No online router, expert or simulation changes. '
                 'One sample per world avoids repeated samples inflating individual-world density.',
        'results': [], 'across_draws': [],
    }
    artifacts = dict(world_ids=worlds, physics=physics[worlds], physics_names=names)
    models_by_name = {}
    for seed in (731, 1731, 2731):
        rng = np.random.default_rng(seed)
        rows = np.array([rng.choice(np.flatnonzero(eligible & (data['world'] == w))) for w in worlds])
        x = z[rows]
        artifacts[f'seed{seed}_source_rows'] = data['source_row'][rows]
        artifacts[f'seed{seed}_latents'] = x
        candidates = [
            (f'hdbscan_min{size}', HDBSCAN, dict(min_cluster_size=size, min_samples=8,
              cluster_selection_method='eom', allow_single_cluster=True, copy=True, n_jobs=1))
            for size in (16, 32, 64)]
        candidates += [('affinity_default_preference', AffinityPropagation,
                        dict(damping=.9, max_iter=1000, convergence_iter=50, random_state=731))]
        candidates += [(f'bayesian_max{maximum}', BayesianGaussianMixture,
                        dict(n_components=maximum, covariance_type='diag', weight_concentration_prior_type='dirichlet_process',
                             weight_concentration_prior=.1, reg_covar=1e-6, max_iter=1000, n_init=3, random_state=731))
                       for maximum in (16,32)]
        for name, constructor, settings in candidates:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model = constructor(**settings).fit(x)
            labels = model.predict(x) if isinstance(model, BayesianGaussianMixture) else model.labels_
            groups, sizes = np.unique(labels[labels >= 0], return_counts=True)
            record = {'name': name, 'sample_seed': seed, 'settings': settings,
                      'clusters': len(groups), 'noise_count': int((labels < 0).sum()),
                      'noise_fraction': float((labels < 0).mean()), 'cluster_sizes': sizes.tolist(),
                      'warnings': [str(w.message) for w in captured]}
            if isinstance(model, BayesianGaussianMixture):
                record.update(weights=model.weights_.tolist(), iterations=int(model.n_iter_),
                              converged=bool(model.converged_), all_component_slots_occupied=len(groups)==model.n_components,
                              components_above_one_world_expected_weight=int((model.weights_ > 1/len(worlds)).sum()))
            elif isinstance(model, AffinityPropagation):
                record.update(iterations=int(model.n_iter_), exemplar_world_ids=worlds[model.cluster_centers_indices_].tolist())
            summary['results'].append(record)
            artifacts[f'{name}_seed{seed}_labels'] = labels
            models_by_name.setdefault(name, []).append(labels)
            print(json.dumps({k:record[k] for k in ('name','sample_seed','clusters','noise_count','cluster_sizes','warnings')}),flush=True)
    for name, labels_by_seed in models_by_name.items():
        rows = []
        for a,b in ((0,1),(0,2),(1,2)):
            first,second=labels_by_seed[a],labels_by_seed[b]
            common=(first>=0)&(second>=0)
            rows.append({'draw_indices':[a,b], 'ari_all_noise_as_one_label':float(adjusted_rand_score(first,second)),
                         'common_nonnoise_worlds':int(common.sum()),
                         'ari_common_nonnoise':float(adjusted_rand_score(first[common],second[common])) if common.sum()>1 else None})
        summary['across_draws'].append({'name':name,'partition_consistency':rows})
    np.savez_compressed(output/'samples_and_labels.npz',**artifacts)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
