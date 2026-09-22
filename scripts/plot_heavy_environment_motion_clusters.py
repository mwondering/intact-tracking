"""Show whether independent motions cluster by physical world in heavy latents.

Read the complete world x motion x policy collection. Fit one joint embedding
without labels; highlight worlds chosen before collection and retain all worlds
for original-space distance and retrieval checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score

POLICIES = ('tracker', 'residual')
MARKERS = ('o', 's', '^', 'v', 'D', 'P', 'X', '*')


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm < 1e-8):
        raise ValueError('Invalid latent')
    return x / norm


def save_figure(fig, output):
    fig.savefig(output.with_suffix('.png'), dpi=190, bbox_inches='tight')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def bootstrap_world(values):
    rng = np.random.default_rng(921)
    values = np.asarray(values)
    means = [values[rng.integers(len(values), size=len(values))].mean()
             for _ in range(2000)]
    return np.quantile(means, [.025, .975]).tolist()


def motion_excluded_retrieval(query, gallery):
    """Every query excludes its motion ID from all gallery environments."""
    n, m, d = query.shape
    world = np.repeat(np.arange(n), m)
    motion = np.tile(np.arange(m), n)
    scores = query.reshape(-1, d) @ gallery.reshape(-1, d).T
    scores[motion[:, None] == motion[None, :]] = -np.inf
    nearest = np.argmax(scores, axis=1)
    top5 = np.argpartition(-scores, 4, axis=1)[:, :5]
    correct = (world[nearest] == world).reshape(n, m)
    top5_correct = (world[top5] == world[:, None]).any(1).reshape(n, m)
    return {
        'top1': float(correct.mean()), 'top5': float(top5_correct.mean()),
        'top1_world_bootstrap_95': bootstrap_world(correct.mean(1)),
        'by_motion_top1': correct.mean(0).tolist(),
        'worlds': n, 'queries': n*m, 'gallery_per_query': n*(m-1),
        'positive_gallery_per_query': m-1, 'chance_top1': 1/n,
    }, nearest.reshape(n, m), correct


def original_geometry(z):
    n, m, d = z.shape
    i, j = np.triu_indices(m, 1)
    within = np.linalg.norm(z[:, i]-z[:, j], axis=-1)
    # Same reference motion on both sides of every between-world comparison.
    between = np.stack([pdist(z[:, motion]) for motion in range(m)])
    within_rms = np.sqrt(np.mean(within**2))
    between_rms = np.sqrt(np.mean(between**2))
    flat = z.reshape(n*m, d)
    return {
        'same_world_cross_motion_rms': float(within_rms),
        'different_world_same_motion_rms': float(between_rms),
        'within_over_between_rms': float(within_rms/between_rms),
        'same_world_cross_motion_median': float(np.median(within)),
        'different_world_same_motion_median': float(np.median(between)),
        'same_world_cross_motion_cosine': float(1-np.mean(within**2)/2),
        'environment_silhouette': float(silhouette_score(flat, np.repeat(np.arange(n), m))),
        'motion_silhouette': float(silhouette_score(flat, np.tile(np.arange(m), n))),
        'pair_distances_are_independent_observations': False,
    }, within, between


def cluster_figure(coords, worlds, selected_worlds, output, *, name, subtitle):
    """coords is [policy, world, motion, 2], from one common fit."""
    _, _, m, _ = coords.shape
    selected = [(int(w), int(np.flatnonzero(worlds == w)[0]))
                for w in selected_worlds if np.any(worlds == w)]
    palette = plt.get_cmap('tab20')
    colors = {w: palette(i) for i, w in enumerate(selected_worlds)}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), sharex=True, sharey=True)
    flat = coords.reshape(-1, 2)
    for ax, policy_ids, title in zip(axes, ((0,), (1,), (0, 1)),
            ('Frozen tracker', 'Residual latent policy', 'Both policies')):
        ax.scatter(*flat.T, s=3, c='#d6dce3', alpha=.12, linewidths=0, rasterized=True)
        for w, row in selected:
            points = coords[list(policy_ids), row]
            for motion in range(m):
                pt = points[:, motion]
                ax.scatter(*pt.T, s=43 if motion != 7 else 62, c=[colors[w]],
                           marker=MARKERS[motion], edgecolors='white', linewidths=.35,
                           alpha=.9, zorder=3)
            center = points.reshape(-1, 2).mean(0)
            ax.annotate(f'E{w:03d}', center, xytext=(4, 5), textcoords='offset points',
                        fontsize=7, color='#253142', zorder=4,
                        bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.6, 'pad':.5})
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(flat[:, 0].min()-3, flat[:, 0].max()+3)
        ax.set_ylim(flat[:, 1].min()-3, flat[:, 1].max()+3)
        for spine in ax.spines.values():
            spine.set_color('#d0d7df')
    fig.suptitle('Latents from different motions in the same environment', fontsize=16, y=.99)
    fig.text(.5, .94, subtitle, ha='center', fontsize=10, color='#485363')
    fig.legend([Patch(facecolor=colors[w]) for w, _ in selected],
               [f'E{w:03d}' for w, _ in selected], title='Color = fixed physical environment',
               loc='lower center', bbox_to_anchor=(.5, .058), ncol=8,
               frameon=False, fontsize=8, title_fontsize=9)
    fig.legend([Line2D([], [], color='#454c57', marker=MARKERS[i], linestyle='', markersize=6)
                for i in range(m)], [f'M{i+1}' for i in range(m)],
               title='Shape = motion (same M1–M8 in every environment)',
               loc='lower center', bbox_to_anchor=(.5, -.018), ncol=m,
               frameon=False, fontsize=8, title_fontsize=9)
    fig.subplots_adjust(left=.025, right=.99, top=.875, bottom=.24, wspace=.05)
    save_figure(fig, output/name)


def main(args):
    root = Path(args.root).resolve()
    output = root/'analysis'
    output.mkdir(exist_ok=False)
    protocol = json.loads((root/'protocol.json').read_text())
    motions, total_worlds = protocol['motion_count'], protocol['worlds']
    if motions != len(MARKERS):
        raise ValueError('This figure expects eight motion identities')
    data, metadata = {}, {}
    reference = None
    for policy in POLICIES:
        for motion in range(motions):
            name = f'{policy}_m{motion}'
            meta = json.loads((root/name/'metadata.json').read_text())
            if not meta['complete'] or not meta['actor_and_encoder_unchanged']:
                raise ValueError('Incomplete or non-frozen collection: '+name)
            if reference is None:
                reference = meta
            for key in ('context_sha256', 'tracker_sha256', 'checkpoint_sha256', 'schema', 'manifest_sha256'):
                assert meta[key] == reference[key], (name, key)
            assert meta['world_metadata']['physics_world_fingerprints'] == reference['world_metadata']['physics_world_fingerprints']
            assert set(meta['fixed_motion_ids']) == {motion}
            metadata[name] = meta
            data[name] = dict(np.load(root/name/'latents.npz', allow_pickle=False))
            np.testing.assert_array_equal(data[name]['truth_normalized'], data['tracker_m0']['truth_normalized'])
            np.testing.assert_array_equal(data[name]['step'], data['tracker_m0']['step'])
    valid = np.zeros((total_worlds, motions), dtype=bool)
    rows = np.full((total_worlds, motions), -1, dtype=np.int64)
    for motion in range(motions):
        a, b = (metadata[f'{p}_m{motion}'] for p in POLICIES)
        assert a['initial_state_sha256'] == b['initial_state_sha256']
        assert a['force_diagnostics']['query_force_sha256'] == b['force_diagnostics']['query_force_sha256']
        full = data[f'tracker_m{motion}']['full'] & data[f'residual_m{motion}']['full']
        valid[:, motion] = full.any(0)
        rows[valid[:, motion], motion] = full[:, valid[:, motion]].argmax(0)
    worlds = np.flatnonzero(valid.all(1))
    if len(worlds) < 64:
        raise ValueError('Too few worlds with all eight independently collected motions')
    raw = np.empty((2, len(worlds), motions, 64), dtype=np.float64)
    history_hashes = np.empty((2, len(worlds), motions), dtype='<U64')
    sample_steps = np.empty((len(worlds), motions), dtype=np.int64)
    for pi, policy in enumerate(POLICIES):
        for motion in range(motions):
            arm = data[f'{policy}_m{motion}']
            selected_rows = rows[worlds, motion]
            raw[pi, :, motion] = arm['z'][selected_rows, worlds]
            history_hashes[pi, :, motion] = arm['history_sha256'][selected_rows, worlds]
            sample_steps[:, motion] = arm['step'][selected_rows]
            assert np.all(arm['motion'][selected_rows, worlds] == motion)
    # All windows came from separate empty-memory rollouts, not adjacent frames.
    unique_hashes = len(set(history_hashes.ravel().tolist()))
    assert unique_hashes == history_hashes.size
    z = unit(raw)
    report = {
        'complete': False, 'worlds': len(worlds), 'requested_worlds': total_worlds,
        'motions_per_world': motions, 'policies': list(POLICIES), 'points': int(z[...,0].size),
        'excluded_worlds': np.flatnonzero(~valid.all(1)).tolist(),
        'full_history_pair_counts_per_motion': valid.sum(0).tolist(),
        'display_worlds_planned': protocol['display_world_ids_chosen_before_collection'],
        'display_worlds_available': [w for w in protocol['display_world_ids_chosen_before_collection'] if w in worlds],
        'same_motion_same_world_initial_state_and_force_sequences_verified': True,
        'same_physics_across_every_motion_and_policy_verified': True,
        'exact_duplicated_history_windows': int(history_hashes.size-unique_hashes),
        'primary_windows': 'One full 350-interaction window per world/motion/policy; earliest common valid timestamp within each tracker/residual pair. No window averaging, repeated-window expansion, or memory from another motion/policy.',
        'geometry': {}, 'retrieval': {}, 'embeddings': {},
        'scope': 'One encoder, two related policies, one new physical seed, eight common training-catalog motions sampled before collecting latents from clips >=700 frames. Not held-out motion families or arbitrary-policy invariance.',
    }
    saved = {'z': z, 'raw_z': raw, 'worlds': worlds, 'primary_rows_all_worlds': rows,
             'valid_world_motion': valid, 'sample_steps': sample_steps, 'history_sha256': history_hashes}
    distances = {}
    for pi, policy in enumerate(POLICIES):
        result, within, between = original_geometry(z[pi])
        report['geometry'][policy] = result
        distances[policy] = within, between
        saved[policy+'_within_distances'] = within
        saved[policy+'_between_distances'] = between
        for target_index, target in enumerate(POLICIES):
            key = policy+'_to_'+target
            metrics, nearest, correct = motion_excluded_retrieval(z[pi], z[target_index])
            report['retrieval'][key] = metrics
            saved[key+'_nearest_index'] = nearest
            saved[key+'_correct'] = correct
    flat = z.reshape(-1, 64)
    for perplexity, seed in ((30,921), (50,922)):
        key = f'tsne_p{perplexity}_s{seed}'
        estimator = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                         init='pca', learning_rate='auto', max_iter=1000)
        xy = estimator.fit_transform(flat)
        saved[key] = xy.reshape(*z.shape[:-1], 2)
        report['embeddings'][key] = {'kl_divergence': float(estimator.kl_divergence_),
            'trustworthiness_k10': float(trustworthiness(flat, xy, n_neighbors=10)),
            'one_joint_fit': True, 'labels_supplied_to_embedding': False}
        cluster_figure(saved[key], worlds, protocol['display_world_ids_chosen_before_collection'],
            output, name='environment_motion_clusters' if perplexity == 30 else 'environment_motion_clusters_sensitivity',
            subtitle=f'Color = environment; shape = motion | 8 independent motions per environment | {len(worlds)} worlds in one joint t-SNE; {len(report["display_worlds_available"])} of 16 preselected worlds highlighted')
    pca = PCA(2).fit(flat)
    saved['pca'] = pca.transform(flat).reshape(*z.shape[:-1], 2)
    report['embeddings']['pca_explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
    cluster_figure(saved['pca'], worlds, protocol['display_world_ids_chosen_before_collection'],
        output, name='environment_motion_clusters_pca',
        subtitle=f'Identical samples and colors; linear PCA instead of t-SNE | 2D variance explained: {pca.explained_variance_ratio_.sum():.1%}')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    quantiles = np.linspace(0, 1, 501)
    for ax, policy in zip(axes, POLICIES):
        within, between = distances[policy]
        ax.plot(np.quantile(within, quantiles), quantiles, label='Same environment, different motion', color='#1670ad', lw=2)
        ax.plot(np.quantile(between, quantiles), quantiles, label='Different environment, same motion', color='#db813a', lw=2)
        ax.set(title=policy.capitalize(), xlabel='Euclidean distance in unit 64D latent', ylabel='Cumulative fraction')
        ax.grid(alpha=.16)
        ratio = report['geometry'][policy]['within_over_between_rms']
        ax.text(.04, .92, f'Within / between RMS = {ratio:.3f}', transform=ax.transAxes)
    axes[1].legend(loc='lower right', fontsize=8, frameon=False)
    fig.suptitle('Do motions from the same environment stay closer in the original latent space?')
    fig.tight_layout()
    save_figure(fig, output/'environment_motion_distances_64d')
    np.savez_compressed(output/'analysis_arrays.npz', **saved)
    report['coverage'] = {key:{'failure_events':meta['failure_events'], 'failure_worlds':meta['failure_worlds'],
        'full_worlds':int(data[key]['full'].any(0).sum())} for key,meta in metadata.items()}
    report.update(complete=True, analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ('worlds','display_worlds_available','geometry','retrieval')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    main(parser.parse_args())
