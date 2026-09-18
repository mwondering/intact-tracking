"""Audit whether DR-supervised cluster centers order physical nominal deviation."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from scipy.stats import spearmanr


def description(values):
    values = np.asarray(values)
    return {'mean': float(values.mean()), 'p10_p50_p90': np.quantile(values, [.1, .5, .9]).tolist()}


def ordered_pair_fraction(radius, severity):
    i, j = np.triu_indices(len(radius), 1)
    product = (radius[i]-radius[j])*(severity[i]-severity[j])
    return float(np.mean((product > 0) + .5*(product == 0)))


def assign(z, centers):
    return np.concatenate([np.square(chunk[:, None]-centers).sum(-1).argmin(1)
                           for chunk in np.array_split(z, 32)])


def main():
    out = Path('runs/limb_context_20260916_dr_cluster_radial_structure')
    out.mkdir(exist_ok=False)
    paired = Path('runs/limb_context_20260916_paired_encoder_dr_clustering/labels_and_inputs.npz')
    base = Path('runs/limb_context_20260916_nominal_cluster_contamination')
    config = Path('runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/run_config.json')
    a = np.load(paired)
    models = np.load(base/'models_and_labels.npz')
    n = np.load(base/'nominal_latents.npz')
    prior = json.loads((base/'summary.json').read_text())
    schema = json.loads(config.read_text())['dr_center_contract']['schema']
    assert schema['names'] == a['names'].tolist()
    p, w, z = a['physics'], a['worlds'], a['latent_dr_label']
    nominal = np.zeros(38)
    nominal[4] = .6
    nominal[5:34] = 1.
    delta = (p-nominal)/(np.array(schema['upper'])-schema['lower'])
    weights = np.asarray(schema['coordinate_weights'])
    physical_distance = np.sqrt((delta**2*weights).sum(1))
    features = {
        'dr_distance': physical_distance,
        'dr_distance_excluding_armature': np.sqrt((delta[:, np.r_[0:5,34:38]]**2 * weights[np.r_[0:5,34:38]]).sum(1)/.9),
        'left_hand_kg': p[:, 34], 'right_hand_kg': p[:, 35],
        'left_shin_kg': p[:, 36], 'right_shin_kg': p[:, 37],
        'total_load_kg': p[:, 34:38].sum(1),
        'friction_abs_deviation': np.abs(p[:, 4]-.6),
        'com_offset_norm_cm': np.linalg.norm(p[:, :3], axis=1)*100,
        'torso_mass_abs_delta_kg': np.abs(p[:, 3])/.12790995401552643,
        'armature_rms_abs_delta': np.sqrt(np.mean((p[:, 5:34]-1)**2, axis=1))}
    nominal_half = np.random.default_rng(1731).permutation(np.unique(n['worlds']))[:256]
    report = {
        'checkpoint': prior['checkpoints']['dr_label'],
        'protocol': 'Reuse saved unwhitened u35857 KMeans centers and held-out labels, K16/64/128, two disjoint 256-world train/test folds. Main mode fits random DR only; sensitivity mode fits 50% nominal. Nominal reference is mean unit latent of TRAIN nominal histories. Rank centers by Euclidean distance to that reference; report actual TEST DR severity using exact training ten-factor metric. No new clustering, simulation or training.',
        'center_ranking': 'Select closest/farthest ceil(N/4) occupied non-nominal clusters by latent radius only, never by physical labels. Nominal modal cluster selected on TRAIN nominal assignments and excluded from rank statistics to avoid a trivial nominal-vs-DR effect. Main means weight actual routed windows; world-equal summaries deduplicate physical worlds within each reported group.',
        'world_centers': 'Descriptive 16-window average of unit latents per held-out physical world, without renormalizing; not KMeans centers or the exact two-view stochastic training center. No motion-independent prototype/query claim.',
        'limits': 'Relative aggregate DR deviation is not a claim that every coordinate increases, that near means nearly zero DR, that individual windows have correct labels, or that expert control improves. Broad independent random DR sampling supplies few environments simultaneously near nominal in all coordinates. Static DR distance, not measured dynamic response.',
        'schema': schema, 'sources': {}, 'cases': [], 'world_center_cases': []}
    for path in (paired, base/'models_and_labels.npz', base/'nominal_latents.npz', base/'summary.json', config, Path(__file__)):
        with path.open('rb') as f:
            report['sources'][str(path)] = hashlib.file_digest(f, 'sha256').hexdigest()
    archive = {'world_physical_distance': physical_distance}
    csv_rows = []
    for fold in (0, 1):
        train = np.isin(w, a['fit_half0_worlds'])
        nt = np.isin(n['worlds'], nominal_half)
        if fold:
            train, nt = ~train, ~nt
        tw = w[~train]
        nominal_center = n['dr_label'][nt].mean(0)
        ids = np.unique(tw)
        environmental_centers = np.stack([z[w == world].mean(0) for world in ids])
        world_radius = np.linalg.norm(environmental_centers-nominal_center, axis=1)
        world_record = {'fold': fold, 'worlds': len(ids),
                        'spearman': float(spearmanr(world_radius, physical_distance[ids]).statistic),
                        'pair_order_agreement': ordered_pair_fraction(world_radius, physical_distance[ids]),
                        'spearman_excluding_armature': float(spearmanr(world_radius, features['dr_distance_excluding_armature'][ids]).statistic)}
        report['world_center_cases'].append(world_record)
        archive[f'fold{fold}_world_ids'] = ids
        archive[f'fold{fold}_world_center_radius'] = world_radius
        for mode in ('existing_dr_only', 'refit_with_nominal50'):
            for k in (16, 64, 128):
                prefix = f'fold{fold}_dr_label_raw_k{k}_{mode}'
                centers = models[prefix+'_centers']
                labels = models[prefix+'_dr_labels']
                assert np.array_equal(labels, assign(z[~train], centers))
                nominal_fit_labels = assign(n['dr_label'][nt], centers)
                nc = int(np.bincount(nominal_fit_labels, minlength=k).argmax())
                earlier = next(r for r in prior['results'] if (r['fold'], r['encoder'], r['metric'], r['k'], r['mode']) == (fold, 'dr_label', 'raw', k, mode))
                assert nc == earlier['modal_nominal_cluster']
                nominal_test_labels = models[prefix+'_nominal_labels']
                radius = np.linalg.norm(centers-nominal_center, axis=1)
                rows = []
                def group_stats(mask):
                    worlds = tw[mask]
                    unique = np.unique(worlds)
                    return {'samples': len(worlds), 'unique_worlds': len(unique),
                            'routed_window_weighted': {name: description(value[worlds]) for name, value in features.items()},
                            'unique_world_weighted': {name: description(value[unique]) for name, value in features.items()}}
                for cluster in range(k):
                    mask = labels == cluster
                    count = int(mask.sum())
                    row = {'cluster': cluster, 'radius': float(radius[cluster]), 'is_nominal_modal_cluster': cluster == nc,
                           'nominal_test_samples': int((nominal_test_labels == cluster).sum()), 'dr_samples': count}
                    if count:
                        row.update(group_stats(mask))
                        csv_rows.append({'fold': fold, 'mode': mode, 'k': k, 'cluster': cluster, 'radius': radius[cluster],
                                         'nominal_cluster': cluster == nc, 'dr_samples': count, 'worlds': row['unique_worlds'],
                                         **{name: val['mean'] for name, val in row['routed_window_weighted'].items()}})
                    rows.append(row)
                occupied = [r for r in rows if r['dr_samples'] and not r['is_nominal_modal_cluster']]
                occupied.sort(key=lambda r: r['radius'])
                num = int(np.ceil(len(occupied)/4))
                near, far = ([r['cluster'] for r in subset] for subset in (occupied[:num], occupied[-num:]))
                cr = np.array([r['radius'] for r in occupied])
                cm = np.array([r['routed_window_weighted']['dr_distance']['mean'] for r in occupied])
                case = {'fold': fold, 'mode': mode, 'k': k, 'nominal_cluster': nc,
                        'occupied_non_nominal_clusters': len(occupied),
                        'radius_vs_cluster_mean_dr_spearman': float(spearmanr(cr, cm).statistic),
                        'cluster_pair_order_agreement': ordered_pair_fraction(cr, cm),
                        'family_mean_spearman': {name: float(spearmanr(cr, [r['routed_window_weighted'][name]['mean'] for r in occupied]).statistic) for name in features},
                        'near_cluster_ids': near, 'far_cluster_ids': far,
                        'near': group_stats(np.isin(labels, near)), 'far': group_stats(np.isin(labels, far)),
                        'clusters': rows}
                # Pure nominal samples are explicitly zero-DR; keep separate from DR-only rank test.
                nrow = rows[nc]
                numerator = sum(physical_distance[tw[labels == nc]])
                case['nominal_cluster_combined_physical_distance_mean'] = float(numerator/(nrow['dr_samples']+nrow['nominal_test_samples']))
                report['cases'].append(case)
                print(json.dumps({'fold': fold, 'mode': mode, 'k': k, 'rho': case['radius_vs_cluster_mean_dr_spearman'],
                                  'pair_order': case['cluster_pair_order_agreement'],
                                  'near_dr_mean': case['near']['routed_window_weighted']['dr_distance']['mean'],
                                  'far_dr_mean': case['far']['routed_window_weighted']['dr_distance']['mean']}), flush=True)
    averages = []
    for mode in ('existing_dr_only', 'refit_with_nominal50'):
        for k in (16, 64, 128):
            records = [r for r in report['cases'] if r['mode'] == mode and r['k'] == k]
            row = {'mode': mode, 'k': k}
            for key in ('radius_vs_cluster_mean_dr_spearman', 'cluster_pair_order_agreement'):
                row[key] = float(np.mean([r[key] for r in records]))
            for side in ('near', 'far'):
                row[side] = {name: float(np.mean([r[side]['routed_window_weighted'][name]['mean'] for r in records])) for name in features}
                row[side+'_world_equal'] = {name: float(np.mean([r[side]['unique_world_weighted'][name]['mean'] for r in records])) for name in features}
            averages.append(row)
    report['averages'] = averages
    np.savez_compressed(out/'world_centers.npz', **archive)
    (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (out/'cluster_parameters.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({'world_centers': report['world_center_cases'], 'main_k16': averages[0], 'main_k64': averages[1]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
