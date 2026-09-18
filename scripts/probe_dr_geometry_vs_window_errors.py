"""Separate true-parameter clustering from learned, single-window geometry errors."""
from pathlib import Path
import hashlib
import json

import numpy as np
from sklearn.cluster import KMeans


def main():
    root = Path('runs')
    paired = root / 'limb_context_20260916_paired_encoder_dr_clustering/labels_and_inputs.npz'
    nominal_dir = root / 'limb_context_20260916_nominal_cluster_contamination'
    config = root / 'limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/run_config.json'
    out = root / 'limb_context_20260916_dr_geometry_vs_window_errors'
    out.mkdir(exist_ok=False)
    a = np.load(paired)
    n = np.load(nominal_dir / 'nominal_latents.npz')
    previous = json.loads((nominal_dir / 'summary.json').read_text())
    schema = json.loads(config.read_text())['dr_center_contract']['schema']
    assert schema['names'] == a['names'].tolist()
    raw = a['physics']
    w = a['worlds']
    z = a['latent_dr_label']
    nominal = np.zeros(38)
    nominal[4] = .6
    nominal[5:34] = 1.
    span = np.array(schema['upper']) - schema['lower']
    x = (raw - nominal) / span * np.sqrt(schema['coordinate_weights'])
    assert raw.shape == (512, 38)
    assert np.all(np.bincount(w) == 16)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.)
    assert np.allclose(np.linalg.norm(n['dr_label'], axis=1), 1.)
    target_nominal = 2 * np.linalg.norm(x, axis=1) / (np.linalg.norm(x, axis=1) + .2)
    nominal_half = np.random.default_rng(1731).permutation(np.unique(n['worlds']))[:256]
    report = {
        'protocol': 'Same 512 random DR worlds and two held-out world folds as prior latent clustering. True DR uses exact training schema (ten equally weighted factors), without unit normalization or nonlinear distance mapping. KMeans sees 256 training DR vectors of weight 1 and nominal vector of weight 256: 50% nominal, 50% random DR. Compare previous nominal50-refit, unwhitened latent results. This removes history inference AND target transformation; not an isolated loss ablation.',
        'window_protocol': 'For each fold, estimate nominal reference from TRAIN nominal windows only. Inspect all 16 cached full-memory windows of the maximum-load held-out DR world previously assigned to the nominal class at least once. These are descriptive, possibly overlapping histories; 16-window means are not identical to the training stochastic two-view centers.',
        'limits': 'Finite cached sample, no new physics or encoder/policy training. Parameter KMeans purity does not prove useful experts or clean bins for every individual DR coordinate. Does not identify why an individual history produces a wrong latent.',
        'checkpoint': previous['checkpoints']['dr_label'],
        'sources': {}, 'parameter_clustering': [], 'window_examples': []}
    for path in (paired, nominal_dir / 'nominal_latents.npz', nominal_dir / 'summary.json', config, Path(__file__)):
        with path.open('rb') as stream:
            report['sources'][str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    arrays = {'physical_vectors': x, 'nominal_target_distances': target_nominal}
    for fold in (0, 1):
        train = np.isin(np.arange(512), a['fit_half0_worlds'])
        nt = np.isin(n['worlds'], nominal_half)
        if fold:
            train, nt = ~train, ~nt
        ids = np.flatnonzero(~train)
        assert train.sum() == len(ids) == 256
        fit = np.concatenate([x[train], np.zeros((1, 38))])
        weights = np.r_[np.ones(256), 256.]
        for k in (16, 64):
            model = KMeans(n_clusters=k, n_init=5, random_state=731, max_iter=300).fit(fit, sample_weight=weights)
            labels = model.predict(x[~train])
            assert np.array_equal(labels, np.linalg.norm(x[~train, None] - model.cluster_centers_, axis=-1).argmin(1))
            nominal_class = int(model.predict(np.zeros((1, 38)))[0])
            same = ids[labels == nominal_class]
            total = raw[same, 34:38].sum(1)
            old = next(r for r in previous['results'] if r['fold'] == fold and r['encoder'] == 'dr_label' and r['metric'] == 'raw' and r['k'] == k and r['mode'] == 'refit_with_nominal50')
            entry = {'fold': fold, 'k': k, 'nominal_class': nominal_class,
                     'true_dr_nominal_worlds': same.tolist(), 'true_dr_nominal_world_fraction': len(same)/256,
                     'true_dr_nominal_max_total_load_kg': float(total.max()) if len(total) else None,
                     'true_dr_nominal_heavy_ge8kg_world_count': int((total >= 8).sum()),
                     'latent_comparison': old['modal_cluster']}
            report['parameter_clustering'].append(entry)
            arrays[f'fold{fold}_k{k}_centers'] = model.cluster_centers_
            arrays[f'fold{fold}_k{k}_test_ids'] = ids
            arrays[f'fold{fold}_k{k}_labels'] = labels
            print(json.dumps({key: value for key, value in entry.items() if key != 'latent_comparison'}), flush=True)
        old = next(r for r in previous['results'] if r['fold'] == fold and r['encoder'] == 'dr_label' and r['metric'] == 'raw' and r['k'] == 64 and r['mode'] == 'existing_dr_only')
        wid = old['modal_cluster']['max_total_world_id']
        assert not train[wid]
        select = w == wid
        ref = n['dr_label'][nt].mean(0)
        observed = np.linalg.norm(z[select] - ref, axis=1)
        saved_labels = a[f'fold{fold}_dr_label_raw_k64_labels']
        selected_labels = saved_labels[w[np.isin(w, ids)] == wid]
        assert len(selected_labels) == int(select.sum()) == 16
        example = {'fold': fold, 'world': int(wid), 'loads_kg': raw[wid, 34:38].tolist(),
                   'dr_distance_to_nominal': float(np.linalg.norm(x[wid])),
                   'target_center_distance_to_nominal': float(target_nominal[wid]),
                   'observed_window_distances_to_training_nominal_mean': observed.tolist(),
                   'observed_16window_mean_distance_to_training_nominal_mean': float(np.linalg.norm(z[select].mean(0) - ref)),
                   'nominal_class_mask': (selected_labels == old['modal_nominal_cluster']).tolist(),
                   'source_rows': a['source_rows'][select].tolist()}
        assert sum(example['nominal_class_mask']) >= 1
        report['window_examples'].append(example)
        print(json.dumps(example), flush=True)
    centers = np.stack([z[w == i].mean(0) for i in range(512)])
    i, j = np.triu_indices(512, 1)
    dp = np.linalg.norm(x[i] - x[j], axis=1)
    target = 2*dp/(dp+.2)
    dz = np.linalg.norm(centers[i] - centers[j], axis=1)
    report['descriptive_all_world_center_geometry'] = {
        'pair_count': len(i), 'pearson_with_raw_dr_distance': float(np.corrcoef(dp, dz)[0, 1]),
        'pearson_with_actual_training_target': float(np.corrcoef(target, dz)[0, 1]),
        'rmse_to_training_target': float(np.sqrt(np.mean((target-dz)**2)))}
    np.savez_compressed(out / 'parameter_models.npz', **arrays)
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
