"""World-disjoint, physical-distance audit of the frozen DR encoder on 16384 worlds.

No policy/encoder optimization. KMeans is fitted only to the designated fit worlds.
Run with PYTHONPATH=.runtime/latent_auto_cluster_deps using the project venv.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

ROOT = Path('runs/limb_context_20260916_dr16384')
OUT = ROOT / 'analysis'
WINDOWS = 16


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def distribution(values):
    x = np.asarray(values)
    if not x.size:
        return {'n': 0}
    return dict(n=int(x.size), mean=float(x.mean()), min=float(x.min()),
                p10=float(np.quantile(x, .1)), p50=float(np.median(x)),
                p90=float(np.quantile(x, .9)), max=float(x.max()))


def prepare():
    rows, audits = [], []
    schema = None
    for shard in range(8):
        path = ROOT / f'shard_{shard:02d}'
        m = json.loads((path / 'metadata.json').read_text())
        assert m['complete'] and m['physics_unchanged']
        if schema is None:
            schema = m['schema']
            checkpoint = m['checkpoint']
            motion_files = m['motion_files']
        assert schema == m['schema'] and checkpoint == m['checkpoint']
        assert motion_files == m['motion_files']
        for name in ('physics', 'latents'):
            assert digest(path / f'{name}.npz') == m[f'{name}_sha256']
        p = np.load(path / 'physics.npz')
        q = np.load(path / 'latents.npz')
        assert p['names'].tolist() == schema['names']
        n = len(p['world'])
        assert len(q['world']) % n == 0
        assert np.array_equal(q['world'].reshape(-1, n), np.broadcast_to(p['world'], (len(q['world']) // n, n)))
        full = ((q['short_steps'] == 50) & (q['long_chunks'] == 30)).reshape(-1, n)
        assert full.sum(0).min() >= WINDOWS
        assert np.array_equal(full.sum(0), m['full_query_counts'])
        selected = np.empty((n, WINDOWS), np.int64)
        rng = np.random.default_rng(16384916 + shard)
        for local in range(n):
            candidates = np.flatnonzero(full[:, local])
            # Same count for every physical world, independent of latent or DR values.
            selected[local] = np.sort(rng.choice(candidates, WINDOWS, replace=False)) * n + local
        z = q['latent'][selected].astype(np.float32)
        assert np.isfinite(z).all()
        norm = np.linalg.norm(z, axis=-1, keepdims=True)
        assert norm.min() > 1e-6
        z /= norm
        rows.append(dict(z=z, physics=p['values'], nominal=p['nominal'], world=p['world'],
                         shard=np.full(n, shard, np.int16), source_rows=selected,
                         step=q['step'][selected], motion=q['motion'][selected],
                         motion_step=q['motion_step'][selected], episode=q['episode'][selected],
                         full_count=full.sum(0)))
        audits.append(dict(shard=shard, dr_worlds=int((~p['nominal']).sum()),
                           nominal_worlds=int(p['nominal'].sum()),
                           numerical_audit=m['numerical_audit'],
                           metadata_sha256=digest(path / 'metadata.json'),
                           full_windows=distribution(full.sum(0))))
    a = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    assert len(np.unique(a['world'])) == len(a['world']) == 18432
    assert a['nominal'].sum() == 2048
    dr = np.flatnonzero(~a['nominal'])
    assert np.unique(a['physics'][dr], axis=0).shape[0] == 16384
    nom = np.zeros(38)
    nom[4], nom[5:34] = .6, 1.
    np.testing.assert_allclose(a['physics'][a['nominal']], np.broadcast_to(nom, (2048, 38)), atol=1e-6)
    span = np.asarray(schema['upper']) - schema['lower']
    weights = np.asarray(schema['coordinate_weights'])
    a['x'] = (a['physics'].astype(np.float64) - nom) / span * np.sqrt(weights)
    a['zmean'] = a['z'].mean(1)
    a['radius'] = np.linalg.norm(a['x'], axis=1)
    rng = np.random.default_rng(73116384)
    fold = np.zeros(len(a['world']), np.int8)
    fold[rng.permutation(dr)[:8192]] = 1
    fold[rng.permutation(np.flatnonzero(a['nominal']))[:1024]] = 1
    a['fold'] = fold
    a['names'] = np.asarray(schema['names'])
    a['span'] = span
    a['weights'] = weights
    np.savez_compressed(OUT / 'selected_worlds.npz', **a)
    radius = a['radius'][dr]
    loads = a['physics'][dr, 34:38]
    coverage = dict(dr_worlds=16384, nominal_worlds=2048, windows_per_world=16,
                    selected_windows=int(a['z'].shape[0] * WINDOWS),
                    dr_radius=distribution(radius), total_load_kg=distribution(loads.sum(1)),
                    radius_below={str(t): int((radius < t).sum()) for t in (.1, .15, .2, .25, .3, .35, .4, .5)},
                    all_four_loads_below_10pct_count=int((loads < np.array([.25, .25, .4, .4])).all(1).sum()),
                    per_parameter={name: distribution(a['physics'][dr, i]) for i, name in enumerate(schema['names'])},
                    selected_unique_motions_per_world=distribution([len(np.unique(v)) for v in a['motion']]),
                    full_windows=distribution(a['full_count']))
    save_json(OUT / 'collection_audit.json', dict(shards=audits, schema=schema, checkpoint=checkpoint,
              motion_files=motion_files, coverage=coverage,
              source_sha256=digest(Path(__file__)), selected_sha256=digest(OUT / 'selected_worlds.npz')))
    print(json.dumps(dict(event='prepared', coverage={k: v for k, v in coverage.items() if k != 'per_parameter'})), flush=True)


def physical_pair_summary(physics, x, i, j):
    raw = physics[i] - physics[j]
    limb = np.abs(raw[:, 34:38]).mean(0)
    return dict(dr_distance_mean=float(np.linalg.norm(x[i] - x[j], axis=1).mean()),
                dr_squared_distance_mean=float(((x[i] - x[j])**2).sum(1).mean()),
                left_hand_kg=float(limb[0]), right_hand_kg=float(limb[1]),
                left_shin_kg=float(limb[2]), right_shin_kg=float(limb[3]),
                hands_sum_absolute_difference_kg=float(np.abs(raw[:, 34:36].sum(1)).mean()),
                shins_sum_absolute_difference_kg=float(np.abs(raw[:, 36:38].sum(1)).mean()),
                total_load_absolute_difference_kg=float(np.abs(raw[:, 34:38].sum(1)).mean()),
                com_vector_difference_cm=float(np.linalg.norm(raw[:, :3], axis=1).mean() * 100),
                torso_mass_difference_kg=float(np.abs(raw[:, 3]).mean() / .12790995401552643),
                friction_difference=float(np.abs(raw[:, 4]).mean()),
                armature_mean_absolute_difference=float(np.abs(raw[:, 5:34]).mean()))


def pair_quality(a, world_ids, labels, seed=183):
    # Anchor uniform over worlds, then over its windows. Partner uniform over
    # windows in the same cluster, conditional on being a DIFFERENT world.
    local_world = np.repeat(world_ids, labels.shape[1])
    flat = labels.ravel()
    rng = np.random.default_rng(seed)
    anchor = rng.integers(len(flat), size=200000)
    partner = np.empty_like(anchor)
    valid = np.ones(len(anchor), bool)
    for c in np.unique(flat):
        destinations = np.flatnonzero(flat == c)
        select = np.flatnonzero(flat[anchor] == c)
        if len(np.unique(local_world[destinations])) < 2:
            valid[select] = False
            continue
        chosen = rng.choice(destinations, len(select))
        bad = local_world[chosen] == local_world[anchor[select]]
        while bad.any():
            chosen[bad] = rng.choice(destinations, bad.sum())
            bad = local_world[chosen] == local_world[anchor[select]]
        partner[select] = chosen
    i, j = local_world[anchor[valid]], local_world[partner[valid]]
    random_j = rng.choice(world_ids, len(i))
    bad = i == random_j
    while bad.any():
        random_j[bad] = rng.choice(world_ids, bad.sum())
        bad = i == random_j
    inside = physical_pair_summary(a['physics'], a['x'], i, j)
    random = physical_pair_summary(a['physics'], a['x'], i, random_j)
    return dict(anchor_coverage=float(valid.mean()), pairs=len(i), within_cluster=inside,
                random_other_world=random,
                reduction={key: 1 - inside[key] / random[key] for key in inside})


def cluster_rows(a, test_dr, test_nom, ld, ln, centers, nominal_ref, train_nom_labels, key):
    mode = int(np.bincount(train_nom_labels.ravel(), minlength=len(centers)).argmax())
    rows = []
    for c in range(len(centers)):
        world_pos, _ = np.where(ld == c)
        ids = test_dr[world_pos]
        p = a['physics'][ids]
        row = dict(model=key, cluster=c, nominal_mode=(c == mode),
                   center_radius=float(np.linalg.norm(centers[c] - nominal_ref)),
                   dr_windows=len(ids), dr_unique_worlds=len(np.unique(ids)),
                   nominal_windows=int((ln == c).sum()),
                   nominal_coverage=float((ln == c).mean()), dr_window_fraction=float((ld == c).mean()))
        features = dict(dr_radius=a['radius'][ids], left_hand_kg=p[:, 34], right_hand_kg=p[:, 35],
                        left_shin_kg=p[:, 36], right_shin_kg=p[:, 37],
                        hands_kg=p[:, 34:36].sum(1), shins_kg=p[:, 36:38].sum(1), total_load_kg=p[:, 34:38].sum(1),
                        com_norm_cm=np.linalg.norm(p[:, :3], axis=1)*100,
                        torso_mass_absolute_change_kg=np.abs(p[:, 3])/.12790995401552643,
                        friction=p[:, 4], friction_absolute_deviation=np.abs(p[:, 4]-.6))
        for name, val in features.items():
            for stat, v in distribution(val).items():
                if stat != 'n':
                    row[f'{name}_{stat}'] = v
        row['dr_heavy_ge8kg_fraction'] = float((features['total_load_kg'] >= 8).mean()) if len(ids) else None
        rows.append(row)
    order = np.argsort(-np.bincount(train_nom_labels.ravel(), minlength=len(centers)))
    counts = np.bincount(train_nom_labels.ravel(), minlength=len(centers))
    stop = np.searchsorted(np.cumsum(counts[order])/counts.sum(), .95) + 1
    union = order[:stop]
    heavy = a['physics'][test_dr, 34:38].sum(1) >= 8
    light = a['radius'][test_dr] < .25
    out = dict(nominal_class=mode, nominal_95pct_classes=union.tolist())
    for name, classes in [('modal', [mode]), ('union95', union)]:
        routed = np.isin(ld, classes)
        nr = np.isin(ln, classes)
        selected = np.where(routed)[0]
        load = a['physics'][test_dr[selected], 34:38].sum(1)
        out[name] = dict(test_nominal_coverage=float(nr.mean()),
                        random_dr_window_fraction=float(routed.mean()),
                        random_dr_world_ever_fraction=float(routed.any(1).mean()),
                        heavy_world_count=int(heavy.sum()), heavy_window_routed_fraction=float(routed[heavy].mean()),
                        heavy_world_ever_routed_fraction=float(routed[heavy].any(1).mean()),
                        near_dr_world_count=int(light.sum()), near_dr_window_routed_fraction=float(routed[light].mean()) if light.any() else None,
                        routed_DR_total_load_kg=distribution(load),
                        balanced_prior_nominal_precision=float(nr.mean()/(nr.mean()+routed.mean())) if nr.mean()+routed.mean() else None)
    return rows, out


def nearest_physical(a, test_ids, fold):
    rng = np.random.default_rng(971 + fold)
    anchors = rng.choice(test_ids, 1024, replace=False)
    lookup = {wid: pos for pos, wid in enumerate(test_ids)}
    self_col = np.array([lookup[i] for i in anchors])
    distance = pairwise_distances(a['x'][anchors], a['x'][test_ids], n_jobs=1)
    distance[np.arange(len(anchors)), self_col] = np.inf
    physical = np.argpartition(distance, 20, axis=1)[:, :20]
    ld = pairwise_distances(a['zmean'][anchors], a['zmean'][test_ids], n_jobs=1)
    ld[np.arange(len(anchors)), self_col] = np.inf
    latent = np.argpartition(ld, 20, axis=1)[:, :20]
    recall = np.array([len(set(i) & set(j))/20 for i, j in zip(physical, latent)])
    return dict(anchors=anchors, anchor_positions=self_col, physical=physical, latent=latent,
                report=dict(anchor_worlds=len(anchors), candidates=len(test_ids)-1,
                            neighbors=20, mean_overlap_count=float(recall.mean()*20),
                            mean_recall=float(recall.mean()), random_expected_overlap=400/(len(test_ids)-1)))


def geometry(a, train_nom, test_dr, fold):
    rng = np.random.default_rng(592 + fold)
    i, j = rng.choice(test_dr, (2, 200000))
    valid = i != j
    i, j = i[valid], j[valid]
    dp = np.linalg.norm(a['x'][i]-a['x'][j], axis=1)
    dz = np.linalg.norm(a['zmean'][i]-a['zmean'][j], axis=1)
    target = 2*dp/(dp+.2)
    nr = a['zmean'][train_nom].mean(0)
    radii = np.linalg.norm(a['zmean'][test_dr]-nr, axis=1)
    return dict(pair_count=len(i), pair_raw_pearson=float(np.corrcoef(dp, dz)[0, 1]),
                pair_target_pearson=float(np.corrcoef(target, dz)[0, 1]),
                pair_spearman=float(spearmanr(dp, dz).statistic),
                pair_target_rmse=float(np.sqrt(((target-dz)**2).mean())),
                nominal_radius_spearman=float(spearmanr(a['radius'][test_dr], radii).statistic),
                nominal_radius_target_rmse=float(np.sqrt(((2*a['radius'][test_dr]/(a['radius'][test_dr]+.2)-radii)**2).mean())))


def fit_models():
    a = dict(np.load(OUT/'selected_worlds.npz'))
    results, all_rows, geometries = [], [], []
    (OUT/'models').mkdir(exist_ok=True)
    for fold in (0, 1):
        train_dr = np.flatnonzero((a['fold'] == fold) & ~a['nominal'])
        test_dr = np.flatnonzero((a['fold'] != fold) & ~a['nominal'])
        train_nom = np.flatnonzero((a['fold'] == fold) & a['nominal'])
        test_nom = np.flatnonzero((a['fold'] != fold) & a['nominal'])
        assert len(train_dr) == len(test_dr) == 8192 and len(train_nom) == len(test_nom) == 1024
        near = nearest_physical(a, test_dr, fold)
        g = geometry(a, train_nom, test_dr, fold)
        g.update(fold=fold, nearest_neighbors=near['report'])
        geometries.append(g)
        save_json(OUT/'geometry.json', geometries)
        small = np.random.default_rng(3481+fold).choice(train_dr, 512, replace=False)
        for rep in ('true_dr', 'world_mean_latent', 'window_latent'):
            value = {'true_dr': a['x'][:, None, :], 'world_mean_latent': a['zmean'][:, None, :], 'window_latent': a['z']}[rep]
            dimension = value.shape[-1]
            for nfit, k in ((512, 16), (8192, 16), (8192, 64)):
                started = time.monotonic()
                key = f'fold{fold}_{rep}_n{nfit}_k{k}'
                tr = small if nfit == 512 else train_dr
                fit_d, fit_n = value[tr].reshape(-1, dimension), value[train_nom].reshape(-1, dimension)
                fit = np.concatenate([fit_d, fit_n])
                weight = np.r_[np.ones(len(fit_d)), np.full(len(fit_n), len(fit_d)/len(fit_n))]
                model = KMeans(n_clusters=k, random_state=731, n_init=5, max_iter=300).fit(fit, sample_weight=weight)
                ld = model.predict(value[test_dr].reshape(-1, dimension)).reshape(len(test_dr), -1)
                ln = model.predict(value[test_nom].reshape(-1, dimension)).reshape(len(test_nom), -1)
                nt = model.predict(fit_n)
                nominal_ref = fit_n.mean(0)
                rows, nominal = cluster_rows(a, test_dr, test_nom, ld, ln, model.cluster_centers_, nominal_ref, nt, key)
                all_rows.extend(rows)
                quality = pair_quality(a, test_dr, ld)
                # Physical nearest-neighbor class agreement: actual windows
                # averaged over both worlds' window routing distributions.
                probs = np.eye(k)[ld].mean(1)
                same = (probs[near['anchor_positions'], None, :] * probs[near['physical']]).sum(-1)
                random_agreement = (probs[near['anchor_positions']] * probs.mean(0)).sum(-1).mean()
                mode_fraction = probs.max(1)
                entry = dict(key=key, fold=fold, representation=rep, fit_dr_worlds=nfit, k=k,
                             fit_nominal_worlds=1024, test_dr_worlds=8192, test_nominal_worlds=1024,
                             nominal_fit_weight=.5, iterations=int(model.n_iter_),
                             cluster_pair_quality=quality, nominal=nominal,
                             window_modal_fraction=distribution(mode_fraction),
                             physical_20nn_same_class_fraction=float(same.mean()),
                             random_same_class_fraction=float(random_agreement),
                             elapsed_seconds=time.monotonic()-started)
                # Radial averages are secondary, explicitly EXCLUDING the nominal modal class.
                modal = nominal['nominal_class']
                ranked = sorted([r for r in rows if r['cluster'] != modal and r['dr_windows']], key=lambda r: r['center_radius'])
                nquarter = max(1, round(len(ranked)/4))
                radial = {}
                for side, group in [('nearest_non_nominal_quarter', ranked[:nquarter]), ('farthest_non_nominal_quarter', ranked[-nquarter:])]:
                    selected = np.isin(ld, [r['cluster'] for r in group])
                    ids = test_dr[np.where(selected)[0]]
                    radial[side] = dict(clusters=[r['cluster'] for r in group], windows=len(ids),
                                        dr_radius=distribution(a['radius'][ids]),
                                        hands_kg=distribution(a['physics'][ids, 34:36].sum(1)),
                                        shins_kg=distribution(a['physics'][ids, 36:38].sum(1)))
                entry['radial_secondary'] = radial
                results.append(entry)
                np.savez_compressed(OUT/'models'/f'{key}.npz', centers=model.cluster_centers_,
                                    train_dr=tr, test_dr=test_dr, train_nom=train_nom, test_nom=test_nom,
                                    dr_labels=ld, nominal_labels=ln, nominal_ref=nominal_ref)
                save_json(OUT/'results.json', results)
                fields = list(dict.fromkeys(field for row in all_rows for field in row))
                with (OUT/'all_cluster_parameters.csv').open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(all_rows)
                print(json.dumps(dict(event='model_complete', key=key, seconds=entry['elapsed_seconds'],
                                      physical_distance_reduction=quality['reduction']['dr_distance_mean'],
                                      nominal=nominal['modal'])), flush=True)
    save_json(OUT/'complete.json', dict(complete=True, model_count=len(results),
              script_sha256=digest(Path(__file__)), results_sha256=digest(OUT/'results.json'),
              selected_sha256=digest(OUT/'selected_worlds.npz')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--threads', type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    while True:
        states = []
        for shard in range(8):
            path = ROOT/f'shard_{shard:02d}'/'metadata.json'
            states.append(path.exists() and json.loads(path.read_text())['complete'])
        if all(states):
            break
        if not args.wait:
            raise RuntimeError(f'Collection unfinished: {states}')
        time.sleep(10)
    with threadpool_limits(limits=args.threads):
        if not (OUT/'selected_worlds.npz').exists():
            prepare()
        fit_models()


if __name__ == '__main__':
    main()
