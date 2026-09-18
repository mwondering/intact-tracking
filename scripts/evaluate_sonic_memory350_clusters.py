"""Evaluate a fixed Sonic motion subset and the existing LAFAN raw-history cache."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_forward_context_clusters import normalized
from analyze_memory350 import sha256
from analyze_memory350_latent_clusters import analyze_subset
from evaluate_memory350_dr_center import center_geometry
from evaluate_memory350_weak_pairs import encode, environment_readout, write_json
from probe_dr_center_neighbor_overlap import split
from intact_tracking.memory350_inference import load_memory350_checkpoint


PROFILES = ('capped_loads', 'common')
FIELDS = {'memory350': 'latent', 'baseline': 'latent_baseline'}


def load_data(folder):
    with np.load(folder / 'latents.npz') as saved:
        data = {key: saved[key] for key in saved.files}
    data.setdefault('source_row', np.arange(len(data['world'])))
    return data


def full_data(data, dr_only=False):
    mask = (data['short_steps'] == 50) & (data['long_chunks'] == 30)
    if dr_only:
        mask &= ~data['nominal']
    return {key: value[mask] for key, value in data.items()}


def encode_reference(root, source, device):
    checkpoints = json.loads((root / 'checkpoints.json').read_text())
    models = {name: load_memory350_checkpoint(info['path'], device=device)
              for name, info in checkpoints.items()}
    for name, model in models.items():
        assert model.sha256 == checkpoints[name]['sha256']
    for profile in PROFILES:
        old = source / profile
        old_meta = json.loads((old / 'metadata.json').read_text())
        assert old_meta['complete']
        saved = load_data(old)
        data = {k: v for k, v in saved.items() if not k.startswith('latent')}
        assert old_meta['checkpoints']['memory350']['sha256'] == models['baseline'].sha256
        data['latent_baseline'] = saved[old_meta['models']['memory350']]
        parts, checks = [], []
        started = time.monotonic()
        for index, info in enumerate(old_meta['queries']):
            query = torch.load(info['path'], map_location='cpu', weights_only=False, mmap=True)
            mask = data['step'] == query['step']
            assert query['format_version'] == 'memory350_raw_query_v1'
            assert np.array_equal(query['world'].numpy(), data['world'][mask])
            assert np.array_equal(query['short_valid'].sum(-1).numpy(), data['short_steps'][mask])
            assert np.array_equal(query['long_valid'].sum(-1).numpy(), data['long_chunks'][mask])
            if index in (0, len(old_meta['queries']) - 1):
                assert sha256(Path(info['path'])) == info['sha256']
                reconstructed = encode(models['baseline'], query)
                error = float(np.sqrt(np.mean(np.sum((normalized(reconstructed) -
                                                      normalized(data['latent_baseline'][mask])) ** 2, axis=1))))
                assert error < .001, error
                checks.append({'step': query['step'], 'unit_rms_reconstruction_error': error})
            z = encode(models['memory350'], query)
            assert np.isfinite(z).all()
            parts.append(z)
            if index % 8 == 0 or index == len(old_meta['queries']) - 1:
                print(json.dumps({'event': 'lafan_encoded', 'profile': profile,
                                  'queries': index + 1, 'seconds': time.monotonic() - started}), flush=True)
        data['latent'] = np.concatenate(parts)
        folder = root / 'lafan' / profile
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / 'latents.npz', **data)
        write_json(folder / 'metadata.json', {
            'models': FIELDS, 'motion_files': old_meta['motion_files'], 'complete': True,
            'profile': profile, 'checkpoints': checkpoints,
            'source_physics': old_meta['source_physics'], 'queries': old_meta['queries'],
            'source_metadata': str(old / 'metadata.json'), 'reconstruction': checks,
        })


def accuracy_stats(correct, worlds, *, seed=20260916):
    labels, inverse = np.unique(worlds, return_inverse=True)
    count = np.bincount(inverse)
    total = np.bincount(inverse, weights=correct.astype(float))
    draws = np.random.default_rng(seed).integers(len(labels), size=(2000, len(labels)))
    bootstrap = total[draws].sum(1) / count[draws].sum(1)
    return {'accuracy': float(correct.mean()),
            'world_balanced_accuracy': float((total / count).mean()),
            'query_world_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist()}


def readout_details(data, metadata, output):
    train, test, ids, truth = split(data, metadata)
    result = {'worlds': len(ids), 'queries': int(test.sum()), 'models': {}}
    arrays = {'center_worlds': ids, 'query_source_row': data['source_row'][test],
              'query_world': data['world'][test], 'truth': truth}
    for name, field in FIELDS.items():
        z = normalized(data[field].astype(np.float64))
        centers = np.stack([z[train & (data['world'] == key)].mean(0) for key in ids])
        q = z[test]
        distance2 = np.maximum(0, (q*q).sum(1)[:, None] + (centers*centers).sum(1)[None] - 2*q@centers.T)
        pred = distance2.argmin(1)
        top5 = np.argpartition(distance2, 4, axis=1)[:, :5]
        row = np.arange(len(truth))
        own = np.sqrt(distance2[row, truth])
        distance2[row, truth] = np.inf
        margin = np.sqrt(distance2.min(1)) - own
        result['models'][name] = {
            'top1': accuracy_stats(pred == truth, data['world'][test]),
            'top5': accuracy_stats((top5 == truth[:, None]).any(1), data['world'][test]),
            'query_to_own_center_mean': float(own.mean()),
            'wrong_minus_correct_margin_mean': float(margin.mean()),
        }
        arrays[name + '_predicted_world'] = ids[pred]
        arrays[name + '_correct_center_distance'] = own
        arrays[name + '_margin'] = margin
    np.savez_compressed(output / 'readout_predictions.npz', **arrays)
    return result


def matched_readout(data_by_dataset, metadata_by_dataset, output):
    splits = {name: split(data, metadata_by_dataset[name]) for name, data in data_by_dataset.items()}
    ids = np.intersect1d(splits['sonic'][2], splits['lafan'][2])
    assert len(ids) > 5
    selected = {name: {'train': [], 'test': []} for name in splits}
    rng = np.random.default_rng(20260916)
    counts = []
    for world in ids:
        rows = {name: {'train': np.flatnonzero(sp[0] & (data_by_dataset[name]['world'] == world)),
                       'test': np.flatnonzero(sp[1] & (data_by_dataset[name]['world'] == world))}
                for name, sp in splits.items()}
        n_train = min(len(x['train']) for x in rows.values())
        n_test = min(len(x['test']) for x in rows.values())
        assert n_train > 0 and n_test > 0
        counts.append((int(world), n_train, n_test))
        for name in rows:
            for key, count in [('train', n_train), ('test', n_test)]:
                selected[name][key].append(rng.choice(rows[name][key], count, replace=False))
    result = {'worlds': len(ids), 'seed': 20260916,
              'contract': 'Identical physical world IDs and candidate centers; equal center and query sample counts per world across datasets. Subsample original valid disjoint-history splits.',
              'counts': counts, 'datasets': {}}
    arrays, correctness = {}, {}
    for name, data in data_by_dataset.items():
        trains = selected[name]['train']
        test = np.concatenate(selected[name]['test'])
        truth = np.searchsorted(ids, data['world'][test])
        arrays[name + '_train_source_rows'] = data['source_row'][np.concatenate(trains)]
        arrays[name + '_query_source_rows'] = data['source_row'][test]
        result['datasets'][name] = {'queries': len(test), 'models': {}}
        correctness[name] = {}
        for model, field in FIELDS.items():
            z = normalized(data[field].astype(np.float64))
            centers = np.stack([z[rows].mean(0) for rows in trains])
            q = z[test]
            distances = (q*q).sum(1)[:, None] + (centers*centers).sum(1)[None] - 2*q@centers.T
            correct = distances.argmin(1) == truth
            top5 = np.argpartition(distances, 4, axis=1)[:, :5]
            result['datasets'][name]['models'][model] = {
                'top1': accuracy_stats(correct, data['world'][test]),
                'top5': accuracy_stats((top5 == truth[:, None]).any(1), data['world'][test]),
            }
            correctness[name][model] = correct
    # Query ordering and counts are paired by world, although the motions differ.
    labels = np.repeat(ids, np.array(counts)[:, 2])
    _, inverse = np.unique(labels, return_inverse=True)
    count = np.bincount(inverse)
    draws = np.random.default_rng(20260916).integers(len(ids), size=(2000, len(ids)))
    result['sonic_minus_lafan_pp'] = {}
    for model in FIELDS:
        gains = np.bincount(inverse, weights=correctness['sonic'][model].astype(float) -
                           correctness['lafan'][model].astype(float))
        bootstrap = 100 * gains[draws].sum(1) / count[draws].sum(1)
        result['sonic_minus_lafan_pp'][model] = {
            'gain': float(100 * gains.sum() / count.sum()),
            'world_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist()}
    arrays['center_worlds'] = ids
    np.savez_compressed(output / 'matched_dataset_readout_rows.npz', **arrays)
    return result


def analyze(root, *, allow_partial=False):
    checkpoints = json.loads((root / 'checkpoints.json').read_text())
    states = {name: torch.load(info['path'], map_location='cpu', weights_only=False, mmap=True)
              for name, info in checkpoints.items()}
    health = {}
    for name, state in states.items():
        assert sha256(Path(checkpoints[name]['path'])) == checkpoints[name]['sha256']
        assert all(torch.isfinite(x).all() for x in state['model'].values() if x.is_floating_point())
        steps = {int(x['step']) for x in state['optimizer']['state'].values() if 'step' in x}
        assert steps == {state['optimizer_steps']}
        assert state['loss_config']['representation_weight'] == .4
        assert state['loss_config']['dr_positive_weight'] == .2
        assert state['loss_config']['dr_distance_scale'] == .2
        health[name] = {'update': state['update'], 'optimizer_steps': state['optimizer_steps'],
                        'model_finite': True, 'loss_config': state['loss_config'],
                        'ddp_agreement': state['distributed_parameter_agreement']}
    summary = {'complete': False, 'checkpoints': checkpoints, 'checkpoint_health': health,
               'motion_manifest': str(root / 'motion_manifest.json'), 'datasets': {},
               'matched_dataset_readout': {}, 'physics_equality': {}, 'precision': 'bfloat16'}
    cached = {}
    script_hash = sha256(Path(__file__))
    for dataset in ('sonic', 'lafan'):
        summary['datasets'][dataset] = {}
        for profile in PROFILES:
            folder = root / profile if dataset == 'sonic' else root / 'lafan' / profile
            meta = json.loads((folder / 'metadata.json').read_text())
            if allow_partial and not meta['complete']:
                continue
            assert meta['complete']
            meta = {**meta, 'models': FIELDS}
            data = load_data(folder)
            for field in FIELDS.values():
                assert np.isfinite(data[field]).all()
            full = full_data(data)
            dr = full_data(data, dr_only=True)
            out = folder / 'analysis'
            out.mkdir(exist_ok=True)
            physics = folder / 'physics.npz' if dataset == 'sonic' else Path(meta['source_physics'])
            signature = {'script_sha256': script_hash,
                         'latents_sha256': sha256(folder / 'latents.npz'),
                         'metadata_sha256': sha256(folder / 'metadata.json')}
            result_path = out / 'cluster_metrics.json'
            existing = json.loads(result_path.read_text()) if result_path.exists() else {}
            if existing.get('input_signature') == signature:
                summary['datasets'][dataset][profile] = existing
                cached[(dataset, profile)] = (dr, meta, physics)
                continue
            geometry = analyze_subset(full, meta, out, 'memory_full')
            readout = environment_readout(dr, meta, disjoint=True)
            details = readout_details(dr, meta, out)
            for name in FIELDS:
                assert details['models'][name]['top1']['accuracy'] == readout['models'][name]['top1_accuracy']
            center = center_geometry(dr, meta, physics, states['memory350']['dr_metric_schema'],
                                     distance_scales={name: .2 for name in FIELDS})
            profile_result = {'geometry': geometry, 'readout': readout, 'readout_details': details,
                              'dr_only_center_geometry': center, 'all_queries': len(data['world']),
                              'full_history_queries': len(full['world']), 'full_history_dr_queries': len(dr['world']),
                              'full_history_dr_fraction': len(dr['world']) / int((~data['nominal']).sum()),
                              'motion_count': len(meta['motion_files']),
                              'motion_ids_observed_full': len(np.unique(full['motion'])),
                              'input_signature': signature}
            write_json(out / 'cluster_metrics.json', profile_result)
            summary['datasets'][dataset][profile] = profile_result
            cached[(dataset, profile)] = (dr, meta, physics)
            write_json(root / 'summary.json', summary)
            print(json.dumps({'event': 'analyzed', 'dataset': dataset, 'profile': profile,
                              'readout': readout}), flush=True)
    for profile in PROFILES:
        if ('sonic', profile) not in cached or ('lafan', profile) not in cached:
            continue
        s, l = cached[('sonic', profile)], cached[('lafan', profile)]
        with np.load(s[2]) as a, np.load(l[2]) as b:
            assert np.array_equal(a['names'], b['names'])
            assert np.array_equal(a['values'], b['values'])
        summary['physics_equality'][profile] = {'exact_names_and_values_match': True,
                                               'sonic_physics': str(s[2]), 'lafan_physics': str(l[2])}
        summary['matched_dataset_readout'][profile] = matched_readout(
            {'sonic': s[0], 'lafan': l[0]}, {'sonic': s[1], 'lafan': l[1]}, root / profile / 'analysis')
    summary['complete'] = len(cached) == 4
    summary['analysis_script_sha256'] = script_hash
    write_json(root / 'summary.json', summary)
    if summary['complete']:
        report_and_plot(root, summary)
    print(json.dumps({'event': 'complete' if summary['complete'] else 'partial_analysis',
                      'output': str(root / 'summary.json')}), flush=True)


def report_and_plot(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    titles = {'capped_loads': 'DR + limb loads', 'common': 'DR without limb loads'}
    chinese = {'capped_loads': 'DR＋四肢负载', 'common': '普通 DR'}
    rows = ['# Sonic 128 动作：Memory350 聚类评测', '',
            '主表统一使用 u32750，保留 u30750 在相同轨迹上的结果作为参考。'
            '两组物理参数与原 LAFAN 诊断逐项相同；Sonic 文件按固定随机种子抽取，'
            '同一源动作的演员、镜像和重复 take 只保留一个版本。', '',
            '只使用完整 short50＋long30×10 历史。已知环境中心与查询使用不同 motion，'
            '并排除原始历史重叠。距离为 64 维单位 latent 的成对欧氏距离 RMS。', '',
            '| 条件 | 指标 | LAFAN 42 motion | Sonic 128 motion |',
            '|---|---|---:|---:|']
    for profile in PROFILES:
        values = [summary['datasets'][dataset][profile] for dataset in ('lafan', 'sonic')]
        for key, label in [('top1_accuracy', '跨 motion Top-1'), ('top5_accuracy', 'Top-5')]:
            vals = [v['readout']['models']['memory350'][key] for v in values]
            rows.append(f'| {chinese[profile]} | {label} | {vals[0]:.2%} | {vals[1]:.2%} |')
        for key, label in [('dr_same_world_cross_motion_disjoint', '同 DR、跨 motion、历史不重叠'),
                           ('dr_different_world_same_motion_near_phase', '不同 DR、同 motion、phase 差≤0.02'),
                           ('dr_different_world_cross_motion', '不同 DR、不同 motion')]:
            vals = [v['geometry']['models']['memory350']['pairs'][key]['unit_distance_rms'] for v in values]
            rows.append(f'| {chinese[profile]} | {label} | {vals[0]:.4f} | {vals[1]:.4f} |')
        vals = [v['geometry']['models']['memory350']['dr_same_world_cross_motion_disjoint_over_between'] for v in values]
        rows.append(f'| {chinese[profile]} | 簇内／簇间 | {vals[0]:.4f} | {vals[1]:.4f} |')
        rows.append(f'| {chinese[profile]} | 中心数／查询数 | '
                    f'{values[0]["readout"]["worlds"]} / {values[0]["readout"]["test_samples"]} | '
                    f'{values[1]["readout"]["worlds"]} / {values[1]["readout"]["test_samples"]} |')
    rows += ['', '## 对齐环境中心与每个环境的样本数量', '',
             '两数据集沿用各自合法的中心／查询划分，取共有物理 world，'
             '再逐 world 对齐用于中心和查询的样本数量。固定 seed=20260916；'
             '这样控制候选中心数量与估计中心的样本量。', '',
             '| 条件 | 共同中心数 | 各组查询数 | LAFAN Top-1 | Sonic Top-1 | Sonic−LAFAN，百分点及95%区间 |',
             '|---|---:|---:|---:|---:|---|']
    for profile in PROFILES:
        m = summary['matched_dataset_readout'][profile]
        a, b = (m['datasets'][name] for name in ('lafan', 'sonic'))
        gain = m['sonic_minus_lafan_pp']['memory350']
        ci = gain['world_bootstrap_ci95']
        rows.append(f'| {chinese[profile]} | {m["worlds"]} | {a["queries"]} | '
                    f'{a["models"]["memory350"]["top1"]["accuracy"]:.2%} | '
                    f'{b["models"]["memory350"]["top1"]["accuracy"]:.2%} | '
                    f'{gain["gain"]:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] |')
    rows += ['', '## 覆盖率', '',
             '| 数据 | 条件 | 完整历史 DR 查询数 | 占全部 DR 查询 | 完整查询覆盖 motion 数 |',
             '|---|---|---:|---:|---:|']
    for dataset in ('lafan', 'sonic'):
        for profile in PROFILES:
            v = summary['datasets'][dataset][profile]
            rows.append(f'| {dataset} | {chinese[profile]} | {v["full_history_dr_queries"]} | '
                        f'{v["full_history_dr_fraction"]:.2%} | {v["motion_ids_observed_full"]} |')
    rows += ['', '## 范围与复核', '',
             '- 这是固定抽样的 128 条 Sonic motion，不能等同于完整 129785 条 Sonic 的穷举评测。',
             '- Sonic 和 LAFAN 都来自训练 motion 库；跨 motion 是指环境中心与查询分开，不是 encoder 未见 motion 的测试。',
             '- 350 步历史可跨 episode 保留；在 Sonic 较短动作上，近期历史更常包含多段 motion。',
             '- 为匹配历史评测，带负载组沿用 uniform，普通 DR 组沿用原 tracker 的 adaptive；每个 profile 内两数据集设置相同。当前训练仍是 uniform。',
             '- 主识别率是已知环境中心的分类，不是新环境路由正确率或 PPO 控制效果。',
             '- Bootstrap 按物理 world 重采样，不包含重复训练、重复动作子集的随机性。',
             '- 同 motion/phase 匹配并不强制实际状态、动作或整个历史一致。', '',
             '配置与数据清单：`motion_manifest.json`、`checkpoints.json`；'
             '完整指标：`summary.json`；配对索引和逐查询预测位于各 profile 的 `analysis/`。', '']
    (root / 'README.md').write_text('\n'.join(rows))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, profile in enumerate(PROFILES):
        for col, dataset in enumerate(('lafan', 'sonic')):
            folder = root / profile if dataset == 'sonic' else root / 'lafan' / profile
            data = full_data(load_data(folder))
            u = normalized(data['latent'].astype(np.float64))
            with np.load(folder / 'analysis/memory_full_pairs.npz') as pairs:
                for key, label, color in [
                    ('dr_same_world_cross_motion_disjoint', 'Same DR, different motion', '#2563eb'),
                    ('dr_different_world_same_motion_near_phase', 'Different DR, matched motion/phase', '#ea580c')]:
                    distance = np.linalg.norm(u[pairs[key + '_left']] - u[pairs[key + '_right']], axis=1)
                    axes[row, col].hist(distance, bins=np.linspace(0, 2, 65), density=True,
                                        histtype='step', linewidth=2, color=color, label=label)
            axes[row, col].set(title=f'{dataset.upper()} | {titles[profile]}',
                               xlabel='64-D unit-latent Euclidean distance', ylabel='Density', xlim=(0, 2))
            axes[row, col].legend(fontsize=8)
    fig.suptitle('Memory350 u32750 | same physical DR worlds | disjoint-history positive pairs', fontsize=13)
    fig.savefig(root / 'distance_distributions.png', dpi=180)
    fig.savefig(root / 'distance_distributions.svg')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    manifest = {'seed': 20260916, 'checkpoint_update': 32750, 'profiles': {},
                'note': 'Real 64-D unit-latent distances, ordered by environment. Selection is independent of latent values.'}
    rng = np.random.default_rng(20260916)
    for row, profile in enumerate(PROFILES):
        datasets = {dataset: full_data(load_data(root / profile if dataset == 'sonic' else root / 'lafan' / profile), True)
                    for dataset in ('lafan', 'sonic')}
        ids = np.intersect1d(np.unique(datasets['lafan']['world']), np.unique(datasets['sonic']['world']))
        worlds = np.sort(rng.choice(ids, 16, replace=False))
        selected = {name: [] for name in datasets}
        ticks, boundaries, names, offset = [], [], [], 0
        for world in worlds:
            candidates = {name: np.flatnonzero(data['world'] == world) for name, data in datasets.items()}
            n = min(8, *(len(values) for values in candidates.values()))
            for name in datasets:
                selected[name].extend(rng.choice(candidates[name], n, replace=False).tolist())
            ticks.append(offset + (n - 1) / 2)
            boundaries.append(offset + n - .5)
            names.append(str(world))
            offset += n
        manifest['profiles'][profile] = {'worlds': worlds.tolist(), 'selected_source_rows': {}}
        for col, (name, data) in enumerate(datasets.items()):
            selected_rows = np.asarray(selected[name])
            u = normalized(data['latent'][selected_rows].astype(np.float64))
            matrix = np.sqrt(np.maximum(0, (u*u).sum(1)[:, None] + (u*u).sum(1)[None] - 2*u@u.T))
            ax = axes[row, col]
            im = ax.imshow(matrix, cmap='magma', vmin=0, vmax=2, interpolation='nearest')
            ax.set_xticks(ticks, names, rotation=90, fontsize=6)
            ax.set_yticks(ticks, names, fontsize=6)
            for edge in boundaries[:-1]:
                ax.axhline(edge, color='white', linewidth=.4, alpha=.5)
                ax.axvline(edge, color='white', linewidth=.4, alpha=.5)
            ax.set_title(f'{name.upper()} | {titles[profile]}', fontsize=11)
            manifest['profiles'][profile]['selected_source_rows'][name] = data['source_row'][selected_rows].tolist()
    fig.colorbar(im, ax=axes, label='True 64-D unit-latent Euclidean distance', shrink=.8)
    fig.suptitle('Same 16 preselected DR worlds per profile | equal samples per world | u32750', fontsize=13)
    fig.savefig(root / 'unit_distance_heatmaps.png', dpi=180)
    fig.savefig(root / 'unit_distance_heatmaps.svg')
    plt.close(fig)
    write_json(root / 'plot_manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('encode-reference', 'analyze'), required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'encode-reference':
        assert args.reference is not None
        encode_reference(args.root.resolve(), args.reference.resolve(), args.device)
    else:
        analyze(args.root.resolve(), allow_partial=args.allow_partial)


if __name__ == '__main__':
    main()
