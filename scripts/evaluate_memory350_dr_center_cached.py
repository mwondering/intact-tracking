"""Compare DR-center checkpoints on the already audited u1500 raw histories."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from analyze_forward_context_clusters import normalized
from analyze_memory350 import sha256
from analyze_memory350_latent_clusters import analyze_subset
from evaluate_memory350_dr_center import center_geometry
from evaluate_memory350_weak_pairs import encode, environment_readout, write_json
from intact_tracking.memory350_inference import load_memory350_checkpoint


def report(summary, output):
    names = summary['checkpoints']
    order = ('earlier', 'baseline', 'memory350', 'response10')
    labels = [f'{"旧 response10" if name == "response10" else "DR-center"} '
              f'u{names[name]["update"]}' for name in order]
    rows = [f'# DR-center u{summary["candidate_update"]} 固定轨迹检查', '',
            '复用 u1500 检查保存的原始状态、动作和 Memory350 历史。所有模型使用相同查询；'
            '各自使用 checkpoint 保存的归一化，BF16 前向。', '',
            '识别任务使用不同 motion family 建立环境中心和查询，并剔除重叠的原始 350 步历史。'
            '距离为真实 64 维单位 latent 的成对欧氏距离 RMS。', '',
            '| 条件 | 指标 | ' + ' | '.join(labels) + ' |',
            '|---|---|' + '---:|' * len(order)]
    notes = []
    for profile, label in [('capped_loads', 'DR＋负载（手2.5/腿4）'), ('common', '普通 DR')]:
        result = summary['profiles'][profile]
        readout = result['readout']
        for key, title in [('top1_accuracy', '跨 motion Top-1'), ('top5_accuracy', '跨 motion Top-5')]:
            values = [readout['models'][name][key] for name in order]
            rows.append(f'| {label} | {title} | ' + ' | '.join(f'{v:.2%}' for v in values) + ' |')
        for key, title in [
            ('nominal_cross_motion', 'nominal 跨 motion'),
            ('dr_same_world_cross_motion_disjoint', '同 DR 跨 motion，历史不重叠'),
            ('dr_different_world_same_motion_near_phase', '异 DR，同 motion、phase 差≤0.02'),
            ('dr_different_world_cross_motion', '异 DR，跨 motion'),
        ]:
            if key not in result['geometry']['models']['memory350']['pairs']:
                continue
            values = [result['geometry']['models'][name]['pairs'][key]['unit_distance_rms'] for name in order]
            rows.append(f'| {label} | {title} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
        for title, values in [
            ('簇内/簇间，越低越好', [result['geometry']['models'][name][
                'dr_same_world_cross_motion_disjoint_over_between'] for name in order]),
            ('DR-only 中心/参数距离 Pearson', [result['dr_only_center_geometry']['models'][name][
                'dr_distance_pearson'] for name in order]),
        ]:
            rows.append(f'| {label} | {title} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
        lo, hi = readout['top1_gain_query_world_ci95_pp']
        notes += ['', f'{label}：{readout["worlds"]} 个中心、{readout["test_samples"]} 个查询。'
                 f'最新相对前一个 checkpoint 的 Top-1 改变 {readout["top1_gain_percentage_points"]:+.2f} '
                 f'个百分点，按查询 world 配对 bootstrap 的 95% 区间 [{lo:+.2f}, {hi:+.2f}]。', '']
    rows += notes
    changes = summary.get('loss_config_changes', {})
    if changes:
        rows += ['本次候选相对 baseline 的损失配置变化：' + '；'.join(
            f'`{key}` {value["previous"]} → {value["candidate"]}' for key, value in changes.items()) + '。',
            f'调参后累计 {summary["candidate_update"] - names["baseline"]["update"]} 个 update；'
            '多项参数同时改变，不能把改进单独归因于某一个损失。', '']
    rows += ['u1500 的表征权重为 0.02；u1863 后为 0.04，同时从四卡改为八卡且保持全局 batch。'
             + ('本次两个较新 checkpoint 的损失配置不同。' if changes else '本次两个较新 checkpoint 的配置相同。')
             + '旧 response10 u15000 的训练轮数、损失和负载范围不同，'
             '只作为成熟模型参考。', '',
             '该任务衡量已知环境中心的跨 motion 识别，不等于未见环境分类或 PPO 控制表现。', '',
             '完整结果：`summary.json`；各 profile 下保存 latent、配对索引和原始查询来源。', '']
    (output / 'README.md').write_text('\n'.join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'previous', 'cache_root', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-loss-retuning', action='store_true',
                        help='Compare the retuned center/positive weights and DR target scale; keep model, normalization, tracker and physical schema identical')
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = args.cache_root.resolve()
    cached_summary = json.loads((cache / 'summary.json').read_text())
    assert cached_summary['complete']
    paths = {'memory350': args.checkpoint.resolve(), 'baseline': args.previous.resolve(),
             'earlier': Path(cached_summary['checkpoints']['memory350']['path']),
             'response10': Path(cached_summary['checkpoints']['baseline']['path'])}
    states = {name: torch.load(path, map_location='cpu', weights_only=False, mmap=True)
              for name, path in paths.items()}
    candidate, previous = states['memory350'], states['baseline']
    for field in ('model_config', 'normalization', 'tracker', 'dr_metric_schema'):
        assert candidate[field] == previous[field], field
    old_loss, new_loss = dict(previous['loss_config']), dict(candidate['loss_config'])
    for loss in (old_loss, new_loss):
        loss.setdefault('dr_positive_weight', 0.)
    changes = {key: {'previous': old_loss.get(key), 'candidate': new_loss.get(key)}
               for key in sorted(old_loss.keys() | new_loss.keys()) if old_loss.get(key) != new_loss.get(key)}
    if changes:
        assert args.allow_loss_retuning, 'Use --allow-loss-retuning for a changed objective'
        assert changes.keys() <= {'representation_weight', 'dr_distance_scale', 'dr_positive_weight'}, changes
    assert candidate['update'] > previous['update'] > states['earlier']['update']
    for state in states.values():
        assert state['model_config'] == candidate['model_config']
        assert state['tracker']['checkpoint_sha256'] == candidate['tracker']['checkpoint_sha256']
    schema = candidate['dr_metric_schema']
    assert schema == states['earlier']['dr_metric_schema']
    health = {}
    for name in ('baseline', 'memory350'):
        state = states[name]
        assert state['optimizer_steps'] == 4 * state['update'] == state['scheduler']['last_epoch']
        assert state['distributed_parameter_agreement']['passed']
        assert len(set(state['distributed_parameter_agreement']['sha256_by_rank'])) == 1
        assert all(torch.isfinite(t).all().item() for t in state['model'].values() if t.is_floating_point())
        optimizer_steps = {int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v}
        assert optimizer_steps == {state['optimizer_steps']}
        health[name] = {'update': state['update'], 'optimizer_steps': state['optimizer_steps'],
                        'learning_rate': state['optimizer']['param_groups'][0]['lr'],
                        'model_finite': True, 'ddp_agreement': state['distributed_parameter_agreement'],
                        'representation_weight': state['loss_config']['representation_weight'],
                        'dr_distance_scale': state['loss_config']['dr_distance_scale'],
                        'dr_positive_weight': state['loss_config'].get('dr_positive_weight', 0.)}
    encoders = {name: load_memory350_checkpoint(paths[name], device=args.device)
                for name in ('memory350', 'baseline', 'earlier')}
    checkpoints = {name: {'path': str(path), 'sha256': sha256(path), 'update': states[name]['update']}
                   for name, path in paths.items()}
    assert checkpoints['earlier']['sha256'] == cached_summary['checkpoints']['memory350']['sha256']
    assert checkpoints['response10']['sha256'] == cached_summary['checkpoints']['baseline']['sha256']
    summary = {'complete': False, 'candidate_update': candidate['update'], 'checkpoints': checkpoints,
               'checkpoint_health': health, 'cache_root': str(cache), 'profiles': {},
               'verification': {}, 'same_raw_queries': True, 'precision': 'bfloat16',
               'loss_config_changes': changes}
    for profile in ('capped_loads', 'common'):
        source = cache / profile
        old_meta = json.loads((source / 'metadata.json').read_text())
        assert old_meta['complete']
        if profile == 'capped_loads':
            assert old_meta['rollout_config']['limb_max_masses_kg'] == [2.5, 2.5, 4., 4.]
            query_paths = sorted((source / 'queries').glob('query_*.pt'))
            physics = source / 'physics.npz'
            expected_hashes = {}
        else:
            query_paths = [Path(q['path']) for q in old_meta['queries']]
            physics = Path(old_meta['source_physics'])
            expected_hashes = {q['path']: q['sha256'] for q in old_meta['queries']}
        assert len(query_paths) == 32
        with np.load(source / 'latents.npz') as saved:
            data = {key: saved[key] for key in saved.files if not key.startswith('latent')}
            data['latent_earlier'] = saved[old_meta['models']['memory350']]
            data['latent_response10'] = saved[old_meta['models']['baseline']]
        parts = {'memory350': [], 'baseline': []}
        query_records, reconstruction = [], []
        for index, path in enumerate(query_paths):
            query = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            mask = data['step'] == query['step']
            assert query['format_version'] == 'memory350_raw_query_v1'
            assert np.array_equal(query['world'].numpy(), data['world'][mask])
            assert np.array_equal(query['short_valid'].sum(-1).numpy(), data['short_steps'][mask])
            assert np.array_equal(query['long_valid'].sum(-1).numpy(), data['long_chunks'][mask])
            if index in (0, len(query_paths) - 1):
                z = encode(encoders['earlier'], query)
                error = float(np.sqrt(np.mean(np.sum((normalized(z) - normalized(data['latent_earlier'][mask])) ** 2, axis=1))))
                assert error < .001, (profile, query['step'], error)
                reconstruction.append({'step': query['step'], 'unit_latent_rms_error': error})
            for name in parts:
                value = encode(encoders[name], query)
                assert np.isfinite(value).all()
                parts[name].append(value)
            digest = sha256(path)
            if expected_hashes:
                assert digest == expected_hashes[str(path)]
            query_records.append({'path': str(path), 'sha256': digest, 'step': query['step']})
            if index % 8 == 0 or index == len(query_paths) - 1:
                print(json.dumps({'event': 'encoded', 'profile': profile, 'queries': index + 1,
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
        data.update({'latent_' + name: np.concatenate(values) for name, values in parts.items()})
        data['source_row'] = np.arange(len(data['world']))
        folder = output / profile
        folder.mkdir()
        meta = {'models': {name: 'latent_' + name for name in ('earlier', 'baseline', 'memory350', 'response10')},
                'checkpoints': checkpoints, 'motion_files': old_meta['motion_files'], 'profile': profile,
                'complete': True, 'source_metadata': str(source / 'metadata.json'),
                'source_metadata_sha256': sha256(source / 'metadata.json'),
                'source_physics': str(physics), 'source_physics_sha256': sha256(physics), 'queries': query_records}
        write_json(folder / 'metadata.json', meta)
        np.savez_compressed(folder / 'latents.npz', **data)
        mask = (data['short_steps'] == 50) & (data['long_chunks'] == 30)
        full = {k: v[mask] for k, v in data.items()}
        analysis = folder / 'analysis'
        analysis.mkdir()
        geometry = analyze_subset(full, meta, analysis, 'memory_full')
        dr = {k: v[~full['nominal']] for k, v in full.items()}
        readout = environment_readout(dr, meta, disjoint=True)
        old_readout = cached_summary['profiles'][profile]['readout']
        assert readout['worlds'] == old_readout['worlds'] and readout['test_samples'] == old_readout['test_samples']
        for name, old_name in [('earlier', 'memory350'), ('response10', 'baseline')]:
            assert readout['models'][name] == old_readout['models'][old_name]
            for pair, metrics in geometry['models'][name]['pairs'].items():
                if 'unit_distance_rms' in metrics:
                    old_value = cached_summary['profiles'][profile]['geometry']['models'][old_name]['pairs'][pair]['unit_distance_rms']
                    assert math.isclose(metrics['unit_distance_rms'], old_value, rel_tol=1e-10, abs_tol=1e-12)
        result = {'geometry': geometry, 'readout': readout,
                  'dr_only_center_geometry': center_geometry(dr, meta, physics, schema,
                      distance_scales={name: states[name]['loss_config'].get('dr_distance_scale', .3)
                                       for name in states})}
        write_json(analysis / 'cluster_metrics.json', result)
        summary['profiles'][profile] = result
        summary['verification'][profile] = {'reconstruction': reconstruction, 'queries': len(query_records),
                                             'old_readout_and_distance_metrics_reproduced': True}
        write_json(output / 'summary.json', summary)
        print(json.dumps({'event': 'profile_complete', 'profile': profile, 'readout': readout}), flush=True)
    summary.update(complete=True, elapsed_seconds=time.monotonic() - started, script_sha256=sha256(Path(__file__)))
    write_json(output / 'summary.json', summary)
    report(summary, output)
    print(json.dumps({'event': 'complete', 'report': str(output / 'README.md')}), flush=True)


if __name__ == '__main__':
    main()
