"""Evaluate a DR-center checkpoint on shared raw histories and capped-load rollouts."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_forward_context_clusters import normalized
from analyze_memory350 import sha256
from analyze_memory350_latent_clusters import analyze_subset
from evaluate_memory350_weak_pairs import encode, environment_readout, write_json
from intact_tracking.memory350_inference import load_memory350_checkpoint


def center_geometry(data, metadata, physics_path, schema, *, distance_scales=None):
    """Measure DR-only geometry, without the easy nominal-versus-DR contrast."""
    from scipy.stats import spearmanr

    family_names = np.array([Path(p).name.split('_subject')[0] for p in metadata['motion_files']])
    family = family_names[data['motion']]
    families = np.random.default_rng(93011).permutation(np.unique(family))
    train = np.isin(family, families[:len(families) // 2]) & (data['step'] <= 1600)
    ids = np.unique(data['world'][train])
    with np.load(physics_path) as physical:
        names = physical['names'].tolist()
        values = physical['values']
    assert ids.min() >= 0 and ids.max() < len(values)
    columns = []
    for name in schema['names']:
        if name in names:
            columns.append(values[ids, names.index(name)])
        else:
            assert '/added_mass_kg/' in name, name
            columns.append(np.zeros(len(ids)))
    raw = np.column_stack(columns)
    unit = (raw - schema['lower']) / (np.array(schema['upper']) - schema['lower'])
    assert np.isfinite(unit).all() and unit.min() >= -1e-5 and unit.max() <= 1 + 1e-5
    features = unit * np.sqrt(schema['coordinate_weights'])
    left, right = np.triu_indices(len(ids), 1)
    d_dr = np.linalg.norm(features[left] - features[right], axis=1)
    target = 2 * d_dr / (d_dr + .3)
    result = {'worlds': len(ids), 'pairs': len(left),
              'reference': 'First half of motion families, collector steps <=1600; DR worlds only.',
              'dr_parameter_distance_mean': float(d_dr.mean()),
              'reference_target_scale': .3,
              'target_distance_mean': float(target.mean()), 'models': {}}
    for name, field in metadata['models'].items():
        scale = .3 if distance_scales is None else distance_scales.get(name, .3)
        assert np.isfinite(scale) and scale > 0
        model_target = 2 * d_dr / (d_dr + scale)
        z = normalized(data[field].astype(np.float64))
        centers = np.stack([z[train & (data['world'] == key)].mean(0) for key in ids])
        distance = np.linalg.norm(centers[left] - centers[right], axis=1)
        result['models'][name] = {
            'dr_distance_pearson': float(np.corrcoef(d_dr, distance)[0, 1]),
            'dr_distance_spearman': float(spearmanr(d_dr, distance).statistic),
            'mapped_target_rmse': float(np.sqrt(np.mean((distance - model_target) ** 2))),
            'target_distance_scale': float(scale),
            'target_distance_mean': float(model_target.mean()),
            'latent_center_distance_mean': float(distance.mean()),
        }
    return result


def analyze_profile(folder, metadata, schema, physics_path):
    with np.load(folder / 'latents.npz') as source:
        data = {k: source[k] for k in source.files}
    data['source_row'] = np.arange(len(data['world']))
    mask = (data['short_steps'] == 50) & (data['long_chunks'] == 30)
    full = {k: value[mask] for k, value in data.items()}
    analysis = folder / 'analysis'
    analysis.mkdir(exist_ok=True)
    geometry = analyze_subset(full, metadata, analysis, 'memory_full')
    np.savez_compressed(analysis / 'memory_full_data.npz', **full)
    dr = {k: value[~full['nominal']] for k, value in full.items()}
    readout = environment_readout(dr, metadata, disjoint=True)
    trend_metadata = {**metadata, 'models': {
        'baseline': metadata['models']['earlier'], 'memory350': metadata['models']['memory350']}}
    trend = environment_readout(dr, trend_metadata, disjoint=True)
    result = {'geometry': geometry, 'readout': readout, 'u1000_to_candidate_readout': trend,
              'dr_only_center_geometry': center_geometry(dr, metadata, physics_path, schema)}
    write_json(analysis / 'cluster_metrics.json', result)
    return result


def report(summary, output):
    rows = ['# DR-center u1500 聚类检查', '',
            '候选：双手上限 2.5 kg / 双小腿 4 kg，nominal50 Memory350 encoder2x，DR 中心关系 + 五步 predictor。', '',
            '同版本 u1000 用于检查训练趋势；旧 response10 u15000 仅作为成熟模型参考，训练轮数、损失、负载范围和归一化不同，不能视为单因素消融。', '',
            '所有距离来自 64 维单位 latent，报告成对欧氏距离的 RMS。环境识别使用不同 motion family 建中心和查询，且原始 350 步历史不重叠。', '',
            '| 条件 | 指标 | DR-center u1000 | DR-center u1500 | 旧 response10 u15000 |',
            '|---|---|---:|---:|---:|']
    for profile, title in (('capped_loads', 'DR＋负载（手2.5/腿4）'), ('common', '普通DR（无负载）')):
        result = summary['profiles'][profile]
        readout = result['readout']
        models = result['geometry']['models']
        for metric, key in (('同DR跨motion，历史不重叠', 'dr_same_world_cross_motion_disjoint'),
                            ('不同DR，同motion、phase差≤0.02', 'dr_different_world_same_motion_near_phase'),
                            ('不同DR，不同motion', 'dr_different_world_cross_motion')):
            values = [models[name]['pairs'][key]['unit_distance_rms'] for name in ('earlier', 'memory350', 'baseline')]
            rows.append(f'| {title} | {metric} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
        values = [models[name]['dr_same_world_cross_motion_disjoint_over_between'] for name in ('earlier', 'memory350', 'baseline')]
        rows.append(f'| {title} | 簇内/簇间（越低越好） | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
        for key in ('top1_accuracy', 'top5_accuracy'):
            values = [readout['models'][name][key] for name in ('earlier', 'memory350', 'baseline')]
            rows.append(f'| {title} | {key}（{readout["worlds"]}中心，{readout["test_samples"]}查询） | ' + ' | '.join(f'{v:.2%}' for v in values) + ' |')
        values = [result['dr_only_center_geometry']['models'][name]['dr_distance_pearson'] for name in ('earlier', 'memory350', 'baseline')]
        rows.append(f'| {title} | DR中心距离/参数距离 Pearson | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    rows += ['', '普通 DR 复用历史固定原始轨迹；2.5/4 kg 负载组新采 512 个固定 DR world，每个 3200 步，42 个诊断 motion 文件。所有 encoder 处理各组内完全相同的原始轨迹，各自使用 checkpoint 中保存的归一化。', '',
             '2.5/4 kg 新轨迹与历史 4/4 kg 诊断集不同，不能直接把本表百分比与历史 81.60% 相减。当前主指标仍是已知环境中心的跨 motion 识别，不等于新环境识别或控制性能。', '',
             '详细结果：`summary.json`；原始 latent、物理标签和历史分别保存在两个 profile 目录或其注明的缓存路径。', '']
    (output / 'README.md').write_text('\n'.join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--earlier', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--common-cache', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    output = args.output.resolve()
    paths = {'memory350': args.checkpoint.resolve(), 'earlier': args.earlier.resolve(), 'baseline': args.reference.resolve()}
    states = {name: torch.load(path, map_location='cpu', weights_only=False, mmap=True) for name, path in paths.items()}
    assert states['memory350']['update'] == 1500 and states['earlier']['update'] == 1000
    assert states['baseline']['update'] == 15000
    for name in ('memory350', 'earlier'):
        assert states[name]['loss_config']['dr_center_objective_version'] == 1
        assert states[name]['loss_config']['representation_weight'] == .02
    for state in states.values():
        assert state['model_config'] == states['memory350']['model_config']
        assert state['tracker']['checkpoint_sha256'] == states['memory350']['tracker']['checkpoint_sha256']
    encoders = {name: load_memory350_checkpoint(path, device='cuda:0') for name, path in paths.items()}
    assert encoders['baseline'].sha256 == 'db5d7bd7df026db163b9eedf588d37dcc81b55551d28afe3744f67a8ab2642ac'
    run_config = json.loads((paths['memory350'].parent / 'run_config.json').read_text())
    schema = run_config['dr_center_contract']['schema']
    summary = {'complete': False, 'candidate_update': 1500,
               'checkpoints': {name: {'path': str(path), 'sha256': encoders[name].sha256,
                                     'update': states[name]['update']} for name, path in paths.items()},
               'same_model': True, 'same_raw_queries_within_profile': True,
               'normalization': 'Each checkpoint uses its own saved normalization.',
               'profiles': {}, 'verification': {}}
    capped = output / 'capped_loads'
    metadata = json.loads((capped / 'metadata.json').read_text())
    assert metadata['complete'] and metadata['rollout_config']['limb_max_masses_kg'] == [2.5, 2.5, 4., 4.]
    assert metadata['checkpoint_sha256'] == encoders['memory350'].sha256
    assert metadata['comparison_checkpoints']['baseline']['sha256'] == encoders['baseline'].sha256
    assert metadata['comparison_checkpoints']['earlier']['sha256'] == encoders['earlier'].sha256
    metadata['models'] = {'memory350': 'latent', 'baseline': 'latent_baseline', 'earlier': 'latent_earlier'}
    query = torch.load(capped / 'queries/query_003200.pt', map_location='cpu', weights_only=False, mmap=True)
    with np.load(capped / 'latents.npz') as data:
        expected = data['latent'][data['step'] == 3200]
    reconstructed = encode(encoders['memory350'], query)
    error = float(np.sqrt(np.mean(np.sum((normalized(reconstructed) - normalized(expected)) ** 2, axis=1))))
    assert error < .001
    with np.load(capped / 'physics.npz') as physical:
        cols = [i for i, name in enumerate(physical['names']) if '/added_mass_kg/' in name]
        loads = physical['values'][:, cols]
    assert len(cols) == 4 and (loads >= 0).all() and (loads <= [2.5, 2.5, 4, 4]).all()
    summary['verification']['capped_loads'] = {'reconstruction_unit_rms': error, 'load_max_kg': loads.max(0).tolist(),
                                              'numerical_check': metadata['numerical_check']}
    summary['profiles']['capped_loads'] = analyze_profile(capped, metadata, schema, capped / 'physics.npz')
    write_json(output / 'summary.json', summary)
    print(json.dumps({'event': 'capped_analyzed', 'readout': summary['profiles']['capped_loads']['readout']}), flush=True)

    source = args.common_cache.resolve()
    folder = output / 'common'
    folder.mkdir(exist_ok=True)
    old_metadata = json.loads((source / 'metadata.json').read_text())
    assert old_metadata['complete'] and old_metadata['arguments']['save_history']
    with np.load(source / 'latents.npz') as saved:
        data = {key: saved[key] for key in saved.files if not key.startswith('latent')}
        calibration_latent = saved['latent'].copy()
    calibration = load_memory350_checkpoint(old_metadata['checkpoint'], device='cuda:0')
    parts = {name: [] for name in encoders}
    query_paths = sorted((source / 'queries').glob('query_*.pt'))
    assert len(query_paths) == 32
    queries, checks = [], []
    for number, path in enumerate(query_paths):
        query = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        mask = data['step'] == query['step']
        assert query['format_version'] == 'memory350_raw_query_v1'
        assert np.array_equal(query['world'].numpy(), data['world'][mask])
        assert np.array_equal(query['short_valid'].sum(-1).numpy(), data['short_steps'][mask])
        assert np.array_equal(query['long_valid'].sum(-1).numpy(), data['long_chunks'][mask])
        if number in (0, 31):
            z = encode(calibration, query)
            error = float(np.sqrt(np.mean(np.sum((normalized(z) - normalized(calibration_latent[mask])) ** 2, axis=1))))
            assert error < .001
            checks.append({'step': query['step'], 'calibration_reconstruction_unit_rms': error})
        for name, encoder in encoders.items():
            value = encode(encoder, query)
            assert np.isfinite(value).all()
            parts[name].append(value)
        queries.append({'path': str(path), 'sha256': sha256(path), 'step': query['step']})
        if number % 8 == 0 or number == 31:
            print(json.dumps({'event': 'common_encoded', 'queries': number + 1}), flush=True)
    fields = {name: 'latent_' + name for name in encoders}
    data.update({fields[name]: np.concatenate(values) for name, values in parts.items()})
    np.savez_compressed(folder / 'latents.npz', **data)
    metadata = {'models': fields, 'checkpoints': summary['checkpoints'],
                'checkpoint': str(paths['memory350']), 'checkpoint_sha256': encoders['memory350'].sha256,
                'checkpoint_update': 1500, 'checkpoint_label': 'DR-center / u1500',
                'comparison_checkpoints': {name: summary['checkpoints'][name] for name in ('baseline', 'earlier')},
                'motion_files': old_metadata['motion_files'], 'profile': 'common', 'complete': True,
                'arguments': {'num_envs': old_metadata['arguments']['num_envs']},
                'source_collection_arguments': old_metadata['arguments'],
                'rollout_config': old_metadata['rollout_config'], 'rollout_metadata': old_metadata['rollout_metadata'],
                'source_metadata': str(source / 'metadata.json'), 'source_metadata_sha256': sha256(source / 'metadata.json'),
                'source_physics': str(source / 'physics.npz'), 'queries': queries}
    write_json(folder / 'metadata.json', metadata)
    summary['verification']['common'] = checks
    summary['profiles']['common'] = analyze_profile(folder, metadata, schema, source / 'physics.npz')
    summary.update(complete=True, elapsed_seconds=time.monotonic() - started, script_sha256=sha256(Path(__file__)))
    write_json(output / 'summary.json', summary)
    report(summary, output)
    print(json.dumps({'event': 'complete', 'report': str(output / 'README.md')}), flush=True)


if __name__ == '__main__':
    main()
