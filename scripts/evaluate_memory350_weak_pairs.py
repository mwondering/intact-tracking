"""Compare encoder2x checkpoints on identical cached histories and A/B validation."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from analyze_forward_context_clusters import normalized
from analyze_memory350 import cluster_ci, error_metrics, sha256
from analyze_memory350_latent_clusters import analyze_subset
from intact_tracking.forward_predictor_objective import _normalized_state_error
from intact_tracking.memory350_inference import load_memory350_checkpoint
from intact_tracking.memory350_model import Memory350Config, Memory350Predictor


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@torch.inference_mode()
def encode(checkpoint, query, *, bf16=True):
    output = []
    device = checkpoint.state_mean.device
    for start in range(0, len(query['world']), 256):
        def normalize(raw):
            raw = raw[start:start + 256].to(device)
            return torch.cat(((raw[..., :71] - checkpoint.state_mean) / checkpoint.state_std,
                              (raw[..., 71:100] - checkpoint.action_mean) / checkpoint.action_std,
                              (raw[..., 100:] - checkpoint.state_mean) / checkpoint.state_std), -1)
        short, long = normalize(query['short']), normalize(query['long'])
        precision = torch.autocast('cuda', dtype=torch.bfloat16) if bf16 else nullcontext()
        with precision:
            value = checkpoint.encoder(short[..., :71], short[..., 71:100], short[..., 100:],
                query['short_valid'][start:start + 256].to(device), long,
                query['long_valid'][start:start + 256].to(device))
        output.append(value.float().cpu().numpy())
    return np.concatenate(output)


def environment_readout(data, metadata, *, disjoint):
    """Fit known-world centers on other motion families, optionally disjoint histories."""
    names = np.array([Path(p).name.split('_subject')[0] for p in metadata['motion_files']])
    family = names[data['motion']]
    families = np.random.default_rng(93011).permutation(np.unique(family))
    train = np.isin(family, families[:len(families) // 2])
    test = ~train
    world = data['world']
    if disjoint:
        train &= data['step'] <= 1600
        test &= data['step'] > 1600
        eligible = np.zeros(len(world), bool)
        for key in np.unique(world[train]):
            refs = train & (world == key)
            end = data['total_chunks'][refs] + (data['short_steps'][refs] + data['pending_steps'][refs] + 9) // 10
            sessions = np.unique(data['memory_session'][refs])
            assert len(sessions) == 1
            eligible |= ((world == key) & (data['total_chunks'] - data['long_chunks'] >= end.max())
                         & (data['step'] >= data['step'][refs].max() + 50)
                         & (data['memory_session'] == sessions[0]))
        test &= eligible
    ids = np.array(sorted(set(world[train]).intersection(world[test])))
    test &= np.isin(world, ids)
    if not len(ids) or not test.any():
        raise RuntimeError('No eligible worlds for cross-motion readout')
    truth = np.searchsorted(ids, world[test])
    result = {'worlds': len(ids), 'test_samples': int(test.sum()), 'uniform_chance': 1 / len(ids),
              'disjoint_history': disjoint, 'models': {},
              'train_motion_families': families[:len(families) // 2].tolist(),
              'test_motion_families': families[len(families) // 2:].tolist()}
    predictions = {}
    for name, field in metadata['models'].items():
        unit = normalized(data[field].astype(np.float64))
        centers = np.stack([unit[train & (world == key)].mean(0) for key in ids])
        distance = (np.square(unit[test]).sum(1)[:, None] + np.square(centers).sum(1)[None]
                    - 2 * unit[test] @ centers.T)
        correct = distance.argmin(1) == truth
        top5 = np.argpartition(distance, min(4, len(ids) - 1), axis=1)[:, :5]
        predictions[name] = correct
        result['models'][name] = {'top1_accuracy': float(correct.mean()),
                                  'top5_accuracy': float((top5 == truth[:, None]).any(1).mean())}
    labels, inverse = np.unique(world[test], return_inverse=True)
    counts = np.bincount(inverse)
    gain = np.bincount(inverse, weights=predictions['memory350'].astype(float) - predictions['baseline'].astype(float))
    draws = np.random.default_rng(20260912).integers(len(labels), size=(2000, len(labels)))
    delta = 100 * gain[draws].sum(1) / counts[draws].sum(1)
    result['top1_gain_percentage_points'] = 100 * float(gain.sum() / counts.sum())
    result['top1_gain_query_world_ci95_pp'] = np.quantile(delta, [.025, .975]).tolist()
    return result


@torch.inference_mode()
def prediction_comparison(paths, states, output):
    models = {}
    for name, state in states.items():
        model = Memory350Predictor(Memory350Config(**state['model_config']))
        model.load_state_dict(state['model'], strict=True)
        models[name] = model.to('cuda:0').eval().requires_grad_(False)
    parts, hashes = [], {}
    for rank in range(4):
        filename = f'validation_broad_rank_{rank}.pt'
        path = paths['baseline'].parent / filename
        hashes[filename] = sha256(path)
        assert sha256(paths['memory350'].parent / filename) == hashes[filename]
        full = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        size = len(full['state'])
        for start in range(0, size, 256):
            batch = {key: (value[start:start + 256] if value.ndim and value.shape[0] == size else value).to('cuda:0')
                     for key, value in full.items()}
            part = {key: batch[key].cpu().numpy() for key in ('world_id', 'is_nominal')}
            part['rank'] = np.full(len(batch['state']), rank)
            part['short_steps'] = batch['history_valid'].sum(-1).cpu().numpy()
            part['long_chunks'] = batch['memory_valid'].sum(-1).cpu().numpy()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                unchanged = _normalized_state_error(batch['state'][:, :1].expand_as(batch['state'][:, 1:]),
                    batch['state'][:, 1:], batch['state_mean'], batch['state_std'], batch['delta_std'])
                part['unchanged_mse'] = unchanged.float().square().mean(-1).cpu().numpy()
                for name, model in models.items():
                    latent = model.context_encoder(batch['history_state'], batch['history_action'],
                        batch['history_next_state'], batch['history_valid'], batch['memory_interactions'], batch['memory_valid'])
                    part[name + '_mse'] = error_metrics(model, batch, latent)
            parts.append(part)
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    assert all(np.isfinite(value).all() for value in data.values())
    np.savez_compressed(output / 'prediction_samples.npz', **data)
    result = {'validation_sha256': hashes, 'groups': {}, 'history_groups': {}}
    dr = ~data['is_nominal']
    conditions = [('groups', 'nominal', ~dr), ('groups', 'dr', dr)]
    conditions += [('history_groups', name, mask) for name, mask in (
        ('dr_full350', dr & (data['short_steps'] == 50) & (data['long_chunks'] == 30)),
        ('dr_full_long_incomplete_short', dr & (data['short_steps'] < 50) & (data['long_chunks'] == 30)),
        ('dr_partial_long', dr & (data['long_chunks'] > 0) & (data['long_chunks'] < 30)),
        ('dr_no_long', dr & (data['long_chunks'] == 0)),
    )]
    for section, group, mask in conditions:
        if not mask.any():
            continue
        a, b, denominator = data['baseline_mse'][:, -1], data['memory350_mse'][:, -1], data['unchanged_mse'][:, -1]
        result[section][group] = {
            'samples': int(mask.sum()), 'worlds': len(np.unique(data['world_id'][mask])),
            'baseline_nmse': float(a[mask].sum() / denominator[mask].sum()),
            'candidate_nmse': float(b[mask].sum() / denominator[mask].sum()),
            'candidate_to_baseline_error_ratio': float(b[mask].sum() / a[mask].sum()),
            'paired_world_ratio_ci95': cluster_ci(data['world_id'], a, b, mask),
        }
    write_json(output / 'prediction.json', result)
    del models
    torch.cuda.empty_cache()
    return result


def validate_checkpoints(states, comparison_kind):
    a, b = states['baseline'], states['memory350']
    for field in ('model_config', 'normalization', 'tracker', 'nominal_a_fraction'):
        if a[field] != b[field]:
            raise ValueError(f'Checkpoint comparison changed {field}')
    if b['nominal_a_fraction'] != .5:
        raise ValueError('Comparison requires nominal50 checkpoints')
    if comparison_kind == 'tuning':
        from intact_tracking.memory350_weak_tuning import MILESTONES, PARENT_UPDATE, validate_tuning_losses
        if a['update'] != PARENT_UPDATE or b['update'] not in MILESTONES:
            raise ValueError('Tuning compares u8000 against u9000 or u10000')
        validate_tuning_losses(a['loss_config'], b['loss_config'])
        for state in states.values():
            if state['optimizer_steps'] != 4 * state['update']:
                raise ValueError('Tuning changed the optimizer-step budget')
            if state['scheduler']['last_epoch'] != state['optimizer_steps']:
                raise ValueError('Tuning reset the restored scheduler')
            if state['optimizer']['param_groups'][0]['lr'] != 1e-5:
                raise ValueError('Tuning must retain the original learning-rate floor')
        for key in ('T_max', 'continuation_min_learning_rate', 'base_lrs', 'eta_min'):
            if a['scheduler'][key] != b['scheduler'][key]:
                raise ValueError(f'Tuning changed scheduler {key}')
        return
    if comparison_kind == 'continuation':
        if a['update'] != 5000 or not 5000 < b['update'] <= 10000 or b['update'] % 1000:
            raise ValueError('Continuation comparisons require u5000 versus a u6000..u10000 milestone')
        if a['loss_config'] != b['loss_config']:
            raise ValueError('Continuation comparison must keep the loss configuration unchanged')
        for state in states.values():
            if state['optimizer_steps'] != 4 * state['update']:
                raise ValueError('Continuation optimizer-step count differs from four steps per update')
            if state['scheduler']['T_max'] != 32000 or state['scheduler']['continuation_min_learning_rate'] != 1e-5:
                raise ValueError('Continuation changed the original cosine schedule or its floor')
        if (b['scheduler']['base_lrs'] != a['scheduler']['base_lrs']
                or b['scheduler']['eta_min'] != a['scheduler']['eta_min']):
            raise ValueError('Continuation changed the original cosine curve')
    else:
        for state in states.values():
            if state['update'] != 5000 or state['optimizer_steps'] != 20000:
                raise ValueError('Matched weak-pair ablation requires two u5000 checkpoints')
    expected_loss = {**a['loss_config'], 'response_distance_scale': .5,
                     'weak_positive_weight': .002, 'weak_negative_weight': .002, 'weak_negative_margin': 1.}
    if b['loss_config'] != expected_loss:
        raise ValueError('Checkpoint loss differs from the weak-pair experiment')


def assess(summary):
    models = summary['profiles']['memory_training']['memory_full']['models']
    baseline, candidate = models['baseline'], models['memory350']
    pair = 'dr_same_world_cross_motion_disjoint'
    within_change = 100 * (candidate['pairs'][pair]['unit_distance_rms'] / baseline['pairs'][pair]['unit_distance_rms'] - 1)
    ratio_change = 100 * (candidate[pair + '_over_between'] / baseline[pair + '_over_between'] - 1)
    readout = summary['readout']['memory_training']['disjoint']
    gain = readout['top1_gain_percentage_points']
    ci = readout['top1_gain_query_world_ci95_pp']
    prediction_change = 100 * (summary['prediction']['groups']['dr']['candidate_to_baseline_error_ratio'] - 1)
    directions = [within_change < 0, ratio_change < 0, gain > 0]
    if all(directions):
        verdict = '在本次 DR＋负载诊断集上，簇内紧凑性、相对分离程度和跨 motion 识别率的变化方向一致改善。'
    elif any(directions):
        verdict = '本次结果有收益和不足，DR 表征指标没有一致改善。'
    else:
        verdict = '本次 DR＋负载诊断未观察到簇内紧凑性、相对分离程度或跨 motion 识别率的改善。'
    text = (verdict + f' 历史不重叠的同 DR 距离变化 {within_change:+.2f}%，簇内／簇间比变化 {ratio_change:+.2f}%；'
            f'跨 motion 且历史不重叠的环境 Top-1 变化 {gain:+.2f} 个百分点（查询 world 配对 bootstrap 95% 区间 [{ci[0]:+.2f}, {ci[1]:+.2f}]）。'
            f'固定验证 DR 五步预测误差变化 {prediction_change:+.2f}%。')
    if summary.get('comparison_kind') == 'tuning':
        statements = []
        for profile, title in [('common', '普通 DR'), ('memory_training', 'DR＋负载')]:
            result = summary['readout'][profile]['disjoint']
            lo, hi = result['top1_gain_query_world_ci95_pp']
            statements.append(f'{title}跨 motion 且历史不重叠的环境 Top-1 变化 '
                              f'{result["top1_gain_percentage_points"]:+.2f} 个百分点'
                              f'（配对 world bootstrap 95% 区间 [{lo:+.2f}, {hi:+.2f}]）')
        text = '；'.join(statements) + f'。固定验证 DR 五步预测误差变化 {prediction_change:+.2f}%。'
    return {'primary_profile': 'memory_training', 'disjoint_within_distance_change_percent': within_change,
            'disjoint_within_over_between_change_percent': ratio_change, 'disjoint_readout_gain_pp': gain,
            'dr_prediction_error_change_percent': prediction_change,
            'all_three_representation_directions_improved': all(directions), 'text': text}


def report(summary, output):
    continuation = summary.get('comparison_kind') == 'continuation'
    tuning = summary.get('comparison_kind') == 'tuning'
    reference_update = summary.get('reference_update', 5000)
    update = summary['update']
    left = f'弱样本 u{reference_update}' if continuation or tuning else '旧扩容 u5000'
    right = f'调参 u{update}' if tuning else f'弱样本 u{update}'
    if tuning:
        description = [
            f'从已检查的原配置 u{reference_update} checkpoint 恢复 encoder、predictor、AdamW 与学习率状态。', '',
            '弱正、弱负权重各从 0.002 翻倍至 0.004；response scale 从 0.5 改为 0.3；margin 保持 1.0。',
            'nominal50、模型、采样、归一化和固定验证相同，学习率沿用原下限 1e-5。模拟器、replay 与弱样本档案重新预热。',
            '主指标是普通 DR 和 DR＋负载的跨 motion、历史不重叠环境识别 Top-1；测量三项调整和继续训练的联合结果。']
    elif continuation:
        description = [
            f'比较同一 nominal50-memory350 扩容弱样本实验的 {reference_update} 与 {update} 轮 checkpoint。', '',
            '继续训练保持所有损失、采样、模型、normalization 和固定验证窗口不变；弱正／负系数各 0.002，margin=1.0，response scale=0.5。',
            '模型、优化器和原 8000 轮 cosine 曲线从 u5000 恢复，学习率达到原下限 1e-5 后保持；模拟器与 replay 重新预热。']
    else:
        description = [
            '比较同为 5000 次训练更新、20000 次 optimizer step 的 nominal50-memory350 扩容模型。', '',
            '新版同时加入弱正/负样本（总系数各 0.002，负样本 margin=1.0），并将 response scale 从 0.75 改为 0.5。',
            '两版模型、种子、数据集、物理参数采样、学习率曲线、normalization 和固定验证窗口相同；本实验测量这组三项修改的联合效果。']
    rows = [
        summary['assessment']['text'], '',
        *description, '', f'| DR 表征指标 | {left} | {right} |', '|---|---:|---:|']
    for profile, title in (('common', '普通 DR'), ('memory_training', 'DR＋四肢负载／扰动')):
        models = summary['profiles'][profile]['memory_full']['models']
        a, b = models['baseline'], models['memory350']
        for label, key in (('同 DR、跨 motion', 'dr_same_world_cross_motion'),
                           ('同 DR、跨 motion、历史不重叠', 'dr_same_world_cross_motion_disjoint'),
                           ('不同 DR、同 motion／phase 相近', 'dr_different_world_same_motion_near_phase'),
                           ('不同 DR、不同 motion', 'dr_different_world_cross_motion')):
            rows.append(f'| {title}：{label} | {a["pairs"][key]["unit_distance_rms"]:.4f} | {b["pairs"][key]["unit_distance_rms"]:.4f} |')
        key = 'dr_same_world_cross_motion_over_between'
        rows.append(f'| {title}：簇内／簇间，越小越好 | {a[key]:.4f} | {b[key]:.4f} |')
        readout = summary['readout'][profile]['disjoint']
        rows.append(f'| {title}：跨 motion 且历史不重叠的环境 Top-1 | {100*readout["models"]["baseline"]["top1_accuracy"]:.2f}% | {100*readout["models"]["memory350"]["top1_accuracy"]:.2f}% |')
    rows += ['', f'| 固定验证第 5 步递归预测误差 | {left} | {right} |', '|---|---:|---:|']
    for key in ('nominal', 'dr'):
        group = summary['prediction']['groups'][key]
        rows.append(f'| {key} 五步 NMSE | {group["baseline_nmse"]:.6f} | {group["candidate_nmse"]:.6f} |')
    for key, title in (('dr_full350', 'DR，完整 350 步历史'),
                       ('dr_full_long_incomplete_short', 'DR，完整 long、short 不足 50'),
                       ('dr_partial_long', 'DR，1～29 个 long chunk'), ('dr_no_long', 'DR，无 long chunk')):
        group = summary['prediction'].get('history_groups', {}).get(key)
        if group:
            rows.append(f'| {title}，{group["samples"]} 个窗口 | {group["baseline_nmse"]:.6f} | {group["candidate_nmse"]:.6f} |')
    rows += ['', '所有几何比较使用相同的原始轨迹和相同样本配对。主结果要求完整 short50＋long30×10。',
             '环境识别是在已知物理环境上，用一部分 motion family 拟合中心，在另一部分 family 测试；另行排除历史重叠。',
             '这是单个训练种子的受控诊断集，不是完整数据集或下游 PPO 收益证明。t-SNE 独立拟合，图上全局距离不能跨模型直接比较。', '',
             '[普通 DR 交互图](common/plots/tsne_interactive.html) · [DR＋负载交互图](memory_training/plots/tsne_interactive.html) · [完整结果](summary.json)', '']
    (output / 'README.md').write_text('\n'.join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--comparison-kind', choices=('weakpairs', 'continuation', 'tuning'), default='weakpairs')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f'Refusing to overwrite comparison: {output}')
    output.mkdir(parents=True)
    started = time.monotonic()
    paths = {'baseline': args.reference.resolve(), 'memory350': args.candidate.resolve()}
    states = {name: torch.load(path, map_location='cpu', weights_only=False, mmap=True) for name, path in paths.items()}
    validate_checkpoints(states, args.comparison_kind)
    update = states['memory350']['update']
    reference_update = states['baseline']['update']
    labels = {'baseline': ('Weak pairs' if args.comparison_kind in ('continuation', 'tuning') else 'Baseline') + f' encoder2x / u{reference_update}',
              'memory350': ('Tuned' if args.comparison_kind == 'tuning' else 'Weak pairs') + f' encoder2x / u{update}'}
    summary = {'complete': False, 'update': update, 'reference_update': states['baseline']['update'],
               'comparison_kind': args.comparison_kind, 'model_labels': labels,
               'checkpoints': {name: {'path': str(path), 'sha256': sha256(path),
                                    'update': states[name]['update'], 'optimizer_steps': states[name]['optimizer_steps']}
                               for name, path in paths.items()},
               'same_model': True, 'same_normalization': True, 'same_raw_queries': True,
               'profiles': {}, 'readout': {}, 'verification': {}}
    summary['prediction'] = prediction_comparison(paths, states, output)
    encoders = {name: load_memory350_checkpoint(path, device='cuda:0') for name, path in paths.items()}
    calibration_path = Path(json.loads((args.cache / 'common/metadata.json').read_text())['checkpoint'])
    calibration = load_memory350_checkpoint(calibration_path, device='cuda:0')
    model_fields = {'baseline': 'latent_baseline', 'memory350': 'latent_weakpairs'}
    for profile in ('common', 'memory_training'):
        source, folder = args.cache / profile, output / profile
        folder.mkdir()
        metadata = json.loads((source / 'metadata.json').read_text())
        assert metadata['complete'] and metadata['arguments']['save_history']
        with np.load(source / 'latents.npz') as cache:
            data = {key: cache[key] for key in cache.files if not key.startswith('latent')}
            calibration_latent = cache['latent'].copy()
        data['source_row'] = np.arange(len(data['world']))
        parts = {name: [] for name in encoders}
        verifications, queries = [], []
        query_paths = sorted((source / 'queries').glob('query_*.pt'))
        assert len(query_paths) == 32
        for number, path in enumerate(query_paths):
            query = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            assert query['format_version'] == 'memory350_raw_query_v1'
            assert query['short'].dtype == query['long'].dtype == torch.float32
            mask = data['step'] == query['step']
            assert np.array_equal(query['world'].numpy(), data['world'][mask])
            assert np.array_equal(query['short_valid'].sum(-1).numpy(), data['short_steps'][mask])
            assert np.array_equal(query['long_valid'].sum(-1).numpy(), data['long_chunks'][mask])
            if number in (0, len(query_paths) - 1):
                reconstructed = encode(calibration, query)
                expected = calibration_latent[mask]
                rms = float(np.sqrt(np.square(normalized(reconstructed) - normalized(expected)).sum(1).mean()))
                assert rms < .001
                verifications.append({'step': query['step'], 'reconstruction_unit_rms': rms,
                                      'bitwise_equal': bool(np.array_equal(reconstructed, expected))})
            for name, encoder in encoders.items():
                value = encode(encoder, query)
                assert np.isfinite(value).all()
                if number == len(query_paths) - 1:
                    full_precision = encode(encoder, query, bf16=False)
                    cosine = float((normalized(value) * normalized(full_precision)).sum(1).mean())
                    assert cosine > .9999
                    verifications.append({'model': name, 'bf16_float32_mean_cosine': cosine})
                parts[name].append(value)
            queries.append({'path': str(path.resolve()), 'sha256': sha256(path), 'step': query['step'], 'samples': int(mask.sum())})
            if number % 8 == 0 or number == len(query_paths) - 1:
                print(json.dumps({'event': 'encoded', 'profile': profile, 'queries': number + 1}), flush=True)
        assert np.array_equal(np.concatenate([np.full(q['samples'], q['step']) for q in queries]), data['step'])
        data.update({model_fields[name]: np.concatenate(values) for name, values in parts.items()})
        np.savez_compressed(folder / 'latents.npz', **data)
        metadata = {
            'models': model_fields, 'checkpoint': str(paths['memory350']),
            'checkpoint_sha256': summary['checkpoints']['memory350']['sha256'],
            'comparison_checkpoints': {'baseline': summary['checkpoints']['baseline']},
            'model_config': states['memory350']['model_config'],
            'arguments': {'num_envs': metadata['arguments']['num_envs']},
            'source_collection_arguments': metadata['arguments'],
            'source_metadata': str(source / 'metadata.json'),
            'source_metadata_sha256': sha256(source / 'metadata.json'),
            'source_latents_sha256': sha256(source / 'latents.npz'),
            'motion_files': metadata['motion_files'], 'profile': profile,
            'queries': queries, 'primary_subset': 'memory_full', 'complete': True,
        }
        write_json(folder / 'metadata.json', metadata)
        analysis = folder / 'analysis'
        analysis.mkdir()
        summary['profiles'][profile] = {}
        for subset, mask in (
            ('memory_full', (data['short_steps'] == 50) & (data['long_chunks'] == 30)),
            ('long_full', data['long_chunks'] == 30)):
            selected = {key: value[mask] for key, value in data.items()}
            result = analyze_subset(selected, metadata, analysis, subset)
            summary['profiles'][profile][subset] = result
            if subset == 'memory_full':
                np.savez_compressed(analysis / 'memory_full_data.npz', **selected)
                dr = {key: value[~selected['nominal']] for key, value in selected.items()}
                summary['readout'][profile] = {name: environment_readout(dr, metadata, disjoint=flag)
                                               for name, flag in (('ordinary', False), ('disjoint', True))}
        summary['verification'][profile] = {'passed': True, 'checks': verifications, 'queries': len(queries)}
        write_json(analysis / 'cluster_metrics.json', summary['profiles'][profile])
        plot_env = dict(os.environ, PYTHONPATH='/tmp/intact-tsne-deps', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/plot_memory350_nominal_comparison.py'),
                        str(folder), '--subset', 'memory_full', '--model', 'baseline=' + labels['baseline'],
                        '--model', 'memory350=' + labels['memory350']], cwd=ROOT, env=plot_env, check=True)
        write_json(output / 'summary.json', summary)
    summary.update(complete=True, elapsed_seconds=time.monotonic() - started,
                   script_sha256=sha256(Path(__file__)))
    summary['assessment'] = assess(summary)
    report(summary, output)
    write_json(output / 'summary.json', summary)
    print(json.dumps({'event': 'complete', 'report': str(output / 'README.md')}), flush=True)


if __name__ == '__main__':
    main()
