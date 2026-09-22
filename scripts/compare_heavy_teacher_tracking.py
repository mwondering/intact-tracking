"""Audit paired heavy tracking for learned/zero latent, RMA and Any2Track."""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from compare_native_tracking import audit_query_forces


ROLES = ('rma_teacher', 'latent', 'vanilla')
ROLE_LABELS = {'rma_teacher': 'RMA teacher', 'latent': 'Latent',
               'vanilla': 'Vanilla', 'any2track': 'Any2Track', 'rma_student': 'RMA student'}
METRICS = {
    'error_joint_pos': '关节角 L2（rad）',
    'error_body_pos': '局部 body 位置（m）',
    'error_body_rot': 'body 姿态（rad）',
    'error_anchor_pos_global': '全局 root 位置（m）',
    'error_body_pos_global': '全局 body 位置（m）',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected_roles(protocol):
    roles = tuple(protocol.get('roles', ROLES))
    if (len(roles) != len(set(roles)) or not set(ROLES).issubset(roles)
            or not set(roles).issubset(ROLE_LABELS)):
        raise ValueError('Expected the three residual methods, optionally with Any2Track or RMA student')
    return roles


def compare(directory, physics, protocol, resamples, seed):
    roles = selected_roles(protocol)
    rows = {role: json.loads((directory / f'{role}_{physics}.json').read_text())
            for role in roles}
    reference = rows['rma_teacher']
    paired = (
        'seed', 'motion_files', 'motion_ids', 'start_frames', 'horizons',
        'metric_names', 'max_steps', 'physics_world_fingerprints', 'is_nominal',
        'reward_contract', 'query_initial_state_sha256', 'global_metric_contract',
        'policy_precision', 'physics', 'added_limb_payload_kg',
        'payload_com_offsets_m', 'failure_term_names',
    )
    checks = {key: all(row[key] == reference[key] for row in rows.values())
              for key in paired}
    if not all(checks.values()):
        raise ValueError(f'{physics}: unmatched fields {checks}')
    manifest = Path(protocol['manifest']).read_text().splitlines()
    for role, row in rows.items():
        selected = protocol['checkpoints'][role]
        if (row['checkpoint_sha256'] != selected['sha256']
                or row['completed_training_updates'] != selected['completed_updates']
                or row['context_sha256'] != selected['context_sha256']
                or row['motion_files'] != manifest
                or row['seed'] != protocol['physics_seed']
                or row['max_steps'] != protocol['query_steps']
                or row['warmup']['steps'] != protocol['warmup_steps']
                or row['warmup']['policy_sha256'] != selected['tracker_sha256']
                or row['warmup']['policy'] != 'frozen tracker'
                or row['episodes'] != protocol['worlds_per_arm']
                or row['repeats_per_motion'] != protocol['repeats']
                or row['arguments']['physics'] != physics
                or row['memory_start'] != 'warm'
                or row['policy_precision'] != 'fp32'
                or not row['arguments']['paired_starts']
                or row['latent_intervention'] != 'correct'
                or row['arguments']['frozen_tracker_only']):
            raise ValueError(f'{role}/{physics}: evaluation differs from selected protocol')
        if row['warmup']['seed'] != reference['warmup']['seed']:
            raise ValueError('Warmup seeds differ')
        for key in ('reference_timeline_audited', 'partial_reset_survivor_state_audited',
                    'partial_reset_survivor_history_audited'):
            if not row[key]:
                raise ValueError(f'Missing runtime audit: {key}')
        if not row['evaluation_diagnostics']['cached_proprio_preserved_after_partial_reset']:
            raise ValueError('Sensor cache was not preserved')
        if role in ('latent', 'vanilla'):
            if row['initial_context']['long_full_fraction'] != 1.:
                raise ValueError('Tracker warmup did not fill every long history')
            if bool(row['initial_context']['latent_input_is_zero']) != (role == 'vanilla'):
                raise ValueError(f'{role} uses the wrong latent input mode')
        elif role == 'any2track':
            if row['initial_context']['Adapter/history_valid_fraction'] != 0.:
                raise ValueError('Any2Track must clear episode history at the query reset')
            if not 0. <= row['final_context']['Adapter/history_valid_fraction'] <= 1.:
                raise ValueError('Invalid Any2Track history diagnostics')
        elif role == 'rma_student':
            if abs(row['initial_context']['Distill/history_valid_fraction']-.02)>1e-6:
                raise ValueError('RMA student query must begin with only the current sensor frame')
            if not 0. <= row['final_context']['Distill/history_valid_fraction'] <= 1.:
                raise ValueError('Invalid RMA student history diagnostics')

    nominal = np.asarray(reference['is_nominal'], bool)
    if (physics == 'nominal' and not nominal.all()) or (
            physics == 'mixed' and not 0.09 <= nominal.mean() <= 0.11):
        raise ValueError('Incorrect evaluation physics population')
    horizons = np.asarray(reference['horizons'])
    lengths = {role: np.asarray(row['episode_lengths']) for role, row in rows.items()}
    common = np.minimum.reduce(list(lengths.values()))
    if (common < 1).any() or any((length > horizons).any() for length in lengths.values()):
        raise ValueError('Invalid episode lengths')
    force_fields = ('query_force_steps', 'query_force_sha256', 'query_force_prefix_sha256')
    forces = {role: {key: row['evaluation_diagnostics'][key] for key in force_fields}
              for role, row in rows.items()}
    force_audits = {}
    for left, right in combinations(roles, 2):
        steps, same_length = audit_query_forces(forces[left], forces[right], int(common.max()))
        force_audits[f'{left}_vs_{right}'] = {'common_steps': steps, 'same_length': same_length}
    checks['common_query_force_sequence'] = True

    means, successes = {}, {}
    for role, row in rows.items():
        with np.load(directory / f'{role}_{physics}.traces.npz') as archive:
            np.testing.assert_array_equal(archive['metric_names'], row['metric_names'])
            np.testing.assert_array_equal(archive['lengths'], lengths[role])
            values = archive['all_metrics'].astype(np.float64)
        own_mask = np.arange(values.shape[1])[None, :] < lengths[role][:, None]
        own_means = np.where(own_mask[..., None], values, 0).sum(1) / lengths[role][:, None]
        np.testing.assert_allclose(own_means, row['per_episode_metrics'], rtol=1e-10, atol=1e-12)
        mask = np.arange(values.shape[1])[None, :] < common[:, None]
        means[role] = np.where(mask[..., None], values, 0).sum(1) / common[:, None]
        if not np.isfinite(means[role]).all():
            raise ValueError('Nonfinite scored metrics')
        successes[role] = 1 - np.asarray(row['failed'], np.float64)
        if np.any((lengths[role] < horizons) & (successes[role] == 1)):
            raise ValueError('Incomplete query incorrectly marked successful')

    motion_ids = np.asarray(reference['motion_ids'])
    groups = {}
    masks = {'all': np.ones(len(nominal), bool)}
    if physics == 'mixed':
        masks.update(hdr_only=~nominal, nominal_only=nominal)
    pairs = (('rma_teacher', 'vanilla'), ('latent', 'vanilla'), ('latent', 'rma_teacher'))
    if 'any2track' in roles:
        pairs += (('any2track', 'vanilla'), ('latent', 'any2track'), ('rma_teacher', 'any2track'))
    if 'rma_student' in roles:
        pairs += tuple(('rma_student',other) for other in roles if other != 'rma_student')
    for name, mask in masks.items():
        ids = np.unique(motion_ids[mask])
        counts = np.asarray([(motion_ids[mask] == i).sum() for i in ids])
        sums = {role: np.asarray([means[role][mask][motion_ids[mask] == i].sum(0) for i in ids])
                for role in roles}
        success_sums = {role: np.asarray([successes[role][mask][motion_ids[mask] == i].sum()
                                        for i in ids]) for role in roles}
        draws_by_pair = {pair: [] for pair in pairs}
        success_draws = {pair: [] for pair in pairs}
        rng = np.random.default_rng(seed)
        for offset in range(0, resamples, 256):
            draws = rng.integers(0, len(ids), size=(min(256, resamples-offset), len(ids)))
            denominator = counts[draws].sum(1)
            samples = {role: sums[role][draws].sum(1)/denominator[:, None] for role in roles}
            for candidate, baseline in pairs:
                pair = (candidate, baseline)
                draws_by_pair[pair].append(100 * (samples[baseline]-samples[candidate]) / samples[baseline])
                success_draws[pair].append(100 * (success_sums[candidate][draws].sum(1)
                    - success_sums[baseline][draws].sum(1)) / denominator)
        comparisons = {}
        for candidate, baseline in pairs:
            pair = (candidate, baseline)
            ci = np.quantile(np.concatenate(draws_by_pair[pair]), [.025, .975], axis=0)
            comparisons[f'{candidate}_vs_{baseline}'] = {
                'candidate': candidate, 'reference': baseline,
                'error_reduction_percent': {metric: float(100 * (
                    1-means[candidate][mask, i].mean()/means[baseline][mask, i].mean()))
                    for i, metric in enumerate(reference['metric_names'])},
                'error_reduction_percent_ci95': {metric: ci[:, i].tolist()
                    for i, metric in enumerate(reference['metric_names'])},
                'success_difference_percentage_points': float(100 * (
                    successes[candidate][mask].mean()-successes[baseline][mask].mean())),
                'success_difference_percentage_points_ci95': np.quantile(
                    np.concatenate(success_draws[pair]), [.025, .975]).tolist(),
            }
        groups[name] = {
            'episodes': int(mask.sum()), 'motions': len(ids),
            'common_prefix_coverage': float((common[mask]/horizons[mask]).mean()),
            'arms': {role: {
                'failures': int((1-successes[role][mask]).sum()),
                'successes': int(successes[role][mask].sum()),
                'success_rate': float(successes[role][mask].mean()),
                'coverage_fraction': float((lengths[role][mask]/horizons[mask]).mean()),
                'metrics': dict(zip(reference['metric_names'], means[role][mask].mean(0).tolist(), strict=True)),
                'own_episode_metrics': dict(zip(reference['metric_names'], np.asarray(
                    rows[role]['per_episode_metrics'])[mask].mean(0).tolist(), strict=True)),
            } for role in roles},
            'comparisons': comparisons,
        }
    return {
        'physics': physics, 'pairing_checks': checks, 'force_audits': force_audits,
        'nominal_worlds': int(nominal.sum()), 'hdr_worlds': int((~nominal).sum()),
        'groups': groups,
        'warmup_trajectory_bitwise_equal': len({row['warmup']['trajectory_sample_sha256']
                                              for row in rows.values()}) == 1,
        'warmup_note': 'Same frozen tracker, seeds and number of steps. Independent simulations need not be bitwise identical. Query state, physical parameters and force sequences are audited exactly. Learned latent preserves long physics memory; Any2Track, when included, clears episode history at the query reset as in training.',
        'initial_history_diagnostics': {role: row['initial_context'] for role, row in rows.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260921)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error('Bootstrap samples must be positive')
    directory = args.directory.resolve()
    protocol = json.loads((directory/'protocol.json').read_text())
    if digest(protocol['manifest']) != protocol['manifest_sha256']:
        raise ValueError('Selected manifest changed')
    selected = protocol['checkpoints']
    roles = selected_roles(protocol)
    mode = protocol.get('comparison_mode', 'matched_updates')
    if mode not in ('matched_updates', 'latest', 'distillation'):
        raise ValueError(f'Unsupported comparison mode: {mode}')
    if set(selected) != set(roles):
        raise ValueError('Selected methods differ from the declared protocol roles')
    if mode == 'matched_updates' and len({x['completed_updates'] for x in selected.values()}) != 1:
        raise ValueError('All methods must use the same update count')
    if len({x['tracker_sha256'] for x in selected.values()}) != 1:
        raise ValueError('Frozen tracker identities differ')
    if selected['latent']['context_sha256'] != selected['vanilla']['context_sha256']:
        raise ValueError('Residual context checkpoints differ')
    for row in selected.values():
        if digest(row['snapshot']) != row['sha256']:
            raise ValueError('Selected checkpoint snapshot changed')
    scenes = {physics: compare(directory, physics, protocol, args.bootstrap_samples, args.seed)
              for physics in protocol['physics']}
    sources = [directory/'protocol.json', directory/'evaluation_source_sha256.json',
               Path(protocol['manifest']), Path(__file__)]
    capture_path=directory/'evaluation_source_capture.json'
    capture=json.loads(capture_path.read_text()) if capture_path.exists() else None
    if capture is not None:sources.append(capture_path)
    sources.extend(directory/f'{role}_{physics}{suffix}' for role in roles
                   for physics in protocol['physics'] for suffix in ('.json', '.traces.npz'))
    result = {
        'protocol': protocol, 'scenes': scenes,
        'aggregation': f'Per-world mean over the common valid trajectory prefix across all {len(roles)} methods, then equal world weights. Failure step is included; padding excluded. Success and coverage use each full query.',
        'bootstrap': {'resamples': args.bootstrap_samples, 'seed': args.seed,
                      'unit': 'motion cluster, keeping repeats together',
                      'scope': 'Conditional on this checkpoint selection and sample; not training-seed uncertainty'},
        'source_sha256': {str(path): digest(path) for path in sources},
        'evaluation_source_capture': capture,
        'limitations': [protocol['scope'], protocol['training_history_caveat'],
                        protocol.get('history_contract', 'Warm-history evaluation after 500 frozen-tracker steps; not a cold-start test.'),
                        ({'latest':'Latest saved checkpoints have different total PPO updates.',
                          'matched_updates':'Same total PPO updates.',
                          'distillation':'Student updates are supervised distillation; teacher is the fixed source, other arms retain their own PPO training histories.'}[mode]),
                        'Pretrained context and auxiliary supervision differ by design.'],
    }
    (directory/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    checkpoint_labels = ', '.join(f'{role}: checkpoint_{selected[role]["filename_iteration"]}.pt '
                                  f'(completed_updates={selected[role]["completed_updates"]})'
                                  for role in roles)
    title = {'latest':'各自最新 checkpoint','matched_updates':'相同 PPO 更新次数',
             'distillation':'RMA student 与固定 teacher 的蒸馏对照'}[mode]
    labels = '、'.join(ROLE_LABELS[role] for role in roles)
    lines = [f'# {labels}：{title}的 tracking 评测', '',
             f'{checkpoint_labels}。', '',
             f'{protocol["motion_count"]} 条随机 motion × {protocol["repeats"]} 个环境；冻结 tracker 预热 {protocol["warmup_steps"]} 步，正式评测最多 {protocol["query_steps"]} 步。',
             f'误差在全部 {len(roles)} 组共同有效轨迹前缀上逐环境平均；成功率按各自完整轨迹计算。', '',
             protocol.get('training_history_caveat_zh', protocol['training_history_caveat']), '']
    if protocol.get('history_contract_zh'):
        lines.extend([protocol['history_contract_zh'], ''])
    if capture and capture['phase']=='after_existing_results':
        lines.extend(['代码指纹在现有评测结果生成后补记；不是评测启动前的代码快照。原始结果与配对审计保留。', ''])
    for physics, scene in scenes.items():
        group = scene['groups']['all']
        arms = group['arms']
        lines.extend([f'## {physics}：{scene["nominal_worlds"]} nominal + {scene["hdr_worlds"]} HDR', '',
                      '| 指标 | '+' | '.join(ROLE_LABELS[role] for role in roles)+' |',
                      '|---|'+'---:|'*len(roles)])
        lines.append('| 成功率 ↑ | '+' | '.join(f'{arms[role]["success_rate"]:.2%} ({arms[role]["successes"]}/{group["episodes"]})' for role in roles)+' |')
        for metric, label in METRICS.items():
            lines.append(f'| {label} ↓ | '+' | '.join(f'{arms[role]["metrics"][metric]:.5f}' for role in roles)+' |')
        lines.append('| 轨迹覆盖率 ↑ | '+' | '.join(f'{arms[role]["coverage_fraction"]:.2%}' for role in roles)+' |')
        lines.extend(['', f'全部 {len(roles)} 组配对检查通过；共同有效轨迹覆盖率 {group["common_prefix_coverage"]:.2%}。', ''])
    lines.extend(['这是可行训练 motion 的抽样测试，不是完整数据集或未见 motion 泛化评测。',
                  'summary.json 保留混合环境子集、按 motion 分组的 95% bootstrap 区间和完整审计。', ''])
    (directory/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
