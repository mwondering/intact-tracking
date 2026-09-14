"""Summarize fixed-trajectory checks at u5000, u6000, ..., u10000."""

import argparse
import json
from pathlib import Path


def row_from_summary(summary):
    row = {'update': summary['update']}
    for profile, prefix in [('common', 'common'), ('memory_training', 'native')]:
        model = summary['profiles'][profile]['memory_full']['models']['memory350']
        for name, pair in [('within', 'dr_same_world_cross_motion'),
                           ('within_disjoint', 'dr_same_world_cross_motion_disjoint'),
                           ('between', 'dr_different_world_same_motion_near_phase')]:
            row[prefix + '_' + name] = model['pairs'][pair]['unit_distance_rms']
        row[prefix + '_ratio'] = model['dr_same_world_cross_motion_over_between']
        row[prefix + '_top1'] = summary['readout'][profile]['disjoint']['models']['memory350']['top1_accuracy']
    prediction = summary['prediction']
    row['dr_nmse'] = prediction['groups']['dr']['candidate_nmse']
    row['nominal_nmse'] = prediction['groups']['nominal']['candidate_nmse']
    for group in ('dr_full350', 'dr_full_long_incomplete_short', 'dr_partial_long', 'dr_no_long'):
        row[group + '_nmse'] = prediction.get('history_groups', {}).get(group, {}).get('candidate_nmse')
    return row


def summarize(root):
    journal = root / 'continuation_005000_010000'
    baseline = root / 'comparison_005000'
    rows = [row_from_summary(json.loads((baseline / 'summary.json').read_text()))]
    audit = baseline / 'prediction_context_audit.json'
    if audit.exists():
        groups = json.loads(audit.read_text())['groups']
        for group in ('dr_full350', 'dr_full_long_incomplete_short', 'dr_partial_long', 'dr_no_long'):
            key = 'dr_partial_long_1_to_29' if group == 'dr_partial_long' else group
            rows[0][group + '_nmse'] = groups[key]['candidate_nmse_by_horizon'][-1]
    for update in range(6000, 10001, 1000):
        folder = journal / f'comparison_{update:06d}'
        if not (folder / 'summary.json').exists():
            continue
        summary = json.loads((folder / 'summary.json').read_text())
        if not summary.get('complete'):
            continue
        rows.append(row_from_summary(summary))
        for group, values in summary['prediction'].get('history_groups', {}).items():
            rows[0][group + '_nmse'] = values['baseline_nmse']
    result = {'checks': rows, 'reference_update': 5000,
              'scheduled_updates': list(range(6000, 10001, 1000)),
              'protocol': 'Identical cached raw histories and fixed validation, unchanged weak-pair model and losses.',
              'convergence_note': 'Inspect successive fixed-query changes; lower training loss or one improved checkpoint alone does not establish convergence.'}
    temp = journal / 'trend.json.tmp'
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temp.replace(journal / 'trend.json')
    lines = [
        '从弱样本版 u5000 继续训练至 u10000；每增加 1000 轮，在相同轨迹和固定验证窗口上检查一次。', '',
        '弱正／负权重各 0.002，negative margin=1.0，response scale=0.5；nominal50、模型、数据与采样设置不变。',
        '恢复模型、优化器和原 8000 轮 cosine 曲线，达到原学习率下限 1e-5 后保持。模拟器、replay 和弱样本档案重新预热。', '',
        '距离为完整 350 步历史下，归一化 64 维 latent 的成对 L2 距离的 RMS。不同 DR 的距离匹配同 motion、phase 差≤0.02。',
        '环境识别用其他 motion family 建立已知环境中心；以下 Top-1 严格排除了原始历史重叠。', '',
        '| 总轮数 | DR＋负载簇内 ↓ | 历史不重叠簇内 ↓ | 簇间 ↑ | 簇内／簇间 ↓ | 跨 motion Top-1 ↑ |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["update"]} | {row["native_within"]:.4f} | {row["native_within_disjoint"]:.4f} | {row["native_between"]:.4f} | {row["native_ratio"]:.4f} | {100 * row["native_top1"]:.2f}% |')
    lines += ['', '| 总轮数 | 普通 DR 簇内 ↓ | 簇间 ↑ | 簇内／簇间 ↓ | 跨 motion Top-1 ↑ |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["update"]} | {row["common_within"]:.4f} | {row["common_between"]:.4f} | {row["common_ratio"]:.4f} | {100 * row["common_top1"]:.2f}% |')
    lines += ['', '预测误差为固定窗口上第 5 步递归预测的 pooled NMSE，越低越好。', '',
              '| 总轮数 | 全部 DR | 完整 350 步 | long30、short 不足 | long 1～29 | 无 long | nominal |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row[key] for key in ('dr_nmse', 'dr_full350_nmse', 'dr_full_long_incomplete_short_nmse',
                                      'dr_partial_long_nmse', 'dr_no_long_nmse', 'nominal_nmse')]
        lines.append('| ' + str(row['update']) + ' | ' + ' | '.join('—' if value is None else f'{value:.5f}' for value in values) + ' |')
    lines += ['', '判断收敛时同时看后续相邻检查的簇内／簇间比、跨 motion 识别率和分历史长度的预测误差；单个 checkpoint 改善不能证明已经收敛。', '',
              '[趋势图](trend.png) · [机器可读趋势](trend.json) · [u5000 完整复核](../comparison_005000/review.md)', '']
    for row in rows[1:]:
        lines.append(f'- [u{row["update"]} 对 u5000 的评估、t-SNE 和距离热图](comparison_{row["update"]:06d}/README.md)')
    (journal / 'README.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    updates = [row['update'] for row in rows]
    panels = [
        ('Unit latent distance / DR + loads', [('native_within_disjoint', 'Within, disjoint'), ('native_between', 'Between, matched phase')]),
        ('Within / between (lower is better)', [('native_ratio', 'DR + loads'), ('common_ratio', 'Common DR')]),
        ('Cross-motion, disjoint environment Top-1', [('native_top1', 'DR + loads'), ('common_top1', 'Common DR')]),
        ('Final-step pooled prediction NMSE', [('dr_nmse', 'All DR'), ('nominal_nmse', 'Nominal')]),
        ('DR NMSE / complete or partial memory', [('dr_full350_nmse', 'Full 350'), ('dr_full_long_incomplete_short_nmse', 'Full long, partial short'), ('dr_partial_long_nmse', 'Long 1-29')]),
        ('DR NMSE / no long memory', [('dr_no_long_nmse', 'No long memory')]),
    ]
    for ax, (title, curves) in zip(axes.flat, panels):
        for key, label in curves:
            ax.plot(updates, [row[key] for row in rows], marker='o', label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Total training updates')
        ax.set_xticks(list(range(5000, 10001, 1000)))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(journal / 'trend.png', dpi=160)
    fig.savefig(journal / 'trend.pdf')
    plt.close(fig)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_root.resolve())))
