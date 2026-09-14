"""Report the u8000 baseline and two cross-motion DR identification checks."""

import argparse
import json
from pathlib import Path

from report_memory350_weak_pairs_continuation import row_from_summary


def summarize(root):
    plan = json.loads((root / 'launch_contract.json').read_text())
    baseline = json.loads(Path(plan['parent_evaluation_summary']).read_text())
    rows = [row_from_summary(baseline)]
    for update in plan['evaluation_updates']:
        path = root / f'comparison_{update:06d}/summary.json'
        if path.exists():
            summary = json.loads(path.read_text())
            if summary.get('complete'):
                row = row_from_summary(summary)
                row['gains_vs_parent'] = {
                    profile: {key: summary['readout'][profile]['disjoint'][key] for key in
                              ('top1_gain_percentage_points', 'top1_gain_query_world_ci95_pp')}
                    for profile in ('common', 'memory_training')}
                rows.append(row)
    result = {'checks': rows, 'reference_update': 8000, 'scheduled_updates': plan['evaluation_updates'],
              'primary_metric': plan['primary_metric'], 'loss_changes': plan['loss_changes'],
              'learning_rate_schedule_unchanged': True}
    tmp = root / 'trend.json.tmp'
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    tmp.replace(root / 'trend.json')
    lines = [
        '从完成检查的原配置 u8000 续训至 u10000；每新增 1000 轮检查跨 motion 的 DR 环境识别。', '',
        '弱正样本权重 0.002 → 0.004；弱负样本权重 0.002 → 0.004；RMS 映射参数 0.5 → 0.3。',
        '恢复 encoder、predictor、AdamW 和原 scheduler，学习率保持原下限 1e-5；nominal50、扩容 memory350、数据和采样保持。',
        '模拟器、replay 和弱样本档案重新预热 500 控制步。新 run 保留父 checkpoint 的 SHA 和独立 W&B 记录。', '',
        '主指标：用一组 motion family 建立已知 DR 环境中心，在另一组 family 上测试 Top-1，排除原始历史重叠。',
        '普通 DR 和 DR＋四肢负载／扰动分别报告；固定 42 条诊断 motion，不代表全部训练 motion 或下游 PPO 性能。', '',
        '| 总轮数 | 新增轮数 | 普通 DR 跨 motion Top-1 | DR＋负载跨 motion Top-1 |',
        '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["update"]} | {row["update"] - 8000} | {100 * row["common_top1"]:.2f}% | {100 * row["native_top1"]:.2f}% |')
    lines += ['', '| 总轮数 | 普通 DR 簇内 | DR＋负载簇内 | DR 完整350步 NMSE | DR 无长期记忆 NMSE |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["update"]} | {row["common_within"]:.4f} | {row["native_within"]:.4f} | '
                     f'{row["dr_full350_nmse"]:.5f} | {row["dr_no_long_nmse"]:.5f} |')
    lines += ['', '这轮测量三项修改与继续训练的联合效果，不能拆分各项参数的独立贡献。', '',
              f'[父模型 u8000 检查]({Path(plan["parent_evaluation_summary"]).with_name("README.md")}) · [趋势数据](trend.json) · [启动配置](launch_contract.json)']
    for row in rows[1:]:
        lines.append(f'- [u{row["update"]} 完整检查、置信区间和 t-SNE](comparison_{row["update"]:06d}/README.md)')
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, title, curves in zip(axes, [
        'Cross-motion, disjoint environment Top-1', 'Same DR, different motion distance', 'DR prediction NMSE'], [
        [('common_top1', 'Common DR'), ('native_top1', 'DR + loads')],
        [('common_within', 'Common DR'), ('native_within', 'DR + loads')],
        [('dr_full350_nmse', 'Full 350'), ('dr_no_long_nmse', 'No long memory')],
    ]):
        for key, label in curves:
            ax.plot([row['update'] for row in rows], [row[key] for row in rows], marker='o', label=label)
        ax.set(title=title, xlabel='Total updates', xticks=[8000, 9000, 10000])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(root / 'trend.png', dpi=160)
    fig.savefig(root / 'trend.pdf')
    plt.close(fig)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    print(json.dumps(summarize(parser.parse_args().run_root.resolve())))
