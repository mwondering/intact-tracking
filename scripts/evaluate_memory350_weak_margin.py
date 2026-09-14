"""Evaluate u11000/u12000 against u10000 using the unchanged frozen diagnostic protocol."""

from pathlib import Path

import evaluate_memory350_weak_pairs as base
from intact_tracking.memory350_weak_margin_tuning import (
    MILESTONES, PARENT_UPDATE, TARGET_TOP1, validate_tuning_losses,
)


_base_report = base.report


def validate_checkpoints(states, comparison_kind):
    if comparison_kind != 'tuning':
        raise ValueError('The margin evaluator is only for the explicit tuning stage')
    a, b = states['baseline'], states['memory350']
    for field in ('model_config', 'normalization', 'tracker', 'nominal_a_fraction'):
        if a[field] != b[field]:
            raise ValueError(f'Checkpoint comparison changed {field}')
    if b['nominal_a_fraction'] != .5:
        raise ValueError('Margin tuning requires nominal50 checkpoints')
    if a['update'] != PARENT_UPDATE or b['update'] not in MILESTONES:
        raise ValueError('Margin tuning compares u10000 against u11000 or u12000')
    validate_tuning_losses(a['loss_config'], b['loss_config'])
    for state in states.values():
        if state['optimizer_steps'] != 4 * state['update']:
            raise ValueError('Margin tuning changed the optimizer-step budget')
        if state['scheduler']['last_epoch'] != state['optimizer_steps']:
            raise ValueError('Margin tuning reset the restored scheduler')
        if state['scheduler']['T_max'] != 32000 or state['scheduler']['continuation_min_learning_rate'] != 1e-5:
            raise ValueError('Margin tuning changed the original cosine horizon or learning-rate floor')
        if any(group['lr'] != 1e-5 for group in state['optimizer']['param_groups']):
            raise ValueError('Margin tuning must retain the original learning-rate floor')
    for key in ('T_max', 'continuation_min_learning_rate', 'base_lrs', 'eta_min'):
        if a['scheduler'][key] != b['scheduler'][key]:
            raise ValueError(f'Margin tuning changed scheduler {key}')


def report(summary, output):
    summary['tuning_stage_kind'] = 'weak_weights_008_margin_11_scale_03'
    summary['evaluator_entry_sha256'] = base.sha256(Path(__file__))
    readout = summary['readout']['memory_training']['disjoint']
    score = readout['models']['memory350']['top1_accuracy']
    summary['target'] = {'profile': 'memory_training', 'metric': 'disjoint_cross_motion_top1',
                         'threshold': TARGET_TOP1, 'candidate_top1': score, 'met_on_fixed_diagnostic': score >= TARGET_TOP1,
                         'worlds': readout['worlds'], 'test_samples': readout['test_samples']}
    _base_report(summary, output)
    path = output / 'README.md'
    text = path.read_text()
    old = '弱正、弱负权重各从 0.002 翻倍至 0.004；response scale 从 0.5 改为 0.3；margin 保持 1.0。'
    assert text.count(old) == 1
    text = text.replace(old, '弱正、弱负权重各从 0.004 翻倍至 0.008；负样本 margin 从 1.0 改为 1.1；response scale 保持 0.3。')
    target_text = (f'本阶段 DR＋负载跨 motion Top-1 目标为 {100 * TARGET_TOP1:.0f}%；'
                   f'当前 {100 * score:.2f}%，' + ('已达到' if score >= TARGET_TOP1 else '尚未达到')
                   + '固定诊断集目标。普通 DR 与冷启动预测误差同时报告。\n\n')
    path.write_text(target_text + text)


def main():
    base.validate_checkpoints = validate_checkpoints
    base.report = report
    base.main()


if __name__ == '__main__':
    main()
