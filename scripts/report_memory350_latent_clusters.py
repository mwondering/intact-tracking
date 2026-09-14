"""Build the readable paired-checkpoint latent report from completed measurements."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    results = {name: json.loads((root / name / 'analysis/cluster_metrics.json').read_text())
               for name in ['common', 'memory_training']}
    verification = json.loads((root / 'inference_verification.json').read_text())
    common_models = results['common']['subsets']['common_full']['models']
    native_models = results['memory_training']['subsets']['common_full']['models']
    lines = ['Memory350 原版 update 22700 与 v12 update 8000 的 latent 实测。', '',
             '两个冻结 encoder 读取同一次 rollout 的交互，各自使用训练时的归一化和历史结构。'
             '每个 profile 运行 3,200 控制步，保持物理参数跨 reset 不变。'
             '共同 motion 集为此前使用的 42 个 LAFAN/Qingtong 文件，是 memory350 全训练目录的一个子集。', '',
             f'主要发现：共同基准中的同 DR 跨 motion 单位化距离，v12 为 '
             f'{common_models["v12"]["pairs"]["dr_same_world_cross_motion"]["unit_distance_rms"]:.3f}，'
             f'Memory350 为 {common_models["memory350"]["pairs"]["dr_same_world_cross_motion"]["unit_distance_rms"]:.3f}。'
             f'带负载组中 Memory350 的簇内/簇间距离比为 '
             f'{native_models["memory350"]["dr_same_world_cross_motion_over_between"]:.3f}，同环境跨 motion 波动较大。'
             '排除重复使用历史交互后，这个趋势仍在。', '',
             f'Memory350 仍有可读出的环境信息：带负载组的跨 motion family 环境中心识别 Top1 为 '
             f'{native_models["memory350"]["dr_world_center_transfer"]["top1_accuracy"]:.1%}，'
             f'v12 为 {native_models["v12"]["dr_world_center_transfer"]["top1_accuracy"]:.1%}。'
             '不同读出指标并不都给出相同排序，不能只根据簇内/簇间距离比判断表征或下游控制整体优劣。', '',
             '下面距离均为 RMS 欧氏距离，即先对样本对的距离平方求均值，再开根号。'
             '单位化表示逐个 latent 除以自身 L2 范数。主要比较只选 v12 完整 100 步、'
             'memory350 完整短期 50 步和长期 30×10 步的同一批 query。', '',
             '“同 motion、相近 phase”指 motion 文件相同、归一化进度差不超过 0.02。'
             '“同一帧”则要求 motion 文件和整数 motion_step 完全相同。'
             '这些条件没有强制实际状态、动作或交互历史相同。', '']
    title = {'common': '共同基准：512 nominal + 512 原始 DR，无额外负载',
             'memory_training': '新版训练物理配置：512 原始 DR + 独立四肢 U(0,4 kg) 负载'}
    selected = [
        ('nominal_cross_motion', 'Nominal，不同 motion'),
        ('dr_same_world_cross_motion', '同一 DR，不同 motion'),
        ('dr_same_world_cross_motion_disjoint', '同一 DR，不同 motion，历史不重叠'),
        ('dr_different_world_same_motion_near_phase', '不同 DR，同 motion、phase 差≤0.02'),
        ('dr_different_world_exact_motion_frame', '不同 DR，同 motion、同一帧'),
        ('nominal_dr_same_motion_near_phase', 'Nominal–DR，同 motion、phase 差≤0.02'),
        ('nominal_dr_exact_motion_frame', 'Nominal–DR，同 motion、同一帧'),
        ('nominal_nominal_exact_motion_frame', 'Nominal–nominal，同 motion、同一帧'),
    ]
    summary = {'checkpoint': verification['checkpoint'], 'checkpoint_sha256': verification['checkpoint_sha256'],
               'profiles': {}, 'verification': verification}
    for profile, result in results.items():
        primary = result['subsets']['common_full']
        lines.extend([f'**{title[profile]}**', '',
                      f'完整共同 query：{primary["n"]:,}；nominal {primary["nominal_samples"]:,}，DR {primary["dr_samples"]:,}。', '',
                      '| 比较条件 | 配对数 | v12 单位化 | Memory350 单位化 | v12 原始 | Memory350 原始 |',
                      '| --- | ---: | ---: | ---: | ---: | ---: |'])
        old, new = primary['models']['v12'], primary['models']['memory350']
        for key, label in selected:
            a, b = old['pairs'].get(key, {}), new['pairs'].get(key, {})
            if not b.get('n'):
                continue
            lines.append(f'| {label} | {b["n"]:,} | {a["unit_distance_rms"]:.4f} | {b["unit_distance_rms"]:.4f} | {a["raw_distance_rms"]:.4f} | {b["raw_distance_rms"]:.4f} |')
        ratio = 'dr_same_world_cross_motion_over_between'
        disjoint_ratio = 'dr_same_world_cross_motion_disjoint_over_between'
        lines.extend(['', f'同环境跨 motion 距离 / 不同环境匹配 motion、phase 距离：v12 **{old[ratio]:.3f}**，Memory350 **{new[ratio]:.3f}**。'
                      f'只使用历史不重叠的同环境样本对时，该比例分别为 **{old[disjoint_ratio]:.3f}** 和 **{new[disjoint_ratio]:.3f}**。较低比例表示相对环境间间隔，同环境波动更小。', '',
                      '| 固定同一 checkpoint 的推理方式 | 同 DR 跨 motion，单位化 RMS | 不同 DR 匹配 motion/phase，单位化 RMS | 二者比值 |',
                      '| --- | ---: | ---: | ---: |'])
        for key, label in [('memory350','正确长期记忆'), ('memory350_without_long','屏蔽长期记忆')]:
            m=primary['models'][key]
            lines.append(f'| {label} | {m["pairs"]["dr_same_world_cross_motion"]["unit_distance_rms"]:.4f} | {m["pairs"]["dr_different_world_same_motion_near_phase"]["unit_distance_rms"]:.4f} | {m[ratio]:.3f} |')
        lines.extend(['', '| 跨 motion family 的环境中心识别 | 可评估 DR world | 测试样本 | Top1 | Top5 | 中心解释方差 R² |',
                      '| --- | ---: | ---: | ---: | ---: | ---: |'])
        for name, m in [('v12', old), ('Memory350', new)]:
            probe=m['dr_world_center_transfer']
            lines.append(f'| {name} | {probe["worlds"]} | {probe["test_samples"]:,} | {probe["top1_accuracy"]:.2%} | {probe["top5_accuracy"]:.2%} | {probe["heldout_motion_center_r2"]:.4f} |')
        lines.extend(['', '环境中心由一半 motion family 的样本估计，再识别另一半 family 的样本。'
                      '新旧模型使用同样的训练、测试划分；这是一项表征读出检查。', '',
                      f'[Memory350 t-SNE]({profile}/plots/tsne_memory350.png) · '
                      f'[新旧 t-SNE 对照]({profile}/plots/tsne_comparison.png) · '
                      f'[交互图]({profile}/plots/tsne_interactive.html) · '
                      f'[真实单位化距离热图]({profile}/plots/unit_distance_heatmaps.png) · '
                      f'[距离分布]({profile}/plots/distance_distributions.png) · '
                      f'[t-SNE 参数对照]({profile}/plots/tsne_settings.png)', ''])
        summary['profiles'][profile] = {'common_full': primary,
            'memory_full': result['subsets']['memory_full'], 'long_full': result['subsets']['long_full']}
    lines.extend(['**采样与数值核对**', '',
        '每 100 步保存一次 query。完整长期记忆在约 400 步形成，之后持续覆盖多个 motion。'
        '由于相邻长期窗口会重叠，另加保守的不重叠检查：较晚样本最旧 chunk 的序号必须越过'
        '较早样本已有 chunk，以及当时短期和 pending 交互未来可能进入的所有 chunk。'
        '该条件已通过跨 reset 的真实交互时间戳校验。样本并非相互独立，置信区间见 JSON 中的描述性 world bootstrap。', '',
        '报告的 common_full 主表用于同 query 比较。另有 memory_full（短期 50 + 长期 300 完整）'
        '和 long_full（长期 300 完整，短期可能为 0–50 步）的指标，以覆盖 reset 后的在线输入情况；'
        '它们都保存在完整结果中。', '',
        f'四个 rank 共 2,048 个原始固定验证窗口复现五步 NMSE：{verification["reproduced_five_step_nmse"]:.10f}，'
        f'训练日志为 {verification["recorded_five_step_nmse"]:.10f}，相对差异 {verification["relative_difference"]:.3g}。'
        '把验证输入还原成原始交互再写入在线 memory bank，得到的 latent 与训练验证路径一致到很小的数值误差。', '',
        '[推理核对](inference_verification.json) · [历史不重叠条件核对](disjoint_control_verification.json) · '
        '[完整汇总](summary.json)', '',
        '这些结果比较两个已经训练好的 checkpoint。它们的训练数据、DR、损失配置和训练轮数也不同，'
        '不能把差异全部归因于 memory 架构。屏蔽长期记忆是在推理时干预，不能代替单独训练的无 memory 对照。'
        '同环境跨 motion 波动同时包含运动状态、历史内容和其他轨迹差异，不能单凭距离把所有变化都归因于 motion 标签。'
        '聚类紧凑程度也不能替代 predictor 或下游 policy 的性能测量。', '',
        't-SNE 标签只用于着色，16 个 DR world 在拟合前随机选定。新旧模型分别拟合 t-SNE，'
        '只能比较颜色混合和局部聚集，不能比较绝对坐标、簇面积或全局空隙；'
        '热图及交互数值给出真实 64 维距离。[算法作者说明](https://lvdmaaten.github.io/tsne/)', '',
        '**复现记录**', '',
        f'Memory350 checkpoint：`{verification["checkpoint"]}`。', '',
        f'SHA256：`{verification["checkpoint_sha256"]}`。', '',
        '各 profile 的 metadata.json 记录完整采样参数和实际物理配置；latents.npz 保存每个 query 的三种 latent、'
        'motion、phase、world 和记忆计数；analysis/*_pairs.npz 保存实际配对及原始 query 行号；'
        'plots/tsne_manifest.json 保存随机选样和拟合设置。', '',
        '脚本：`scripts/probe_memory350_latent_clusters.py`、`scripts/analyze_memory350_latent_clusters.py`、'
        '`scripts/plot_memory350_latent_clusters.py`、`scripts/verify_memory350_latent_inference.py`、'
        '`scripts/report_memory350_latent_clusters.py`。', ''])
    (root/'README.md').write_text('\n'.join(lines))
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    sources = [Path('scripts')/name for name in ['probe_memory350_latent_clusters.py','analyze_memory350_latent_clusters.py',
        'plot_memory350_latent_clusters.py','verify_memory350_latent_inference.py','report_memory350_latent_clusters.py']]
    (root/'source_sha256.json').write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print(str(root/'README.md'))


if __name__ == '__main__':
    main()
