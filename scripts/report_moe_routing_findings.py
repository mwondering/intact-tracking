"""Present completed routing diagnostics and held-out payload decoding together."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    root = Path(parser.parse_args().root)
    routing = json.loads((root/'summary.json').read_text())
    decoder = json.loads((root/'payload_decoder/summary.json').read_text())
    cluster = json.loads((root/'cluster_metric_probe/summary.json').read_text())
    provenance = json.loads((root/'context_payload_training_provenance.json').read_text())
    primary = decoder['modes']['sample']['mean_of_two_direction_metrics']
    plt.rcParams.update({'font.size': 11, 'savefig.dpi': 170})
    labels = ['Left hand', 'Right hand', 'Left shin', 'Right shin']
    methods = [('expert_id', 'Actual expert ID (16)', '#c94a45'),
               ('unit_latent_ridge', 'Full 64-D latent: linear', '#2468a2'),
               ('unit_latent_mlp', 'Full 64-D latent: MLP', '#31886b')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for j, (name, label, color) in enumerate(methods):
        for ax, key in zip(axes, ('r2_per_limb', 'mae_kg_per_limb')):
            values = primary[name][key]
            bars = ax.bar(np.arange(4)+(j-1)*.25, values, .24, label=label, color=color)
            ax.bar_label(bars, labels=[f'{v:.3f}' if key == 'r2_per_limb' else f'{v:.2f}' for v in values], fontsize=8, padding=3)
    axes[0].set(title='Load information available to a decoder', ylabel='Held-out R² (1 = perfect, 0 = mean predictor)', ylim=(-.1, 1.12))
    axes[1].set(title='Absolute mass prediction error', ylabel='Mean absolute error (kg)', ylim=(0, 1.18))
    for ax in axes:
        ax.set_xticks(np.arange(4), labels)
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.1), fontsize=9)
    fig.suptitle('Shared MoE u1000 | actual sampled-action trajectories | independent DR-seed test, both directions')
    fig.savefig(root/'payload_information_loss.png')
    fig.savefig(root/'payload_information_loss.svg')
    plt.close(fig)
    arrays = []
    for train, test in ((30402, 30401), (30401, 30402)):
        with np.load(root/f'payload_decoder/sample_train{train}_test{test}.npz') as f:
            arrays.append({key: f[key] for key in ('targets', 'expert_id', 'unit_latent_ridge')})
    data = {key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    for row, name in enumerate(('unit_latent_ridge', 'expert_id')):
        for col, ax in enumerate(axes[row]):
            limit = 2.5 if col < 2 else 4.
            ax.hexbin(data['targets'][:, col], data[name][:, col], gridsize=45, bins='log', mincnt=1, cmap='Blues')
            ax.plot([0, limit], [0, limit], '--', color='#be5046', lw=1)
            lower = min(0., data['unit_latent_ridge'][:, col].min(), data['expert_id'][:, col].min())-.1
            upper = max(limit, data['unit_latent_ridge'][:, col].max(), data['expert_id'][:, col].max())+.1
            ax.set(xlim=(0, limit), ylim=(lower, upper), xlabel='True load (kg)', ylabel='Predicted load (kg)',
                   title=f"{labels[col]} | R² = {primary[name]['r2_per_limb'][col]:.3f}")
    fig.suptitle('Top: full latent + linear decoder | Bottom: actual expert ID + training-cluster mean\n65,536 instantaneous predictions on unseen DR worlds; no test-world fitting')
    fig.savefig(root/'payload_decoding_scatter.png')
    plt.close(fig)
    lines = ['# 完整探索结果：latent 有负载信息，16 类硬路由大量丢失', '',
             '本次诊断的是共享 encoder 的 MoE 第1000轮。其 context encoder 确实在四肢负载上训练过：nominal50、Memory350扩容版、response10/predictor5、第15000轮。训练时四肢各自独立U(0,4) kg；本次PPO/诊断手部U(0,2.5) kg、小腿U(0,4) kg。', '',
             f"Policy SHA256：`{routing['sample']['checkpoint_sha256']}`。Context SHA256：`{provenance['context_sha256']}`。共享和无共享MoE的训练记录均指向同一context；本次交互测量只针对共享版。见[训练来源审计](context_payload_training_provenance.json)。", '',
             '## 1. 完整 latent 能否读出四肢负载', '',
             '可以。用实际闭环轨迹中的64维latent拟合一个线性回归器，便能预测未参与拟合的环境的重量。两批独立DR seed交换训练/测试；每次768个world拟合、256个world选超参数、另一批1024个world测试。每world32个充分记忆的瞬时样本。', '',
             '下表为sample动作下两个交换方向的平均R²；mean动作结果一致。R²=1为完美预测，0相当于均值预测，负值表示更差，不是分类正确率。', '',
             '| 输入/解码器 | 左手 | 右手 | 左小腿 | 右小腿 |', '|---|---:|---:|---:|---:|']
    for name, label, _ in methods:
        lines.append('| '+label+' | '+' | '.join(f'{v:.3f}' for v in primary[name]['r2_per_limb'])+' |')
    lines.extend(['', '线性读取完整latent的平均绝对误差分别为'+ '/'.join(f'{v:.3f}' for v in primary['unit_latent_ridge']['mae_kg_per_limb'])+' kg。原始latent与单位归一化latent的解码结果几乎相同，信息丢失不能主要归于归一化抹掉了幅值。', '',
                  '![负载信息](payload_information_loss.png)', '',
                  '[瞬时预测散点密度图](payload_decoding_scatter.png) · [完整解码协议、双向结果及kg误差](payload_decoder/README.md)', '',
                  '## 2. 是在线中心不准，还是距离度量偏向其他参数', '',
                  '按原距离重新拟合16个中心，或增加到64个中心，小腿负载仍没有得到有效分组。原checkpoint的测试inertia并不差于重新拟合16中心。这更支持当前距离几何与负载分组目标不一致，而非仅仅在线中心没有收敛。', '',
                  '| 分组方式 | 左手R² | 右手R² | 左小腿R² | 右小腿R² |', '|---|---:|---:|---:|---:|'])
    for name, values in cluster['mean_of_two_direction_metrics'].items():
        lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in values['r2_per_limb'])+' |')
    lines.extend(['', '当前expert划分对水平COM和摩擦的方差解释率约74%/74%/69%，对两条小腿仅约1%/2%。白化后负载分组有部分改善，但小腿仍较弱。按真实重量分16类能获得约0.71–0.75的负载R²；这是使用真实标签的参照，也牺牲了其他DR参数的分组能力，不能直接当作更好的控制器。', '',
                  '[聚类对照完整协议](cluster_metric_probe/README.md) · [每expert内的负载范围](expert_payload_ranges_mean.png)', '',
                  '## 3. 同一DR的实际路由是否稳定', '',
                  '| 指标 | mean动作 | sample动作 |', '|---|---:|---:|'])
    for label, fn in [
        ('同expert/不同world的单位latent L2', lambda x:x['geometry']['latent']['within_expert_different_worlds']['mean']),
        ('不同expert/不同world的单位latent L2', lambda x:x['geometry']['latent']['between_experts_different_worlds']['mean']),
        ('充分记忆且同motion，每步切换率(%)', lambda x:x['temporal']['rates']['steady_full_memory_within_motion']['switch_fraction']*100),
        ('充分记忆且同motion，每秒切换次数', lambda x:x['temporal']['rates']['steady_full_memory_within_motion']['switches_per_second']),
        ('同world不同motion的主expert一致率(%)', lambda x:x['temporal']['cross_motion']['same_expert_across_motion_fraction_world_weighted']['mean']*100),
        ('切换后下一步返回原expert的比例(%)', lambda x:x['temporal']['switch_details']['immediate_A_B_A_return_fraction']*100)]:
        lines.append('| '+label+' | '+' | '.join(f'{fn(routing[mode]):.3f}' for mode in ('mean','sample'))+' |')
    lines.extend(['', '全程冻结中心及policy，所有静态DR开始/结束逐项相同。因此这里测到的是输入变化导致的实际路由切换，与中心更新造成的重分配是两回事。每秒次数是总体平均，各world不均匀。跨motion一致率也不是DR分类准确率。', '',
                  '[逐步路由图](expert_routes_mean.png) · [全部簇内DR与切换统计](README.md)', '',
                  '## 对后续方案的影响', '',
                  '当前MoE的latent只进入router，expert head只接收压缩obs和tracker action，连续latent没有送进head。实际路径将可解码负载的64维表示压成一个16选1编号，特别损失了小腿重量信息；重新聚类/增加中心数量都未能自动解决。', '',
                  '后续最有针对性的对照是让expert保留连续latent输入，并另行比较按负载/控制相关信息定义的路由度量；路由稳定性应独立比较有滞回或固定分配的版本。这里没有执行新policy训练，也没有把新聚类中心替换进旧expert。', '',
                  '这些测量明确定位了信息丢失和路由变化，但不能单独证明它们解释了全部控制性能差距。独立policy与共享MoE还存在每expert数据量、连续DR覆盖等差异；需要控制实验衡量各因素对tracking的实际贡献。', ''])
    (root/'FINDINGS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
