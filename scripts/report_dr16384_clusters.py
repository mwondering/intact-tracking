"""Generate auditable tables and figures from the completed 16384-world audit."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('runs/limb_context_20260916_dr16384')
OUT = ROOT/'analysis'
NAMES = {'true_dr': '真实 DR 参数', 'world_mean_latent': '16 窗口平均 latent', 'window_latent': '单窗口 latent'}
ENGLISH = {'true_dr': 'True DR', 'world_mean_latent': 'Mean latent', 'window_latent': 'Window latent'}


def main():
    results = json.loads((OUT/'results.json').read_text())
    assert len(results) == 18 and json.loads((OUT/'complete.json').read_text())['complete']
    audit = json.loads((OUT/'collection_audit.json').read_text())
    a = np.load(OUT/'selected_worlds.npz')
    coverage = audit['coverage']
    groups = {}
    for r in results:
        groups.setdefault((r['representation'], r['fit_dr_worlds'], r['k']), []).append(r)
    def avg(rows, path):
        def get(row):
            for key in path.split('.'):
                row = row[key]
            return row
        return float(np.mean([get(r) for r in rows]))
    agg = []
    for (rep, nfit, k), rows in groups.items():
        q = 'cluster_pair_quality.'
        pooled_routed_load = []
        for row in rows:
            saved = np.load(OUT/'models'/f"{row['key']}.npz")
            pos = np.where(saved['dr_labels'] == row['nominal']['nominal_class'])[0]
            pooled_routed_load.append(a['physics'][saved['test_dr'][pos],34:38].sum(1))
        pooled_routed_load = np.concatenate(pooled_routed_load)
        record = dict(representation=rep, fit_dr_worlds=nfit, k=k,
                      distance_within=avg(rows,q+'within_cluster.dr_distance_mean'),
                      distance_random=avg(rows,q+'random_other_world.dr_distance_mean'),
                      distance_reduction=avg(rows,q+'reduction.dr_distance_mean'),
                      left_hand_difference=avg(rows,q+'within_cluster.left_hand_kg'),
                      right_hand_difference=avg(rows,q+'within_cluster.right_hand_kg'),
                      left_shin_difference=avg(rows,q+'within_cluster.left_shin_kg'),
                      right_shin_difference=avg(rows,q+'within_cluster.right_shin_kg'),
                      com_difference_cm=avg(rows,q+'within_cluster.com_vector_difference_cm'),
                      torso_difference_kg=avg(rows,q+'within_cluster.torso_mass_difference_kg'),
                      friction_difference=avg(rows,q+'within_cluster.friction_difference'),
                      nominal_coverage=avg(rows,'nominal.modal.test_nominal_coverage'),
                      dr_nominal_fraction=avg(rows,'nominal.modal.random_dr_window_fraction'),
                      heavy_nominal_fraction=avg(rows,'nominal.modal.heavy_window_routed_fraction'),
                      heavy_ever_nominal_fraction=avg(rows,'nominal.modal.heavy_world_ever_routed_fraction'),
                      nominal_cluster_dr_load_p90=float(np.quantile(pooled_routed_load,.9)) if len(pooled_routed_load) else None,
                      nominal_cluster_dr_load_max=float(pooled_routed_load.max()) if len(pooled_routed_load) else None,
                      modal_window_fraction=avg(rows,'window_modal_fraction.mean'),
                      physical20nn_class_agreement=avg(rows,'physical_20nn_same_class_fraction'),
                      random_class_agreement=avg(rows,'random_same_class_fraction'))
        agg.append(record)
    (OUT/'aggregate.json').write_text(json.dumps(agg,indent=2)+'\n')
    with (OUT/'aggregate.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(agg[0]));writer.writeheader();writer.writerows(agg)
    lines = ['# 16,384 个随机 DR 环境：冻结 encoder 的聚类评估', '',
             '## 协议与可解释范围', '',
             '- 16,384 个独立随机 DR 参数组合，另加 2,048 个 nominal 历史对照；不是 16,384 个聚类中心。',
             '- mjwarp 在 8 张 GPU 上各采集 3,200 步；使用原 frozen tracker，当前 DR 监督 encoder u35857，全程没有训练 encoder 或 PPO。',
             '- 每个世界随机选 16 个完整的 350 步历史窗口，不依赖参数、latent 或跟踪好坏。窗口可重叠，不作为独立环境。',
             '- 按世界分成 8,192/8,192 两半，交换拟合和评估；nominal 各 1,024。以下主表是两个方向的均值。',
             '- KMeans 用 K=16 和 64、5 次初始化；fit nominal 与随机 DR 总权重各 50%。评估中 nominal 与随机 DR 单独统计。',
             '- 真实参数使用训练标签的 38 维归一化加权距离；4 项负载、3 个 COM 坐标、躯干质量、摩擦、armature 组各占 1/10。armature 组包含 29 个坐标。',
             '- 单窗口 latent 先单位归一化；平均 latent 是同一世界 16 个单位 latent 的均值，不再单位归一化。',
             '- 真参数参照同时绕过了 encoder 和 2d/(d+0.2) 标签映射，不是只改变 loss 权重的因果对照。',
             '- 使用相同 42 个 LAFAN motion 文件的独立初始相位和环境；世界留出不等于 motion 文件留出。没有专家训练或闭环 latent 路由收益测试。', '',
             '## 随机采样覆盖', '',
             '| 项目 | 16,384 环境结果 |', '|---|---:|',
             f"| 综合 DR 距离 min / P10 / median / P90 / max | {coverage['dr_radius']['min']:.4f} / {coverage['dr_radius']['p10']:.4f} / {coverage['dr_radius']['p50']:.4f} / {coverage['dr_radius']['p90']:.4f} / {coverage['dr_radius']['max']:.4f} |"]
    for t, count in coverage['radius_below'].items():
        lines.append(f'| 综合距离 < {t} | {count} / 16,384（{count/16384:.3%}） |')
    lines += [f"| 四项负载同时 < 各自上限 10% | {coverage['all_four_loads_below_10pct_count']} |", '',
              '这里保留了原随机分布，没有刻意增加弱 DR 样本。增加数量提高了覆盖和估计精度，不会自动改变各区域的概率。', '',
              '## 相同类别是否意味着参数更接近', '',
              '每次均匀抽一个留出 DR 世界及其一个窗口，再从同类窗口抽另一个物理世界；排除同一世界自配对。每个模型 20 万对。nominal 不参与此表，以免 nominal 对 nominal 的零距离制造改善。', '',
              '“缩小比例” = 1 − 同类参数距离 / 随机两环境参数距离；越大表示同类更相似。不是准确率，也不表示控制性能。', '',
              '| 中心输入 | 拟合 DR 数 | K | 同类综合距离 | 相比随机缩小 | 左手差 kg | 右手差 kg | 左小腿差 kg | 右小腿差 kg |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        lines.append(f"| {NAMES[r['representation']]} | {r['fit_dr_worlds']} | {r['k']} | {r['distance_within']:.4f} | {r['distance_reduction']:.2%} | {r['left_hand_difference']:.3f} | {r['right_hand_difference']:.3f} | {r['left_shin_difference']:.3f} | {r['right_shin_difference']:.3f} |")
    random = results[0]['cluster_pair_quality']['random_other_world']
    lines += ['', f"随机对照约为：综合距离 {random['dr_distance_mean']:.4f}，左/右手差 {random['left_hand_kg']:.3f}/{random['right_hand_kg']:.3f} kg，左/右小腿差 {random['left_shin_kg']:.3f}/{random['right_shin_kg']:.3f} kg。", '',
              '## nominal 类与重载混入', '',
              'nominal 类只按拟合 nominal 的最常见类别确定，再在留出世界评估。重载定义为四项附加负载之和 ≥ 8 kg。最大值仅描述本样本极端情况，不代替混入概率。', '',
              '| 输入 | K | nominal 被接纳 | 随机 DR 窗口进入 nominal 类 | 重载窗口进入 nominal 类 | 重载世界至少一次进入 | 类内 DR 总负载 P90 kg | 类内 DR 总负载最大 kg |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        if r['fit_dr_worlds'] != 8192:
            continue
        p90='—' if r['nominal_cluster_dr_load_p90'] is None else f"{r['nominal_cluster_dr_load_p90']:.3f}"
        maximum='—' if r['nominal_cluster_dr_load_max'] is None else f"{r['nominal_cluster_dr_load_max']:.3f}"
        lines.append(f"| {NAMES[r['representation']]} | {r['k']} | {r['nominal_coverage']:.2%} | {r['dr_nominal_fraction']:.3%} | {r['heavy_nominal_fraction']:.3%} | {r['heavy_ever_nominal_fraction']:.3%} | {p90} | {maximum} |")
    lines += ['', '平均 latent 与真参数每世界只有一个标签，因此“至少一次”和单次比例相同；单窗口结果反映 16 个观察时刻，不能视为无限时间失效率。P90 和最大值合并两个留出方向的被路由 DR 窗口计算；没有 DR 混入时记为 —。', '',
              '## 连续 latent 的邻居与窗口稳定性', '']
    for g in json.loads((OUT/'geometry.json').read_text()):
        near=g['nearest_neighbors']
        lines.append(f"- Fold {g['fold']}：对每个世界找真实参数最近的 20 个邻居，平均 latent 最近的 20 个邻居平均保留 {near['mean_overlap_count']:.2f} 个；随机预期 {near['random_expected_overlap']:.3f} 个。DR 两两距离与平均 latent 距离 Spearman={g['pair_spearman']:.3f}；世界到 nominal 距离排序 Spearman={g['nominal_radius_spearman']:.3f}。")
    for r in agg:
        if r['representation']=='window_latent' and r['fit_dr_worlds']==8192:
            lines.append(f"- K={r['k']}：同一世界 16 个窗口中，落在该世界最常见类别的平均比例为 {r['modal_window_fraction']:.2%}。这是稳定性描述，不是物理参数识别准确率。")
    lines += ['', '## 文件', '',
              '- `all_cluster_parameters.csv`：全部类别（包含 nominal 类）的世界数、窗口数、参数 min/P10/median/P90/max；类别编号只在该模型内部有效。',
              '- `results.json`：每个留出方向的详细指标、nominal 95% 覆盖类别并集、近/远非 nominal 类统计。',
              '- `selected_worlds.npz`：原始参数、选中窗口、全局 world ID、latent、划分；`models/`：拟合中心及留出标签。',
              '- `collection_audit.json`：物理参数、样本数、数值精度和元数据哈希检查。',
              '- `../protocol.json` 与 `../shard_*/queries/`：协议及可回溯的原始历史。', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    dr=~a['nominal']
    axes[0,0].hist(a['radius'][dr],bins=np.linspace(0,.75,46),density=True,color='#537db7',alpha=.8,label='16,384 random DR')
    old=np.load('runs/limb_context_20260916_paired_encoder_dr_clustering/labels_and_inputs.npz')['physics']
    nom=np.zeros(38);nom[4]=.6;nom[5:34]=1.
    od=np.linalg.norm((old-nom)/a['span']*np.sqrt(a['weights']),axis=1)
    axes[0,0].hist(od,bins=np.linspace(0,.75,26),density=True,histtype='step',color='#bf6339',label='Previous 512 DR')
    axes[0,0].set(xlabel='True DR distance to nominal',ylabel='Density',title='More samples, same sampling distribution');axes[0,0].legend()
    colors={'true_dr':'#3c9961','world_mean_latent':'#d69c30','window_latent':'#537db7'}
    for rep in NAMES:
        rr=sorted([r for r in agg if r['representation']==rep and r['k']==16],key=lambda r:r['fit_dr_worlds'])
        axes[0,1].plot([r['fit_dr_worlds'] for r in rr],[100*r['distance_reduction'] for r in rr],'-o',color=colors[rep],label=ENGLISH[rep])
    axes[0,1].set(xlabel='DR worlds used to fit centers (K=16)',ylabel='Within-class DR distance reduction (%)',xticks=[512,8192],title='Same held-out worlds for both fit sizes');axes[0,1].legend()
    x=np.arange(4)
    for ri,rep in enumerate(NAMES):
        r=next(r for r in agg if r['representation']==rep and r['k']==16 and r['fit_dr_worlds']==8192)
        vals=[r[k] for k in ['left_hand_difference','right_hand_difference','left_shin_difference','right_shin_difference']]
        axes[1,0].bar(x+(ri-1)*.23,vals,.23,color=colors[rep],label=ENGLISH[rep])
    baseline=[random[k] for k in ['left_hand_kg','right_hand_kg','left_shin_kg','right_shin_kg']]
    axes[1,0].scatter(x,baseline,marker='_',s=350,color='black',label='Random pair')
    axes[1,0].set(xticks=x,xticklabels=['L hand','R hand','L shin','R shin'],ylabel='Mean absolute load difference (kg)',title='Do same-class environments have similar loads?');axes[1,0].legend(fontsize=8)
    r=[r for r in results if r['representation']=='window_latent' and r['fit_dr_worlds']==8192 and r['k']==16][0]
    data=np.load(OUT/'models'/f"{r['key']}.npz")
    pos=np.where(data['dr_labels']==r['nominal']['nominal_class'])[0]
    load=a['physics'][data['test_dr'][pos],34:38].sum(1)
    sl=np.sort(load)
    axes[1,1].plot(sl,np.arange(1,len(sl)+1)/len(sl),color=colors['window_latent'])
    axes[1,1].axvline(8,color='#bb4444',linestyle='--',label='Heavy threshold: 8 kg')
    axes[1,1].set(xlabel='Total added limb load (kg)',ylabel='Cumulative fraction of routed DR windows',ylim=(0,1.01),title='DR windows routed to nominal class (fold 0, K=16)');axes[1,1].legend(fontsize=8)
    fig.savefig(OUT/'physical_cluster_audit.png',dpi=180)
    fig.savefig(OUT/'physical_cluster_audit.svg')
    print(json.dumps(agg,indent=2),flush=True)


if __name__=='__main__':
    main()
