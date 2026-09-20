"""Summarize online nominal/DR separation and learned residual magnitudes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


def auc(negative, positive):
    n, p = len(negative), len(positive)
    if not n or not p:
        return None
    ranks = rankdata(np.r_[negative, positive])
    return float((ranks[n:].sum()-p*(p+1)/2)/(n*p))


def summaries(data, mask, threshold):
    nominal = data['is_nominal']
    radius, residual = data['radius'], data['residual']
    rsq = np.mean(residual.astype(np.float64)**2, axis=-1)
    rrms = np.sqrt(rsq)
    latent_norm = np.linalg.norm(data['latent'], axis=-1)
    unit = data['latent']/np.maximum(latent_norm[...,None], 1e-12)
    result, world_rsq = {}, {}
    for name, label in [('nominal', nominal), ('dr', ~nominal)]:
        selected = mask & label[None]
        counts = selected.sum(axis=0)
        worlds = counts > 0
        if not worlds.any():
            continue
        score = radius[selected]
        world_rsq[name] = (rsq*selected).sum(0)[worlds]/counts[worlds]
        center = unit[selected].mean(0)
        result[name] = {
            'worlds':int(worlds.sum()), 'world_frames':int(selected.sum()),
            'radius_mean':float(score.mean()), 'radius_rms':float(np.sqrt(np.mean(score**2))),
            'radius_p05':float(np.quantile(score,.05)), 'radius_median':float(np.median(score)),
            'radius_p95':float(np.quantile(score,.95)), 'radius_max':float(score.max()),
            'radius_min':float(score.min()), 'latent_raw_norm_mean':float(latent_norm[selected].mean()),
            'center_anchor_distance':float(np.linalg.norm(center-data['anchor'])),
            'within_center_rms':float(np.sqrt(np.square(unit[selected]-center).sum(-1).mean())),
            'residual_action_rms':float(np.sqrt(rsq[selected].mean())),
            'residual_frame_rms_median':float(np.median(rrms[selected])),
            'residual_frame_rms_p95':float(np.quantile(rrms[selected],.95)),
            'residual_world_equal_rms':float(np.sqrt(world_rsq[name].mean())),
            'base_action_rms':float(np.sqrt(np.mean(data['base_rms'][selected].astype(np.float64)**2))),
            'residual_pd_radians_rms':float(np.sqrt(np.mean(data['residual_pd_rms'][selected].astype(np.float64)**2))),
            'near_zero_fraction':{str(t):float((rrms[selected] <= t).mean()) for t in (.005,.01,.02,.05)},
            'mean_residual_per_joint':residual[selected].mean(0).tolist(),
            'rms_residual_per_joint':np.sqrt(np.square(residual[selected].astype(np.float64)).mean(0)).tolist(),
            'saturation_coordinate_fraction':float((np.abs(residual[selected]) >= .95*.25).mean()),
            'anchor_direction_change_action_rms':float(np.sqrt(np.mean(data['anchor_direction_action_delta_rms'][selected].astype(np.float64)**2))),
            'exploration_rms':float(np.sqrt(np.mean(data['exploration_rms'][selected].astype(np.float64)**2))),
        }
        result[name]['residual_over_base_rms'] = result[name]['residual_action_rms']/result[name]['base_action_rms']
    if 'nominal' not in result or 'dr' not in result:
        return result
    neg, pos = radius[mask & nominal[None]], radius[mask & ~nominal[None]]
    nominal_recall, dr_recall = float((neg < threshold).mean()), float((pos >= threshold).mean())
    count = mask.sum(0); present = count > 0
    world_radius = (radius*mask).sum(0)/np.maximum(count,1)
    rng=np.random.default_rng(7201)
    n, d = world_rsq['nominal'], world_rsq['dr']
    draw_n=rng.choice(n,(3000,len(n))).mean(-1)
    draw_d=rng.choice(d,(3000,len(d))).mean(-1)
    result['comparison']={
        'frame_radius_auroc':auc(neg,pos),
        'world_mean_radius_auroc':auc(world_radius[present & nominal],world_radius[present & ~nominal]),
        'nominal_recall_at_fixed_threshold':nominal_recall,
        'dr_recall_at_fixed_threshold':dr_recall,
        'balanced_accuracy_at_fixed_threshold':(nominal_recall+dr_recall)/2,
        'full_radius_ranges_disjoint':bool(neg.max()<pos.min()),
        'residual_nominal_over_dr_rms':result['nominal']['residual_action_rms']/result['dr']['residual_action_rms'],
        'world_equal_residual_nominal_over_dr_rms':float(np.sqrt(n.mean()/d.mean())),
        'world_equal_residual_ratio_ci95':np.quantile(np.sqrt(draw_n/draw_d),[.025,.975]).tolist(),
        'uncertainty':'Resample worlds independently within each class; repeated frames are not treated as independent trials',
    }
    return result


def main(directory):
    directory=Path(directory)
    protocol=json.loads((directory/'protocol.json').read_text())
    result={'protocol':protocol,'arms':{},'distance_definition':'Euclidean distance between current 64D unit latent and fixed nominal unit anchor',
            'residual_definition':'Learned mean correction in normalized command units; excludes exploration noise',
            'causal_limit':'Descriptive association between environment class and correction; correction magnitude alone does not establish benefit or harm'}
    offline=directory/'offline_encoder_u7179'
    if (offline/'summary.json').exists():
        previous=json.loads((offline/'summary.json').read_text())
        with np.load(offline/'u7179_broad.npz') as saved:
            distance=np.linalg.norm(saved['z']-np.asarray(previous['anchor']),axis=-1)
            subsets={}
            for title,key in [('full350','full'),('all_usable','usable')]:
                neg=distance[saved[key] & saved['nominal']]
                pos=distance[saved[key] & ~saved['nominal']]
                subsets[title]={'nominal_samples':len(neg),'dr_samples':len(pos),'auroc':auc(neg,pos),
                                'nominal_radius_rms':float(np.sqrt(np.mean(neg**2))),
                                'nominal_radius_max':float(neg.max()),'dr_radius_min':float(pos.min())}
        result['same_encoder_offline_reference']={'checkpoint':previous['checkpoints']['7179'],
            'groups':subsets,'scope':'Same encoder on cached frozen-tracker histories; different motion/DR samples and policy, CPU fp32'}
    fig, axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for row,mode in enumerate(['sampled','mean']):
        metadata=json.loads((directory/f'{mode}.json').read_text())
        with np.load(directory/f'{mode}.npz') as source:
            data={k:source[k] for k in source.files}
        assert metadata['completed_updates']==protocol['completed_updates']
        assert metadata['mode']==mode and metadata['recorded_world_frames']==data['radius'].size
        assert metadata['nominal_worlds']==int(data['is_nominal'].sum())
        assert all(np.isfinite(x).all() for x in data.values())
        np.testing.assert_allclose(np.linalg.norm(data['latent']/np.linalg.norm(data['latent'],axis=-1,keepdims=True)
                                                -data['anchor'],axis=-1), data['radius'],atol=2e-6,rtol=2e-6)
        assert np.abs(data['residual']).max() <= .250001
        if mode=='mean':
            assert not data['exploration_rms'].any()
        full=(data['short_count']==50)&(data['long_count']==30)&(data['history5_count']==5)
        masks={'all_warm':np.ones_like(full),'full350':full,
               'long_full_short_partial':(data['long_count']==30)&(data['short_count']<50),
               'first_five_frames':data['history5_count']<5}
        groups={key:summaries(data,mask,protocol['distance_threshold']) for key,mask in masks.items() if mask.any()}
        result['arms'][mode]={'metadata':metadata,'groups':groups,
            'full350_fraction':float(full.mean()),
            'motion_ids_observed':len(np.unique(data['motion_id'])),
            'failures_during_collection':{'nominal':int(data['failures'][data['is_nominal']].sum()),
                                          'dr':int(data['failures'][~data['is_nominal']].sum())}}
        radius=data['radius'];rrms=np.sqrt(np.mean(data['residual']**2,axis=-1))
        rng=np.random.default_rng(90)
        for name,label,color in [('Nominal',data['is_nominal'],'#2563eb'),('DR',~data['is_nominal'],'#ea580c')]:
            use=full & label[None]
            x,y=radius[use],rrms[use]
            axes[row,0].hist(x,bins=np.linspace(0,2,101),density=True,histtype='step',linewidth=1.6,label=name,color=color)
            axes[row,1].hist(y,bins=np.linspace(0,.25,101),density=True,histtype='step',linewidth=1.6,label=name,color=color)
            choice=rng.choice(len(x),min(3500,len(x)),replace=False)
            axes[row,2].scatter(x[choice],y[choice],s=4,alpha=.18,color=color,label=name,rasterized=True)
        axes[row,0].axvline(protocol['distance_threshold'],ls='--',color='gray',lw=1)
        axes[row,0].set(xlabel='Unit latent distance to nominal anchor',ylabel='Density',xlim=(0,2))
        axes[row,1].set(xlabel='Learned residual RMS (command units)',ylabel='Density',xlim=(0,.25))
        axes[row,2].set(xlabel='Unit latent distance to nominal anchor',ylabel='Learned residual RMS',xlim=(0,2),ylim=(0,.25))
        title='Training-mode sampled actions' if mode=='sampled' else 'Evaluation-mode mean actions'
        axes[row,0].set_title(title+'\nFull 350-step memory')
        axes[row,1].set_title('Residual excludes exploration noise')
        axes[row,2].set_title('Subsampled world/step observations')
        for ax in axes[row]:
            ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle(f"PPO u{protocol['completed_updates']} / encoder u7179: online nominal vs DR")
    for key in ('checkpoint_sha256','context_sha256','tracker_sha256','motion_catalog_sha256'):
        assert result['arms']['sampled']['metadata'][key] == result['arms']['mean']['metadata'][key]
    if 'same_encoder_offline_reference' in result:
        assert result['same_encoder_offline_reference']['checkpoint']['sha256']==result['arms']['mean']['metadata']['context_sha256']
    fig.savefig(directory/'online_latent_residual.png',dpi=180)
    fig.savefig(directory/'online_latent_residual.pdf')
    plt.close(fig)
    (directory/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=[f"# PPO u{protocol['completed_updates']} 在线 nominal/DR latent 与 residual",'',
           '固定当前 PPO checkpoint 和 encoder u7179，在新采样的 1024 个环境中在线运行；每组 103 个 nominal、921 个原生 DR，无额外负载。',
           '使用训练 rank 0 的完整 27560 条 motion，并恢复该 checkpoint 保存的 adaptive sampling 统计。两种动作模式分别由自身策略预热 1000 步，再采集 1000 步，每 5 步记录一次；每组记录 204800 个 world-frame。原始终止项和 500 步 episode 上限保留。',
           '这是固定 checkpoint 的新在线轨迹，未读取或更改正在训练的进程缓冲区，也未进行梯度更新。仅覆盖一个训练数据分片；重复时间帧并非独立环境。',
           'latent 距离使用当前 64D latent 归一化后到固定单位锚点的距离；policy 仍接收原始五帧 latent。residual 指学到的均值补偿，统计时排除探索噪声。',
           '二分类阈值预先固定为 0.4，来自旧离线数据 nominal 最大 0.100、DR 最小 0.786 之间的间隔，未根据本次在线结果调优。','']
    for mode,title in [('sampled','训练模式：带当前策略探索噪声'),('mean','评测模式：确定性均值动作')]:
        arm=result['arms'][mode];g=arm['groups']['full350'];c=g['comparison']
        lines.extend([f'## {title}','',f"完整记忆帧占比：{arm['full350_fraction']:.2%}。以下主表仅使用完整 350 步记忆及 5 帧队列。",'',
                      '| 指标 | nominal | DR |','|---|---:|---:|'])
        labels=[('worlds','环境数'),('world_frames','记录帧数'),('radius_mean','到锚点距离均值'),
                ('radius_median','到锚点距离中位数'),('radius_p95','到锚点距离 P95'),
                ('residual_action_rms','Residual RMS（命令单位）'),('residual_frame_rms_p95','单帧 residual RMS P95'),
                ('base_action_rms','Tracker action RMS'),('residual_pd_radians_rms','缩放后 PD 角度补偿 RMS（rad，SP 滤波前）'),
                ('anchor_direction_change_action_rms','同观测替换为 nominal 方向的动作变化 RMS')]
        for key,title in labels:
            lines.append(f"| {title} | {g['nominal'][key]:.5f} | {g['dr'][key]:.5f} |")
        lines.extend([f"| 单帧 residual RMS ≤ 0.01 的比例 | {g['nominal']['near_zero_fraction']['0.01']:.2%} | {g['dr']['near_zero_fraction']['0.01']:.2%} |",'',
                      f"按到锚点距离分类的 frame AUROC：{c['frame_radius_auroc']:.6f}；先对每个环境取距离均值后的 AUROC：{c['world_mean_radius_auroc']:.6f}。",
                      f"固定阈值 0.4：nominal 识别率 {c['nominal_recall_at_fixed_threshold']:.2%}，DR 识别率 {c['dr_recall_at_fixed_threshold']:.2%}，balanced accuracy {c['balanced_accuracy_at_fixed_threshold']:.2%}。",
                      f"nominal residual RMS / DR residual RMS：{c['residual_nominal_over_dr_rms']:.3f}。",
                      f"计入 warm 阶段所有记录帧（含 episode/motion 切换后短期历史未满）时，AUROC 为 {arm['groups']['all_warm']['comparison']['frame_radius_auroc']:.6f}。",''])
    lines.extend(['替换方向的敏感度只在同一观测上重算动作，没有将替换后的动作施加给环境，不能视作闭环收益实验。',
                  'nominal 中非零 residual 本身不等于错误：原 tracker 也可能存在可纠正误差。此处只判断表征分离和补偿幅度，未用幅度推断 tracking 改善/退化的因果。','',
                  '图：![Online latent/residual distributions](online_latent_residual.png)',''])
    if 'same_encoder_offline_reference' in result:
        old=result['same_encoder_offline_reference']['groups']['full350']
        lines.extend(['## 同一个 u7179 encoder 的离线对照','',
                      f"补测同一批冻结 tracker 留出历史，完整历史下 nominal 到锚点 RMS 为 {old['nominal_radius_rms']:.5f}，nominal 最大距离 {old['nominal_radius_max']:.5f}，DR 最小距离 {old['dr_radius_min']:.5f}，AUROC {old['auroc']:.6f}。",
                      '离线与在线的 motion/DR 样本、轨迹策略和数值精度不同；它用于确认同一 encoder 的离线分离能力，不用于单独归因在线变化。',''])
    (directory/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({mode:{'full350_fraction':arm['full350_fraction'],**arm['groups']['full350']} for mode,arm in result['arms'].items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    main(parser.parse_args().directory)
