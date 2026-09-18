"""Summarize the complete latent-partition exploration and validation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('runs/limb_context_20260916_latent_environment_partitions')
SOURCE=Path('runs/limb_context_20260916_dr16384/analysis')


def read(path):return json.loads(path.read_text())


def main():
    old=read(ROOT/'dynamics_evaluation/results.json')
    fresh=read(ROOT/'fresh_validation/results.json')
    smooth=read(ROOT/'fresh_ema_validation/results.json')
    confirmation=read(ROOT/'confirmation/results.json')
    physical=read(ROOT/'causal_ema/results.json')
    base=read(SOURCE/'results.json')
    rows=[]
    for method,k in [('latent_radius_fixed',16),('window_latent',16),('response_metric',16),
                     ('true_dr',64),('window_latent',64),('response_metric',64),('selected_blend',64),('causal200',64)]:
        rr=[r for r in fresh+smooth if r['method']==method and r['k']==k]
        assert len(rr)==2
        rows.append(dict(method=method,k=k,response_reduction=float(np.mean([r['reduction'] for r in rr])),
                         old_state_reduction=float(np.mean([r['old_state_reduction'] for r in rr])),
                         block_reduction={name:float(np.mean([r['block_reduction'][name] for r in rr])) for name in rr[0]['block_reduction']}))
    f=read(ROOT/'confirmation/per_family.json');e=f
    families=sorted(set(x['family'] for x in e));family_rows=[]
    for family in families:
        def sums(source,method):
            rr=[x for x in source if x['family']==family and x['method']==method and x['k']==64]
            return np.mean([np.sum(x['within']) for x in rr]),np.mean([np.sum(x['random']) for x in rr])
        raw,ref=sums(f,'window_latent');new,ref2=sums(e,'causal200')
        np.testing.assert_allclose(ref,ref2)
        family_rows.append(dict(family=family,raw_squared=raw,new_squared=new,random_squared=ref,
                                raw_reduction=float(1-np.sqrt(raw/ref)),new_reduction=float(1-np.sqrt(new/ref))))
    rng=np.random.default_rng(83116);draw=rng.integers(len(families),size=(10000,len(families)))
    raw=np.array([x['raw_squared'] for x in family_rows]);new=np.array([x['new_squared'] for x in family_rows]);ref=np.array([x['random_squared'] for x in family_rows])
    delta=np.sqrt(raw[draw].sum(1)/ref[draw].sum(1))-np.sqrt(new[draw].sum(1)/ref[draw].sum(1))
    uncertainty=dict(paired_family_bootstrap_delta_95pct=np.quantile(delta,[.025,.975]).tolist(),
                     replicates=10000,families=8,note='Resamples motion families, not a guarantee over arbitrary motions or all possible DR.')
    classes=read(ROOT/'confirmation/per_class.json');class_results=[]
    for fold in (0,1):
        entries=[]
        for c in range(64):
            cc=[x for x in classes if x['fold']==fold and x['cluster']==c and x['squared_distance'] is not None]
            if not cc:continue
            ratio=float(np.sqrt(np.mean([x['squared_distance'] for x in cc])/np.mean([x['random_squared_distance'] for x in cc])))
            entries.append(dict(cluster=c,ratio_to_random=ratio,observed_dr_worlds=cc[0]['worlds']))
        class_results.append(dict(fold=fold,classes=entries,closer_than_random=sum(x['ratio_to_random']<1 for x in entries)))
    rr=[x for x in base if x['representation']=='window_latent' and x['fit_dr_worlds']==8192 and x['k']==64]
    comparative={}
    for name in rr[0]['cluster_pair_quality']['within_cluster']:
        comparative[name]=dict(raw=float(np.mean([x['cluster_pair_quality']['within_cluster'][name] for x in rr])),
                               new=float(np.mean([x['cluster_pair_quality']['within_cluster'][name] for x in physical])))
    runtime=read(ROOT/'export/runtime_verification.json');assert runtime['passed']
    summary=dict(fresh_response=rows,confirmation=confirmation,per_family=family_rows,uncertainty=uncertainty,
                 physical_comparison=comparative,per_class=class_results,runtime=runtime,
                 mean_modal_window_fraction=float(np.mean([x['modal_window_fraction'] for x in physical])),
                 mean_heavy_nominal_window_fraction=float(np.mean([x['nominal']['modal']['heavy_window_routed_fraction'] for x in physical])))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    labels={'latent_radius_fixed':'Latent radius bins','window_latent':'Raw latent KMeans','true_dr':'True DR KMeans',
            'response_metric':'Response metric','selected_blend':'Response + DR metric','causal200':'Combined + causal EMA'}
    fig,ax=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    methods=['window_latent','causal200']
    xx=np.arange(2)
    ax[0].bar(xx,[100*np.mean([r['reduction'] for r in confirmation if r['method']==m]) for m in methods],color=['#597fba','#36805a'])
    ax[0].set(xticks=xx,xticklabels=[labels[m] for m in methods],ylabel='Within-class response difference reduction (%)',title='Final confirmation: new worlds and states, K=64')
    ax[0].tick_params(axis='x',rotation=25)
    keys=['left_hand_kg','right_hand_kg','left_shin_kg','right_shin_kg'];x=np.arange(4)
    ax[1].bar(x-.18,[comparative[k]['raw'] for k in keys],.36,label='Raw latent',color='#597fba')
    ax[1].bar(x+.18,[comparative[k]['new'] for k in keys],.36,label='Combined + causal EMA',color='#36805a')
    ax[1].set(xticks=x,xticklabels=['L hand','R hand','L shin','R shin'],ylabel='Mean absolute load difference (kg)',title='Same-class load similarity');ax[1].legend(fontsize=8)
    ax[2].barh(np.arange(8),[100*(x['new_reduction']-x['raw_reduction']) for x in family_rows],color='#36805a')
    ax[2].axvline(0,color='black',linewidth=.6)
    ax[2].set(yticks=np.arange(8),yticklabels=families,xlabel='Improvement over raw latent (percentage points)',title='Final confirmation by motion family')
    fig.savefig(ROOT/'comparison.png',dpi=180);fig.savefig(ROOT/'comparison.svg')
    main_reduction=np.mean([x['reduction'] for x in confirmation if x['method']=='causal200'])
    raw_reduction=np.mean([x['reduction'] for x in confirmation if x['method']=='window_latent'])
    text=['# 利用 latent 聚类相似动力学环境：实验记录','',
          '推荐候选：冻结 encoder，组合“实际响应标定距离”和“latent 读出的 DR 参数距离”，64 类；固定 DR 场景下加因果平滑。没有更新 encoder、predictor 或 PPO，没有训练专家或更改现有策略。','',
          '## 方法','',
          '1. 对单位化 latent 拟合两个离线标定映射：64维线性响应映射，以及38维仿真参数读出。后者包含4负载、COM、质量、摩擦、armature；没有把读出不准的维度从评价里删除。',
          '2. 响应标定标签由 mjwarp 回放得到：各 DR 使用同一初态、同一10步物理PD目标序列，保留四肢相对位置/速度、root位置/速度、root姿态/角速度、无接触门控的足部XY位置/速度。8个块等权，尺度来自标定nominal motion的单步变化。',
          '3. 两路特征按拟合DR窗口方差归一化，拼接 sqrt(0.65)×响应特征 与 sqrt(0.35)×参数估计特征，再用KMeans分64类。推理只需要latent。',
          '4. 针对当前物理参数固定的任务，特征加因果EMA，时间常数200个控制步。使用每100步一次的全部有效历史查询，首次有效窗口直接初始化；没有使用未来窗口。控制dt=0.02s，实际验证的查询间隔2s、平滑时间常数4s。', '',
          '## 划分与选择','',
          '- 16,384独立随机DR，加2,048 nominal控制。每世界16个完整历史窗口，用8,192/8,192世界互斥两折。',
          '- 动力学探针取2,048个随机DR，两折各1,024；96组初态覆盖24个motion文件。响应映射仅用拟合世界的7个motion family、44组初态；另外8个family、52组初态用于评价。',
          '- 组合权重只由各折拟合世界+拟合motion的响应和物理相似性选取。候选必须让两小腿负载差各缩小至少8%，且总体参数距离不比原latent差；两个折在K64都独立选择0.35。K16仅一折有可行候选，不能报告为稳定满足条件。',
          '- 另从本次新采集的nominal历史抽64组初态、17个motion文件，全部属于未用于响应标定的8个family；固定映射、中心、权重和尺度重放，未据此重拟合。',
          '- 因果平滑是在探索中追加的，因此另做最终确认：冻结最终模型哈希，换用2,048个未参与前述响应探针的DR世界，并从query2800另抽64组初态。最终确认后没有再调参。',
          '- 编码器本身可能在原训练里见过这些motion文件；这里的motion留出针对本次响应标定，不声称全新motion域泛化。', '',
          '## 探索阶段的新初态对照','',
          '每个世界的16个窗口决定其路由概率；仅配对不同DR世界。比较同类与随机配对在相同初态、动作下的响应RMS差异。不是预测准确率或控制奖励。下表为两折均值。','',
          '|方法|K|响应差异比随机缩小|原70维状态指标缩小|','|---|---:|---:|---:|']
    for row in rows:text.append(f"|{labels[row['method']]}|{row['k']}|{row['response_reduction']:.2%}|{row['old_state_reduction']:.2%}|")
    text += ['', '## 最终冻结方案确认：新DR世界、新初态', '',
             f"主要结果：{raw_reduction:.2%} → {main_reduction:.2%}；同类响应差异在原latent基础上再下降 {1-(1-main_reduction)/(1-raw_reduction):.2%}。8个family成对重采样的提升95%区间为 {uncertainty['paired_family_bootstrap_delta_95pct'][0]*100:.2f}–{uncertainty['paired_family_bootstrap_delta_95pct'][1]*100:.2f} 个百分点。", '',
             '**没有所有指标同时改善。** 新方法的原70维状态指标比原latent弱；探索阶段walk4、最终确认walk2的主要响应指标出现回退。完整逐family、逐块结果都保留。手部/COM的参数相似性有退步，躯干质量差仍接近随机，不能宣称所有DR参数都被完整区分。','',
             '## 同类物理参数差','', '|项目|原latent K64|组合距离+因果平滑 K64|','|---|---:|---:|']
    for key in ['left_hand_kg','right_hand_kg','left_shin_kg','right_shin_kg','torso_mass_difference_kg','com_vector_difference_cm','friction_difference']:
        d=comparative[key];text.append(f"|{key}|{d['raw']:.4f}|{d['new']:.4f}|")
    text += ['',f"同一世界最常见类别覆盖其16个窗口的平均比例为 {summary['mean_modal_window_fraction']:.2%}。重载(总负载≥8kg)进入nominal类约占重载窗口 {summary['mean_heavy_nominal_window_fraction']:.4%}，仍非零。", '',
             '最终确认中，两折各64个类别的平均类内响应距离均低于随机配对；最弱类别分别约为随机的85.4%/89.3%。这不是对每对成员、每条motion的保证。按类检查仅包含DR世界，没有用大量零差异nominal成员抬高结果。探索阶段的nominal误路由类更弱，这也是追加因果平滑并另做冻结确认的原因。','',
             '## 为什么单纯按距nominal分档还不够','',
             '分档与已有监督的nominal距离目标一致，确实能区分偏离幅度；实测0.05宽度的档位约58%–60%预测正确、97%–98%在正确或相邻档。',
             '但半径相同并不表示响应方向相同。即使直接使用真实DR半径分档，原探针上的类内响应差异也只比随机缩小约3.5%–3.8%。因此本目标需要保留多维信息，半径不能单独代表环境相似性。','',
             '## 导出与运行','',
             '- `export/partition_k64.npz`：已验证fold0分类器，只需latent；64类，nominal类ID=1。绑定u35857 encoder SHA。',
             '- `export/environment_bank.npz`：所有已测世界的显式参数、类别、路由频率和拟合/留出标记，可作为后续专家的环境参数库。类别取16个选中窗口的因果路由众数，不保证每时刻不变。',
             '- `export/runtime_verification.json`：GPU实际推理、因果状态重放、物理session重置、无有效历史时不输出类别和checkpoint匹配检查。',
             '- `src/intact_tracking/latent_environment_partition.py`：独立PyTorch运行时；未接入任何正在运行的policy。','',
             '```python',
             'from intact_tracking.latent_environment_partition import LatentEnvironmentPartition',
             'partition = LatentEnvironmentPartition(',
             '    "runs/limb_context_20260916_latent_environment_partitions/export/partition_k64.npz",',
             '    encoder_sha256=encoder_checkpoint.sha256,',
             ').to(device)',
             'decision = partition.route_causal(',
             '    latent, elapsed_control_steps=100, valid=full_history_mask,',
             '    reset=physics_session_changed_mask,',
             ')',
             'class_id = decision["class_id"]  # 未有任何有效历史时为 -1',
             '```','',
             '平滑仅在固定DR、每100控制步查询的场景验证。更频繁查询、DR突变的适应延迟、专家训练和闭环控制收益尚未验证。本次完成的是分组方法、环境库和离线/运行时一致性验证。','',
             '详细数据：`summary.json`、`dynamics_evaluation/`、`blends/selection.json`、`fresh_validation/`、`fresh_ema_validation/`、`causal_ema/`。所有mjwarp原始状态和特征保留。']
    (ROOT/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary['fresh_response'],indent=2),flush=True)


if __name__=='__main__':main()
