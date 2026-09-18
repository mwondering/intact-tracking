"""Summarize both held-out motion/DR directions without selecting a winning split."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('runs/limb_context_20260916_motion_metric_probe')
reports=[json.load(open(P/f'diagnostics{s}.json')) for s in ('','_reverse')]
rows=[]
for scenario in ['load','mass','com','friction','random_full','random_load']:
 for method in ['old','new','new_targeted','old_recalibrated']:
  for k in [8,16,32]:
   rr=[r for rep in reports for r in rep['results'] if (r['scenario'],r['method'],r['k'])==(scenario,method,k)]
   if not rr:continue
   row={'scenario':scenario,'method':method,'k':k,'distance_reduction_mean':float(np.mean([r['overall_distance_reduction'] for r in rr])),'heavy_nominal_fraction_mean':float(np.mean([r['heavy_nominal_fraction'] for r in rr])),'distance_reduction_each_direction':[r['overall_distance_reduction'] for r in rr],'heavy_nominal_fraction_each_direction':[r['heavy_nominal_fraction'] for r in rr]};rows.append(row)
(P/'two_direction_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
with (P/'two_direction_summary.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['scenario','method','k','distance_reduction_mean','heavy_nominal_fraction_mean'])
 for r in rows:w.writerow([r[key] for key in ['scenario','method','k','distance_reduction_mean','heavy_nominal_fraction_mean']])
def get(sc,method):return next(r for r in rows if r['scenario']==sc and r['method']==method and r['k']==16)
lines=['# Motion 实测结论：新四组标签尚不适合直接替换旧标签','',
 '已完成 mjwarp GPU 配对回放，没有启动 encoder 或 PPO 训练。旧 baseline/transformer 已分别在 u4342/u1774 保存退出，checkpoint_final.pt 保留。测试结束 GPU 0–7 均空闲。','',
 '24条motion、15个动作家族、每条4个历史窗口，共96个anchor；每个anchor回放186种物理配置，共17856条10-step轨迹（0.2秒）。使用冻结tracker先前实际产生的物理PD目标，相同状态/动作跨全部DR配置重放。包含逐项扫描、64组随机四类DR、64组仅随机负载。encoder bias、armature不随机化；没有外力扰动。','',
 '新指标严格使用：四肢相对root位置与相对运动速度（扣除root平移和旋转）；root位置/线速度；root相对姿态旋转向量/角速度；不加接触条件的左右脚水平位移/速度。新标签四组等权，组内位置/速度等权。除四肢相对root外，root误差使用与旧实现相同的世界方向；足部切向取世界水平面。','',
 '旧指标为当前response10的70维物理状态响应，使用u15000归一化统计。新标签尺度由训练motion的nominal单步变化RMS拟合，每块一个尺度，位置/角度下限0.001、速度/角速度下限0.01。另有旧信号重新标定的控制组。没有根据测试结果调整这些设置。','',
 '分成两组不重叠motion家族，交换训练/测试方向重复。随机组合测试还把64组DR分成不相交32/32，交换方向；nominal是共同参考例外。聚类只读响应向量，不读DR参数。这里评价的是响应标签空间，未经过latent训练，也未应用距离标签的饱和映射；不能把数值直接当作训练后latent表现。','',
 '## 单项指标是否保留物理区分能力','',
 '以下16类；分别仅改变该类DR。旧指标使用完整旧响应，新指标只使用提议对应的那组信号。百分比表示同簇参数差异比随机配对缩小多少，越高越好；是两次方向平均，不是准确率。','',
 '|DR变化|旧完整响应|新对应分组|判断|','|---|---:|---:|---|']
for sc,title,verdict in [('load','四肢负载','未改善'),('mass','躯干质量','小幅、非两折一致的参数紧凑性收益；强弱分离两折均改善'),('com','COM偏移','明显丢失参数结构'),('friction','摩擦','未改善')]:
 lines.append(f'|{title}|{get(sc,"old")["distance_reduction_mean"]:.1%}|{get(sc,"new_targeted")["distance_reduction_mean"]:.1%}|{verdict}|')
lines+=['','## 多项随机变化，测试未见DR和未见motion','', '|场景|旧同簇距离缩小|新同簇距离缩小|旧重载进入nominal|新重载进入nominal|','|---|---:|---:|---:|---:|']
for sc,title in [('random_full','四类DR同时随机'),('random_load','仅四肢负载随机')]:
 old,new=get(sc,'old'),get(sc,'new');lines.append(f'|{title}|{old["distance_reduction_mean"]:.1%}|{new["distance_reduction_mean"]:.1%}|{old["heavy_nominal_fraction_mean"]:.1%}|{new["heavy_nominal_fraction_mean"]:.1%}|')
lines+=['','随机测试中重载指四肢附加质量合计≥8kg。随机的同簇距离按四类参数范围归一化后等权；相同DR的不同motion样本允许配对，属于需要聚到一起的真实相同参数。这里的参数接近与之前对“强动态作用接近”的目标仍有区别，重量大不保证当前窗口响应一定大。','',
 '## 对原假设的修正','',
 '固定motion逐项增加负载时，旧指标和新指标都能体现负载增大。主测试方向：旧指标左右手单调窗口比例均100%，左右小腿约90%/94%；新指标约100%/100%/90%/98%。两者负载幅度Spearman中位数均1，强扰动响应均明显高于重复nominal的数值噪声。因此不能再将负载聚类差直接解释为10步内负载尚未起作用。','',
 '同一负载在不同motion下的响应幅度和方向会变化。主测试中，左小腿0.8kg旧响应幅度P10–P90约0.031–0.317，4kg约0.122–1.004，有明显重叠；新指标也有类似重叠。运动激励和响应距离表达方式需要继续处理。','',
 'COM对关节调整、线性运动等状态也有作用，仅保留root姿态和角速度会丢弃旧标签中可用信号。摩擦只用足部位置和速度且无接触条件，本批motion下也未获得更好的参数分组。这是此次定义、窗口、归一化与motion集合的实测结论，不是证明所有body指标设计都无效。','',
 '建议保留躯干质量分支继续诊断；不要直接用这版四组标签替换旧标签训练。先解决跨motion响应可比性，并给COM等分组补回实际有辨识力的信号。未在本次结果上继续调权重挑最优结果。','',
 '## 验证与产物','',
 '独立核对了全部配置的实际质量、COM、14个足部摩擦值；全部初始状态一致到1e-6量级；用scipy独立重算4356个旋转差；重算132种配置的物理距离和重载混入统计。完整检查见verification.json。重复nominal存在GPU接触求解数值差异，响应噪声在analysis.json/diagnostics.json中逐项报告，没有将其当作DR信号。','',
 '- simulation.npz/json：全部物理状态、body特征、motion与DR配置。','- physics_audit.npz：实际设置的物理参数。','- responses.npz：新旧响应向量及尺度。','- diagnostics.json、diagnostics_reverse.json：两次独立划分方向。','- two_direction_summary.csv/json：全部8/16/32类对照。','- cross_motion_amplitudes.json：轻重负载跨motion幅度重叠。','',
 '![两方向平均](targeted_comparison.png)','']
(P/'FINDINGS.md').write_text('\n'.join(lines))
fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(4)
for method,shift,label in [('old',-.18,'Old full response'),('new_targeted',.18,'New corresponding group')]:
 vals=[100*get(sc,method)['distance_reduction_mean'] for sc in ['load','mass','com','friction']];axes[0].bar(x+shift,vals,.36,label=label)
axes[0].set_xticks(x,['Limb load','Torso mass','COM','Friction']);axes[0].set_ylabel('Within-cluster parameter distance reduction (%)');axes[0].set_title('One DR family varied, K=16');axes[0].legend(fontsize=8)
for method,shift,label in [('old',-.18,'Old full response'),('new',.18,'New four-group response')]:
 vals=[100*get(sc,method)['distance_reduction_mean'] for sc in ['random_full','random_load']];axes[1].bar(np.arange(2)+shift,vals,.36,label=label)
axes[1].set_xticks([0,1],['Random four-family DR','Random limb loads']);axes[1].set_ylabel('Within-cluster parameter distance reduction (%)');axes[1].set_title('Unseen motion families + unseen DR, K=16');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(P/'targeted_comparison.png',dpi=180)
print('\n'.join(lines[14:27]))
