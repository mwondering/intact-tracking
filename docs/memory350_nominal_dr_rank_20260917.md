# Memory350：nominal 响应距离 + DR 环境中心排序

新增入口：`python -m intact_tracking.cli.forward_memory_nominal_dr_rank_train`。
在现有 nominal-direction / response10 版本上增加一项 DR 参数监督，保留其余六项 loss、Memory350 网络以及推理接口。

## 新增监督的含义

设两个环境对的 DR 参数距离满足 `d_theta(i,j) < d_theta(k,l)`，则软约束环境中心满足
`d_z(i,j) < d_z(k,l)`。可以比较共享一个环境的两对，也可以比较不共享环境的两对。
新项学习相对远近，不把参数距离映射为指定的 latent 绝对距离，也不使用固定负样本 margin。

因此，两个中心都可以接近 nominal，同时在附近保持差异。例如，两者都距 nominal 0.2，
彼此距离可以为 0.1 或 0.3；新项不会要求它们必须相距 1.1。
这是软排序偏好，并不保证所有参数排序与所有逐片段响应约束都能同时完全满足。

### 参数距离

复用 `memory350_dr_center_rollout.py` 中已实现的 38 维物理标签及固定范围归一化：

- 四肢附加载荷：4 维。
- 躯干 COM 偏移：3 维。
- 躯干相对质量、共享足底摩擦：各 1 维。
- 关节 armature scale：29 维。

归一化为 `x_k = (theta_k - lower_k) / (upper_k - lower_k)`。
四个负载、三个 COM 轴、质量、摩擦各占一个因子，29 个 armature 共同占一个因子；
十个因子等权，每个因子内部取均方。因此：

`d_theta(i,j) = sqrt(sum_k w_k * (x_ik - x_jk)^2)`。

使用采样范围而非当前 batch 的统计量。确切的维度、范围、权重写入 `dr_metric_schema`。
不重复计入由负载确定的惯量/COM；encoder bias 和时变外力沿用旧 DR 标签版本的排除规则。
这项监督使用的是这 38 维物理度量，并非全部 67 个随机采样坐标的无权欧氏距离。

### 环境中心

沿用当前跨 motion 弱正样本的原始历史归档。有效样本必须满足：

1. 当前片段与归档片段属于同一个 DR world / physics session。
2. motion 不同，两段均有完整的 short50 + long30×10 历史，且交互不重叠。
3. 使用当前 encoder 重新编码两段历史，两端均参与梯度计算。

先分别单位化 latent，再平均；同一个微批内重复出现的 world/session 合并：

`c_i = mean_rows((normalize(z_current) + normalize(z_archive)) / 2)`。

平均后不再次单位化，`d_z(i,j) = ||c_i - c_j||_2`。
这是当前采样片段对环境中心的估计，不是该环境全部 motion 的精确均值；没有缓存旧 encoder 的 latent。
nominal 样本不参与新排序项，继续由原有固定方向与响应 loss 监督。
物理参数变化会使旧历史和跨 motion 归档失效，避免混合两个不同环境的中心。

### 排序 loss

在每个 GPU 的每个微批内枚举有效 DR 中心的无序对，按 `d_theta` 排序。
用排序索引间隔 `1, 2, 4, ...` 比较近邻和远邻排序，避免枚举所有环境对之间的组合。
若共有 P 个环境对，则比较开销为 `O(P log P)`；不跨 GPU 收集 latent。

对于物理距离较小的环境对 a 和较大的环境对 b，定义：

```text
gap = d_theta[b] - d_theta[a]
weight = gap                  if gap > min_gap else 0
L_rank = sum(weight * tau * softplus((d_z[a] - d_z[b]) / tau)) / sum(weight)
```

参数距离相同或过于接近时不规定顺序。没有有效比较时返回可反传的零，记录比较数为零。
`min_gap` 是参数距离差的筛选阈值，`tau` 是软排序温度；二者都不是 latent 的最小间隔。
softplus 在排序正确后仍有较小梯度，因此它与原有 loss 的相对强度仍需由实验评估。

## 七项 loss 与默认系数

| 项目 | 总系数 |
| --- | ---: |
| 五步 teacher-forced prediction | 1 |
| 五步 recursive prediction | 0.5 |
| 同 world/episode/motion 的 ±5 步局部正样本 | 0.01 |
| 自身十步 A/B 响应监督单位 latent 到固定 nominal 方向的距离 | 0.04 |
| nominal 固定方向 | 0.01 |
| 同 DR、跨 motion 弱正样本 | 0.008 |
| **DR 参数距离 → 环境中心距离排序** | **0.002** |

A/B 标签和目标公式保持不变：`target = 2 * RMS(A-B) / (RMS(A-B) + 0.3)`，
其中 RMS 使用原有归一化后的十步、70 维物理状态差。A/B 项仍按片段计算，没有改成环境均值。
旧固定 margin 弱负样本仍关闭。

新增参数：

```text
--dr-center-rank-weight 0.002
--dr-rank-temperature 0.1
--dr-rank-min-gap 0.01
```

将新权重设为零可以关闭新项梯度。DR 参数仅作为训练标签，不拼接进 encoder 或 predictor，
policy 仍只需要原有 history-only encoder。

## 训练与恢复

新入口沿用每卡约 10% 完全 nominal、其余环境每个 DR 坐标独立 50% nominal / 50% 范围采样的设置；
默认四肢最大负载为 `[2.5, 2.5, 4, 4]` kg。正式训练仍默认持续到用户停止。
卡数与环境数由 torchrun 和 `--num-envs` 指定，未在本次实现中自动切换现有训练任务。

从头训练时使用新入口和新的输出目录；例如在原有 4 卡、8192 环境配置中替换模块名为：

```text
intact_tracking.cli.forward_memory_nominal_dr_rank_train
```

其余训练参数可保持原配置，包括 `--num-envs 8192 --nominal-fraction 0.1`
和 `--dr-nominal-probability 0.5`。建议初始 `--warmup-steps 1000`，
以积累跨 motion 历史；warmup 会继续等待，直到每个 rank 的训练/验证 replay 均至少有三个有效 DR world。

如果明确选择从已有 nominal-direction checkpoint 增加新项，使用 `--resume ... --resume-new-stage`
及独立输出目录，生成含 DR 标签的新验证探针；默认要求原有六项 loss 完全一致。
需要同时调整权重时，显式增加 `--resume-retune-weights`：仅允许调整 A/B 距离项的
`--representation-relation-weight`、nominal 的 `--nominal-anchor-weight` 和
新增项的 `--dr-center-rank-weight`。局部正样本、跨 motion 正样本、预测损失、目标距离映射、
排序温度与参数度量均继续严格检查；新阶段记录 `new_stage_loss_changes`。
例如保留 `--representation-weight 0.01`，将 `--representation-relation-weight` 从 4 改为 8，
只把 A/B 距离项总系数从 0.04 改为 0.08，不改变局部正样本系数。
同版本普通恢复严格检查所有 loss、参数度量和采样归档配置，并恢复原始 nominal 锚点。
checkpoint 保存 `dr_center_rank_supervision` 与 `dr_metric_schema`；`run_config.json` 保存完整排序约定。

关注 `dr_center_rank_loss`、`dr_center_rank_weighted_loss`、`dr_center_rank_worlds`、
`dr_center_rank_pairs`、`dr_center_rank_comparisons`、`dr_center_rank_accuracy`、
`dr_center_rank_tie_fraction`。没有比较的微批不能把零 loss 解读为已经学会排序。
这些指标在原有训练及固定验证探针中记录；多卡汇总的数量是各 rank 平均值。

## 验证

相关四组测试共 66 项通过；新增 14 项覆盖：远近排序及梯度、固定 nominal 半径下的可区分性、
重复 world 聚合、单位化、无有效比较、标签不进入 predictor、原有六项 loss 保持一致、
恢复配置约束、十步标签与 DR 标签对齐、跨 motion 历史不重叠及物理参数变化失效。

真实 GPU 短程训练与恢复的验证产物位于
`runs/memory350_nominal_dr_rank_20260917_check/`；这仅验证实现链路，不代表训练收敛或下游 policy 已改善。

使用空闲 GPU 4、5，每卡 128 环境，完整 Memory350 网络。先训练两个 update，
再从 `update_000001.pt` 恢复至 update 3，保留原固定验证样本和归一化。
恢复前后 nominal 锚点逐项一致，两卡模型参数一致；新 checkpoint 可由现有 history-only 推理入口加载。
`verification.json` 的 46 项产物核对全部通过，包括逐项加权和与两卡实际探针的有效中心。

恢复后的 update 3：训练排序 loss 为 0.113454，乘系数后为 0.000226909，
每卡平均 193 个有效排序比较；固定验证排序 loss 为 0.101062，每卡平均 85 个比较。
验证脚本为同目录的 `verify.py`，启动参数记录在 `launch.json`、`resume_launch.json`。
短训采用有上限的 smoke 配置；当前正式四卡任务没有切换，也没有增加训练上限。
