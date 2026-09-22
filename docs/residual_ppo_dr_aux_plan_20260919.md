# Residual actor 共享隐藏层 DR 辅助预测（2026-09-19）

> **已纳入最新定稿 Pipeline（2026-09-20）。** 本文保留实现阶段的设计与验证记录；最终采用配置、训练状态与产物见 [定稿文档](final_pipeline_20260920.md)。

状态：共享隐藏层、辅助头、PPO 损失、rollout 标签和 W&B 指标已实现。新的 proprio122 native 训练入口默认启用。entropy=0.005、初始 std=1.0、无界 residual 保持生效；辅助监督仅保留 COM x/y/z 和摩擦，质量、Kp、Kd、armature 均不参与，仿真环境中的这些 DR 保留。现有训练进程不会自动切换到新代码。

共享 residual actor 的最后一个 128 维隐藏层，分别连接现有 29 维无界动作头和独立的 92 维线性 DR 预测头：

\[
h_t=f_\theta(h_{\rm tracker}(o_t),z_{t-4:t},a_t^{\rm tracker}),
\quad \Delta a_t=W_a h_t+b_a,
\quad \hat p_t=W_p h_t+b_p.
\]

共享的是可训练 residual actor 的隐藏层。辅助头的输入不能 detach，否则损失只训练读出器，无法影响策略表征。冻结的 tracker 和 context encoder 仍通过 detached 输入供给特征；DR 真值只作为监督标签，不进入 actor 的推理输入。critic 保持独立，不使用它的特权输入为这个辅助头提供捷径。推理动作不依赖辅助预测头。

辅助任务塑造共享表征的思路有 [UNREAL](https://arxiv.org/abs/1611.05397) 等工作的依据，但这里的结构、监督目标与权重是针对本仓库的设计，训练效果尚待对照实验验证。

**标签和重点**

当前 native schema 恰好为 92 维，顺序如下。按参数名称绑定维度并校验 schema，避免只依赖位置常量。

| 标签 | 0-based 索引 | 默认组权重 |
| --- | --- | ---: |
| COM x | 0 | 1 |
| COM y | 1 | 1 |
| COM z | 2 | 1 |
| 躯干相对质量扰动 | 3 | **0，已确认排除** |
| 足部摩擦系数 | 4 | 1 |
| Kp，29 关节 | 5–33 | **0，已确认排除** |
| Kd，29 关节 | 34–62 | **0，已确认排除** |
| Armature，29 关节 | 63–91 | **0，已确认排除** |

保留 92 维接口便于复用参数名称与范围，但默认只监督 COM 三轴和摩擦对应的 4 个输出。质量和三个 motor 组在误差计算前即被排除，不计算损失、不计入误差统计，对应输出不能当作有效参数估计使用。当前有效组权重合计为 4。

每个标签直接由实际仿真物理参数读取，包括 nominal world 恢复后的参数值，并按 context checkpoint 中 native schema 的固定 DR 上下界归一化。初始化时校验环境的名称、顺序和范围一致。已有 `dr_metric` 乘过 `sqrt(coordinate_weights)`，因此不直接用于回归标签，避免重复加权。

\[
L_{DR}=\frac{1}{B}\sum_{i=1}^B q_i
\frac{\sum_g w_g\,\operatorname{mean}_{j\in g}(\hat p_{ij}-\tilde p_{ij})^2}{\sum_g w_g},
\qquad L=L_{PPO}+0.05L_{DR}.
\]

PPO 中包含系数为 0.005 的 entropy 奖励，即最小化损失中的 `-0.005 * entropy`。辅助系数默认 0.05，通过 `--dr-aux-coef` 调整；`--dr-aux-motor-weight` 控制三个 motor 组各自的权重，默认 0，launcher 也显式传入 0。只有显式设为正值时才重新监督 motor 组，并在各 29 维组内部取均值。`--dr-aux-coef=0` 时不创建辅助头或存储辅助标签，便于消融。

历史权重 `q = (有效 short 步数 + 有效 long chunk 数 × 10) / 350`，限制在 `[0, 1]`。空历史权重为零；episode/motion 边界保留的有效 long memory 仍计入可用历史；物理参数失效后的旧 session 不计入。分母使用 minibatch 样本数，不除以权重之和，因此全批次历史不足时确实降低辅助损失。标签和历史权重以独立 observation 字段保存到 rollout，并随同一 minibatch 索引重排；它们不属于 actor 或 critic 的输入组。

动作头保留零初始化，DR 头使用独立随机种子和非零权重、0.5 bias，不改变动作、critic 或环境的 RNG 流。这样第一步辅助梯度即可到达共享层。动作 rollout 不计算 DR 输出头；PPO 更新直接复用当前 actor forward 的隐藏层，不重复采样动作。

多卡使用标准平均梯度；即使某个 rank 的历史全为空，该 rank 也保留零辅助梯度路径。指标先汇总各卡误差和样本权重，再计算全局均值，避免平均各卡误差均值造成偏差。

**判断标准**

当前 [u7179 参数读出报告](../runs/144000_exp/evaluation_proprio122_u7179_parameter_decoding/REPORT.md) 的留出环境测试 R² 为 COM x/y/z：0.9773/0.9864/0.8871，摩擦：0.9609，支持优先尝试这些目标。读出精度表示参数信息可恢复，不等同于该参数对跟踪误差的因果影响大小。

共享层能预测参数，也不保证动作头会利用这些信息。需要同时检查：独立 DR world 上的逐参数 MAE/R²；打乱或清零 latent 对预测与动作的影响；匹配 motion、初态和 DR 的无辅助损失/有辅助损失跟踪结果。是否采用方案最终由跟踪误差、失败率和回报改善决定，不能只看训练集 DR loss 降低。

已接入 W&B 的 `AuxDR/*` 分组，横轴仍为 `completed_updates`，实验 group 保持 `144000_exp`；新 launcher 的 run name 带 `auxdr`：

- `loss`、`weighted_loss`、`coefficient`：原始辅助损失、乘系数后的损失及系数。
- `com_x/y/z_mse_normalized`、`friction_mse_normalized`：重点参数归一化 MSE；对应物理 MAE 使用 `com_x/y/z_mae_m`、`friction_mae_coefficient`。
- 默认不记录 Kp、Kd、armature 的预测误差；仅在显式启用 motor 监督时记录 `kp/kd/armature_mse_normalized`、`kp/kd/armature_mae_scale`。
- `history_weight_mean`、`valid_fraction`：平均历史权重、非零历史权重的样本比例。
- `shared_ppo_gradient_norm`、`shared_aux_gradient_norm`、`shared_gradient_ratio`：每次 update 第一个 minibatch 上，对最后共享隐藏层激活的梯度范数及其比值。辅助梯度包含 0.05 系数；这不是所有 trunk 参数的梯度范数。初始动作头为零时 PPO 对隐藏层的梯度也为零，此时比值记零且 `shared_gradient_ratio_valid=0`，不应解读为辅助梯度弱。
- `policy_kl`：本次 PPO update 内各 minibatch 的旧策略到当前策略 KL，固定学习率下也记录。

误差指标按历史权重加权、除以总历史权重；辅助训练损失按总样本数归一化。误差为 rollout 训练样本上的、优化过程中累计的误差，不是留出 world 的验证结果。零权重组不提供预测指标。

**代码、兼容性和验证**

- `src/intact_tracking/residual_dr_aux.py`：按名称绑定监督维度、范围归一化、分组损失、历史权重及实际环境标签。
- `src/intact_tracking/memory350_tracker_action_policy.py`：共享隐藏层、独立 DR 输出头、PPO 辅助项、梯度与误差统计。
- `src/intact_tracking/memory350_proprio_policy.py`：按 observation 附加标签和历史权重。
- `src/intact_tracking/cli/memory350_proprio_native_policy_train.py`：默认系数与配置；`scripts/run_144000_residual.py` 显式传入默认值。

新 checkpoint 保存辅助头及 optimizer 状态；推理时无需 DR 真值，使用 `actor.predict_dr(obs)` 可选读取归一化参数预测。关闭辅助任务时保留旧模型的 state-dict 键和初始化行为，旧 checkpoint 可按原配置评估或恢复。启用辅助任务改变了模型和 optimizer，不能作为旧 checkpoint 的严格原配置续训；应使用独立输出目录开始新实验。辅助任务开启状态、完整 schema 和权重保存于 run config，并参与恢复配置一致性检查。

初版相关回归测试共 **100 项通过**，覆盖监督梯度路径、质量零梯度、历史快照与失效、rollout/minibatch 对齐、完整 PPO 更新、checkpoint 恢复、无标签推理、双进程 CPU/Gloo 同步，以及 W&B 分组透传；另已读取实际 u7179 checkpoint，确认 92 维 schema 与分组。motor 权重清零后，**25 项相关测试通过**，进一步验证三个 motor 组的监督梯度为零、PPO 更新不会改变其输出头参数、零权重标签不影响损失且不会显示为有效误差指标。真实 checkpoint 审计脚本也已支持按保存配置附加辅助标签。尚未运行新的 GPU 训练对照，不能据此声称跟踪效果已提升。
