# Residual PPO 早期短 episode 诊断（2026-09-19）

对象：`runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919`。训练从零初始化 residual actor、critic 和 optimizer，entropy=0.005、初始 std=1.0、无界 residual、DR auxiliary coefficient=0.05，仅监督 COM xyz 和摩擦。正式训练仍在继续，没有更新次数上限。本次诊断没有修改正式训练参数或重启训练。

现有证据最支持：**高探索噪声与作用于采样动作的强平滑惩罚组合，使正常跟踪也得到持续负回报；提前失败减少后续负回报，而无界 residual 允许策略大幅偏离冻结 tracker。** 这不是完整训练消融后的唯一因果结论，也不能由早期 checkpoint 推断最终收敛情况。

## 1. 训练中实际发生了什么

| 完成 PPO 更新数 | 平均 episode 步数 | 平均 episode 总奖励 | 动作 std | residual 均值 RMS |
|---|---:|---:|---:|---:|
| 16 | 310.98 | -194.26 | 0.974 | 0.207 |
| 31 | 130.59 | -70.73 | 0.954 | 0.385 |
| 63 | 13.43 | -6.67 | 0.912 | 0.844 |
| 101 | 11.68 | -4.31 | 0.826 | 1.057 |
| 198 | 8.92 | -1.64 | 0.664 | 1.448 |

后续主训练**没有改变配置**：u310 时平均 episode 已回升至 19.32 步、平均回报 +0.315；u335 时进一步回升至 53.76 步、+2.899；u349 达到 185.25 步、+12.761，std 为 0.401。因此这是早期明显退化及随后开始回升的记录，不能称为已经不可恢复的训练崩溃。

继续监控至 u402，平均 episode 为 474.26 步、回报 +72.79，std 为 0.311。u401 周期 checkpoint 的 actor/critic 参数均有限，53 个冻结 tracker tensor 与初始版本一致，88 个未监督 DR 输出行仍完全不变，4 个受监督输出行发生学习；entropy、初始 std、辅助权重、无界 residual 和无更新上限的配置均通过复核。详细趋势见 [持续监控记录](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/monitor/recovery_trend.json)。这仍不是与冻结 tracker 的后期性能对照。

奖励总和变得接近零，同时 episode 急剧缩短、残差增大。因此不能把总奖励上升或 value loss 下降解释为跟踪能力改善。现有检查未发现 NaN/Inf、训练进程退出、冻结 tracker 权重改变或 nominal 物理参数改变。

`Episode_Reward/*` 在 MJLab 中记录的是 episode 累积加权奖励除以固定的最大 episode 时长（本实验为 10 秒），并非除以实际存活时间。episode 从 311 步缩至 9 步，会使许多正负奖励分项同时趋近零。该日志方式与 tracker 保持一致，不能仅凭这些曲线判断每步控制质量。

u101 的早期确定性诊断使用 16 条训练 motion × 2 次 native DR，共 32 个 episode：冻结 tracker 失败 0/32，residual 失败 32/32；覆盖率分别为 100% 和 5.97%。只比较每对共同有效时段，局部 body 位置误差为 0.02074 → 0.11964 m。参数、查询初始状态和共同计分时段的推力配对校验通过；两 GPU 的预热仿真轨迹不逐位一致。此结果证明该早期均值策略在这个小样本上变差，不代表未见 motion 泛化或最终训练效果。

数据：[训练日志](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/ppo_8gpu8192/metrics.jsonl)、[配对报告](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/diagnostics/update_000101_short_episode_force_prefix/REPORT.md)。

## 2. 是否预测损失压过 PPO

在线日志统计的是每轮首个 minibatch 在最后共享**激活**上的梯度，已经包含 0.05 系数。截至 u198，有效记录的 aux/PPO 比例最高约 1.98%，u198 为 0.35%。激活梯度的范数不能替代共享**参数**梯度的范数，因为不同样本回传到同一参数时会叠加或抵消。

因此另取 u201 checkpoint，独立构建 2048 个 native DR 环境，冻结 tracker 预热 384 步、当前随机策略运行 128 步，然后采集 24 步，共 49152 个真实 on-policy transition。该批量等于正式训练单 rank 一个 minibatch 的样本数。保持参数不变，分别求 PPO 与加权 DR loss 的梯度：

| 范围 | PPO 梯度范数 | 加权 DR 梯度范数 | DR / PPO |
|---|---:|---:|---:|
| 全部共享参数 | 1.48548 | 0.09576 | 6.45% |
| 最后共享隐藏层权重 | 0.82310 | 0.08944 | 10.87% |
| latent/action 输入权重 | 0.21489 | 0.00378 | 1.76% |
| 最后共享激活 | 0.00505 | 0.0000152 | 0.30% |

全部共享参数的梯度余弦为 -0.0030，方向近乎正交。该样本没有预测梯度在数量级上压过 PPO 的证据。它是独立本地 rollout、裁剪前且未跨 rank 平均的梯度，不是正式训练所有 minibatch 或所有历史更新的测量；也不排除辅助目标对学习方向有较小影响。

初始化时 action 输出层为零，所以**第一个 minibatch 的共享 trunk PPO 梯度本来就是零**，预测头可先为 trunk 提供梯度；此时比值无定义，不能拿在线日志的 0 当作有效比值。PPO 仍可更新 action 输出层。之后的有效在线记录以及 u201 参数梯度没有持续压制现象。

数据：[完整梯度测量](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/diagnostics/gradient_and_noise_000201/current201.json)。

## 3. 零更新噪声对照：回报符号翻转

读取同一个初始 checkpoint（残差均值为零），采用相同 seed、motion 清单、native DR 构建和预热设置，只在独立副本上改变 std；每组 2048 环境 × 24 步，**不执行任何 optimizer step**。因此此处奖励差异无需通过辅助损失或 PPO 参数更新产生。

| 每个控制步的统计 | std=1.0 | std=0.25 |
|---|---:|---:|
| 总奖励均值 | **-0.67952** | **+0.18148** |
| 负奖励比例 | **99.9756%** | **0.0977%** |
| 全部正奖励项之和 | +0.17949 | +0.24160 |
| action_rate_l2 | -0.30745 | -0.02166 |
| waist_action_rate_l2 | -0.43696 | -0.02691 |
| waist_joint_acc_l2 | -0.08507 | -0.00564 |

当前普通 action-rate 权重为 0.3，腰部 action-rate 权重为 -5.0，二者都对采样动作的变化计罚。采样噪声方差随 std 平方增加；std 从 0.25 提到 1.0 时方差增加 16 倍。本次对照中，仅两个 action-rate 项就从每步约 -0.0486 增至 -0.7444，远大于全部正奖励。两次模拟并不声称逐位一致，但 checkpoint、实验配置和仅改变 std 的条件均已记录。

回报为负时，提前终止可以成为 PPO 的有利行为。例如恒定负奖励 r、折扣 gamma 的长度 T 回报为 `r * (1 - gamma**T) / (1 - gamma)`；T 越短，损失越少。当前真实失败切断未来回报，并没有专门补偿“放弃未来负回报”的失败惩罚。训练中同时出现“episode 变短、总奖励提高、残差增大”，与这一机制吻合。该机制的训练级因果份额仍需同种子消融确认，不能用两组零更新 rollout 完全证明。

原 tracker 的 144000 checkpoint 实际 std 均值为 0.23645；std=1.0 是新的探索初值，不能把两种噪声下相同奖励权重的效果直接视为等价。当前奖励函数与该 source checkpoint 保持一致，未找到 action-rate 移植错误。source checkpoint 的 algorithm 和 task overrides 没有设置 `clamp_rewards_min`，原类默认 None；不能依据原仓库后来 YAML 中的 0.0 声称此处漏掉了原训练使用的奖励裁剪。

数据：[std=1.0](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/diagnostics/gradient_and_noise_000201/initial_std1.json)、[std=0.25](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/diagnostics/gradient_and_noise_000201/initial_std025.json)。

## 4. 修复方向与本次改动范围

优先检验奖励尺度与探索噪声的兼容性。若保留 std=1.0、entropy=0.005 和无界 residual，可对两个 action-rate 惩罚共同改用确定性策略均值，或给强平滑惩罚设置训练初期的权重调度；再检验失败终止是否仍提供回报上的捷径。这些会改变训练目标，应单独记录和对照，不能伪称“与原 tracker 完全一致”的代码修复。当前没有依据先将 DR loss 大幅降权。

补充进行了同轨迹奖励重算：初始 checkpoint、std=1.0、2048 环境 × 24 步，在**完全相同的采样动作、物理状态和终止事件**上，仅将两个 action-rate 项改算为策略均值之差。实际回报均值 -0.67902，重算后 -0.07268；负奖励比例由 99.9736% 降为 70.2271%。均值动作 action-rate 与腰部 action-rate 分别为 -0.05518、-0.08239 / 步。原采样奖励已从动作历史独立重构并核对。

这说明直接惩罚采样噪声贡献很大，**但仅将平滑罚切换到均值仍不足以让初始高噪声策略的平均回报转正**，物理加速度、跟踪误差和均值本身的变化仍有代价。因此没有将这个候选方案直接上线；其训练效果也尚未验证。主训练既然已出现持续回升，目前保留原配置继续观察。

数据：[同轨迹奖励重算](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/diagnostics/initial_std1_mean_action_counterfactual/result.json)。

本次只修复了配对评测比较器：某策略提前全部失败时，两个运行长度不同，原本比较完整推力 hash 会误拒绝。现在记录累计 hash，并严格校验覆盖全部共同计分时段的推力前缀；旧结果若长度不同且没有前缀 hash，仍要求重新评测。5 项定向测试及真实 GPU 配对重跑通过。梯度探针也在三个真实 checkpoint 副本上完成，参数摘要一致、optimizer step=0；u201 重算 log-prob 最大误差约 1.14e-5。

正式训练持续进行，仅保留短 episode 预警；上述诊断没有降低探索、恢复 residual 限幅、修改 auxiliary 权重或修改奖励。

std=0.25 的副本只用于检验噪声对奖励符号的影响，不是对正式训练的修改，也不意味着应当用降低探索来解决问题。强探索是否产生有效学习，还应由更长训练趋势、DR/nominal 跟踪成功率及正确/置零/打乱 latent 的闭环对照判断。
