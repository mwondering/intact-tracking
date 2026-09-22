# RMA student 的训练 reward 为什么高于 teacher

2026-09-22。主要区别是 teacher PPO 训练采样高斯动作，student 在线蒸馏执行确定性动作；同时两套训练日志的 episode 统计窗口不同。训练曲线不能直接作为 student 优于 teacher 的证据。

核对 reward 定义及权重、termination、500 步 episode、起始状态规则、uniform sampling、完整过滤数据 manifest，均一致；四 rank 的 DR 物理 prototype 与负载样本指纹也一致。Teacher 的 timeout bootstrap 只写入 PPO 内部 rewards 副本，没有混入记录的环境 reward。

## 训练日志的证据

以下对每个 update 的日志做算术平均：student 更新 798–897，teacher 更新 10206–10305，各 100 次更新。它们不是相同初始状态的配对轨迹。

| 指标 | Teacher PPO | Student 蒸馏 |
|---|---:|---:|
| `Train/mean_reward` | 112.01745 | 118.65998 |
| `Train/mean_episode_length` | 497.13438 | 490.30966 |
| `Episode_Reward/action_rate_l2` | -0.72752 | -0.21325 |
| `Episode_Reward/waist_action_rate_l2` | -0.12512 | -0.06550 |
| `Episode_Reward/joint_pos_tracking` | 0.86645 | 0.85623 |
| `Metrics/motion/error_joint_pos` | 0.49894 | 0.54958 |
| `Metrics/motion/error_body_pos` | 0.03856 | 0.04732 |
| `Metrics/motion/error_anchor_pos` | 0.21947 | 0.24793 |

Teacher 这段训练的平均动作 std 为 `0.18608`，`PPO.act()` 使用 `stochastic_output=True`。Student 使用 `actor(obs)`，执行确定性总动作。原 checkpoint 的 `action_rate_source=sampled`，因此 teacher 的探索噪声进入 action-rate 惩罚。Student 动作变化惩罚明显减小，与总 reward 更高相符；同期的关节/body/root tracking 误差并未更好。上述窗口差异意味着这些日志不能精确分解噪声造成的因果贡献。

`Train/mean_reward` 的统计口径也不同：teacher 的 RSL logger 每卡保留最近 100 个已完成 episode，四卡合并最多 400 个；student 对当前 24 步 rollout 内所有已完成 episode 做四卡汇总。奖励都是实际 episode 累计 return，但样本窗口和权重不同。

Student 的 tracker 和 residual controller 已继承固定 teacher `checkpoint_9000.pt`（9001 次 PPO 更新）。Student 横轴从 0 开始，仅表示 adaptation 的监督更新，不能与 teacher 从零训练时的同轮数作预算比较。蒸馏仅优化 64 维 embedding MSE，reward 只用于观察行为。

## 同样执行确定性动作的配对评测

使用已经完成的 student `checkpoint_500.pt` 与固定 teacher9000：256 条 motion × 2 个环境，冻结 tracker 预热 500 步、查询最多 500 步。motion、起始状态、DR、推力序列等配对检查通过。Student 查询开始清空历史并保留当前帧。

| 场景 | Teacher 平均 episode return | Student 平均 episode return | Teacher / Student 成功率 |
|---|---:|---:|---:|
| Nominal | 98.90911 | 98.77777 | 100% / 100% |
| Mixed：52 nominal + 460 HDR | 95.60618 | 95.14121 | 99.6094% / 99.6094% |

这是各自完整查询的 return，包含失败后提前结束的影响；motion 剩余时长不同，因此也不能将这里的绝对 return 与固定 500 步训练 episode 的 return 直接对照。这次统一动作执行方式的评测中，student reward 接近且略低于 teacher。

## 记录与修复

- [本次诊断数据](../runs/144000-exp-heavy/baselines/rma_student_uniform/reward_diagnosis_20260922.json)
- [第 500 轮完整配对报告](../runs/144000-exp-heavy/baselines/rma_student_uniform/evaluations/update_000500/REPORT.md)
- [配对审计、误差与置信区间](../runs/144000-exp-heavy/baselines/rma_student_uniform/evaluations/update_000500/summary.json)
- 动作采样：[residual_policy.py](../src/intact_tracking/residual_policy.py)、训练调用：[heavy_rma_student_train.py](../src/intact_tracking/cli/heavy_rma_student_train.py)
- Teacher 四卡日志汇总：[memory350_policy_train.py](../src/intact_tracking/cli/memory350_policy_train.py)

排查时发现原始 8 组评测已经完成，但 watcher 缺少 `evaluation_source_sha256.json`，导致最终报告生成失败。已修复自动记录，并重新生成报告、恢复 watcher；训练未停止。第 500 轮代码指纹如实标为结果生成后补记，不能视作评测开始前的快照；后续评测在开始前记录并核对指纹。另修复 watcher 在耗时评测后使用旧 progress 时间而误报停滞的问题。
