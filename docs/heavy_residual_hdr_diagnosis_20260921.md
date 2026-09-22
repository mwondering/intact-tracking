# Heavy residual HDR 差距诊断（2026-09-21）

已定位到的现象：从 checkpoint_2000 到 checkpoint_4200，latent 条件 residual 策略在同一 HDR 查询集上的跟踪精度退步，重负载时差距尤其明显。冻结 context encoder 的 checkpoint/hash 相同，变化发生在 residual PPO 的策略/价值/预测头训练及其状态分布中。目前仍不能把因果责任唯一归到预测损失系数、删除 COM 监督或 PPO 续训。

诊断复用前次评测的 256 条 motion、每条 2 个世界、seed=20260921。重新评测了父 checkpoint_2000 的 latent/baseline，并运行 latent 置零、配对交换和自身策略预热三个干预。每对比较均核对查询初始状态、物理/控制器参数、motion 起始帧和有效查询区间的外力序列；误差在共同有效前缀上比较。六次仿真评测及一次梯度探针均完成，未对正在训练的进程执行优化或修改参数。

**同一 HDR 条件下的训练变化**

| 指标 | Latent：2000 → 4200 误差变化 | Baseline：2000 → 4200 误差变化 |
|---|---:|---:|
| 关节角 L2 | +6.91% | -2.20% |
| 局部 body 位置 | +9.95% | -5.24% |
| body 姿态 | +16.00% | -3.96% |
| 全局 root 位置 | +7.01% | -2.57% |
| 全局 body 位置 | +7.72% | -3.03% |

Latent 五项误差增加的 motion-cluster bootstrap 95% 区间均大于 0；baseline 关节、局部位置、姿态三项下降区间均小于 0，全局位置变化区间跨 0。与此同时 latent 成功数从 506/512 增至 509/512，baseline 从 504/512 增至 510/512：精度变化与成功率变化不能混为一谈，也不构成 PPO 全面崩溃的证据。

在这次完全相同 HDR 配置的父 checkpoint 重测中，2000 的 latent 比 baseline 的关节误差低 3.46%、姿态误差低 6.43%；4200 时相对优势已经反转。

**Latent 干预**

以下均在同一 latent 4200 策略上进行；数值是相对于正常 latent 的误差变化，正数表示干预后更差。

| 干预 | 关节角 | 局部 body 位置 | body 姿态 | 全局 root 位置 | 成功数：原始 → 干预 |
|---|---:|---:|---:|---:|---:|
| HDR：latent 置零 | +51.41% | +40.85% | +58.62% | +42.41% | 509 → 503 |
| HDR：交换配对世界的 latent | +12.51% | +21.75% | +25.74% | +17.79% | 509 → 507 |
| HDR：用自身策略预热历史 | -1.32% | -2.41% | -2.32% | -1.63% | 509 → 510 |
| Nominal：latent 置零 | +138.49% | +163.77% | +283.61% | +155.47% | 512 → 509 |

交换实验仅在两个世界都仍有效、motion 相位一致且历史完整时执行，实际覆盖候选轨迹有效步数的 86.57%。置零是已训练条件策略的分布外干预，不能等同于从头训练的零 latent baseline。
这些结果说明该策略确实依赖 latent，正确匹配的 latent 也优于错误匹配的 latent；不能解释成“策略完全没学会使用 latent”。但这不证明该 encoder 提供了足以超过独立 baseline 的信息。自身策略预热能小幅改善 HDR，说明存在一定历史分布影响，但其幅度不足以解释原来约 6%–14% 的差距。独立运行的预热轨迹不是逐位一致，查询初始条件和外力已匹配。

**重负载分层**

对 HDR 世界按四肢质量除以各自上限后的平均值分成四分位。每组 128 个世界，下表比较 latent 4200 与 baseline 4200；这是分层描述，不是只改变质量的因果实验。

| 负载组 | 平均总附加质量 | 关节误差增加 | 局部位置误差增加 | 姿态误差增加 |
|---|---:|---:|---:|---:|
| 第 1 四分位 | 4.14 kg | -0.34% | +3.31% | +3.21% |
| 第 2 四分位 | 5.84 kg | +3.30% | +9.55% | +8.52% |
| 第 3 四分位 | 7.12 kg | +6.49% | +15.05% | +18.85% |
| 第 4 四分位 | 8.93 kg | +12.43% | +25.08% | +21.12% |

重负载下的条件控制能力是目前最明显的短板。在共同有效长度至少 450 步的固定 251 个世界中，300–450 步区间仍有约 9.7% 的局部位置误差差距，因此不能仅归因于查询开始时短历史为空。

**预测损失与 PPO 梯度**

对 checkpoint_4200 的独立 512 世界探针使用匹配的训练 DR，先运行自身随机策略 512 步，再采集 24 步 PPO rollout；不执行 optimizer.step。共享参数梯度范数：PPO=1.058025，加权辅助损失=0.010438，比值 **0.99%**，夹角余弦 0.0230。最后共享隐藏激活的梯度比为 **0.104%**。
训练日志 4102–4201 的最后共享激活梯度比均值为 **0.114%**。这些观测不支持当前存在辅助梯度碾压 PPO；它们不排除辅助目标在长时间累计训练中影响优化方向。参数探针只覆盖一个独立 batch，日志覆盖每次更新的第一 minibatch。

| 训练指标（各取 100 次更新均值） | 截至 2001 | 截至 4201 |
|---|---:|---:|
| 动作 std | 0.18034 | 0.26076 |
| Residual action RMS | 0.19675 | 0.25930 |
| 躯干 COM x MAE | 0.00970 m | 0.00818 m |
| 左手负载质量 MAE | 0.22663 kg | 0.21934 kg |
| 左小腿负载质量 MAE | 0.58320 kg | 0.55365 kg |

预测精度有所改善，HDR 动作跟踪却退步；预测物理参数与学会正确的控制补偿不是同一个训练目标。动作 std 没有继续塌缩，且错误 latent/零 latent 干预会明显损害性能，因此“没有探索”或“完全不使用 latent”都缺乏当前证据支持。

**仍未分开的原因**

2000 之后同时发生了继续 PPO、adaptive sampler 刷新、负载 COM 辅助监督删除和剩余预测损失系数 ×5。当前对照只能确认退步发生在这段续训中，不能把责任唯一分配给其中某项。需要从同一 2000 checkpoint、同一 sampler 重置和相同训练设置建立保持旧辅助目标的续训对照，再分别比较 mass-only 的 0.1 与 0.5，才能区分这些因果。不能仅凭当前 loss 标量或梯度大小直接宣布 ×5 无影响或宣布它就是根因。

[完整诊断数据及置信区间](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/diagnosis_4200_20260921/diagnosis.json) · [运行命令](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/diagnosis_4200_20260921/processes.json) · [梯度探针](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/diagnosis_4200_20260921/latent4200_gradients.json) · [前次评测报告](/data_zcy/wxy/intact-tracking/docs/heavy_residual_latest_eval_20260921.md)

复现统计：

```bash
.venv/bin/python scripts/analyze_heavy_residual_diagnosis.py runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/diagnosis_4200_20260921
```
