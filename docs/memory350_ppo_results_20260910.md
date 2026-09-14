# Memory350 latent residual PPO：最终对照结果

**本轮没有证明 Memory350 latent + FiLM 相对无 latent baseline 带来整体 tracking 优势。** 独立 U(0,4 kg) 的主要评估场景中，cold 的共同有效时段 body/joint 误差分别高 1.87% / 2.86%；warm 的 body 点估计高 0.27%，95% 区间包含零，joint 高 2.21%。同时，均匀负载下 anchor 位置误差降低约 9%–10%，五项速度误差也更低，因此结果体现了指标间的取舍。

0 kg 下 latent 的十项 tracking 误差均优于 baseline；2 kg 和 4 kg 下，body/joint 精度更差。这不等于 residual 无效：两种 residual 在 4 kg 下均优于冻结 tracker 的十项误差，但在 0 kg 下均损失精度。Memory350 减轻了低负载的精度损失，未超过 baseline 的高负载 body/joint 精度。

两组均已完成固定第 5000 轮训练。baseline、Memory350 和冻结 tracker 各完成八项最终测试，每项 4096 条固定配对 motion，共 24 项。结论只覆盖这一个配对训练 seed 和本协议，不作未见 motion 泛化或跨训练 seed 稳健性的声明。

## 主要误差与持续追踪

下表 body 均值及 body/joint 降幅均使用每对策略的共同有效时段，先按 motion 求均值再平均。降幅为正表示 latent 更好，方括号为 2000 次按 motion 配对 bootstrap 的 95% 区间。同一策略与不同对照取交集时，统计窗口会不同，均值不应跨对照直接拼接。

| 场景 | Baseline body（m） | Latent body（m） | Body 降幅及 95% CI | Joint 降幅及 95% CI |
|---|---:|---:|---|---|
| uniform_cold | 0.04555 | 0.04640 | -1.87% [-2.26, -1.44] | -2.86% [-3.12, -2.61] |
| uniform_warm | 0.04558 | 0.04570 | -0.27% [-0.68, +0.13] | -2.21% [-2.48, -1.96] |
| all_0_cold | 0.04239 | 0.03720 | +12.23% [+11.89, +12.57] | +8.52% [+8.29, +8.75] |
| all_0_warm | 0.04237 | 0.03550 | +16.21% [+15.87, +16.54] | +10.96% [+10.72, +11.19] |
| all_2_cold | 0.04392 | 0.04501 | -2.48% [-2.87, -2.09] | -3.53% [-3.76, -3.30] |
| all_2_warm | 0.04395 | 0.04451 | -1.26% [-1.60, -0.92] | -3.14% [-3.37, -2.92] |
| all_4_cold | 0.06713 | 0.07161 | -6.68% [-7.18, -6.18] | -5.19% [-5.53, -4.87] |
| all_4_warm | 0.06711 | 0.06988 | -4.12% [-4.60, -3.61] | -3.54% [-3.87, -3.21] |

失败率和覆盖率差值以下均为 latent 减 baseline，单位为百分点。

| 场景 | 失败率 baseline → latent | 失败率差及 95% CI | 覆盖率 baseline → latent | 覆盖率差及 95% CI |
|---|---|---|---|---|
| uniform_cold | 1.78% → 1.95% | +0.17 [-0.12, +0.46] | 99.01% → 98.85% | -0.16 [-0.33, +0.01] |
| uniform_warm | 1.81% → 2.03% | +0.22 [-0.07, +0.51] | 99.01% → 98.83% | -0.18 [-0.36, -0.01] |
| all_0_cold | 0.17% → 0.15% | -0.02 [-0.12, +0.05] | 99.90% → 99.92% | +0.02 [-0.02, +0.07] |
| all_0_warm | 0.17% → 0.07% | -0.10 [-0.20, -0.02] | 99.90% → 99.95% | +0.05 [+0.01, +0.11] |
| all_2_cold | 0.95% → 1.03% | +0.07 [-0.15, +0.29] | 99.46% → 99.43% | -0.04 [-0.15, +0.09] |
| all_2_warm | 0.90% → 1.07% | +0.17 [-0.05, +0.39] | 99.48% → 99.40% | -0.09 [-0.22, +0.04] |
| all_4_cold | 9.94% → 10.77% | +0.83 [+0.27, +1.42] | 94.04% → 93.42% | -0.63 [-0.97, -0.29] |
| all_4_warm | 10.03% → 10.67% | +0.63 [+0.00, +1.22] | 94.00% → 93.54% | -0.46 [-0.80, -0.10] |

均匀负载的失败率差异区间均包含零，不能认定失败率已经上升或下降；warm 覆盖率降低 0.18 个百分点，区间为 [-0.36,-0.01]。4 kg cold 失败率上升 0.83 个百分点，区间为 [0.27,1.42]，cold/warm 覆盖率均降低。0 kg warm 的失败次数由 7/4096 降至 3/4096。

预先固定的强改善标准要求两个 uniform 场景的 body/joint 降幅区间均大于零，且失败率差区间上限不大于零；本轮不满足。该标准不替代对其余 tracking 指标的解释。

## 其他 tracking 指标

0 kg 的全部十项误差均更低。2 kg 的 anchor 位置 cold/warm 降低 9.64% / 8.55%，4 kg 降低 13.40% / 14.12%，均匀负载降低 9.98% / 9.31%。在 2 kg、4 kg 和均匀负载下，joint 速度、anchor 线速度/角速度、body 线速度/角速度五项误差也均更低。

Body 旋转误差在 2 kg cold/warm 高 2.64% / 3.53%，4 kg 高 1.87% / 2.30%，均匀负载高 1.84% / 2.44%。4 kg warm 的 anchor 旋转误差高 2.76%，cold 差异区间包含零；2 kg 和均匀负载的 anchor 旋转误差更低。

这些辅助误差按各自实际存活时段统计，可能受提前失败影响，必须结合失败率和覆盖率解释。Anchor 世界位置误差与对齐参考下的 body 位置误差口径不同，两者改善方向可以不同。本轮没有保存全部十项误差的逐步 trace，只有 body/joint 可做共同有效时段比较。

[全部十项指标、共同有效时段指标与区间](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/final_005000.md)；[完整 336 行 CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/final_005000.csv)。

## 与冻结 tracker 比较

以下降幅为各 residual 相对冻结 tracker 的共同有效时段误差降幅；正值有利于 residual。

| 场景 | Baseline body / joint 降幅 | Memory350 body / joint 降幅 | 失败率 frozen → baseline → latent |
|---|---|---|---|
| uniform_cold | +3.20% / -8.96% | +1.40% / -12.06% | 2.42% → 1.78% → 1.95% |
| uniform_warm | +3.23% / -8.92% | +2.98% / -11.34% | 2.17% → 1.81% → 2.03% |
| all_0_cold | -80.69% / -55.15% | -58.57% / -41.92% | 0.12% → 0.17% → 0.15% |
| all_0_warm | -80.61% / -55.10% | -51.37% / -38.10% | 0.10% → 0.17% → 0.07% |
| all_2_cold | +5.49% / -8.98% | +3.16% / -12.84% | 1.00% → 0.95% → 1.03% |
| all_2_warm | +5.47% / -9.06% | +4.31% / -12.47% | 1.20% → 0.90% → 1.07% |
| all_4_cold | +22.91% / +13.95% | +17.63% / +9.42% | 13.53% → 9.94% → 10.77% |
| all_4_warm | +22.94% / +14.02% | +19.80% / +10.96% | 13.84% → 10.03% → 10.67% |

4 kg 下，baseline body 降低约 22.9%、joint 约 14.0%；Memory350 body 降低 17.6%–19.8%、joint 降低 9.4%–11.0%，两者十项误差均更低、失败率也更低。0 kg 下，baseline body 高约 80.6%–80.7%，Memory350 高约 51.4%–58.6%；两者全部十项误差都更高。0 kg 仍保留原 tracker DR，不是完全 nominal。

2 kg 与均匀负载下，两种 residual 相对冻结 tracker 的 body 略有改善，joint 更差；anchor 位置及部分线速度改善，旋转与部分角速度存在代价。完整区间见上方全表，不能把相对 baseline 的局部收益直接解释为全面超过冻结 tracker。

## 失败触发项与解释边界

4 kg cold：baseline 407 次失败中 399 次触发 EE z 位置，latent 441 次中 437 次触发；warm 对应 baseline 411 次中 404 次、latent 437 次中 429 次。均匀负载 cold 的 baseline/latent 失败数为 73/80，warm 为 74/83，也主要由 EE 位置触发。同一失败可能同时触发多个条件，不能把各条件计数相加作为失败总数。

训练按方案关闭 EE 位置 termination，评估保留冻结 tracker 原始完整失败标准。这些记录描述的是已观察到的终止触发，不能直接称为摔倒次数，也无法据截断轨迹判断关闭该终止项后能否恢复。[24 项失败触发明细](../runs/limb_context_20260910_memory350_ppo/artifacts/failure_breakdown/final_005000.md)。

## 实验设置与证据

baseline 使用 GPU 0/1，Memory350 使用 GPU 2/3，每卡 8192 环境；训练完整覆盖 motion_data_full 的 129827 条 motion、48085337 帧。两组使用原冻结 tracker 的六项 DR 与观测噪声，另在双手和两侧小腿中部各独立 U(0,4 kg) 加载，启动时固定、没有 nominal 子集。小腿负载位于 knee_link 局部坐标 (0,0,-0.15)，手部位于 wrist_yaw_link 的 (0.12,0,0)。

Residual actor/critic 均从头初始化；actor 原始输入 1645 维、主干 512→256→128，critic 原始输入 6330 维、主干 1024→512→512。Baseline 没有 context encoder、latent 或 FiLM；Memory350 的 64 维 latent 经 actor/critic 各自的 64→256→逐层 gain/bias 分支调制，强度为 0.5*tanh。FiLM 增加 1546496 个训练参数，因此本对照不单独控制额外网络容量。

冻结 encoder 采用 PPO 前选定的 stage1 第 22700 轮 checkpoint，使用 nominal counterfactual 监督表征。Short50 与 long300 不重叠，长期包含最多 30 个 10-step 连续片段，reset 丢弃不完整片段并保留完整记忆。本轮表征损失权重为 0.01、relation 权重 2.0、response_distance_scale=0.75。Tracker、encoder 与 context 归一化均冻结，PPO 不执行 predictor。

训练 seed=121，episode 上限 1000，rollout=24，actor/critic LR=1e-4/5e-4，5 epochs、4 minibatches、entropy=0.0002、residual scale=0.25；前 1000 轮 uniform，从精确第 1000 轮 checkpoint 恢复模型/优化器/归一化至 adaptive，failure rewind 关闭。公共初始 trunk 使用相同 seed，residual 输出均值初始为零，action std=0.25。

最终测试为训练目录中的固定 4096 个 motion，每个 motion 一个固定起点，seed=20001；冷启动历史为空。Warm 先由同一冻结 tracker 收集 500 步，再 reset 到固定 query，清空短期并保留长期；实际四个 warm 场景起始长期均为 30 个完整片段。所有对照的物理参数、query 初始状态、motion 与起点已配对。GPU 接触仿真的实际 warm 轨迹摘要分别保留，不声称逐位相同。Warm/cold 本身也不替代单独的长期记忆消融。

最终清单通过 PPO 前的固定随机抽样得到，包含 4094 条 sonic_filtered 和两条 LaFAN；周期测试及 4400 轮 2 kg 补测使用其前 512 条。补测曾观察到 2 kg body/joint 更差、anchor 更好，该补测保留在记录中，不替代最终第 5000 轮结果。[Motion 清单复核](../runs/limb_context_20260910_memory350_ppo/artifacts/motion_selection_reaudit.json)；[4400 轮 2 kg 补测](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/supplemental_update_004400_2kg.md)。

已核验两组 5000 次更新、两 rank 最终参数一致、每个优化参数累计 100000 步、53 个 tracker 权重/缓冲张量与原 checkpoint 逐位相同、encoder 冻结、全部周期测试及最终 24 项轨迹/点估计一致。实际第 0 轮的两组主干逐位匹配按约定 seed 重新生成的新 actor/critic；初始 optimizer state 为空，actor 输出层及 FiLM 调制头为零。W&B 两组各保存 1–5000 轮完整记录，最终 summary 的 1344 个数值与本地结果一致。

调度器曾因一次瞬时 progress 文件读取失败退出，随后附着原训练进程恢复调度；PPO worker 未因该事件重启，训练和评估源码未改动。唯一记录的运行中源码修订为 supervisor 恢复逻辑。[恢复记录](../runs/limb_context_20260910_memory350_ppo/artifacts/supervisor_recovery_20260910/verified.json)。

[最终原始比较 JSON](../runs/limb_context_20260910_memory350_ppo/final_results.json)；[24 项轨迹与点估计核对](../runs/limb_context_20260910_memory350_ppo/artifacts/final_output_audits/final_reconciliation.json)；[完整输入审计](../runs/limb_context_20260910_memory350_ppo/artifacts/protocol_audits/final_inputs.json)；[最终 checkpoint 张量审计](../runs/limb_context_20260910_memory350_ppo/artifacts/checkpoint_state_audits/baseline_005000_film_005000.json)；[实际初始 checkpoint 核验](../runs/limb_context_20260910_memory350_ppo/artifacts/final_output_audits/initial_checkpoints.json)；[训练 W&B 完成审计](../runs/limb_context_20260910_memory350_ppo/artifacts/wandb_training_completion_audit.json)；[最终 W&B 汇总核验](../runs/limb_context_20260910_memory350_ppo/artifacts/wandb_final_summary_audit.json)。

[W&B 最终比较](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/elxgffpv)；[baseline 训练](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/m350ppo-baseline-121-20260910)；[Memory350 训练](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/m350ppo-film-121-20260910)。

## 完整周期曲线

![100–5000 轮配对曲线](../runs/limb_context_20260910_memory350_ppo/artifacts/plots/tracking_progress_update_005000.png)

[曲线 PDF](../runs/limb_context_20260910_memory350_ppo/artifacts/plots/tracking_progress_update_005000.pdf)。曲线使用周期 512-motion 协议，最终结论使用上方 4096-motion 协议。
