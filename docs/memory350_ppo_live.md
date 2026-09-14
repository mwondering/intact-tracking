# Memory350 residual PPO 实时状态

**2026-09-14 当前任务：八个独立固定 DR 专家 B。** GPU 0–7 各运行一个不带 latent 的 residual MLP，每卡 8192 环境，全部 129827 motions，从零直接 adaptive，保留末端高度 termination，不设训练上限。每 1000 轮在各自完整固定 DR 上与同一个通用 A update 4000 比较。配置见 [八专家实验](fixed_dr_specialists8_20260914.md)，实时进度见 `runs/limb_context_20260914_fixed_dr_specialists8/state.json`。此前的 MLP/MoE 已分别在 4020/3300 轮保存退出；以下是历史记录，不代表当前占卡配置。

2026-09-10：阶段一 Memory350 与 Short50 已保存退出，分别完成 22871 / 10631 updates。此阶段二任务使用阶段一固定验证集最优 Memory350 update 22700；详见 context_selection.json。

正式监督器已启动，GPU 0–1 baseline，2–3 Memory350 FiLM，每卡 8192 环境、每组 5000 PPO updates。完整数据集、原始 tracker DR + 四肢独立 U(0,4 kg)、1000-step episode、关闭训练 ee_body_pos termination。前 1000 updates uniform，然后从精确 checkpoint 切换 adaptive。实验仍在执行，尚无最终性能结论。

W&B：项目 intact-preview-v2，group memory350-trackerdr-residual-20260910；已验证账号 2486344338@qq.com。

- [Baseline](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/m350ppo-baseline-121-20260910)
- [Memory350 latent](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/m350ppo-film-121-20260910)

每 100 updates 测 512 motions 的 0/4 kg cold/warm；最终 4096 motions 的 0/2/4 kg 及独立 uniform，加冻结 tracker 控制。逐步轨迹、失败率、覆盖率、配对 bootstrap 全部自动落盘。正式协议在 memory350_ppo_plan_20260910.md，最终报告将写入 memory350_ppo_results_20260910.md。

检查依据：24 项 CPU 测试通过；双卡每卡 8192 环境的 baseline 与 FiLM 均完成 2 updates + 精确恢复 adaptive 1 update；两 rank 参数/optimizer/normalizer 一致，公共网络权重相同。初始 critic 归一化均值最大差 7.45e-9，平方和与计数相同。冻结缓存推理在真实 checkpoint、512/8192 worlds、满历史下与原始 encoder 输出完全相同。

运行目录：runs/limb_context_20260910_memory350_ppo。supervisor_state.json 记录进程及完成轮数；logs/ 存训练和评估终端日志；ppo/*/metrics.jsonl 和 endpoint_eval_metrics.jsonl 存完整训练/测试数据。GPU 4–7 不用于此实验。

## 首个 100 轮配对结果

2026-09-10 17:13 UTC：两组第 100 轮的 512-motion 周期测试均完成，训练状态保护审计通过，PPO 已直接继续。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 5.69% / 6.80%；在 4 kg cold/warm 分别升高 7.96% / 6.70%。4 kg 失败率差异的配对置信区间均跨零。此时不能认定总体提升，保持已固定的对照设置，继续至 5000 updates。

同轮次自动配对汇总在 memory350_ppo_progress.md；完整中间结果在 progress_comparisons/，终端汇总在 logs/paired_progress.log。

## 200 轮配对结果

2026-09-10 17:25 UTC：两组 200 轮 512-motion 测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 19.09% / 22.38%，在 4 kg cold/warm 分别升高 10.20% / 8.31%。4 kg cold 失败率 baseline 8.59%、latent 10.74%，差 +2.15 个百分点，配对 95% CI 为 [+0.78, +3.71] 个百分点；4 kg warm 为 8.40% / 8.79%，差异区间跨零。当前只能确认不同负载下存在取舍，尚不能判断最终训练分布上的总体收益。继续按固定设置训练至 5000 updates。

## 300 轮配对结果

2026-09-10 17:37 UTC：300 轮的 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 18.84% / 21.87%，在 4 kg cold/warm 分别升高 8.74% / 8.87%。4 kg cold 失败率 baseline 7.62%、latent 9.18%，差 +1.56 个百分点（95% CI +0.20 至 +3.13）；warm 为 7.42% / 9.38%，差 +1.95 个百分点（95% CI +0.39 至 +3.52）。当前早期大负载结果仍不支持 latent 优势；保持协议继续训练，最终仍用 5000 轮、4096-motion 的 0/2/4 kg 与 uniform 测试判定。

## 400 轮配对结果

2026-09-10 17:49 UTC：400 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 20.03% / 23.82%，在 4 kg cold/warm 分别升高 7.92% / 7.42%。4 kg cold 失败率 baseline 8.20%、latent 8.40%；warm 为 8.01% / 8.79%，两项失败率差异的配对置信区间均跨零。4 kg 误差差距较 300 轮略缩小，仍未消除；两组继续原定训练。

## 500 轮配对结果

2026-09-10 18:01 UTC：500 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 20.33% / 24.12%，在 4 kg cold/warm 分别升高 7.05% / 5.24%。4 kg cold 失败率 baseline 6.64%、latent 7.81%；warm 为 6.64% / 8.20%。冷启动失败率差的 95% CI 为 -0.59 至 +2.93 个百分点，warm 为 0 至 +3.13 个百分点。4 kg 的误差差距继续缩小，但尚未转为优势。两组保持既定训练与最终评估协议。W&B 服务端在 17:53 UTC 再次确认两组在线同步，训练及周期测试数据均可见。

## 600 轮配对结果

2026-09-10 18:13 UTC：600 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 20.99% / 24.19%，在 4 kg cold/warm 分别升高 7.51% / 5.16%。4 kg cold 失败率 baseline 7.62%、latent 8.79%；warm 为 7.23% / 8.20%，两项配对差异区间均跨零。整体格局与 500 轮接近，当前不能认定大负载收益。两组仍处于 uniform 前缀，将按既定协议在各自 1000 轮后切换 adaptive，再继续至 5000 轮。

## 700 轮配对结果

2026-09-10 18:25 UTC：700 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 20.44% / 23.80%，在 4 kg cold/warm 分别升高 6.20% / 4.61%；joint 误差对应降低 12.27% / 13.65%、升高 4.47% / 4.13%。4 kg cold 失败率 baseline 7.23%、latent 8.59%；warm 为 7.81% / 8.40%。大负载误差差距较 600 轮略缩小，仍不支持大负载 tracking 优势。继续既定协议；两组各自完成 1000 updates 后精确恢复到 adaptive，并训练至 5000 updates 后进行最终全工况评估。

## 800 轮配对结果

2026-09-10 18:37 UTC：800 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 19.80% / 23.07%，在 4 kg cold/warm 分别升高 6.90% / 4.13%；joint 误差对应降低 11.79% / 12.86%、升高 4.59% / 3.18%。4 kg cold 失败率 baseline 6.84%、latent 9.18%；warm 为 6.64% / 8.98%。两项大负载失败率均高 2.34 个百分点。大负载表现仍落后，不据此更改预定训练或测试协议；继续至 5000 updates 后判断全部工况。18:32 UTC 再次从 W&B 服务端确认两组运行中，训练与周期评估日志持续同步。

## Baseline 第 1000 轮 adaptive 切换审计

2026-09-10 18:48 UTC：baseline 完成第 1000 轮及四项周期测试，两个 rank 的 actor、critic 和 critic normalizer 摘要一致；随后从同一已评估 checkpoint 精确恢复模型、优化器及归一化状态。恢复前后配置仅 resume 路径和 sampling.active_mode 改变，DR、数据集、环境数、奖励、termination 与网络设置一致。第 1001 条更新起 adaptive_enabled=true，failure rewind 仍关闭，目标保持 5000 updates。GPU 0/1 上的新训练进程已实际开始推进。完整证据：phase_transition_audits/baseline/verified.json、uniform_completion.json、uniform_run_config.json、adaptive_run_config.json。

## 900 轮配对结果

2026-09-10 18:49 UTC：900 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 18.92% / 22.30%，在 4 kg cold/warm 分别升高 4.95% / 3.01%；joint 误差对应降低 11.33% / 12.27%、升高 3.83% / 3.09%。4 kg cold 失败率 baseline 7.03%、latent 8.59%；warm 为 7.81% / 8.01%。all_4_cold 失败率差 +1.56 个百分点，95% CI [+0.00, +3.32]；all_4_warm 失败率差 +0.20 个百分点，95% CI [-1.37, +1.76]。大负载误差差距缩小，但尚未反超。两组 900 轮 checkpoint 均来自 uniform 前缀，比较未混入 adaptive 训练。18:50 UTC W&B 服务端确认 baseline 同一 run 已续训至 1035 轮，adaptive_enabled=true，1000 轮测试可见；Memory350 同步至 913 轮，900 轮测试可见。

## 1000 轮配对结果

2026-09-10 19:02 UTC：两组均匀采样阶段结束时的 1000 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 19.19% / 22.85%，在 4 kg cold/warm 分别升高 4.94% / 1.71%；joint 误差对应降低 11.33% / 12.51%、升高 4.11% / 2.66%。4 kg cold 失败率 baseline 8.20%、latent 8.79%；warm 两组均为 8.40%。Warm 大负载 body 误差差距较 900 轮缩小，但配对区间仍指向 latent 误差较高；当前不能宣称总体 tracking 优势。Memory350 的双卡参数一致性审计通过，已启动在 GPU 2/3 上从第 1000 轮精确恢复的进程，等待全量数据加载和实际恢复验证。两组继续向 5000 updates 推进。

## Memory350 第 1000 轮 adaptive 切换审计

2026-09-10 19:08 UTC：Memory350 从已完成周期测试的第 1000 轮 checkpoint 精确恢复，模型、优化器与归一化状态校验通过，第 1001 条更新起 adaptive_enabled=true，failure rewind 仍关闭。恢复前后配置只有 resume 路径和 sampling.active_mode 改变；冻结 encoder 的 checkpoint/归一化、网络、DR、数据集和环境数保持一致。GPU 2/3 上的新进程已实际开始更新。两组均已完成 1000 uniform updates，进入剩余 4000 adaptive updates 的训练，目标仍为各 5000 updates。完整证据：phase_transition_audits/film/verified.json、uniform_completion.json、uniform_run_config.json、adaptive_run_config.json。

19:09 UTC：W&B 服务端确认 baseline 与 Memory350 的同一 run 均为 running、adaptive_enabled=true，恢复后日志已持续同步；Memory350 服务端已记录到 1020 updates 和第 1000 轮测试。证据：phase_transition_audits/film/wandb_resumed.json、wandb_live_audit.json。

## 1100 轮配对结果

2026-09-10 19:21 UTC：两组各完成 100 次 adaptive 更新后的 1100 轮 512-motion 测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 17.47% / 20.99%，在 4 kg cold/warm 分别升高 3.98% / 2.57%；joint 误差对应降低 10.37% / 11.31%、升高 3.85% / 3.01%。4 kg cold 失败率 baseline 9.18%、latent 8.79%，差 -0.39 个百分点（95% CI -2.34 至 +1.37）；warm 为 8.59% / 9.57%，差 +0.98 个百分点（95% CI -0.59 至 +2.54）。冷启动大负载误差差距略缩小，warm 略回升，仍未出现全面优势。两组周期评估的训练状态保护审计均通过，PPO 已继续。

[100–1100 轮累计趋势图](../runs/limb_context_20260910_memory350_ppo/artifacts/plots/tracking_progress_update_001100.png)；[PDF](../runs/limb_context_20260910_memory350_ppo/artifacts/plots/tracking_progress_update_001100.pdf)。图中所有面板正值均代表 latent 更好，置信区间只反映配对 motion 抽样，不包含训练 seed 波动。原始配对 JSON 与输入哈希保留在 progress_comparisons/ 和图像同名 JSON 中。

## 1200 轮配对结果

2026-09-10 19:34 UTC：1200 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 17.22% / 21.24%，在 4 kg cold/warm 分别升高 3.96% / 1.60%；joint 误差对应降低 10.58% / 11.75%、升高 3.64% / 2.72%。4 kg cold 失败率 baseline 8.59%、latent 9.96%；warm 为 8.98% / 8.20%。all_4_cold 失败率差 +1.37 个百分点，95% CI [-0.20, +3.12]；all_4_warm 失败率差 -0.78 个百分点，95% CI [-2.34, +0.78]。大负载 warm 的误差差距略缩小，但 body/joint 尚未反超；失败率不能只看单轮点估计。两组评估状态保护通过，已恢复 PPO，继续既定 5000-update 及最终全工况评估协议。

## 1300 轮配对结果

2026-09-10 19:47 UTC：1300 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 17.30% / 20.73%，在 4 kg cold/warm 分别升高 2.80% / 1.06%；joint 误差对应降低 10.47% / 11.39%、升高 2.65% / 2.47%。4 kg cold 失败率 baseline 8.40%、latent 8.59%，差 +0.20 个百分点（95% CI -1.17 至 +1.56）；warm 为 8.98% / 8.01%，差 -0.98 个百分点（95% CI -2.73 至 +0.78）。4 kg warm 的 body 降幅区间首次包含零（-2.32% 至 +0.17%），但 joint 降幅区间仍为负（-3.33% 至 -1.64%），不能视为整体 tracking 持平或优势。两组本轮表中绝对 body 误差均较 1200 轮上升，相对差距缩小尚不能说明 latent 自身持续改善。19:48 UTC W&B 服务端再次确认两组 running、adaptive_enabled=true，分别同步至 1589/1322 updates，1500/1300 轮测试可见。继续预定训练和最终评估。

## 1400 轮配对结果

2026-09-10 19:59 UTC：1400 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 16.61% / 20.03%，在 4 kg cold/warm 分别升高 4.49% / 2.06%；joint 误差对应降低 10.11% / 11.09%、升高 3.92% / 2.86%。4 kg cold 失败率 baseline 9.18%、latent 8.40%；warm 为 8.40% / 8.20%。all_4_cold 失败率差 -0.78 个百分点，95% CI [-2.54, +0.98]；all_4_warm 失败率差 -0.20 个百分点，95% CI [-1.95, +1.56]。4 kg body 差距较 1300 轮扩大，两项 body 降幅区间重新均为负（cold -5.75% 至 -3.29%；warm -3.37% 至 -0.70%）。前一轮 warm 接近持平尚未形成持续趋势，当前仍不支持大负载的稳定 tracking 优势。周期测试状态保护通过，继续两组各 5000 updates 及最终全工况评估。

## 1500 轮配对结果

2026-09-10 20:12 UTC：1500 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 15.78% / 19.08%，在 4 kg cold/warm 分别升高 3.56% / 0.51%；joint 误差对应降低 9.19% / 10.28%、升高 3.15% / 1.98%。4 kg cold 失败率 baseline 8.79%、latent 9.38%；warm 为 8.98% / 8.01%。all_4_cold 失败率差 +0.59 个百分点，95% CI [-1.17, +2.34]；all_4_warm 失败率差 -0.98 个百分点，95% CI [-2.73, +0.78]。4 kg warm 的 body 降幅区间包含零（-1.69% 至 +0.67%），但 joint 仍较差，不能据此宣称整体优势。两组周期评估的训练状态保护通过，训练继续向各 5000 updates 推进。

## 1600 轮配对结果

2026-09-10 20:25 UTC：1600 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 16.18% / 19.13%，在 4 kg cold/warm 分别升高 4.63% / 2.32%；joint 误差对应降低 9.66% / 10.71%、升高 3.70% / 2.87%。4 kg cold 失败率 baseline 9.38%、latent 9.96%；warm 为 8.98% / 9.18%。all_4_cold 失败率差 +0.59 个百分点，95% CI [-0.59, +1.95]；all_4_warm 失败率差 +0.20 个百分点，95% CI [-1.18, +1.56]。4 kg body 差距较 1500 轮扩大，cold/warm 降幅区间分别为 -5.95% 至 -3.25%、-3.52% 至 -1.09%，均仍指向 latent 误差较大。保留记忆工况接近持平尚不稳定；当前结果仍为低负载有位置误差优势、高负载落后。两组周期评估状态保护通过，训练继续至各 5000 updates，再执行最终完整评估。

## 1700 轮配对结果与完整指标补充

2026-09-10 20:38 UTC：1700 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 16.87% / 19.22%，在 4 kg cold/warm 分别升高 5.68% / 2.92%；joint 误差对应降低 9.81% / 10.32%、升高 3.94% / 2.91%。4 kg cold 失败率两组均为 8.98%；warm 为 baseline 8.79%、latent 8.59%。all_4_cold 失败率差 +0.00 个百分点，95% CI [-1.95, +1.95]；all_4_warm 失败率差 -0.20 个百分点，95% CI [-1.76, +1.37]。两组评估状态保护通过。

此前阶段汇报集中在 body/joint 位置主指标，表述范围偏窄。原有 JSON 一直同时保存其他 tracking 指标，本次完整展开后，4 kg 下可见明确的指标取舍：按各自有效跟踪步数统计，anchor 位置误差在 cold/warm 分别降低 20.49% / 23.97%，joint 速度降低 6.65% / 8.16%，anchor 线速度降低 10.64% / 13.33%，body 线速度降低 7.41% / 9.65%；anchor/body 旋转误差则更高。1000、1500、1700 轮的 anchor 与多项速度改善方向一致。因此“高负载 body/joint 位置尚无优势”不能扩展为“所有 tracking 指标都无改善”。

anchor 位置指标直接比较世界坐标中的参考与机器人 anchor；body 位置指标使用对齐后的参考 body 位置。两者含义不同。除 body/joint 外，其余误差当前按各自有效步数统计，可能受提前失败影响，须结合失败率和覆盖率解释；本轮 4 kg 两项失败率差异区间均包含零。评估协议、数据和预定主指标保持原样，最终报告须完整呈现这些取舍。

[全部 tracking 指标](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_001700.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_001700.csv)。导出脚本为 scripts/report_memory350_tracking_details.py，最终结果生成后用 --final 导出全部 0/2/4 kg 与 uniform、含冻结 tracker 的对照。20:44 UTC W&B 服务端确认两组 running、adaptive_enabled=true，训练及周期测试继续同步。

## 1800 轮完整指标对照

2026-09-10 20:50 UTC：1800 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 14.99% / 17.73%，在 4 kg cold/warm 分别升高 4.62% / 3.19%；joint 误差对应降低 9.38% / 10.09%、升高 3.23% / 2.78%。4 kg cold 失败率 baseline 9.96%、latent 10.16%；warm 为 9.18% / 8.79%。all_4_cold 失败率差 +0.20 个百分点，95% CI [-1.56, +1.76]；all_4_warm 失败率差 -0.39 个百分点，95% CI [-1.76, +0.98]。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 20.40% / 21.21%，joint 速度降低 6.74% / 7.43%，anchor 线速度降低 10.37% / 11.34%，body 线速度降低 7.39% / 8.15%。速度和世界坐标 anchor 位置优势延续，body/joint 位置仍较差；应解释为指标取舍。辅助指标可能受到提前失败影响，继续同时报告失败率与覆盖率。两组评估状态保护通过，训练持续推进。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_001800.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_001800.csv)。

## 1900 轮完整指标对照

2026-09-10 21:03 UTC：1900 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 14.28% / 17.16%，在 4 kg cold/warm 分别升高 4.21% / 3.03%；joint 误差对应降低 8.63% / 9.12%、升高 2.77% / 2.62%。4 kg cold 失败率 baseline 9.77%、latent 8.79%；warm 为 9.77% / 9.96%。两项 4 kg 失败率差异的配对 95% CI 均跨零。0 kg 下 baseline 未失败，latent 在 cold/warm 均为 1/512 失败，需继续观察。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 21.86% / 24.58%，joint 速度降低 8.19% / 7.63%，anchor 线速度降低 11.65% / 12.26%，body 线速度降低 8.58% / 8.67%；anchor/body 旋转误差仍较高。位置与速度之间的取舍延续，不能合并为所有 tracking 指标都改善或都退步。辅助指标可能受到提前失败影响，本轮 4 kg 覆盖率差异区间均跨零。两组评估状态保护通过，继续原定各 5000 updates 与最终 4096-motion 全工况测试。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_001900.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_001900.csv)。

## 第 1900 轮 latent 使用诊断

固定其他 observation、把 latent 换成相邻环境的 latent 后，Memory350 actor 的 deterministic residual RMS 变化为 0.08154；置零 latent 后为 0.06708。正常 residual RMS 为 0.19993。该诊断表明 actor 对 latent 有响应，不能据此推断 tracking 必然改善，也不能单独归因于长期 memory。100、1000、1500、1900 轮均保持非零响应。

两组 residual 都逐渐接近幅度上限：1900 轮 baseline 有 46.07%、Memory350 有 42.82% 的采样 residual 分量达到上限的 95% 以上。这是训练分布上的诊断，不改变当前对照的 residual_scale=0.25 或其他超参数。证据：[诊断数据](../runs/limb_context_20260910_memory350_ppo/artifacts/latent_use_diagnostic_update_001900.json)。

## 2000 轮完整指标对照

2026-09-10 21:16 UTC：2000 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.86% / 16.63%，在 4 kg cold/warm 分别升高 4.35% / 5.43%；joint 误差对应降低 7.71% / 8.16%、升高 3.18% / 3.93%。0 kg 下两组 cold/warm 均无失败。4 kg cold 失败率 baseline 9.38%、latent 10.16%，差 +0.78 个百分点（95% CI -1.17 至 +2.54）；warm 为 8.59% / 10.55%，差 +1.95 个百分点（95% CI +0.20 至 +3.71）。本轮 warm 失败率差异区间高于零，不能把其较低的部分辅助误差直接等同于整体 tracking 收益。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 24.45% / 28.19%，joint 速度降低 8.04% / 6.94%，anchor 线速度降低 11.90% / 11.42%，body 线速度降低 8.56% / 8.19%；anchor/body 旋转误差更高。4 kg warm 覆盖率 baseline 94.32%、latent 93.29%，差 -1.03 个百分点（95% CI -2.08 至 +0.02）。世界坐标 anchor 位置和速度改善、body/joint 位置与部分旋转变差的取舍延续。两组第 2000 轮评估的训练状态保护均通过，仍保持各 5000 updates 和最终 4096-motion 全工况评估计划。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002000.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002000.csv)。

## 2100 轮完整指标对照

2026-09-10 21:29 UTC：2100 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 14.11% / 17.11%，在 4 kg cold/warm 分别升高 5.72% / 4.77%；joint 误差对应降低 7.99% / 8.45%、升高 3.93% / 3.79%。0 kg 下两组 cold/warm 均无失败。4 kg cold 失败率 baseline 9.38%、latent 8.40%，差 -0.98 个百分点（95% CI -2.54 至 +0.59）；warm 为 9.77% / 9.38%，差 -0.39 个百分点（95% CI -2.15 至 +1.17）。上一轮 warm 失败率偏高没有在本轮延续，两项失败率差异区间均跨零；不能据此认定稳定的失败率优势。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 22.86% / 26.63%，joint 速度降低 8.92% / 7.76%，anchor 线速度降低 12.51% / 12.06%，body 线速度降低 9.45% / 8.82%。4 kg cold/warm 覆盖率为 baseline 93.89% / 94.02%、latent 94.40% / 94.22%，差异区间均跨零。anchor 位置和速度改善、body/joint 位置与部分旋转变差的取舍仍在。两组本轮评估的训练状态保护均通过，继续预定各 5000 updates 和最终 4096-motion 全工况评估。

21:25 UTC W&B 服务端确认两组 running、adaptive_enabled=true；baseline 已同步至 2495 轮和第 2400 轮测试，Memory350 已同步至 2098 轮和第 2000 轮测试。21:21 UTC 曾遇到一次进度文件短暂不可读，立即复查确认同一批进程持续运行；只给独立监测脚本加入最多 0.4 秒的读取重试，训练代码和进程未变。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002100.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002100.csv)。

## 2200 轮完整指标对照

2026-09-10 21:42 UTC：2200 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.48% / 16.23%，在 4 kg cold/warm 分别升高 5.71% / 4.09%；joint 误差对应降低 7.26% / 7.69%、升高 3.92% / 3.36%。0 kg 下两组 cold/warm 均无失败。4 kg cold 失败率 baseline 9.38%、latent 9.77%，差 +0.39 个百分点（95% CI -1.17 至 +2.15）；warm 为 9.77% / 8.01%，差 -1.76 个百分点（95% CI -3.52 至 +0.00）。Warm 失败率点估计较低，但区间边界包含零，结合此前波动，尚不能认定稳定的失败率优势。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 23.48% / 24.01%，joint 速度降低 7.93% / 8.16%，anchor 线速度降低 11.53% / 12.04%，body 线速度降低 8.55% / 8.93%。4 kg cold/warm 覆盖率为 baseline 94.30% / 93.96%、latent 93.81% / 94.82%，差异区间均跨零。位置与速度指标之间的取舍延续，两组本轮评估的训练状态保护均通过；继续各 5000 updates 和最终 4096-motion 的固定 0/2/4 kg、独立 uniform 负载及冻结 tracker 对照。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002200.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002200.csv)。

## 2300 轮完整指标对照

2026-09-10 22:00 UTC：2300 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.69% / 16.30%，在 4 kg cold/warm 分别升高 7.12% / 5.14%；joint 误差对应降低 7.16% / 7.62%、升高 4.70% / 3.86%。0 kg cold 两组均为 1/512 失败，warm 均无失败。4 kg cold 失败率 baseline 9.18%、latent 9.96%，差 +0.78 个百分点（95% CI -0.78 至 +2.34）；warm 为 8.98% / 9.38%，差 +0.39 个百分点（95% CI -1.37 至 +2.15）。上一轮 warm 失败率点估计的优势未延续，两项差异区间均跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 21.96% / 24.15%，joint 速度降低 7.27% / 8.02%，anchor 线速度降低 10.11% / 11.22%，body 线速度降低 7.63% / 8.47%。4 kg cold/warm 覆盖率为 baseline 93.92% / 94.07%、latent 93.54% / 94.29%，差异区间均跨零。Anchor 位置和速度收益仍在，body/joint 位置差距较 2200 轮扩大。两组本轮评估状态保护通过，保持各 5000 updates 和最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002300.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002300.csv)。

## 21:53 UTC 调度读取故障及接管恢复

原调度进程 58635 因读取短暂不可见的 baseline progress.json 抛出 FileNotFoundError 而退出。两组 PPO launcher 5504/19374 和四个训练 worker 全程保持原 PID 与启动时间，训练未被重启。修复调度器对临时进度文件读取错误的处理，并新增 --attach-running：核验现有进程命令、GPU 分配及启动标识，训练进程退出后必须有新的 5000-update 完成记录、双卡有限参数一致性和最终 checkpoint 才进入评估。非父进程无法读取原 launcher 的退出码，接管后用进程退出和上述完成证据共同判断。

21:58 UTC 新调度进程 9911 接管成功，后续心跳和两组更新继续推进。4 项进程生命周期测试通过，涵盖活进程不能仅凭完成文件视为结束、旧 1000 轮记录不能通过、非有限完成记录被拒绝、PID 复用及命令不匹配检查。启动时再次审计保护源码、冻结 encoder 和原 smoke 证据。与启动前实现清单相比只有调度脚本改变，训练、推理和评估代码的哈希均未变；旧清单保留，修订证据见 [恢复审计目录](../runs/limb_context_20260910_memory350_ppo/artifacts/supervisor_recovery_20260910/verified.json) 与 [源码审计](../runs/limb_context_20260910_memory350_ppo/artifacts/supervisor_recovery_20260910/source_audit.json)。

## 2400 轮完整指标对照

2026-09-10 22:08 UTC：2400 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 14.32% / 16.87%，在 4 kg cold/warm 分别升高 6.98% / 5.62%；joint 误差对应降低 7.12% / 7.63%、升高 4.96% / 3.83%。0 kg cold 为 baseline 0/512、latent 1/512 失败，warm 两组均无失败。4 kg cold 失败率 baseline 10.16%、latent 10.74%，差 +0.59 个百分点（95% CI -1.17 至 +2.34）；warm 为 9.57% / 9.77%，差 +0.20 个百分点（95% CI -1.76 至 +2.15）。两项 4 kg 失败率差异区间均跨零，尚无稳定优势。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 23.65% / 24.80%，joint 速度降低 8.18% / 8.37%，anchor 线速度降低 10.98% / 12.05%，body 线速度降低 8.15% / 8.96%。4 kg cold/warm 覆盖率为 baseline 93.49% / 93.76%、latent 93.09% / 93.83%，差异区间均跨零。与 2300 轮相同，anchor 位置和速度有收益，body/joint 位置仍较差。两组本轮评估的训练状态保护通过，接管后的调度保持正常；继续各 5000 updates 和最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002400.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002400.csv)。

## 2500 轮完整指标对照

2026-09-10 22:21 UTC：2500 轮 512-motion 配对测试完成，两组均已完成至少一半的固定训练预算。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 14.45% / 16.76%，在 4 kg cold/warm 分别升高 6.66% / 6.00%；joint 误差对应降低 7.50% / 7.98%、升高 4.57% / 4.07%。0 kg cold 为 baseline 0/512、latent 1/512 失败，warm 两组均无失败。4 kg cold 失败率 baseline 9.57%、latent 9.38%，差 -0.20 个百分点（95% CI -1.76 至 +1.37）；warm 为 9.38% / 8.59%，差 -0.78 个百分点（95% CI -2.73 至 +1.17）。两项失败率差异区间均跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 22.94% / 24.66%，joint 速度均降低 7.92%，anchor 线速度降低 11.37% / 12.24%，body 线速度降低 8.31% / 8.99%。4 kg cold/warm 覆盖率为 baseline 93.81% / 94.04%、latent 94.04% / 94.60%，差异区间均跨零。训练过半，anchor 位置和速度收益仍在，body/joint 位置尚未反超；保持原定 5000 轮 checkpoint 与最终完整评估，不根据周期测试改选训练设置。两组本轮评估状态保护通过。

22:13 UTC W&B 服务端确认两组同一 run 持续 running、adaptive_enabled=true，baseline 同步至 2903 轮及第 2900 轮测试，Memory350 同步至 2478 轮及第 2400 轮测试。调度接管后未改变训练进程，22:16 UTC GPU 0–3 和磁盘余量检查正常。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002500.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002500.csv)。

## 2600 轮完整指标对照

2026-09-10 22:33 UTC：2600 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.72% / 16.99%，在 4 kg cold/warm 分别升高 6.72% / 5.04%；joint 误差对应降低 6.98% / 7.72%、升高 4.55% / 3.50%。0 kg cold 为 baseline 0/512、latent 1/512 失败，warm 为 baseline 1/512、latent 0/512。4 kg cold 失败率 baseline 8.98%、latent 9.77%，差 +0.78 个百分点（95% CI -0.59 至 +2.15）；warm 为 9.38% / 9.18%，差 -0.20 个百分点（95% CI -1.95 至 +1.37）。两项 4 kg 失败率差异区间均跨零，仍无稳定优势。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 23.27% / 23.39%，joint 速度降低 6.71% / 7.31%，anchor 线速度降低 10.25% / 11.95%，body 线速度降低 7.17% / 8.57%。4 kg cold/warm 覆盖率为 baseline 94.11% / 93.95%、latent 93.68% / 94.43%，差异区间均跨零。Warm 的 body/joint 位置差距较 2500 轮有所缩小，尚未反超；anchor 位置和速度收益继续存在。两组本轮评估状态保护通过，保持原定各 5000 updates 与最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002600.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002600.csv)。

## 2700 轮完整指标对照

2026-09-10 22:46 UTC：2700 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.45% / 16.35%，在 4 kg cold/warm 分别升高 8.56% / 6.24%；joint 误差对应降低 6.64% / 7.17%、升高 5.50% / 4.19%。0 kg cold 为 baseline 0/512、latent 1/512 失败，warm 两组均无失败。4 kg cold 失败率 baseline 9.77%、latent 10.16%，差 +0.39 个百分点（95% CI -1.17 至 +1.76）；warm 为 9.18% / 8.98%，差 -0.20 个百分点（95% CI -1.95 至 +1.56）。两项 4 kg 失败率差异区间均跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 23.68% / 25.22%，joint 速度降低 6.62% / 7.01%，anchor 线速度降低 10.64% / 10.93%，body 线速度降低 7.46% / 7.97%。4 kg cold/warm 覆盖率为 baseline 93.92% / 93.93%、latent 93.54% / 94.28%，差异区间均跨零。上一轮 body/joint 位置差距缩小没有延续，本轮两项高负载工况的差距均扩大；anchor 位置和速度收益仍然存在。两组本轮评估状态保护通过，继续固定 5000-update 预算和最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002700.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002700.csv)。

## 2800 轮完整指标对照

2026-09-10 22:58 UTC：2800 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.98% / 16.31%，在 4 kg cold/warm 分别升高 7.77% / 5.02%；joint 误差对应降低 7.07% / 7.76%、升高 5.33% / 3.76%。0 kg 两组 cold/warm 均无失败。4 kg cold 失败率 baseline 9.38%、latent 11.13%，差 +1.76 个百分点（95% CI -0.20 至 +3.71）；warm 两组均为 8.79%，差异区间为 -1.76 至 +1.56 个百分点。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 23.91% / 24.46%，joint 速度降低 6.68% / 7.36%，anchor 线速度降低 10.38% / 11.80%，body 线速度降低 7.31% / 8.56%。4 kg cold 覆盖率 baseline 94.07%、latent 92.95%，差 -1.12 个百分点（95% CI -2.17 至 -0.01）；本轮该差异区间低于零，提示需特别结合提前失败解释辅助误差。Warm 覆盖率为 94.44% / 94.37%，差异区间跨零。位置与速度指标的取舍仍在，两组本轮评估状态保护通过，继续原定各 5000 updates 和最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002800.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002800.csv)。

## 2900 轮完整指标对照

2026-09-10 23:12 UTC：2900 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.49% / 15.95%，在 4 kg cold/warm 分别升高 6.30% / 4.63%；joint 误差对应降低 7.68% / 8.32%、升高 4.52% / 3.50%。0 kg 两组 cold/warm 均无失败。4 kg cold 失败率 baseline 9.38%、latent 10.16%，差 +0.78 个百分点（95% CI -1.17 至 +2.54）；warm 为 8.59% / 9.18%，差 +0.59 个百分点（95% CI -0.98 至 +2.15）。两项失败率差异区间均跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 25.18% / 25.28%，joint 速度降低 7.43% / 7.27%，anchor 线速度降低 11.09% / 12.06%，body 线速度降低 8.06% / 8.90%。4 kg cold 覆盖率 baseline 94.20%、latent 93.50%，差 -0.71 个百分点（95% CI -1.85 至 +0.46）；warm 为 94.81% / 94.06%，差 -0.75 个百分点（95% CI -1.76 至 +0.24）。上一轮 cold 覆盖率差异区间低于零的现象本轮未延续，但点估计仍偏低。Body/joint 位置差距较 2800 轮缩小，尚未反超；anchor 位置和速度收益仍在。两组本轮评估状态保护通过，继续固定 5000-update 预算和最终完整评估。

23:08 UTC W&B 服务端确认两组同一 run 持续 running、adaptive_enabled=true；baseline 已同步至 3420 轮及第 3400 轮测试，Memory350 已同步至 2898 轮及第 2800 轮测试。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002900.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_002900.csv)。

## 3000 轮完整指标对照

2026-09-10 23:26 UTC：3000 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.90% / 16.71%，在 4 kg cold/warm 分别升高 7.11% / 6.53%；joint 误差对应降低 8.37% / 9.05%、升高 4.84% / 4.70%。0 kg cold/warm 均为 baseline 1/512、latent 0/512 失败。4 kg cold/warm 两组失败率均为 9.57%；候选减对照的差异区间分别为 -1.95 至 +1.76、-1.76 至 +1.56 个百分点。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 22.42% / 23.15%，joint 速度降低 6.22% / 5.61%，anchor 线速度降低 9.85% / 9.72%，body 线速度降低 7.12% / 6.93%。4 kg cold 覆盖率 baseline 94.01%、latent 93.72%，差 -0.29 个百分点（95% CI -1.39 至 +0.90）；warm 为 94.26% / 93.77%，差 -0.49 个百分点（95% CI -1.53 至 +0.51）。4 kg body 旋转误差 cold/warm 升高 2.11% / 2.75%，warm anchor 旋转误差升高 3.26%。高负载下位置、旋转与速度指标的取舍继续存在；body/joint 位置差距较 2900 轮扩大，两组本轮评估状态保护通过。继续固定各 5000 updates 和最终完整评估。

23:25 UTC：baseline 已进入 3600 轮周期测试，Memory350 已到 3030 轮；调度心跳、四个训练 worker 和两组有限损失检查正常，GPU 0–3 显存余量正常。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003000.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003000.csv)。

## 3100 轮完整指标对照

2026-09-10 23:36 UTC：3100 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 14.05% / 16.99%，在 4 kg cold/warm 分别升高 8.02% / 5.72%；joint 误差对应降低 8.99% / 9.81%、升高 5.67% / 4.30%。0 kg cold 为 baseline 0/512、latent 1/512 失败，warm 为 baseline 1/512、latent 0/512。4 kg cold 失败率 baseline 8.79%、latent 9.77%，差 +0.98 个百分点（95% CI -0.59 至 +2.54）；warm 为 9.18% / 10.16%，差 +0.98 个百分点（95% CI -0.78 至 +2.73）。两项 4 kg 失败率差异区间均跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 24.11% / 23.84%，joint 速度降低 5.84% / 6.51%，anchor 线速度降低 9.96% / 10.26%，body 线速度降低 6.74% / 7.50%。4 kg cold 覆盖率 baseline 94.54%、latent 93.63%，差 -0.90 个百分点（95% CI -1.90 至 +0.07）；warm 为 94.23% / 93.68%，差 -0.55 个百分点（95% CI -1.62 至 +0.54）。4 kg body 旋转误差 cold/warm 升高 2.91% / 2.03%，两项 anchor 旋转误差差异区间均跨零。相比 3000 轮，warm 的 body/joint 位置差距缩小，cold 差距扩大；高负载 anchor 位置和速度收益继续存在，尚不能宣称全面改善。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003100.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003100.csv)。

## 3200 轮完整指标对照

2026-09-10 23:49 UTC：3200 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.34% / 16.38%，在 4 kg cold/warm 分别升高 8.90% / 7.94%；joint 误差对应降低 8.95% / 9.94%、升高 6.42% / 5.87%。0 kg cold/warm 均为 baseline 1/512、latent 0/512 失败。4 kg cold 失败率 baseline 8.40%、latent 9.18%，差 +0.78 个百分点（95% CI -0.98 至 +2.54）；warm 为 8.01% / 10.94%，差 +2.93 个百分点（95% CI +0.98 至 +4.88）。本轮 warm 失败率差异区间高于零，需继续关注是否延续；此前 2000 轮也曾出现同方向区间，随后未持续。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 22.55% / 23.75%，joint 速度降低 4.83% / 3.49%，anchor 线速度降低 8.49% / 6.52%，body 线速度降低 5.58% / 4.22%。4 kg cold 覆盖率 baseline 94.75%、latent 93.92%，差 -0.83 个百分点（95% CI -1.91 至 +0.32）；warm 为 94.87% / 93.19%，差 -1.68 个百分点（95% CI -2.81 至 -0.57）。Warm 覆盖率差异区间也低于零，应特别结合提前失败解释按各自有效步数统计的辅助误差。4 kg body 旋转误差 cold/warm 升高 3.37% / 4.32%，warm anchor 旋转误差升高 4.75%，warm anchor 角速度降幅区间跨零。相比 3100 轮，4 kg 两种状态的 body/joint 位置差距均扩大，部分速度优势缩小；不能据 anchor 改善宣称高负载整体更好。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003200.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003200.csv)。

## 3300 轮完整指标对照

2026-09-11 00:02 UTC：3300 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.65% / 16.58%，在 4 kg cold/warm 分别升高 7.50% / 5.62%；joint 误差对应降低 9.38% / 10.50%、升高 5.33% / 4.16%。0 kg 两组 cold/warm 均无失败。4 kg cold 失败率 baseline 9.57%、latent 9.77%，差 +0.20 个百分点（95% CI -1.37 至 +1.56）；warm 两组均为 8.98%，差异区间为 -1.76 至 +1.95 个百分点。上一轮 warm 失败率差异区间高于零的现象本轮未延续。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 22.93% / 21.90%，joint 速度降低 6.80% / 6.46%，anchor 线速度降低 10.77% / 10.12%，body 线速度降低 7.65% / 7.50%。4 kg cold 覆盖率 baseline 94.05%、latent 93.65%，差 -0.39 个百分点（95% CI -1.28 至 +0.52）；warm 为 94.22% / 94.17%，差 -0.04 个百分点（95% CI -1.19 至 +1.12）。两项覆盖率差异区间均跨零。4 kg body 旋转误差 cold/warm 升高 1.79% / 2.01%，warm anchor 旋转误差升高 2.34%。相比 3200 轮，高负载 body/joint 位置差距缩小、速度降幅回升，尚未反超 body/joint 位置；anchor 位置收益仍在。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

2026-09-11 00:00 UTC W&B 服务端确认两组原 run 均为 running、adaptive_enabled=true，baseline 同步至 3898 轮及第 3800 轮测试，Memory350 同步至 3298 轮及第 3200 轮测试；本地随后已完成第 3900 / 3300 轮评估并继续更新。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003300.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003300.csv)。

## 3400 轮完整指标对照

2026-09-11 00:15 UTC：3400 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.13% / 15.75%，在 4 kg cold/warm 分别升高 8.34% / 6.08%；joint 误差对应降低 9.15% / 10.21%、升高 5.63% / 4.05%。0 kg cold 为 baseline 1/512、latent 0/512 失败，warm 两组均无失败。4 kg cold 失败率 baseline 8.79%、latent 9.77%，差 +0.98 个百分点（95% CI -0.78 至 +2.73）；warm 为 8.59% / 8.98%，差 +0.39 个百分点（95% CI -1.37 至 +2.15）。两项 4 kg 失败率差异区间均跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 20.37% / 21.09%，joint 速度降低 5.26% / 5.90%，anchor 线速度降低 8.36% / 9.35%，body 线速度降低 5.63% / 6.57%。4 kg cold 覆盖率 baseline 94.54%、latent 94.07%，差 -0.47 个百分点（95% CI -1.44 至 +0.51）；warm 为 94.86% / 94.17%，差 -0.68 个百分点（95% CI -1.71 至 +0.34）。两项覆盖率差异区间均跨零。4 kg body 旋转误差 cold/warm 升高 2.39% / 2.37%，warm anchor 旋转误差升高 2.95%。相比 3300 轮，高负载 body 位置差距扩大，warm joint 位置差距略缩小；anchor 位置和速度仍有收益，降幅有所回落。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

00:13 UTC 检查：baseline 已完成第 4000 轮评估并训练至 4037 轮，Memory350 已完成第 3400 轮评估并继续更新。四个训练 worker 和调度仍为原进程，未发生重启。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003400.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003400.csv)。

## 3500 轮完整指标对照

2026-09-11 00:28 UTC：3500 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 12.89% / 15.58%，在 4 kg cold/warm 分别升高 7.17% / 4.38%；joint 误差对应降低 9.15% / 10.30%、升高 4.62% / 3.32%。0 kg cold 两组均无失败，warm 为 baseline 1/512、latent 0/512。4 kg cold 失败率 baseline 8.40%、latent 10.94%，差 +2.54 个百分点（95% CI +0.78 至 +4.49）；warm 为 9.18% / 9.38%，差 +0.20 个百分点（95% CI -1.37 至 +1.76）。本轮 cold 失败率差异区间高于零，warm 区间跨零；需继续观察这种退化是否延续。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 24.56% / 21.71%，joint 速度降低 5.64% / 6.87%，anchor 线速度降低 9.47% / 11.05%，body 线速度降低 6.32% / 7.87%。4 kg cold 覆盖率 baseline 94.91%、latent 93.16%，差 -1.74 个百分点（95% CI -2.85 至 -0.77）；warm 为 94.35% / 94.09%，差 -0.26 个百分点（95% CI -1.21 至 +0.68）。Cold 覆盖率差异区间也低于零，应结合提前失败解释按各自有效步数统计的辅助误差。4 kg body 旋转误差 cold/warm 升高 1.60% / 1.05%，cold anchor 旋转误差降低 2.59%，warm anchor 旋转误差差异区间跨零。相比 3400 轮，高负载 body/joint 位置差距均缩小，anchor 位置和速度仍有收益；失败率尚无稳定优势，不能据误差降幅宣称整体改善。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

00:15 UTC GPU 0–3 显存分别约 38.8 / 38.8 / 41.9 / 42.1 GiB，磁盘剩余约 17.3 TiB；资源余量正常，审计见 [资源记录](../runs/limb_context_20260910_memory350_ppo/artifacts/resource_audits/1789085750.json)。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003500.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003500.csv)。

## 3600 轮完整指标对照

2026-09-11 00:40 UTC：3600 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.81% / 16.86%，在 4 kg cold/warm 分别升高 7.95% / 5.05%；joint 误差对应降低 9.02% / 10.23%、升高 5.68% / 3.87%。0 kg cold/warm 均为 baseline 1/512、latent 0/512 失败。4 kg cold 失败率 baseline 8.40%、latent 10.55%，差 +2.15 个百分点（95% CI +0.39 至 +3.91）；warm 为 8.59% / 8.79%，差 +0.20 个百分点（95% CI -1.17 至 +1.56）。Cold 失败率差异区间连续两个 checkpoint 高于零；这些测试使用同一批 motions，不能视为独立训练复现。Warm 差异区间仍跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 21.88% / 22.16%，joint 速度降低 4.65% / 6.20%，anchor 线速度降低 7.29% / 10.13%，body 线速度降低 5.03% / 7.33%。4 kg cold 覆盖率 baseline 94.69%、latent 93.46%，差 -1.23 个百分点（95% CI -2.34 至 -0.15）；warm 为 94.54% / 94.35%，差 -0.19 个百分点（95% CI -0.91 至 +0.58）。Cold 覆盖率差异区间也连续两个 checkpoint 低于零，应结合提前失败解释辅助误差。4 kg body 旋转误差 cold/warm 升高 2.89% / 2.06%，两项 anchor 旋转误差差异区间均跨零。相比 3500 轮，高负载 body/joint 位置差距扩大，部分速度降幅回落；anchor 位置和速度仍有收益，尚未获得全面改善。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003600.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003600.csv)。

## 3700 轮完整指标对照

2026-09-11 00:53 UTC：3700 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.92% / 16.93%，在 4 kg cold/warm 分别升高 7.26% / 6.00%；joint 误差对应降低 8.21% / 9.37%、升高 5.67% / 4.88%。0 kg cold 两组均无失败，warm 为 baseline 1/512、latent 0/512。4 kg cold 失败率 baseline 9.38%、latent 8.98%，差 -0.39 个百分点（95% CI -1.95 至 +1.17）；warm 两组均为 9.18%，差异区间为 -1.56 至 +1.56 个百分点。前两个 checkpoint 的 cold 失败率差异区间高于零的现象本轮未延续，尚无稳定的失败率优势或退化结论。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 19.53% / 19.19%，joint 速度降低 5.74% / 5.55%，anchor 线速度降低 9.48% / 10.20%，body 线速度降低 6.47% / 7.19%。4 kg cold 两组覆盖率均约 94.25%，差异区间为 -0.99 至 +0.99 个百分点；warm 为 baseline 94.32%、latent 94.27%，差 -0.04 个百分点（95% CI -0.96 至 +0.85）。两项覆盖率差异区间均跨零。4 kg body 旋转误差 cold/warm 升高 2.37% / 2.41%，cold anchor 旋转误差降低 3.53%，warm anchor 旋转误差差异区间跨零。相比 3600 轮，cold body 位置差距缩小、warm body/joint 位置差距扩大，anchor 位置降幅回落；位置、旋转与速度指标的取舍仍在。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003700.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003700.csv)。

## 3800 轮完整指标对照

2026-09-11 01:05 UTC：3800 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.66% / 16.81%，在 4 kg cold/warm 分别升高 7.64% / 5.51%；joint 误差对应降低 7.94% / 9.31%、升高 5.79% / 4.55%。0 kg 两组 cold/warm 均无失败。4 kg cold 失败率 baseline 8.59%、latent 9.38%，差 +0.78 个百分点（95% CI -0.78 至 +2.34）；warm 为 9.18% / 8.59%，差 -0.59 个百分点（95% CI -1.95 至 +0.78）。两项失败率差异区间均跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 18.61% / 17.12%，joint 速度降低 4.58% / 4.85%，anchor 线速度降低 7.69% / 7.88%，body 线速度降低 5.18% / 5.56%。4 kg cold 覆盖率 baseline 94.81%、latent 93.84%，差 -0.98 个百分点（95% CI -2.01 至 -0.01）；warm 为 94.27% / 94.42%，差 +0.14 个百分点（95% CI -0.64 至 +0.93）。Cold 覆盖率差异区间略低于零，应结合提前失败解释辅助误差。4 kg body 旋转误差 cold/warm 升高 2.49% / 2.81%，cold anchor 旋转误差降低 2.79%，warm anchor 旋转误差差异区间跨零。相比 3700 轮，cold body/joint 位置差距略扩大、warm 相对差距缩小，anchor 位置和部分速度降幅继续回落；相对差距缩小不代表 latent 绝对误差下降。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

00:55 UTC W&B 服务端确认两组原 run 均为 running、adaptive_enabled=true：baseline 同步至 4413 轮及第 4400 轮测试，Memory350 同步至 3738 轮及第 3700 轮测试。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003800.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003800.csv)。

## 3900 轮完整指标对照

2026-09-11 01:18 UTC：3900 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.84% / 17.11%，在 4 kg cold/warm 分别升高 7.21% / 5.15%；joint 误差对应降低 8.60% / 10.34%、升高 5.58% / 4.60%。0 kg 两组 cold/warm 均无失败。4 kg cold 失败率 baseline 7.62%、latent 9.38%，差 +1.76 个百分点（95% CI +0.00 至 +3.52）；warm 为 8.59% / 9.77%，差 +1.17 个百分点（95% CI -0.59 至 +3.12）。Cold 差异区间下界为零，warm 区间跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 20.20% / 19.89%，joint 速度降低 4.86% / 4.52%，anchor 线速度降低 7.88% / 7.63%，body 线速度降低 5.38% / 5.24%。4 kg cold 覆盖率 baseline 95.13%、latent 94.01%，差 -1.12 个百分点（95% CI -2.15 至 -0.16）；warm 为 94.58% / 93.93%，差 -0.64 个百分点（95% CI -1.70 至 +0.35）。Cold 覆盖率差异区间连续两个 checkpoint 低于零，应继续结合提前失败解释按各自有效步数统计的辅助误差；相同 motions 的连续 checkpoint 不构成独立训练复现。4 kg body 旋转误差 cold/warm 升高 2.58% / 3.45%，cold anchor 旋转误差降低 2.41%，warm anchor 旋转误差升高 2.38%。相比 3800 轮，高负载 body 位置相对差距略缩小、anchor 位置降幅回升；body/joint 位置尚未反超。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

01:14 UTC 评估期间 GPU 0–3 显存约 40–44 GiB，磁盘剩余约 17.3 TiB；资源余量正常，审计见 [资源记录](../runs/limb_context_20260910_memory350_ppo/artifacts/resource_audits/1789089253.json)。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003900.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_003900.csv)。

## 4000 轮完整指标对照

2026-09-11 01:31 UTC：4000 轮 512-motion 配对测试完成，两组均已完成至少 80% 的固定 PPO 预算。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.93% / 16.93%，在 4 kg cold/warm 分别升高 7.46% / 7.73%；joint 误差对应降低 8.42% / 10.05%、升高 5.86% / 6.27%。0 kg cold 两组均无失败，warm 为 baseline 1/512、latent 0/512。4 kg cold 失败率 baseline 8.40%、latent 9.57%，差 +1.17 个百分点（95% CI -0.59 至 +3.12）；warm 为 7.81% / 9.57%，差 +1.76 个百分点（95% CI +0.00 至 +3.52）。Cold 差异区间跨零，warm 区间下界为零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 19.26% / 19.45%，joint 速度降低 5.08% / 3.75%，anchor 线速度降低 8.78% / 6.35%，body 线速度降低 5.80% / 4.24%。4 kg cold 覆盖率 baseline 94.99%、latent 93.93%，差 -1.06 个百分点（95% CI -2.24 至 +0.04）；warm 为 95.29% / 94.02%，差 -1.27 个百分点（95% CI -2.30 至 -0.29）。此前连续两个 checkpoint 的 cold 覆盖率区间低于零的现象本轮未延续，但点估计仍较低；本轮 warm 覆盖率区间低于零，仍须结合提前失败解释辅助误差。4 kg body 旋转误差 cold/warm 升高 2.46% / 3.98%，cold anchor 旋转误差降低 3.36%，warm anchor 旋转误差升高 4.50%。相比 3900 轮，高负载 body/joint 位置差距均扩大，warm 的变化更明显；anchor 位置和部分速度仍有收益，尚无全面改善。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004000.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004000.csv)。

## 4100 轮完整指标对照

2026-09-11 01:43 UTC：4100 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 14.39% / 17.61%，在 4 kg cold/warm 分别升高 8.30% / 5.29%；joint 误差对应降低 8.21% / 9.87%、升高 6.20% / 4.97%。0 kg 两组 cold/warm 均无失败。4 kg cold 失败率 baseline 7.42%、latent 9.96%，差 +2.54 个百分点（95% CI +0.78 至 +4.49）；warm 为 9.18% / 9.77%，差 +0.59 个百分点（95% CI -1.17 至 +2.34）。Cold 差异区间高于零，warm 区间跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 21.33% / 19.65%，joint 速度降低 4.62% / 5.12%，anchor 线速度降低 7.62% / 7.70%，body 线速度降低 5.16% / 5.40%。4 kg cold 覆盖率 baseline 95.37%、latent 93.49%，差 -1.88 个百分点（95% CI -3.07 至 -0.81）；warm 为 94.48% / 93.83%，差 -0.66 个百分点（95% CI -1.66 至 +0.30）。本轮 cold 覆盖率区间低于零，warm 区间跨零，应继续结合提前失败解释辅助误差。4 kg body 旋转误差 cold/warm 升高 2.49% / 2.74%，两项 anchor 旋转误差差异区间均跨零。相比 4000 轮，cold body/joint 位置相对差距扩大、warm 差距缩小；anchor 位置和部分速度仍有收益，高负载尚未获得全面改善。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004100.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004100.csv)。

## 4200 轮完整指标对照

2026-09-11 01:56 UTC：4200 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 14.84% / 18.04%，在 4 kg cold/warm 分别升高 8.24% / 6.25%；joint 误差对应降低 8.56% / 10.31%、升高 6.25% / 5.52%。0 kg cold 两组均无失败，warm 为 baseline 0/512、latent 1/512，覆盖率分别为 100.00% / 99.99%。4 kg cold 失败率 baseline 8.20%、latent 10.35%，差 +2.15 个百分点（95% CI +0.59 至 +3.91）；warm 为 8.59% / 8.98%，差 +0.39 个百分点（95% CI -1.17 至 +1.95）。Cold 差异区间连续两个 checkpoint 高于零，warm 区间跨零；相同 motions 的连续 checkpoint 不构成独立训练复现。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 18.67% / 15.06%，joint 速度降低 4.67% / 4.09%，anchor 线速度降低 7.59% / 5.74%，body 线速度降低 4.99% / 3.82%。4 kg cold 覆盖率 baseline 94.80%、latent 93.45%，差 -1.35 个百分点（95% CI -2.41 至 -0.43）；warm 为 94.54% / 94.10%，差 -0.44 个百分点（95% CI -1.37 至 +0.53）。Cold 覆盖率区间也连续两个 checkpoint 低于零，应继续结合提前失败解释辅助误差。4 kg body 旋转误差 cold/warm 升高 2.85% / 3.31%，两项 anchor 旋转误差差异区间均跨零。相比 4100 轮，cold body 位置相对差距基本持平、warm 差距扩大，anchor 位置降幅回落；高负载尚未获得全面改善。两组本轮评估状态保护通过，继续固定各 5000 updates 和最终完整评估。

01:49 UTC W&B 服务端确认两组原 run 均为 running、adaptive_enabled=true：baseline 同步至 4918 轮及第 4900 轮测试，Memory350 同步至 4179 轮及第 4100 轮测试。已核对调度器将完成的每组固定 5000 轮 checkpoint 交给最终评估；baseline 最终评估完成后，再使用 0/1 卡评估冻结 tracker。只读监测已增加三组最终八项测试的进度，训练和评估代码未修改。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004200.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004200.csv)。


## Baseline 完成 5000 轮并进入最终评估

2026-09-11 01:58 UTC baseline 完成固定 5000 updates，两个 rank 均完成 5000，actor/critic/critic normalizer 参数摘要一致且数值有限。100–5000 每 100 轮共 50 次评估均通过训练状态保护；5000 checkpoint 的实际 SHA 与评估记录一致。训练 launcher 和两个 worker 正常退出后，调度器于 01:58:50 UTC 自动在 GPU 0/1 启动最终 4096-motion 八项测试。02:07 UTC 已完成六项，Memory350 正进行第 4300 轮周期评估，仍由原 GPU 2/3 worker 运行。完成审计见 [baseline completion audit](../runs/limb_context_20260910_memory350_ppo/artifacts/baseline_completion_audit/verified.json)。

02:10 UTC baseline 最终八项评估全部完成，每项 4096 motions；已核对全部 JSON 的 checkpoint、协议、motion 列表、seed、起始类型、负载、时间线和 summary 一致性，并记录 JSON/逐步 trace 的 SHA。逐步 trace 形状及跨策略配对仍由最终汇总复核。baseline 的独立 U(0,4 kg) cold/warm 失败率分别为 1.782% / 1.807%，覆盖率 99.007% / 99.012%；4 kg 为 9.937% / 10.034%，覆盖率 94.045% / 94.000%。这些是 baseline 单组数据，尚不能据此判断 Memory350 的增益。调度器已自动在 GPU 0/1 启动冻结 tracker 最终评估。详见 [baseline 最终评估审计](../runs/limb_context_20260910_memory350_ppo/artifacts/baseline_completion_audit/final_evaluation_verified.json)。

02:07 UTC W&B 服务端确认 baseline run 为 finished，completed_updates 与最后评估 checkpoint 均为 5000；Memory350 原 run 仍为 running，已同步至 4298。

## 4300 轮完整指标对照

2026-09-11 02:13 UTC：4300 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 14.32% / 18.18%，在 4 kg cold/warm 分别升高 6.39% / 5.39%；joint 误差对应降低 8.29% / 10.22%、升高 4.80% / 4.47%。0 kg cold 失败次数 baseline 1/512、latent 0/512，warm 为 baseline 0/512、latent 2/512；两项差异区间均含零。4 kg cold 失败率 baseline 8.59%、latent 9.96%，差 +1.37 个百分点（95% CI +0.00 至 +2.73）；warm 为 8.20% / 9.38%，差 +1.17 个百分点（95% CI -0.39 至 +2.93）。Cold 差异区间下界本轮回到零，不能延续前两轮区间严格高于零的说法。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 20.02% / 17.59%，joint 速度降低 5.52% / 4.97%，anchor 线速度降低 9.03% / 7.51%，body 线速度降低 6.21% / 5.20%。4 kg cold 覆盖率 baseline 94.64%、latent 93.31%，差 -1.33 个百分点（95% CI -2.31 至 -0.40）；warm 为 94.81% / 94.07%，差 -0.73 个百分点（95% CI -1.79 至 +0.20）。Cold 覆盖率差异区间连续三个 checkpoint 低于零，应结合提前失败解释辅助误差；这些相同 motions 的连续测试不是独立训练复现。4 kg body 旋转误差 cold/warm 升高 1.80% / 2.59%，cold anchor 旋转误差降低 2.56%，warm anchor 旋转误差差异区间跨零。相比 4200 轮，高负载 body/joint 位置相对差距缩小、anchor 位置降幅回升，但高负载尚未获得全面改善。两组本轮评估状态保护通过；baseline 已完成 5000 updates 及最终八项测试，Memory350 继续固定 5000 updates，冻结 tracker 最终评估正在 GPU 0/1 运行。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004300.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004300.csv)。

## 4300 轮曲线与阶段性输入复核

已导出截至 4300 轮的六项曲线：共同有效时段 body/joint 位置、各自有效时段 anchor 位置/body 线速度，以及失败率和覆盖率。图中正值均有利于 latent；辅助误差应结合提前失败解释。曲线并不替代全部十项 tracking 指标表。

![截至 4300 轮的配对训练曲线](../runs/limb_context_20260910_memory350_ppo/artifacts/plots/tracking_progress_update_004300.png)

[PDF](../runs/limb_context_20260910_memory350_ppo/artifacts/plots/tracking_progress_update_004300.pdf)；[图表来源 SHA](../runs/limb_context_20260910_memory350_ppo/artifacts/plots/tracking_progress_update_004300.json)。

02:15 UTC 重新核对 28 个相关源码 SHA、冻结 tracker/context checkpoint SHA、两组数据集完整覆盖、公共 PPO 参数、reward、termination、初始化 seed 和 motion sampling 配置；均匹配启动协议。唯一已记录源码修订为调度器恢复修复，训练和评估源码未变化。见 [阶段性输入审计](../runs/limb_context_20260910_memory350_ppo/artifacts/protocol_audits/inputs_1789092926.json)。02:14 UTC GPU 2/3 显存约 42 GiB，GPU 0/1 冻结 tracker 评估显存约 6.6 GiB，磁盘剩余约 17.3 TiB，见 [资源记录](../runs/limb_context_20260910_memory350_ppo/artifacts/resource_audits/1789092864.json)。这些为阶段性审计，Memory350 5000 轮及其最终评估仍未完成。

## 冻结 tracker 最终对照完成：baseline 的收益与代价

2026-09-11 02:20 UTC 冻结 tracker 最终八项测试全部完成，评估进程正常退出。02:22 UTC 已核对 baseline 与 frozen tracker 每项 4096 个 motion 的动作、起点、物理参数、query 初始状态、warm-up 策略/seed 完全配对；所有逐步 body/joint trace 形状均为 (4096,1000,2)，数值有限，长度与 JSON 一致。接触仿真 warm 轨迹摘要仍按实际运行分别保留，不要求逐位一致。GPU 0/1 已完成本轮安排，Memory350 继续由原 GPU 2/3 worker 运行。

下表候选是 baseline residual，对照是冻结 tracker。正降幅表示 residual 更好；body/joint 使用两组共同有效时段。

| 场景 | Body 降幅 | Joint 降幅 | 失败率 frozen → baseline | 覆盖率差（百分点） |
|---|---:|---:|---|---:|
| all_0_cold | -80.69% | -55.15% | 0.12% → 0.17% | -0.03 |
| all_4_cold | +22.91% | +13.95% | 13.53% → 9.94% | +2.49 |
| all_0_warm | -80.61% | -55.10% | 0.10% → 0.17% | -0.04 |
| all_4_warm | +22.94% | +14.02% | 13.84% → 10.03% | +2.62 |
| all_2_cold | +5.49% | -8.98% | 1.00% → 0.95% | +0.03 |
| uniform_cold | +3.20% | -8.96% | 2.42% → 1.78% | +0.44 |
| all_2_warm | +5.47% | -9.06% | 1.20% → 0.90% | +0.16 |
| uniform_warm | +3.23% | -8.92% | 2.17% → 1.81% | +0.33 |

4 kg 下 baseline 的全部十项 tracking 误差均更低，body 降低约 22.9%、joint 约 14.0%，anchor 位置约 35.1%–35.7%；失败率 cold/warm 分别下降 3.59 / 3.81 个百分点，95% CI 分别为 [-4.30,-2.83] / [-4.52,-3.08]，覆盖率分别提高 2.49 / 2.62 个百分点。

0 kg 下全部十项误差均更高；共同有效时段 body 从约 0.02346 m 增至 0.04237–0.04238 m，增幅约 80.6%–80.7%，joint 增幅约 55.1%。这说明本次无 latent residual 的大负载适应伴随明显的零附加负载精度损失；0 kg 仍保留原 tracker DR，并非完全 nominal。

2 kg 下 body 降低约 5.5%，但 joint 误差升高约 9.0%；独立 U(0,4 kg) 下 body 仅降低约 3.2%，joint 升高约 8.9%。二者 anchor 位置分别降低约 29% / 25%，部分线速度指标也改善，但旋转和部分角速度指标变差。Uniform cold 失败率下降 0.63 个百分点（95% CI -1.03 至 -0.27）；warm 下降 0.37 个百分点（95% CI -0.73 至 +0.00）。辅助误差按各自存活时段统计，仍需结合失败率/覆盖率解释。

这是已完成 baseline 对照的结果，不能替代尚未完成的 Memory350 第 5000 轮结果，也不能把相对 baseline 的增益直接理解为全面超过冻结 tracker。完整十项指标及 95% 区间见 [对照全表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/control_baseline_005000.md)、[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/control_baseline_005000.csv)，配对与逐步 trace 审计见 [verified.json](../runs/limb_context_20260910_memory350_ppo/artifacts/frozen_tracker_completion_audit/verified.json)。

## 4400 轮完整指标对照

2026-09-11 02:26 UTC：4400 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 13.91% / 17.74%，在 4 kg cold/warm 分别升高 7.10% / 4.11%；joint 误差对应降低 8.63% / 10.78%、升高 5.15% / 3.58%。0 kg cold/warm 失败次数均为 baseline 1/512、latent 0/512，差异区间均含零。4 kg cold 失败率 baseline 8.40%、latent 9.77%，差 +1.37 个百分点（95% CI -0.20 至 +3.12）；warm 为 8.20% / 9.38%，差 +1.17 个百分点（95% CI -0.39 至 +2.73），两项区间均跨零。

按各自有效步数统计，4 kg anchor 位置误差 cold/warm 分别降低 18.99% / 17.25%，joint 速度降低 5.04% / 4.95%，anchor 线速度降低 7.90% / 8.30%，body 线速度降低 5.44% / 5.55%。4 kg cold 覆盖率 baseline 94.72%、latent 93.58%，差 -1.14 个百分点（95% CI -2.19 至 -0.18）；warm 为 94.67% / 93.98%，差 -0.69 个百分点（95% CI -1.64 至 +0.21）。Cold 覆盖率区间连续四个 checkpoint 低于零，但相同 motions 的连续测试不构成独立训练复现。4 kg body 旋转误差 cold/warm 升高 2.08% / 2.38%；cold anchor 旋转误差降低 3.56%，warm 差异区间跨零。相比 4300 轮，cold body/joint 相对差距扩大，warm 差距缩小，anchor 降幅略回落。两组本轮评估状态保护通过；baseline 与冻结 tracker 的最终八项评估均已完成，Memory350 继续完成剩余 600 updates 及最终八项评估。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004400.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004400.csv)。

## 按用户要求补测 4400 轮 2 kg

2026-09-11 02:31–02:36 UTC：在空闲 GPU 0/1 对 baseline 与 Memory350 的固定 4400 轮 checkpoint 补测双手及小腿中部各 2 kg，保留原 tracker DR；使用与周期 0/4 kg 相同的 512 条 motion、seed 20001 和起点，cold/warm 分别配对。评估进程全部正常退出，checkpoint、物理参数、起点、warm-up 策略和逐步 trace 检查通过。原 GPU 2/3 的 Memory350 训练保持运行，补测期间从 4459 推进至 4499，随后进行第 4500 轮周期评估。该补测不改变固定第 5000 轮的最终评价协议。

| 2 kg 场景 | Body 误差变化 | Joint 误差变化 | Anchor 位置误差变化 | 失败次数 baseline → latent | 覆盖率 baseline → latent |
|---|---:|---:|---:|---|---|
| all_2_cold | 高 2.81% | 高 3.76% | 低 10.65% | 2/512 → 5/512 | 99.89% → 99.41% |
| all_2_warm | 高 1.39% | 高 3.67% | 低 9.90% | 2/512 → 7/512 | 99.89% → 99.24% |

共同有效时段 body 降幅 95% CI：cold [-3.94,-1.73]%、warm [-2.37,-0.43]%；joint 降幅 CI：cold [-4.42,-3.13]%、warm [-4.28,-3.05]%，均指向 latent 误差更高。失败率差值 cold +0.59 个百分点（95% CI +0.00 至 +1.37），warm +0.98 个百分点（95% CI +0.20 至 +1.95）。覆盖率 cold 低 0.48 个百分点（95% CI -1.05 至 -0.06），warm 低 0.66 个百分点（95% CI -1.30 至 -0.14）。

按各自有效步数统计，anchor 位置在 cold/warm 降低 10.65% / 9.90%，anchor 旋转降低 5.62% / 2.66%，anchor 线速度降低 4.54% / 4.72%，body 线速度降低 1.96% / 1.83%；body 旋转误差则高 2.81% / 3.90%。Cold joint 速度与 body 角速度小幅改善，warm 两者差异区间跨零。辅助误差可能受提前失败影响，须与覆盖率一起解释。当前 4400 轮在 2 kg 下仍是局部指标收益，尚无整体 tracking 改善。

[全部十项指标与区间](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/supplemental_update_004400_2kg.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/supplemental_update_004400_2kg.csv)；[原始配对结果与 SHA](../runs/limb_context_20260910_memory350_ppo/supplemental/update_004400_2kg/results.json)。

## Motion 清单与 baseline 命名澄清

用户询问测试 motion 来源后，重新从完整目录枚举 129827 个文件并用 random.Random(209101).sample(sorted_catalog,4096) 重建抽样，精确复现最终清单；周期和 4400 轮 2 kg 补测使用其前 512 条。512 条中 sonic_filtered 511 条、LaFAN walk2_subject3.motion.npz 1 条；4096 条中 sonic_filtered 4094 条、LaFAN 2 条（walk2_subject3、fight1_subject5）。这是按 motion 文件的固定抽样，训练仍完整覆盖整个数据集。实际 512 条测试的起点中 497 条为第 1 帧，其余 15 条使用固定较晚起点；不能理解为所有 motion 都从随机帧开始。验证见 [motion selection audit](../runs/limb_context_20260910_memory350_ppo/artifacts/motion_selection_reaudit.json)。

用户指出 baseline 名称含 memory350 容易混淆。已直接加载双方第 4400 轮 checkpoint 复核实际权重：baseline context_checkpoint/context_sha256 均为 null、latent_dimensions=0，actor/critic 均无 FiLM condition/heads 参数；latent 版本使用冻结的 22700 轮 Memory350 encoder 与 64 维 latent，actor/critic 各有独立 FiLM。两组 actor 公共主干为 1645→512→256→128→29，critic 公共主干为 6330→1024→512→512→1（隐藏层带归一化）；输入和公共宽度一致。Baseline 共用 wrapper 时只带一个全零占位字段，actor 与 critic 均不读取它，也不运行 Memory350 历史编码。名称中的 memory350 原为整组对照实验前缀。

本轮验证的是“无 latent”与“Memory350 latent + FiLM”的整体差异，FiLM 带来的额外参数未单独控制，不能把全部差异独立归因于表征的环境语义。实际结构审计见 [checkpoint architecture audit](../runs/limb_context_20260910_memory350_ppo/artifacts/baseline_latent_architecture_audit.json)。W&B baseline 展示名已澄清为 baseline_no_latent_trackerdr_121，run ID、分组和已有训练数据保留；见 [展示名修改记录](../runs/limb_context_20260910_memory350_ppo/artifacts/baseline_wandb_display_name.json)。

## 4500 轮完整指标对照

2026-09-11 02:40 UTC：4500 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 12.62% / 16.43%，在 4 kg cold/warm 分别升高 6.81% / 4.21%；joint 对应降低 8.22% / 10.47%、升高 5.47% / 4.23%。0 kg cold 失败次数 baseline 1/512、latent 0/512，warm 两者均为零。4 kg cold 失败率 baseline 7.81%、latent 9.18%，差 +1.37 个百分点（95% CI -0.39 至 +2.93）；warm 为 8.98% / 8.79%，差 -0.20 个百分点（95% CI -1.76 至 +1.17），区间均跨零。

辅助指标按各自有效步数统计：4 kg anchor 位置 cold/warm 降低 16.78% / 17.21%，joint 速度降低 4.84% / 5.32%，anchor 线速度降低 7.00% / 7.61%，body 线速度降低 4.75% / 5.24%。4 kg cold 覆盖率 baseline 95.11%、latent 94.03%，差 -1.08 个百分点（95% CI -2.12 至 -0.05）；warm 为 94.44% / 94.23%，差 -0.20 个百分点（95% CI -1.10 至 +0.67）。Cold 覆盖率区间连续五个 checkpoint 低于零，相同 motions 的连续测试不构成独立训练复现。Body 旋转 cold/warm 误差高 2.01% / 2.23%；cold anchor 旋转低 3.45%，warm 区间跨零。高负载仍存在指标取舍，不能从 warm 失败率的单轮点估计认定持续跟踪已改善。两组本轮评估状态保护通过，Memory350 继续固定 5000 updates。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004500.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004500.csv)。

## Latent 距离与 FiLM 的讨论：本轮配置保持不变

用户询问不同环境 latent 距离更远是否有利于 FiLM。已直接读取本轮冻结的 22700 update context checkpoint：representation_weight=0.01、representation_relation_weight=2.0、response_distance_scale=0.75。当前损失先对 latent 做单位归一化，再匹配单位向量欧氏距离与 2*d_response/(d_response+s)；减小 s 才会增大目标距离，整体放大 latent 数值不改变此损失使用的距离。FiLM 根据 latent 生成每层缩放/偏置，并通过 0.5*tanh 有界调制。合理区分动力学响应可能有帮助，但更大的 embedding 距离不保证更有效的策略调制，也不能据当前 2/4 kg 结果认定距离不足或 tanh 已饱和。

后续可考虑 0.75→0.5 的单因素对照，并检查预测误差、调制差异和饱和程度；本轮没有修改此参数、重训 encoder 或改变 PPO 协议，仍完成固定 5000 轮和原定最终测试。FiLM 原理参考 [Perez et al.](https://arxiv.org/abs/1709.07871)，具体归一化、距离公式和有界调制以本项目代码为准。

02:48 UTC 已将 4400 轮 2 kg 补测的完整指标、95% 区间和 28 行统计表同步到同一 W&B group 的独立评估 run：[paired_4400_2kg_tracking_summary](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/md072vv2)。服务端复核 run 已 finished，checkpoint_update=4400 与关键指标一致；评估没有修改两个训练 run 的记录。

## 4600 轮完整指标对照

2026-09-11 02:53 UTC：4600 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 12.49% / 15.98%，在 4 kg cold/warm 分别升高 7.10% / 3.94%；joint 对应降低 7.89% / 9.95%、升高 5.22% / 3.47%。0 kg 两组 cold/warm 均无失败且覆盖率 100%。4 kg cold 失败率 baseline 8.01%、latent 9.57%，差 +1.56 个百分点（95% CI +0.00 至 +3.12）；warm 为 7.81% / 8.59%，差 +0.78 个百分点（95% CI -0.59 至 +2.15）。

辅助指标按各自有效步数统计：4 kg anchor 位置 cold/warm 降低 15.29% / 15.73%，joint 速度降低 4.76% / 4.83%，anchor 线速度降低 7.71% / 7.14%，body 线速度降低 5.10% / 4.65%。4 kg cold 覆盖率 baseline 94.96%、latent 93.74%，差 -1.22 个百分点（95% CI -2.23 至 -0.25）；warm 为 95.02% / 94.48%，差 -0.54 个百分点（95% CI -1.45 至 +0.33）。Cold 覆盖率区间连续六个 checkpoint 低于零；相同 motions 的连续测试不构成独立训练复现。Body 旋转 cold/warm 误差高 1.95% / 2.07%；cold anchor 旋转低 4.00%，warm 区间跨零。相比 4500 轮，cold body 相对差距扩大、warm 差距略缩小，anchor 位置降幅回落，仍未获得高负载全面改善。两组本轮评估状态保护通过，Memory350 继续固定 5000 updates。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004600.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004600.csv)。

## 实际 checkpoint 冻结状态与优化步数核验

02:58 UTC 对 baseline 第 5000 轮和 Memory350 第 4600 轮 checkpoint 做实际张量核验：两组 tracker 的全部 53 个权重/缓冲张量与原始冻结 checkpoint 逐位相同；baseline 和 latent 优化器分别含 23 / 39 个 trainable 参数张量，与 residual actor/critic 的参数形状、数量和学习率逐项匹配，没有 tracker 参数。每个 optimizer state 的步数分别为 100000 / 92000，精确对应 completed_updates × 5 epochs × 4 minibatches，参数及优化器 moments 均有限。

实际加载的冻结 context 处于 eval 模式，所有 encoder 参数 requires_grad=False；其配置为 short50 + chunk10×30、64 维 latent，文件 SHA 与 PPO 前固定的选择一致。此次为 checkpoint 状态审计，Memory350 第 5000 轮及其最终测试尚待完成；到 5000 轮后会对最终 checkpoint 再核验。见 [审计记录](../runs/limb_context_20260910_memory350_ppo/artifacts/checkpoint_state_audits/baseline_005000_film_004600.json)。

03:01 UTC W&B 服务端再次复核：baseline run 为 finished，5000 轮训练与第 5000 轮测试完整同步，展示名为 baseline_no_latent_trackerdr_121；Memory350 原 run 为 running，训练同步至 4698、第 4600 轮测试已上传，两组 adaptive_enabled=true。见 [服务端审计](../runs/limb_context_20260910_memory350_ppo/wandb_live_audits/1789095661.json)。

## 4700 轮完整指标对照

2026-09-11 03:05 UTC：4700 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 12.37% / 16.16%，在 4 kg cold/warm 分别升高 6.70% / 4.38%；joint 对应降低 7.77% / 10.03%、升高 5.01% / 3.87%。0 kg cold 失败次数 baseline 1/512、latent 0/512，warm 两组均为零。4 kg cold 失败率 baseline 8.40%、latent 10.35%，差 +1.95 个百分点（95% CI +0.20 至 +3.71）；warm 为 8.59% / 9.38%，差 +0.78 个百分点（95% CI -0.78 至 +2.34）。Cold 失败率差异区间本轮再次严格高于零。

辅助指标按各自有效步数统计：4 kg anchor 位置 cold/warm 降低 18.85% / 14.68%，joint 速度降低 4.89% / 3.89%，anchor 线速度降低 7.79% / 6.66%，body 线速度降低 5.34% / 4.52%。4 kg cold 覆盖率 baseline 94.81%、latent 93.44%，差 -1.37 个百分点（95% CI -2.45 至 -0.34）；warm 为 94.59% / 94.14%，差 -0.45 个百分点（95% CI -1.31 至 +0.42）。Cold 覆盖率区间连续七个 checkpoint 低于零，相同 motions 的连续测试不构成独立训练复现。Body 旋转 cold/warm 误差高 1.76% / 2.51%；cold anchor 旋转低 2.68%，warm 区间跨零。相比 4600 轮，cold body 相对差距略缩小、warm 差距扩大；anchor 位置 cold 降幅回升、warm 降幅回落。两组本轮评估状态保护通过，Memory350 继续完成最后 300 updates 与最终八项测试。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004700.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004700.csv)。

## 已观察到的失败触发项

03:14 UTC 复核第 4700 轮与 4400 轮 2 kg 补测的每条轨迹：failed 标记均与至少一个 failure_term 触发一致。4700 轮 4 kg cold，baseline 43 次失败中 42 次为 EE z 位置、1 次为 anchor 姿态；latent 53 次全部为 EE z 位置。Warm 对应 baseline 44 次（43 次 EE、1 次姿态）、latent 48 次全部 EE。该轮高负载的额外失败主要对应末端高度偏差终止，不能直接称为额外摔倒。

4400 轮 2 kg cold 的 baseline 2 次、latent 5 次失败也均触发 EE z 位置；warm 的 baseline 2 次、latent 7 次均触发 EE，其中 latent 有 1 次同时触发 anchor z 位置。一次失败可以触发多个条件，计数不能相加当作失败总数。实际终止配置中 anchor 和 EE 位置条件均为 z-only，阈值 0.5 m。

训练已按协议关闭 EE 位置终止，评估保留原 tracker 的完整终止条件。记录在触发处结束，因此无法据此推断关闭评估 EE 终止后能否恢复；本轮保持原定评估标准，并在最终结果中同时报告这些触发项。见 [4700 轮触发项](../runs/limb_context_20260910_memory350_ppo/artifacts/failure_breakdown/update_004700.md)、[2 kg 补测触发项](../runs/limb_context_20260910_memory350_ppo/artifacts/failure_breakdown/supplemental_update_004400_2kg.md)，对应 JSON 保存完整组合计数与输入 SHA。

03:14 UTC 资源检查：Memory350 仍由原 GPU 2/3 worker 运行，显存约 43.8/44.0 GiB，正在第 4800 轮周期测试；磁盘剩余约 17.3 TiB。GPU 0/1 已完成本实验的 baseline、冻结 tracker 和补测任务，当前由其他目录的训练进程占用，不属于本轮待完成的 Memory350 工作，未操作这些进程。见 [资源记录](../runs/limb_context_20260910_memory350_ppo/artifacts/resource_audits/1789096475.json)。

## 4800 轮完整指标对照

2026-09-11 03:18 UTC：4800 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 11.98% / 15.70%，在 4 kg cold/warm 分别升高 4.69% / 3.71%；joint 对应降低 8.14% / 10.29%、升高 3.90% / 2.76%。0 kg cold/warm 均为 baseline 1 次失败、latent 0 次。4 kg cold 失败率 baseline 8.01%、latent 9.38%，差 +1.37 个百分点（95% CI -0.20 至 +2.93）；warm 为 8.01% / 9.57%，差 +1.56 个百分点（95% CI +0.20 至 +3.12）。Cold 失败率区间本轮跨零，warm 则严格高于零。

辅助指标按各自有效步数统计：4 kg anchor 位置 cold/warm 降低 14.07% / 15.83%，joint 速度降低 5.01% / 4.20%，anchor 线速度降低 8.41% / 7.04%，body 线速度降低 5.95% / 5.00%。4 kg cold 覆盖率 baseline 94.97%、latent 93.84%，差 -1.13 个百分点（95% CI -2.10 至 -0.15）；warm 为 94.98% / 93.94%，差 -1.04 个百分点（95% CI -1.97 至 -0.21）。Cold 覆盖率区间连续八个 checkpoint 低于零，相同 motions 的连续测试不构成独立训练复现。Cold body 旋转误差差异区间跨零，warm 高 1.31%；cold anchor 旋转低 4.66%，warm 差异区间跨零。相比 4700 轮，高负载 body/joint 相对差距缩小，但 warm 失败率及覆盖率的区间不利于 latent，仍不能认定高负载整体改善。

本轮 4 kg cold 的 baseline 41 次、latent 48 次失败均为 EE z 位置触发；warm 为 baseline 41 次全部 EE、latent 49 次（48 次 EE、1 次 anchor 姿态）。两组本轮 checkpoint SHA、评估唯一记录和训练状态保护均通过复核。Memory350 继续完成最后 200 updates 与最终八项测试。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004800.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004800.csv)；[触发项](../runs/limb_context_20260910_memory350_ppo/artifacts/failure_breakdown/update_004800.md)；[checkpoint 复核](../runs/limb_context_20260910_memory350_ppo/artifacts/periodic_checkpoint_audits/update_004800.json)。

## 最终轨迹与汇总的独立核对：已完成对照部分

03:22 UTC 对 baseline 和冻结 tracker 的 16 项最终测试逐一核对：每项 4096 条轨迹、1000-step body/joint trace 的有限性、长度、失败标记和三项终止触发一致；用 float64 重新聚合逐步 trace，检查各 motion 的 body/joint 均值；再核对全部十项均值、失败率和覆盖率。按配对共同有效时段重新计算 body/joint 点估计，并核对其余十项、失败率、覆盖率的对照点估计，均通过。原报告在 float32 trace 上聚合共同有效时段，独立核对采用 float64，允许对应的微小舍入差异。

此检查只覆盖已完成的 baseline 与 frozen tracker，未将待完成的 Memory350 视作通过。待 final_results.json 完成后，将以相同检查覆盖全部 24 项测试。Bootstrap 区间仍由已记录的配对抽样程序及已有测试支持；本次新增核对针对原始轨迹、配对条件与报告点估计。见 [对照轨迹与汇总核对](../runs/limb_context_20260910_memory350_ppo/artifacts/final_output_audits/control_reconciliation.json)。

## 4900 轮完整指标对照

2026-09-11 03:31 UTC：4900 轮 512-motion 配对测试完成。共同有效时段 body 误差：latent 在 0 kg cold/warm 分别降低 12.49% / 16.18%，在 4 kg cold/warm 分别升高 5.53% / 3.07%；joint 对应降低 8.18% / 10.47%、升高 3.92% / 2.67%。0 kg cold 为 baseline 0 次失败、latent 1 次，warm 为 baseline 1 次、latent 0 次，差异区间均含零。4 kg cold 失败率 baseline 9.18%、latent 10.35%，差 +1.17 个百分点（95% CI -0.39 至 +2.93）；warm 为 7.81% / 9.38%，差 +1.56 个百分点（95% CI +0.00 至 +3.12）。本轮两项失败率差异区间均含零，不能延续 4800 轮 warm 区间严格高于零的表述。

辅助指标按各自有效步数统计：4 kg anchor 位置 cold/warm 降低 16.87% / 16.28%，joint 速度均降低 4.28%，anchor 线速度降低 6.67% / 6.56%，body 线速度降低 4.77% / 4.64%。4 kg cold 覆盖率 baseline 94.27%、latent 93.31%，差 -0.96 个百分点（95% CI -1.94 至 +0.02）；warm 为 94.91% / 93.99%，差 -0.92 个百分点（95% CI -1.98 至 +0.12）。覆盖率点估计仍不利于 latent，但两项区间均跨零，cold 不再延续之前八个 checkpoint 区间连续低于零的情况。Body 旋转误差 cold/warm 高 1.20% / 2.09%；cold anchor 旋转差异区间跨零，warm 误差高 2.40%。相比 4800 轮，cold body 差距扩大、warm 缩小，整体仍为指标取舍。

4 kg cold 的 baseline 47 次、latent 53 次失败全部触发 EE z 位置；warm 为 baseline 40 次全部 EE、latent 48 次（47 次 EE、1 次 anchor 姿态）。本轮双方 checkpoint SHA、唯一评估记录和训练状态保护检查通过。Memory350 原 GPU 2/3 worker 已恢复并继续完成最后 100 updates；固定 5000 轮最终结果仍待完成。

[完整 tracking 表](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004900.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_004900.csv)；[触发项](../runs/limb_context_20260910_memory350_ppo/artifacts/failure_breakdown/update_004900.md)；[checkpoint 复核](../runs/limb_context_20260910_memory350_ppo/artifacts/periodic_checkpoint_audits/update_004900.json)。

## 5000 轮周期测试与两组训练完成

2026-09-11 03:43 UTC：Memory350 完成 5000 updates 和该轮 512-motion 周期测试，两个训练 worker 正常退出。共同有效时段 body 误差相对 baseline 在 0 kg cold/warm 降低 12.11% / 15.92%，4 kg 则升高 7.13% / 4.80%；joint 对应降低 8.38% / 10.65%、升高 5.61% / 4.23%。0 kg cold 两组均无失败，warm 为 baseline 0、latent 1/512，区间含零。4 kg cold 失败率 baseline 8.59%、latent 10.35%，差 +1.76 个百分点（95% CI +0.20 至 +3.52）；warm 为 9.38% / 9.18%，差 -0.20 个百分点（95% CI -1.76 至 +1.37）。

按各自有效步数统计，4 kg anchor 位置 cold/warm 降低 17.03% / 13.36%，joint 速度降低 3.72% / 4.08%，anchor 线速度降低 5.39% / 5.73%，body 线速度降低 3.87% / 4.21%；body 旋转误差则高 2.34% / 2.51%，warm anchor 旋转高 2.88%，cold anchor 旋转区间跨零。4 kg cold 覆盖率 baseline 94.65%、latent 93.57%，差 -1.08 个百分点（95% CI -2.07 至 -0.21）；warm 为 94.26% / 94.08%，差 -0.17 个百分点（95% CI -1.04 至 +0.65）。高负载仍有位置、速度及持续追踪间的取舍。4 kg cold 的 baseline 44 次、latent 53 次失败均触发 EE z 位置；warm 为 baseline 48 次全部 EE、latent 47 次（46 次 EE、1 次 anchor 姿态）。

这是周期 512-motion 协议结果，不能替代正在运行的 4096-motion 最终协议。[完整周期指标](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_005000.md)；[CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/update_005000.csv)；[失败触发项](../runs/limb_context_20260910_memory350_ppo/artifacts/failure_breakdown/update_005000.md)。

03:39 UTC 实际加载双方最终 checkpoint：tracker 全部 53 个权重/缓冲张量与原始冻结 checkpoint 逐位一致，优化器均只包含 residual actor/critic，每个参数 optimizer step=100000，精确对应 5000×5×4；所有参数与优化器 moments 均有限。Memory350 encoder 与归一化统计保持冻结。实际张量证据见 [5000 轮 checkpoint 状态审计](../runs/limb_context_20260910_memory350_ppo/artifacts/checkpoint_state_audits/baseline_005000_film_005000.json)。

03:46 UTC 完成记录复核：target=completed=5000、complete=true、stopped=false，无待进行 sampling transition，两 rank 的 actor、critic 与 critic normalizer 摘要一致。第 100–5000 轮的 50 次周期测试均齐全、训练状态保护通过，此外保留第 0 轮初始化测试，因此总评估记录为 51 条。Memory350 最终评估进程 PID 9576 已由调度器在 GPU 2/3 启动，原 worker 与周期汇总进程均正常退出；见 [训练完成记录](../runs/limb_context_20260910_memory350_ppo/artifacts/film_completion_audit/verified.json)。

完整输入复核重新枚举并 stat 全部 129827 个 NPZ，文件路径/大小/mtime 清单 SHA 与启动时相同（该清单不是全部 NPZ 内容哈希）。两 rank 共加载 48085337 帧，公共初始化 trunk、PPO 参数、reward、termination、原 tracker 六项 DR、观测扰动与四肢负载位置一致，实际每 rank 的四肢质量 SHA 在两组间一致。28 个相关源文件也匹配已记录版本，仍只有此前说明的 supervisor 恢复修订。见 [最终输入审计](../runs/limb_context_20260910_memory350_ppo/artifacts/protocol_audits/final_inputs.json)。

03:48 UTC W&B 服务端检查两组均为 finished，completed_updates=5000、最后测试 checkpoint=5000；逐页读取各自完整 history，1–5000 每轮均存在且无重复。见 [训练 W&B 完成审计](../runs/limb_context_20260910_memory350_ppo/artifacts/wandb_training_completion_audit.json)。最终 4096-motion 比较与 summary run 仍待完成。

已导出并检查完整 100–5000 轮配对曲线，包含共同有效时段 body/joint、各自有效时段 anchor 位置/body 线速度、失败率与覆盖率。正值均有利于 latent；图中区间是相同 512 条 motions 上的逐 checkpoint 配对 bootstrap，不是独立训练重复。

![完整 5000 轮周期曲线](../runs/limb_context_20260910_memory350_ppo/artifacts/plots/tracking_progress_update_005000.png)

[PDF](../runs/limb_context_20260910_memory350_ppo/artifacts/plots/tracking_progress_update_005000.pdf)；[输入 SHA 与图表口径](../runs/limb_context_20260910_memory350_ppo/artifacts/plots/tracking_progress_update_005000.json)。

## 全部完成：4096-motion 最终结果与交付核验

2026-09-11 03:57 UTC，baseline、Memory350 与冻结 tracker 的 24 项最终评估全部完成，调度器完成配对统计及 W&B summary 后正常退出。两组 residual 都使用固定第 5000 轮 checkpoint。最终报告已整理为 [memory350_ppo_results_20260910.md](memory350_ppo_results_20260910.md)。

本轮没有证明 Memory350 latent + FiLM 相对 baseline 带来整体 tracking 优势。独立均匀负载下，cold body/joint 误差分别高 1.87% / 2.86%；warm body 点估计高 0.27%，区间包含零，joint 高 2.21%。均匀负载 anchor 位置误差 cold/warm 分别降低 9.98% / 9.31%，五项速度误差也更低，但 warm 覆盖率下降 0.18 个百分点。0 kg 的十项 tracking 误差均优于 baseline，2/4 kg 的共同有效时段 body/joint 则更差，需保留这些取舍。

对照冻结 tracker 时，两种 residual 在 4 kg 下的十项误差均更低、失败率也更低，但在 0 kg 下均损失精度。Memory350 减轻了这部分低负载损失，未超过 baseline 的高负载 body/joint 精度。全部数值、配对置信区间和失败触发明细见最终报告；EE z 位置是主要终止触发，不能把触发次数直接称为摔倒次数。

最终检查覆盖全部 24 项 × 4096 条轨迹：checkpoint、物理参数、query 状态、motion/start、warm-up 协议和记忆初态配对；逐步 body/joint 与逐 motion 均值、全部十项汇总、失败标记和配对点估计一致。主报告的 body 均值与降幅现统一为共同有效时段，原自动生成报告也已保留。完整指标 CSV 为 336 行。

额外直接读取双方第 0 轮 checkpoint，按 seed 10128/20124 重建新 actor/critic，主干逐位匹配、两组公共主干一致；初始 optimizer state 为空，actor 输出层与 FiLM 头为零。W&B 两组完整 1–5000 轮 history 无缺失或重复，最终 [paired_5000_tracking_summary](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/elxgffpv) 的 1344 个数值与本地一致。最终目标的 20 项要求均核验通过，实验任务进程已退出；结论为当前配置存在局部收益，没有整体优势。

[完整目标核验](../runs/limb_context_20260910_memory350_ppo/artifacts/goal_completion_audit/verified.md)；[实际初始化权重](../runs/limb_context_20260910_memory350_ppo/artifacts/final_output_audits/initial_checkpoints.json)；[24 项轨迹核验](../runs/limb_context_20260910_memory350_ppo/artifacts/final_output_audits/final_reconciliation.json)；[最终完整 CSV](../runs/limb_context_20260910_memory350_ppo/artifacts/tracking_details/final_005000.csv)。
