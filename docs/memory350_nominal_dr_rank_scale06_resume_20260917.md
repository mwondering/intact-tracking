# Memory350：u9435 评估及 scale=0.6 续训

2026-09-17，按用户要求先固定当前 checkpoint 做评估，再从该 checkpoint 恢复训练。原任务正常保存到 u9435 后退出；新任务首个 u9436 已完成，22 项恢复检查通过。

## 评估（调整 scale 之前）

对比 u8250 与 u9435，两者 scale 均为 0.3，使用相同留出查询与缓存。

| 指标 | u8250 | u9435 |
| --- | ---: | ---: |
| 普通 DR 跨 motion Top-1 | 33.41% | 34.39% |
| 普通 DR 跨 motion Top-5 | 61.96% | 63.61% |
| DR＋负载跨 motion Top-1 | 47.12% | 50.05% |
| DR＋负载跨 motion Top-5 | 68.45% | 71.02% |
| 当前分布 DR 参数距离与中心距离 Spearman | 0.2791 | 0.3148 |
| 完整历史 nominal 到固定锚点距离 RMS | 0.10089 | 0.12424 |
| 完整历史 A/B 响应与到锚点距离 Spearman | 0.93255 | 0.93762 |
| 五步 DR 预测 NMSE | 0.017389 | 0.017283 |
| 五步 nominal 预测 NMSE | 0.008512 | 0.008228 |

负载缓存 Top-1 提升 2.93 个百分点，配对 world bootstrap 95% 区间 [+1.92,+3.91]。普通 DR 的小幅提升区间包含零。nominal 锚定有所变松。缓存是旧连续 DR 分布、四肢负载 U(0,4kg)，与当前采样不同；环境识别结果不能直接代表 PPO 控制性能。

完整评估：[u9435 报告](../runs/limb_context_20260917_memory350_nominal_dr_rank_resume_4x8192/clustering_u9435/README.md)。18 项评估核验通过。评估脚本现读取各 checkpoint 保存的 scale；回归检查原有所有报告字段不变。

## 本次续训

- 源：`runs/limb_context_20260917_memory350_nominal_dr_rank_resume_4x8192/stage1_8192/update_009435.pt`。
- 新目录：`runs/limb_context_20260917_memory350_nominal_dr_rank_scale06_resume_4x8192`。
- 唯一监督配置变化：`response_distance_scale` 从 0.3 改为 0.6。目标距离为 `2D/(D+0.6)`。
- 保留 Memory350、模型、优化器、调度器、归一化与固定 nominal 方向。新阶段重建 replay 和验证探针。
- GPU 0–3，每卡 8192 环境，其中完整 nominal 819 个（训练806、验证13）；其他环境每项参数独立以50%概率取 nominal、50%概率在原 DR 范围采样。
- 无更新上限，无自动早停；`updates=8000` 是继承的调度器 horizon，不是停止条件。学习率继承为 1e-5。

| Loss | 权重 |
| --- | ---: |
| Teacher-forced prediction | 1 |
| Recursive prediction | 0.5 |
| 局部正样本 | 0.01 |
| DR 到 nominal 的十步 A/B 响应距离 | 0.08 |
| nominal 固定方向 | 0.01 |
| 同 DR 跨 motion 弱正样本 | 0.008 |
| DR 参数中心距离排序 | 0.02 |

弱负样本为0；此前讨论的替代 DR 对比方案尚未加入。

首个 checkpoint u9436 的 optimizer_steps=37744，源为37740；四卡参数一致、模型已更新且有限值、七项 loss 求和正确。`train_verification.json` 保存22项检查。只读监控位于新目录 `monitor/status.json`，不会自动停止或重启训练。

[W&B](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/m350scale06-b95850dd4b39)
