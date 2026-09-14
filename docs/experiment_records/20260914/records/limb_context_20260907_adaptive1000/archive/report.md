# 四肢负载实验归档：保留 ee_body_pos

本批次按用户要求结束。seed 121 的 baseline / FiLM 均完成 5000 PPO updates；seed 122 分别保存于 1000 / 379，seed 123 和 concat/constant 已取消。
结果为完整数据集训练后的固定 4096 motions/starts 测试，不是全目录 IID 评估或跨训练种子结论。

| 每部位负载 | Frozen body (cm) | Baseline body (cm) | FiLM body (cm) | FiLM body 变化 | FiLM joint 变化 | Baseline / FiLM 失败数 |
|---|---:|---:|---:|---:|---:|---:|
| 0 kg | 2.248 | 4.386 | 3.856 | -12.08% | -4.91% | 3 / 3 |
| 2 kg | 4.542 | 4.540 | 4.380 | -3.53% | -1.65% | 28 / 36 |
| 4 kg | 8.617 | 7.068 | 6.902 | -2.34% | -1.62% | 413 / 370 |

FiLM 在三档负载均降低平均 body/joint 误差。4 kg 失败数下降，0 kg 持平；2 kg 的失败数上升，所以不能把收益解释为所有指标全面提升。
相对冻结 tracker，残差策略仍表现为重载补偿与 nominal 精度的取舍。共同存活时段、置信区间及逐步轨迹见 archive/summary.json 和原始结果。

训练配置：每组 2 GPU × 8192 env，完整 129827 motions；独立 U(0,4) 四肢负载，其余 DR 关闭。uniform 1000 updates 后 adaptive 4000；可训练 actor / critic 均从头初始化。
冻结 stage1 选自 update 7800，nominal 监督表征；64 维 latent 经独立 FiLM 分支进入 actor / critic。训练和评估保留 ee_body_pos。

新实验：runs/limb_context_20260908_no_ee_4gpu；仅修改训练终止项，并按用户要求改为每组 4 GPU × 8192 env，从头训练。相较本批次每轮样本量翻倍，跨批次差距变化不能只归因于 termination。
所有模型、每轮日志与原始评估保留；不将未完成的种子作为失败或成功证据。
