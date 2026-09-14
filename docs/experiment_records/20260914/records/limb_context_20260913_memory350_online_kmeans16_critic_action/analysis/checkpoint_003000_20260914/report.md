# MoE 第 3000 轮 checkpoint 评测

结论：MoE 尚未整体优于不带 latent 的 MLP baseline。0 kg 时 joint 误差改善，但 body 误差变大；4 kg 时 root 位置误差改善，body 和 root 姿态误差仍更大。正确 latent 在连续 DR 交换测试中带来的增益较小。

两组均为完成 3000 次 PPO 更新的 checkpoint。每个固定负载场景含相同的 512 个 motion、初始状态和物理环境；固定负载是四条肢体各加指定质量，仍保留 tracker 的其余 DR。热启动先由冻结 tracker 交互 500 步，测试最多 1000 步。

Body/joint 使用两组共同有效时段；root 使用各自有效步数。Body 与 root 位置单位为米，joint 是 29 关节角度误差向量的 L2 范数；root 姿态单位为弧度。所有指标越低越好。

| 场景 | Body：baseline → MoE | Joint：baseline → MoE | Root 位置：baseline → MoE | Root 姿态：baseline → MoE | 失败率：baseline → MoE |
|---|---:|---:|---:|---:|---:|
| 每肢 0 kg，冷启动 | 0.04350 → 0.04524 | 0.60783 → 0.57835 | 0.20217 → 0.19889 | 0.08139 → 0.08429 | 0.20% → 0.00% |
| 每肢 0 kg，热启动 | 0.04363 → 0.04553 | 0.60840 → 0.58281 | 0.20187 → 0.20197 | 0.08142 → 0.08427 | 0.20% → 0.20% |
| 每肢 4 kg，冷启动 | 0.05952 → 0.06025 | 0.84090 → 0.84004 | 0.48196 → 0.45273 | 0.08495 → 0.09154 | 5.86% → 5.86% |
| 每肢 4 kg，热启动 | 0.05927 → 0.06003 | 0.83863 → 0.83866 | 0.48734 → 0.46280 | 0.08463 → 0.09194 | 6.05% → 5.66% |

热启动下，MoE 的 body 误差在 0 kg 增大 4.36%（motion bootstrap 95% CI：3.58%–5.15%），在 4 kg 增大 1.29%（0.03%–2.51%）。0 kg 的 joint 误差降低 4.21%，4 kg joint 基本持平。4 kg 的 root 位置误差降低 5.04%，root 姿态误差增大 8.64%；其失败率差异为 −0.39 个百分点（95% CI：−1.56 至 +0.59），不足以说明失败率已可靠改善。

## 连续随机 DR 与 latent 交换

使用另外固定的 128 个 motion，每个 motion 两组相同运动起点、独立连续 DR 的物理环境，共 256 个 episode。交换测试只在两个环境均存活且两份历史完整时交换 latent；保持各自的状态、参考轨迹与动作执行环境。

连续 DR 下共同有效时段 body：baseline 0.04373 m → MoE 0.04610 m，MoE 误差变化 +5.44%；joint 变化 -0.32%。
将 MoE 的正确 latent 换成另一 DR 环境的 latent，body 误差增加 1.39%（95% CI：0.14% 至 2.78%），joint 增加 1.35%（-0.24% 至 3.75%）。两种 latent 下失败率均为 0%；交换覆盖 86.30% 的有效步数。

## 路由与实现检查

Checkpoint 中有 16×64 个中心元素，中心更新计数为 3000；actor/critic 路由状态一致，二者编码器与 head 无共享参数，critic 的 tracker action 输入为 29 维。第 3000 轮 16 个 expert 均使用，样本占比 4.77%–8.39%，有效 expert 数 15.80/16；最近 100 轮始终有 16 个 active expert。
训练观测上打乱 latent 的动作变化 RMS 为 0.143，残差动作 RMS 为 0.203。这表明路由会影响策略输出；该打乱操作没有匹配 motion，不能单独作为正确环境信息有用的证据。闭环收益以配对 latent 交换测试为准。

当前更准确的判断是：路由正常、latent 已影响输出，正确环境信息带来的闭环增益仍小，整体控制性能尚未超过 MLP。测试结论仅针对当前单个训练 seed 与固定评测子集；区间描述 motion 间的不确定性，不代表跨训练 seed 的稳定性。训练继续，未改训练参数。

## 文件

- MoE checkpoint：`/data_zcy/wxy/intact-tracking/runs/limb_context_20260913_memory350_online_kmeans16_critic_action/ppo/concat_121/checkpoint_update_003000.pt`。
- Baseline checkpoint：`/data_zcy/wxy/intact-tracking/runs/limb_context_20260913_memory350_online_kmeans16_critic_action/ppo/baseline_121/checkpoint_update_003000.pt`。
- `paired_results.json`：四场景配对 bootstrap、完整 root/body/joint 指标、失败项。
- `latent_usage_comparison.json`：连续 DR 配对对照与 latent 交换。
- `router_audit.json`：中心、占用率、动作敏感性与输入检查。
- `metrics.csv`：完整指标。
