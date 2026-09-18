# 全部固定 DR 混合后，latent 距离分布是否更有层次

结论：近 nominal 的覆盖明显增加，整体距离分布比原来均衡，但八档仍远未等频。encoder 保持不变，不能把采样覆盖的改善解释为表征辨识能力提升。

新采样每个固定标量参数独立以 50% 概率取 nominal，否则保留原随机值；足底摩擦继续共享。四肢负载、COM 各轴、躯干质量、每关节 armature 和 encoder bias 均参与混合，力脉冲和观测噪声保留。

16,384 个 DR 环境各在 mjwarp GPU 上交互 50 秒；原 tracker 和 response10/u15000 冻结。按每环境全程成熟 unit latent 的均值计算环境中心，再计算到原 nominal 中心的欧氏距离。沿用原八档等宽边界，中心不再归一化，每环境只归一档。两个旧方案均使用同样前 50 秒结果。

| 距离档，由近到远 | 原均匀采样 | 仅负载混合 | 全部固定 DR 混合 | 新平均总负载 |
|---|---:|---:|---:|---:|
| C0 | 0.01% | 1.85% | 10.10% | 0.98 kg |
| C1 | 0.03% | 1.21% | 3.00% | 1.60 kg |
| C2 | 0.04% | 1.98% | 3.23% | 1.66 kg |
| C3 | 0.15% | 3.55% | 4.88% | 1.98 kg |
| C4 | 1.16% | 8.89% | 9.85% | 2.44 kg |
| C5 | 13.29% | 42.07% | 34.47% | 3.17 kg |
| C6 | 77.55% | 39.34% | 33.47% | 4.65 kg |
| C7 | 7.78% | 1.09% | 1.00% | 7.54 kg |

近端 C0–C3：0.23% → 8.60% → 21.20%。C5+C6：90.84% → 81.41% → 67.94%。最大单档占比：77.55% → 42.07% → 34.47%。目标等频时每档应为 12.5%。

新分布主要补充了很靠近 nominal 的环境；C1–C3 仍然各只有约 3%–5%。参数逐项混合不会直接使综合 latent 距离均匀：实际仍有 75.19% 的环境至少一只手带载，87.75% 的环境至少一个 COM 轴有偏移。

平均负载随档位上升，重载 >8 kg 比轻载 <2 kg 的中心更远的概率为 98.05%，与仅负载混合的 98.24% 接近。这次改变了采样分布，没有重训网络。

个别负载组合仍可落入近端。C0 中负载最大的环境是 world 200433：双手零负载，双小腿分别 2.502 和 2.509 kg，合计 5.010 kg；全程中心距离为 0.164。C2 的最大总负载为 7.131 kg，也是双手零负载、双小腿带载。因此不能把 C0 当作“所有物理参数都小”的严格类别。

## 核验与数据

实际固定参数与上一轮保存的 16,384 环境采样表逐项完全一致；全部环境完成 2,500 控制步。累计 340,137 个成熟查询，每环境 17–22 个，平均覆盖 3.16 个 motion。物理参数和 encoder bias 全程固定。额外 2,048 个 nominal 对照未计入 DR 分布，其中留出的 1,024 个对照有 99.90% 落入 C0。

独立重放了 512 条保存的原始历史，unit latent 最大 L2 误差为 3.11e-7；重新计算全部环境中心、距离与档位后完全一致。

三组沿用相同初始种子与原连续参数抽样，但两种混合方案使用不同 nominal 掩码，motion/reset 路径也可随动力学变化。结果是各采样分布的比较，不是逐环境固定负载的单因素消融。

- [对比图](../runs/limb_context_20260917_all_fixed_dr_half_nominal_50s/analysis/latent_distance_comparison.png)
- [完整表格](../runs/limb_context_20260917_all_fixed_dr_half_nominal_50s/analysis/README.md)
- [统计 JSON](../runs/limb_context_20260917_all_fixed_dr_half_nominal_50s/analysis/summary.json)
- [环境参数和中心](../runs/limb_context_20260917_all_fixed_dr_half_nominal_50s/analysis/environment_comparison.npz)
- [独立验证结果](../runs/limb_context_20260917_all_fixed_dr_half_nominal_50s/analysis/verification.json)

采集命令：`scripts/collect_per_limb_half_zero_latents.py --sampling all-fixed-half-nominal --shard N`，N 为 0–7，各进程绑定对应 GPU。分析使用 `scripts/analyze_all_fixed_half_nominal_50s.py`；验证使用 `scripts/verify_per_limb_half_zero_50s.py --root runs/limb_context_20260917_all_fixed_dr_half_nominal_50s`。
