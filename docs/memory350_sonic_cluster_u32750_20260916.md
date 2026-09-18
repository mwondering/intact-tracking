# u32750：Sonic 与 LAFAN 的聚类对照

2026-09-16。固定训练中的 u32750，另保留 u30750 作相同轨迹上的近期对照。完成 Sonic 新轨迹采集、原 LAFAN 缓存的 u32750 重编码、物理参数一致性校验、原始历史重建校验与共同环境／样本数对齐。八卡训练继续无上限运行，没有改训练参数或数据配比。

**Sonic 诊断子集上的已知环境识别明显优于 LAFAN：带负载 Top-1 99.77%，普通 DR 81.06%。两组改善主要表现为同环境跨 motion 漂移更小，簇间距离基本相同。此前仅依据 LAFAN 判定普通 DR 的整体区分能力不足，适用范围应收窄到该诊断分布。**

## 同一 u32750 的结果

| 指标 | 负载 LAFAN | 负载 Sonic | 普通 DR LAFAN | 普通 DR Sonic |
|---|---:|---:|---:|---:|
| Top-1 | 94.01% | **99.77%** | 55.90% | **81.06%** |
| Top-5 | 98.15% | **100.00%** | 77.35% | **93.49%** |
| 同 DR、跨 motion、历史不重叠距离 | 0.3033 | **0.1405** | 0.3557 | **0.2156** |
| 不同 DR、同 motion、phase 差≤0.02 距离 | 1.2636 | 1.2793 | 0.8840 | 0.8821 |
| 簇内／簇间 | 0.2400 | **0.1098** | 0.4024 | **0.2445** |
| 已知环境中心数 | 278 | 485 | 409 | 481 |
| 查询数 | 2704 | 2580 | 3152 | 2735 |

距离是 64 维单位 latent 的成对欧氏距离 RMS。中心与查询使用不同当前 motion，并严格排除原始 350 步历史重叠。主表只使用完整 short50＋long30×10 的查询。

按查询 world bootstrap，Sonic Top-1 的 95% 区间为：负载 [99.50%, 99.96%]，普通 DR [78.12%, 83.85%]。普通 DR 的点估计超过 80%，区间仍跨过 80%；这不是对所有 Sonic 动作稳定超过 80% 的保证。

## 控制环境中心数量和每个环境的样本量

同 profile 的 Sonic、LAFAN 使用逐项完全相同的固定物理 DR 参数。取两个评测可用 world 的交集，再逐 world 对齐中心和查询的样本数量：

| 条件 | 共同中心数 | 各组查询数 | LAFAN Top-1 | Sonic Top-1 | 差值及95%区间，百分点 |
|---|---:|---:|---:|---:|---|
| DR＋负载 | 262 | 1234 | 94.41% | **99.76%** | +5.35 [+3.66, +7.41] |
| 普通 DR | 390 | 1862 | 54.62% | **82.38%** | +27.77 [+23.51, +32.00] |

因此，结果差异不能仅用候选中心数量或每个中心样本数不同来解释。这里控制了物理环境和样本数量，但 motion 内容、长度和交互历史分布仍然不同。

## 数据与历史覆盖

从 129785 条 Sonic 文件的随机排列中，取最先出现的 128 个不同源动作家族，每个只保留一个演员／镜像／重复 take 版本。固定 seed=20260916，没有按识别率、跟踪成绩或长度筛选。128 条 motion 全部出现在完整历史查询中；motion 长度中位数 287 帧，范围 99–2171 帧。

每组均采集 3200 控制步，每 100 步保存一次原始历史和 latent。负载组为 512 个 DR world，手部各 0–2.5 kg、小腿各 0–4 kg；普通 DR 为 512 nominal＋512 DR，主识别率只计算其中 DR。

完整历史 DR 查询占全部 DR 查询的比例：负载 Sonic 62.83%、LAFAN 85.49%；普通 DR Sonic 67.46%、LAFAN 80.66%。本次结论条件于完整历史，不涵盖全部冷启动和短历史查询。

为复用历史对照，带负载 rollout 使用 uniform，普通 DR rollout 继承原 tracker 的 adaptive；每个 profile 内 Sonic 与 LAFAN 设置一致。当前八卡训练依然是 uniform。

长期记忆跨 episode 保留。完整历史且有 400 步回看记录的负载查询中，过去 400 步未发生 episode reset 的比例：Sonic 6.31%、LAFAN 60.66%。Sonic 中近期发生 reset 的查询更多，历史更常跨多个动作。查询时的 motion 被分到不同中心／测试集合，并不保证整段跨 episode 历史只包含其中一组 motion。原始交互历史的不重叠约束仍严格成立。

在负载组已有分类结果中，Sonic 过去 400 步无 reset 的 159 个查询 Top-1 为 98.74%，其余 2421 个查询为 99.83%。这是条件子样本描述，不是独立训练或完整历史 motion 隔离实验。

训练 motion 库按文件数量约 99.97% 为 Sonic，这与 Sonic 表现更好相符；但本次没有隔离训练占比、motion 激励和历史构成的因果作用。两套 motion 均在训练库内，这也不是 encoder 未见 motion 的泛化评测。高环境识别率不直接等于负载轻重路由率或 PPO 收益。

## 近期 checkpoint 对照

相同 Sonic 查询上，u30750→u32750：负载 Top-1 99.81%→99.77%，普通 DR 80.48%→81.06%；普通 DR 增量 +0.59 个百分点，95% 区间 [−0.44, +1.66]。近期变化较小，不能把 Sonic 与 LAFAN 的差异解释成这 2000 轮训练带来的提升。

## 校验与产物

u32750 SHA-256：`d53b35dabd244262a8643bb35a5e74236ece9bbb67a67cc8651748c1f8a5e8e2`。

两 checkpoint 的参数均有限，Adam 步数与 checkpoint 记录一致。Sonic 两组第 400/1600/3200 步的原始历史重新编码，与采集 latent 的单位距离 RMS 误差均为 0。LAFAN 第 100/3200 步的 u30750 缓存重建误差为 0，其历史 Top-1/Top-5 完全复现。物理参数没有变化，Memory350 的 physics-session invalidation 为 0。两组采集进程均正常结束。

- [完整表格及方法](../runs/limb_context_20260915_dr_center_weight04_scale02_positive02/sonic_cluster_check_032750/README.md)
- [完整数值与 bootstrap](../runs/limb_context_20260915_dr_center_weight04_scale02_positive02/sonic_cluster_check_032750/summary.json)
- [真实 64 维距离分布](../runs/limb_context_20260915_dr_center_weight04_scale02_positive02/sonic_cluster_check_032750/distance_distributions.png)
- [相同随机 16 个 world 的真实距离热图](../runs/limb_context_20260915_dr_center_weight04_scale02_positive02/sonic_cluster_check_032750/unit_distance_heatmaps.png)
- [动作清单与源文件哈希](../runs/limb_context_20260915_dr_center_weight04_scale02_positive02/sonic_cluster_check_032750/motion_manifest.json)
- [episode 历史审计](../runs/limb_context_20260915_dr_center_weight04_scale02_positive02/sonic_cluster_check_032750/capped_episode_history_audit.json)

同目录还保存原始 query、latent、物理参数、配对索引、逐查询预测、两 checkpoint 信息、实际分析代码快照、采样合同、重建校验、图像选点清单和训练健康记录。
