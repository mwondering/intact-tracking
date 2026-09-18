**截至 2026-09-16 的实验正向收益汇总**

依据仓库实验文档、[2026-09-14 静态归档](experiment_records/20260914/README.md)及主要运行目录的原始比较 JSON 整理。此次只分析已有结果，没有启动新的 GPU 评测或改变训练。归档中的“当前运行”仅代表当时状态；以下使用已完成的对照结果，不将启动测试或孤立训练 reward 作为胜出证据。

历史上确实取得过控制收益。较值得保留的结论是：固定环境专门训练能明显改善全局轨迹；低维真实物理参数可以为 actor 提供增量信息；learned latent 在零额外负载下多次改善 body/joint 精度。尚未找到 learned latent 单策略在完整连续 DR 分布上，同时稳定改善主要 tracking 指标与失败率的充分证据。

这里有三类不同的 baseline：原始冻结 tracker、同训练预算的无 latent DR residual、同架构但环境信息置为常量的控制器。各实验只与自己的对照比较。除注明“全局”外，后期 body position 多为消除水平位置/yaw 差异后的局部 tracking 指标；它与 root/global body 的改善方向可能相反。0 kg 表示额外负载为零；在保留背景 DR 的实验中不等于 nominal physics。

**相对无 latent 或无环境信息对照，已观察到的收益**

| 策略与实验 | 对照和预算 | 已观察到的正向收益 | 代价与证据范围 |
|---|---|---|---|
| 真实 7 维物理参数 + 低秩 actor，9/7 | 同架构常量输入；每对同初始化、原奖励、固定 checkpoint500；训练 seed76/77 | 合并 body 误差降低 **2.83%**、joint **4.70%**；失败 **35→24/2016** | 两个训练种子方向一致，是环境输入贡献较直接的证据；失败差区间跨零，部分速度指标不改善；输入是真实参数，不是 learned latent |
| 100-step context + FiLM，9/8–9 | 无 latent residual；两组均 u5000；每种子每场景 4096 motions | 0 kg：seed121/122 body 降低 **11.19%/10.85%**，joint **8.89%/7.13%**；2 kg：body 降低 **4.63%/1.42%** | 0 kg 优势在两个完成种子复现。2 kg 失败分别 **46→49、36→40**；4 kg 收益未跨种子复现 |
| Memory350 + FiLM，9/10 | 无 latent residual；u5000，最终每场景 4096 motions | 0 kg cold/warm 的共同存活段 body 降低 **12.23%/16.21%**、joint **8.52%/10.96%**；warm 失败 **7→3**。Uniform 的 anchor 位置约降低 **9%–10%** | Uniform 的 body/joint 未整体改善；4 kg body 高 **6.68%/4.12%**，joint 也更差。单训练种子；FiLM 同时增加参数量；anchor 使用各自存活窗口 |
| Memory350 压缩观测后拼接 latent，9/13 | 参数量和初始权重匹配，baseline 的 latent 槽为零；从头 adaptive、原 EE termination | u2000、0 kg cold/warm 的共同存活段 body 降低 **19.35%/19.48%**、joint **9.46%/9.31%**；4 kg anchor 降低 **15.1%/13.3%** | 4 kg body 高 **2.95%/2.74%**、joint 高 **2.78%/2.93%**；失败率差区间跨零。停止前最新共同周期评测 u3600 仍有低负载收益，但幅度缩小，见下文 |
| 八个固定 DR 独立 MLP 专家，9/14 | 共享通用 MLP；每个 policy 均 u1000、1.96608 亿 PPO transitions；512 motions × 8 DR | 全局补测中 body **20.7632→17.1627 cm，降低 17.34%**；root 同样降低 **17.34%**；对齐 body 降低 **9.11%**；失败 **22→12/4096** | 7/8 DR 的全局位置改善，单训练种子。Joint 总体未改善；八个专家合计 **8 倍样本**，且按已知测试环境选专家，不是已实现的 latent 路由方案 |
| 共享 encoder 的 K-means16 MoE，9/14 | 同预算 u1000 共享 MLP；cold/warm 各 512 motions × 8 DR | Cold 全局 body **20.7035→20.2815 cm，降低 2.04%**，motion 配对 95% 区间 **[0.10%,4.16%]**；失败 **22→15** | Warm 仅降低 **0.80%**，区间跨零，失败 **21→23**；局部 body/joint/姿态更差。Cold 只弥补 baseline→专家位置差距约 **11.77%** |
| 直接调制冻结 tracker 的 obs+latent FiLM，9/16 | baseline 同样有 obs 条件化 FiLM；共同 u250，128 motions × 2 连续 DR | 对齐 body **3.1652→3.0423 cm，降低 3.88%**，motion 配对 95% 区间 **[2.45%,5.27%]** | 全局 body/root/joint 分别高 **6.71%/7.55%/4.18%**，return 略低；是早期局部收益，不能列为整体胜出 |

上述各行的依据：[7 维输入受控实验](adaptation_audit_results.md)、[7 维原始合并结果](../runs/adaptation_goal/eval_v2/input_control_940xx/compact_two_training_pairs_summary.json)、[FiLM 两种子](limb_context_seed122_latest.md)、[Memory350 最终结果](memory350_ppo_results_20260910.md)、[压缩拼接实验](memory350_compressed_adaptive_scratch_20260913.md)、[固定 DR 专家](residual_uniform_shared_vs_specialists_20260914.md)、[共享 MoE](residual_uniform_moe16_8x1024_20260914.md)、[直接 tracker FiLM](memory350_tracker_film_early_latent_benefit_20260916.md)。

真实 7 维物理输入的 body/joint 误差比 95% 区间分别为 [0.95801,0.98348] / [0.94333,0.96184]。Bootstrap 同时重采样训练对、评测种子和动作簇，但独立训练仍只有两对。同研究的另一组 FiLM actor 输入开关实验仅有 joint 约 **0.78%** 的明确小改善，body 约 **0.66%** 的区间跨零；64 维逐关节物理编码版本反而更差。因此不能把“提供更多物理维度”视为已有正向策略。

较早保留原 EE termination 的 100-step FiLM，seed121/u5000 在 0/2/4 kg 的 body 分别降低 **12.08%/3.53%/2.34%**；4 kg 失败 **413→370/4096**。后续去掉训练 EE termination 的 seed121，4 kg body 降低 **3.34%**、失败 **502→449**；但 seed122 body 高 **0.84%**、失败 **407→423**。前后每轮训练样本量也发生变化，不能归因于去掉 termination。[原 termination 结果](limb_context_original_termination_results.md)；[新批次 seed121](limb_context_no_ee_latest.md)；[seed122 原始比较](../runs/limb_context_20260908_no_ee_4gpu/additional_eval/seed122_update_005000_all_0_2_4/comparison.json)。

**压缩拼接与 grid256 的后续记录**

以下从各自 `ppo_comparison.json` 的最后一个共同评测轮次计算，取 `(baseline − latent)/baseline`。这张表使用各自 episode 存活时段的点估计，没有重新加载逐步轨迹计算共同前缀或置信区间；精度与前述完整配对报告的口径不同。它补充后续趋势，避免只选择收益最大的中期 checkpoint。

| 变体 | 最后共同周期评测 | 0 kg body 降低，cold/warm | 0 kg joint 降低，cold/warm | 4 kg body 变化，cold/warm |
|---|---:|---:|---:|---:|
| 连续 DR，压缩后拼接 | u3600 | **15.13%/14.34%** | **4.58%/4.17%** | 高 **3.12%/2.56%** |
| grid256，共享完整静态 DR 参数 | u900 | **22.95%/21.94%** | **21.04%/20.82%** | 高 **3.15%/2.47%** |
| grid256，另加 tracker action 输入 | u1300 | **23.83%/23.68%** | **12.30%/11.97%** | 高 **3.77%/2.85%** |

原始结果：[连续 DR](../runs/limb_context_20260913_memory350_compressed_ee_adaptive_scratch/ppo_comparison.json)、[grid256](../runs/limb_context_20260913_memory350_compressed_grid256_adaptive_scratch/ppo_comparison.json)、[grid256 + tracker action](../runs/limb_context_20260913_memory350_compressed_grid256_tracker_action_adaptive_scratch/ppo_comparison.json)。每项均是当次 latent 对自己的 baseline，不是不同变体之间的严格消融；不能据此单独认定离散 DR、tracker action 或压缩 MLP 导致提升。Tracker-action 的这两个版本在所列 u1300 均标为 TF32，后来检查发现与 batch 尺寸相关的策略概率数值偏差，保留该限制。[数值及 latent 通路审计](../runs/limb_context_20260913_memory350_compressed_grid256_tracker_action_adaptive_scratch/analysis_latent_pipeline_audit_001000/audit_report.md)。

**相对冻结 tracker 的收益**

这些结果说明训练补偿分支或微调有用，但不单独证明 latent 带来增益。

| 方案 | 已观察到的收益 | 适用边界 |
|---|---|---|
| 普通无 latent DR residual，Memory350 u5000 对照组 | 4 kg cold/warm，共同存活段 body 降低 **22.91%/22.94%**，joint **13.95%/14.02%**；失败率 cold **13.53%→9.94%**、warm **13.84%→10.03%** | 0 kg body 高约 **80.6%–80.7%**。残差补偿有明显重载收益，但没有保持原 tracker 的低负载精度 |
| LaFAN 直接微调原控制 MLP，C 混合负载策略 | 最大负载 body 降低 **46.49%**、joint **18.95%**，失败 **263→73/960** | Nominal body/joint 高 **45.67%/53.78%**；40 条 motion、单训练种子、u1000；不是 residual，也不是 learned latent |
| 同一 LaFAN 实验的环境专用微调 B | 最大负载 body **11.198→5.625 cm**，失败 **263→40/960**；优于混合策略 C 的 **5.993 cm、73 次失败** | 只在对应重载端点较强，nominal 退化；与全数据集实验不能直接比较绝对数值 |
| 7 维物理 teacher 的历史 context student | 对冻结 DR tracker，body 降低 **8.22%**、joint **2.11%**，失败 **15→10/1008**；替代 teacher 的物理输入后 body/joint 均值基本保持 | Student 控制器继承 teacher，102 项非 encoder 张量一致；说明已有控制能力可迁移到历史输入，没有隔离出 latent 相对无 latent residual 的额外收益；仍明显弱于固定 nominal 目标 |

依据：[Memory350 最终原始结果](../runs/limb_context_20260910_memory350_ppo/final_results.json)、[LaFAN A/B/C](lafan_abc_results.md)、[LaFAN 原始结果](../runs/abc_lafan_20260907_r2/comparison.json)、[context 替换与基线核对](adaptation_audit_results.md)、[student 对冻结 tracker 原始结果](../runs/adaptation_goal/eval_v2/context_replacement_950xx/student_vs_frozen_dr.json)。

还有一个较早的组件级正结果：同 nominal 环境下，特权前端、零 residual 初始化相对原 tracker 的 body/joint 低 **9.46%/5.50%**；随后 DR 训练的 teacher400 却高于原 tracker **4.68%/5.74%**。它说明此前部分收益来自特权前端，也说明后续训练可能损失这些收益；不能记为 learned latent 或已部署策略的胜出。[原始诊断](adaptation_audit_results.md)。

**latent 的信息是否真的被用于控制**

有闭环干预证据，作用量通常较小。这类比较回答“正确环境信息是否帮助当前策略”，与“该策略是否胜过另一条训练出来的 baseline”是两个问题。

| 实验 | 同一策略的跨 DR latent 交换结果 | 能支持的结论 |
|---|---|---|
| grid256 + tracker action，u1000 | 同 motion/phase 交换，使共同前缀 body 误差增加 **2.10%**，motion 配对 95% 区间 **[1.38%,2.98%]**；交换覆盖 **85.9%** | 正确 latent 有小幅闭环作用；仍需保留当时 TF32 数值问题的限制 |
| 在线 K-means16，u3000 | 交换使 body 增加 **1.39%**，区间 **[0.14%,2.78%]**；joint 增加 **1.35%** 的区间跨零 | 该 MoE 利用了少量环境信息；但同次连续 DR 对照的 body 仍比 MLP 高 **5.44%** |
| 直接 tracker FiLM，u250 | 同 motion/phase 交换使局部 body 增加 **1.72%**，全局 body/root 差异区间跨零 | 可测的局部作用；不足以解释成总体控制优势，置零干预也没有一致变差 |

依据：[grid256 审计](../runs/limb_context_20260913_memory350_compressed_grid256_tracker_action_adaptive_scratch/analysis_latent_pipeline_audit_001000/audit_report.md)、[MoE u3000 完整报告](../runs/limb_context_20260913_memory350_online_kmeans16_critic_action/analysis/checkpoint_003000_20260914/report.md)、[tracker FiLM 干预](memory350_tracker_film_early_latent_benefit_20260916.md)。仅动作 RMS 随 latent 改变、梯度非零或专家占用均衡，不足以替代这些闭环对照。

**已有表征/预测收益，尚不能计作控制收益的策略**

- 长期 Memory350：匹配训练 u7500 时，在“已有长期、短期不足 50 步”的验证子集，预测误差比 Short50 低 **18.15%**；但全部窗口点估计高 **6.27%**，长期为空时高 **105.29%**，整体差异区间跨零。说明长记忆在部分历史条件下有价值，不证明 PPO 更好。[Short50 对照](short50_prediction_results.md)
- DR-center 与同 DR 跨 motion 正样本训练：u11071→u30750，固定历史负载识别 Top-1 **88.54%→94.12%**、普通 DR **38.93%→54.28%**；相同固定数据的 DR 五步 NMSE 下降 **7.25%**。同时改变权重并继续训练，没有等训练时长的原权重对照；是表征/预测进展。[固定历史及预测比较](memory350_dr_center_u30750_20260916.md)
- Payload256 Top-5 的离线标定识别率、专家动作分化和 Sonic 高识别率，均没有替代对应 baseline 的正式闭环 tracking 对照。[Payload256 记录](payload256_top5_moe_20260916.md)；[Sonic 表征检查](memory350_sonic_cluster_u32750_20260916.md)

**不能据现有记录列为整体正向结果的改动**

- 取消 MoE encoder 共享：u1000 独立 MoE 相对共享 MoE，全局 body 在 cold/warm 高 **15.95%/15.14%**，失败更多；各 expert 分摊的数据少，结论限于该预算。[对照结果](residual_independent_moe16_8x1024_20260914.md)
- 仿真五步 preview：已有单 motion 真实/屏蔽预览对照，body/joint 差异接近零且区间跨无改善；相对冻结 tracker 的进步不能归因于 preview。全数据方案文档主要记录工程验证。[初步效果](simulator_preview_preliminary.md)；[全数据协议](preview_full_dataset.md)
- 关闭 EE termination、恢复 EE termination、uniform/adaptive 切换、加入 tracker action、解除 residual 限幅及提升腕部力矩：有实现和后续模型结果，但未找到能单独识别这些因素控制收益的严格配对结论；部分实验同时改变样本量、动作约束或 DR 范围。
- 当前 29-token Transformer：现有文档与本次检索没有提供正式闭环配对胜出结果；启动测试和训练曲线不能作为正收益确认。[配置与实现记录](memory350_temporal29_transformer_20260916.md)
- 早期改奖励 teacher 曾有优于固定 nominal 的平均误差，但失败风险不达标，且已不属于后来的“原奖励不变”约束；早期 v1 评测还有局部 reset 污染问题。无效的第一版 LaFAN A/B/C 也不计入正式证据。[审计与历史约束](adaptation_status.md)；[无效 LaFAN 运行](../runs/abc_lafan_20260907/INVALID_RUN.md)

这些记录提供了两个有价值的实验参照：同架构的真实 7 维参数/常量输入，用于判断环境信息是否能改善控制；固定 DR 专家与同预算通用策略，用于测量专门训练的收益。Learned latent 的低负载改善和正确/错配闭环差异则表明它曾经发挥作用，但现有收益还没有稳定覆盖完整 DR 分布。后续评估应同时保留局部 body/joint、全局 body/root 和失败率，避免单个指标改善掩盖其他退化。
