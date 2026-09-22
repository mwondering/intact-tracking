# Heavy residual 最新 checkpoint 配对评测（2026-09-21）

结论：本次初测中，latent 在 Nominal 的五项主要误差上均优于零 latent baseline，但在 HDR 中均劣于 baseline。同为 checkpoint_4200.pt 时仍呈现相同趋势，因此差异不能仅由最新 baseline 多训练了 700 次更新解释。

本次评测固定使用启动时已完整保存的 checkpoint 快照；后台两组四卡训练继续运行，未调整训练设置。

| 评测模型 | checkpoint 文件编号 | 内部 completed_updates | SHA-256 |
|---|---:|---:|---|
| latent | 4200 | 4201 | `9e5105a0afc14b2e55c12bb77781db40fce682e5cf8ce3e3e822fbb02062a4e0` |
| baseline | 4900 | 4901 | `59b6f2bd5e42d3146ae77a70d0d8a7d32bfc2bc535cf3a8b1b74aa8f9d6b7a4e` |
| baseline_matched | 4200 | 4201 | `b01386efb606f79b9793a16620f05331faf3428cef2d4f2c4d3d86374cef871b` |

latent 保留已训练的 context encoder，预测辅助损失系数为 0.5，仅监督躯干 COM xyz、摩擦系数和四肢负载质量这 8 个坐标；baseline 输入 latent 恒为 0，辅助损失系数为 0。两者均从各自 2000 编号 checkpoint 恢复并重置过 adaptive sampling。本对比同时包含 latent 输入与辅助监督差异，不能当作仅有 latent 输入不同的消融。

**评测协议**

- 使用此前初测相同的 256 条随机 motion，来自过滤后的 220,480 条训练目录；每条 2 次，共 512 个环境/模型/物理设置。不是完整数据集评测，也不是留出的 motion 测试集。
- Nominal 与 HDR 各自评测全部 512 个环境。Nominal 恢复编译模型的物理参数，附加负载为 0，关闭执行器 DR、encoder bias 与外力脉冲；保留协议中的本体感知观测噪声。
- HDR 使用训练的完整 native DR 与四肢负载：双手各 0–2.5 kg、双小腿各 0–4 kg，负载 COM xyz 各自独立采样于 ±5 cm；256 个质量组合档位各有 2 个环境，组合质量、质心、惯量通过运行时检查。
- seed=20260921，确定性均值动作，policy FP32；冻结 context 推理沿用 BF16 autocast。每个世界先用冻结 tracker 预热 500 步，填满长期历史，再重置到配对查询起点；短历史清空，长期历史保留。查询最长 500 步（10 秒），遇到 motion 结束或失败时停止计分。
- 两组使用相同 motion、起始帧、查询初始 qpos/qvel、物理与控制器参数、观测噪声种子、查询外力序列。独立仿真的预热历史并非逐位一致；baseline 不使用 context。
- 误差先在每个配对环境的共同有效轨迹前缀上求均值，再对环境等权平均。失败步计入，补零不计入；成功率与覆盖率使用各自完整查询。
- 95% 区间通过 motion 聚类 bootstrap（10,000 次）计算，保留每条 motion 的两次重复；仅反映这一 checkpoint 对、一个物理种子下的 motion 抽样不确定性，不代表多训练种子的置信区间。

**最新对最新：latent 4200 / baseline 4900**

下表“误差下降”定义为 100 × (baseline − latent) / baseline；负数表示 latent 更差。

| 环境 | 指标 | Baseline 4900 | Latent 4200 | 误差下降 | 下降比例 95% 区间 |
|---|---|---:|---:|---:|---:|
| Nominal | 关节角 L2（rad） | 0.43816 | 0.39085 | +10.80% | [+9.83%, +11.75%] |
| Nominal | 局部 body 位置（m） | 0.03347 | 0.02697 | +19.42% | [+17.38%, +21.52%] |
| Nominal | body 姿态（rad） | 0.12655 | 0.09644 | +23.79% | [+22.29%, +25.23%] |
| Nominal | 全局 root 位置（m） | 0.14134 | 0.12142 | +14.09% | [+9.37%, +18.81%] |
| Nominal | 全局 body 位置（m） | 0.14498 | 0.12381 | +14.60% | [+10.08%, +19.11%] |
| HDR | 关节角 L2（rad） | 0.50111 | 0.53056 | -5.87% | [-7.68%, -4.25%] |
| HDR | 局部 body 位置（m） | 0.03514 | 0.04001 | -13.87% | [-16.60%, -11.33%] |
| HDR | body 姿态（rad） | 0.13815 | 0.15633 | -13.16% | [-15.03%, -11.30%] |
| HDR | 全局 root 位置（m） | 0.19741 | 0.21434 | -8.58% | [-15.34%, -2.06%] |
| HDR | 全局 body 位置（m） | 0.20157 | 0.22177 | -10.02% | [-16.75%, -3.60%] |

| 环境 | Baseline 成功数 / 比例 | Latent 成功数 / 比例 | Baseline 覆盖率 | Latent 覆盖率 |
|---|---:|---:|---:|---:|
| nominal | 510/512 / 99.61% | 512/512 / 100.00% | 99.8379% | 100.0000% |
| hdr | 511/512 / 99.80% | 509/512 / 99.41% | 99.9178% | 99.8326% |

五项误差下降比例的 motion-bootstrap 95% 区间在 Nominal 全部大于 0，在 HDR 全部小于 0。成功数仅相差 2 个环境，成功率差的区间包含 0，不据此宣称成功率有稳定差异。

**相同训练更新数：latent 4200 / baseline 4200**

| 环境 | 指标 | Baseline 4200 | Latent 4200 | 误差下降 |
|---|---|---:|---:|---:|
| nominal | 关节角 L2（rad） | 0.43994 | 0.39086 | +11.16% |
| nominal | 局部 body 位置（m） | 0.03383 | 0.02697 | +20.28% |
| nominal | body 姿态（rad） | 0.12760 | 0.09644 | +24.42% |
| nominal | 全局 root 位置（m） | 0.14205 | 0.12142 | +14.52% |
| nominal | 全局 body 位置（m） | 0.14586 | 0.12381 | +15.12% |
| hdr | 关节角 L2（rad） | 0.50185 | 0.53049 | -5.71% |
| hdr | 局部 body 位置（m） | 0.03518 | 0.04001 | -13.74% |
| hdr | body 姿态（rad） | 0.13798 | 0.15631 | -13.29% |
| hdr | 全局 root 位置（m） | 0.19687 | 0.21408 | -8.74% |
| hdr | 全局 body 位置（m） | 0.20143 | 0.22152 | -9.97% |

同更新数对比下，Nominal 成功数仍为 baseline 510/512、latent 512/512；HDR 为 baseline 510/512、latent 509/512。这里各项误差使用与该 baseline 配对的共同有效前缀，因此 latent 均值可能与上一张表略有不同。

**检查与解释边界**

四个配对比较（两种物理设置 × 最新/同更新数）均通过查询初始状态、逐环境物理/控制器参数、motion 时间线、奖励定义和完整 500 步外力序列检查。两种物理设置的长期历史在查询前均填满；Nominal 的实际世界数为 512/512，HDR 的实际世界数为 512/512。
本次结果没有支持“latent 在 HDR 下提高跟踪质量”的结论，但也不足以定位原因是预测损失、latent 使用方式还是训练阶段。不能仅凭本次评测认定 PPO 崩溃。之前 2000 编号评测使用的是 460 HDR + 52 Nominal 的混合世界，本次使用两个纯物理设置；不能直接把两次均值的差异解释为从 2000 开始的训练变化。

**复现与原始数据**

- [完整汇总与全部指标](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/summary.json)
- [Checkpoint、数据集与种子协议](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/protocol.json)
- [六次评测的完整命令](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/processes.json)
- [评测代码哈希](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/evaluation_source_sha256.json)
- [配对比较脚本](/data_zcy/wxy/intact-tracking/scripts/compare_heavy_residual_checkpoints.py)

- [latent / nominal 结果](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/latent_nominal.json) · [逐步 traces](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/latent_nominal.traces.npz)
- [latent / hdr 结果](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/latent_hdr.json) · [逐步 traces](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/latent_hdr.traces.npz)
- [baseline / nominal 结果](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/baseline_nominal.json) · [逐步 traces](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/baseline_nominal.traces.npz)
- [baseline / hdr 结果](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/baseline_hdr.json) · [逐步 traces](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/baseline_hdr.traces.npz)
- [baseline_matched / nominal 结果](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/baseline_matched_nominal.json) · [逐步 traces](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/baseline_matched_nominal.traces.npz)
- [baseline_matched / hdr 结果](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/baseline_matched_hdr.json) · [逐步 traces](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921/baseline_matched_hdr.traces.npz)

```bash
.venv/bin/python scripts/compare_heavy_residual_checkpoints.py runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_latest_4200_4900_20260921
```
