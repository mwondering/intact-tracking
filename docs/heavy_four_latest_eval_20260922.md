# 四组最新 checkpoint 的实验结果（2026-09-22）

本次汇总 latent、vanilla、RMA teacher 和 Any2Track baseline 四组。当前抽样结果中，latent 与 RMA teacher 的表现接近，五项主要 tracking 误差都低于 vanilla；Any2Track 当前 checkpoint 的 tracking 误差和失败数明显更多。

快照锁定时间：`2026-09-22T00:11:35.496104+00:00`。选取当时各组最新完整 checkpoint，后台训练继续运行。

| 实验 | Checkpoint 文件编号 | 实际 PPO 更新次数 | 当前输入/适配方式 |
|---|---:|---:|---|
| Latent | 9400 | 9401 | 冻结 HDR context encoder + residual；辅助监督系数 0.5 |
| Vanilla（零 latent） | 11800 | 11801 | 相同 residual 结构，latent 恒为 0；无辅助监督 |
| RMA teacher | 8700 | 8701 | 真实 108 维 DR 参数经可训练编码器输入 residual |
| Any2Track baseline | 4600 | 4601 | 79 步本体感知/动作历史，世界模型训练 encoder，tracker 逐层 adapter |

RMA 与 Any2Track 从零开始全程 uniform；latent 与 vanilla 前 2001 次更新使用 adaptive，之后分别进行 7400、9800 次 uniform 更新。此处比较各自最新模型，训练轮数和采样历史不同。

四组当前均为 uniform motion sampling、关闭 failure rewind、每组 4 × 8192 环境、训练 seed=121，使用相同完整过滤数据集 220,480 条 motion。所有快照中冻结 tracker 的权重均已逐张量验证，仍与原始 144000 完全一致，actor 参数均为有限值。

**评测协议**

- 沿用之前的 256 条随机可行训练 motion，每条 2 次，共 512 个环境/模型/物理设置。四组 × 两种物理设置，共完成八项评测。
- Nominal：全部 512 个世界为名义物理参数、无附加负载，关闭执行器 DR、encoder bias 和外力脉冲，保留本体感知观测噪声。
- 混合 DR：实际 52 nominal + 460 HDR；HDR 使用原生 DR 和四肢负载，双手各 0–2.5 kg、双小腿各 0–4 kg，负载 COM xyz 独立 ±5 cm，256 个质量组合档位内连续采样，更新合成质量、质心和惯量。
- 评测 seed=20260921；确定性均值动作，policy FP32；冻结 context 沿用 BF16。冻结 tracker 预热 500 步，正式查询最多 500 步（10 秒），遇到失败或 motion 结束时停止计分。
- 四组均用冻结 tracker 预热 500 步再重置到配对起点。Latent 保留物理参数长历史并清空短历史；Any2Track 按训练协议清空 79 步因果历史，从查询起点在线积累，不把重置前轨迹拼入新 episode。
- 四组的 motion、起始帧、查询初始 qpos/qvel、逐环境物理与控制器参数、奖励/终止定义和完整 500 步外力均通过一致性检查。独立仿真的预热轨迹不是逐位一致。
- 误差按每个世界中四组共同有效的轨迹前缀求均值，再对世界等权平均，失败步计入、padding 不计入。成功率和覆盖率按各自完整查询计算。
- 95% 区间采用按 motion 分组的 10,000 次配对 bootstrap，两次重复保留在同一组。不是全数据集或未见 motion 泛化评测，也不是多训练种子的统计结果。

**Nominal**

| 指标 | Latent 9400 | Vanilla（零 latent） 11800 | RMA teacher 8700 | Any2Track baseline 4600 |
|---|---:|---:|---:|---:|
| 成功率 ↑ | 100.00%（512/512） | 99.61%（510/512） | 100.00%（512/512） | 97.66%（500/512） |
| 关节角整体 L2（rad） ↓ | 0.37749 | 0.42325 | 0.37795 | 1.62562 |
| 局部 body 位置（m） ↓ | 0.02603 | 0.03210 | 0.02559 | 0.06073 |
| body 姿态（rad） ↓ | 0.09066 | 0.11728 | 0.09025 | 0.31420 |
| 全局 root 位置（m） ↓ | 0.11408 | 0.13274 | 0.11164 | 0.28678 |
| 全局 body 位置（m） ↓ | 0.11668 | 0.13583 | 0.11378 | 0.30344 |
| 轨迹覆盖率 ↑ | 100.00% | 99.84% | 100.00% | 98.79% |

四组共同有效轨迹覆盖率：98.79%。

**混合 DR：52 nominal + 460 HDR**

| 指标 | Latent 9400 | Vanilla（零 latent） 11800 | RMA teacher 8700 | Any2Track baseline 4600 |
|---|---:|---:|---:|---:|
| 成功率 ↑ | 99.80%（511/512） | 99.41%（509/512） | 99.61%（510/512） | 96.68%（495/512） |
| 关节角整体 L2（rad） ↓ | 0.45341 | 0.47500 | 0.45184 | 1.72778 |
| 局部 body 位置（m） ↓ | 0.03254 | 0.03368 | 0.03155 | 0.06537 |
| body 姿态（rad） ↓ | 0.11937 | 0.12789 | 0.11789 | 0.33376 |
| 全局 root 位置（m） ↓ | 0.15098 | 0.16814 | 0.15292 | 0.32692 |
| 全局 body 位置（m） ↓ | 0.15568 | 0.17237 | 0.15672 | 0.34528 |
| 轨迹覆盖率 ↑ | 99.90% | 99.76% | 99.82% | 98.03% |

四组共同有效轨迹覆盖率：98.03%。

**Latent 与 RMA 的差距**

下表为 latent 相对 RMA 的误差下降百分比，正值表示 latent 更好。

| 环境 | 指标 | 误差下降 | 95% 区间 |
|---|---|---:|---:|
| nominal | 关节角整体 L2（rad） | +0.12% | [-0.40%, +0.62%] |
| nominal | 局部 body 位置（m） | -1.73% | [-2.65%, -0.82%] |
| nominal | body 姿态（rad） | -0.46% | [-1.37%, +0.31%] |
| nominal | 全局 root 位置（m） | -2.19% | [-5.53%, +1.01%] |
| nominal | 全局 body 位置（m） | -2.55% | [-5.84%, +0.58%] |
| mixed | 关节角整体 L2（rad） | -0.35% | [-1.06%, +0.34%] |
| mixed | 局部 body 位置（m） | -3.12% | [-4.41%, -1.88%] |
| mixed | body 姿态（rad） | -1.25% | [-2.10%, -0.40%] |
| mixed | 全局 root 位置（m） | +1.27% | [-2.59%, +4.89%] |
| mixed | 全局 body 位置（m） | +0.66% | [-3.12%, +4.23%] |

Nominal 下，latent 与 RMA 都完成全部 512 个查询。RMA 的局部 body 位置略好，其余四项差异的区间包含 0。混合 DR 下，RMA 的局部 body 位置和姿态更好；latent 的全局 root/body 位置均值略低，但区间包含 0。两者不能据此排出全面优劣。

Nominal 下，latent 相对 vanilla 的五项误差下降为 10.81%–22.70%，全部五项下降的 95% 区间均大于 0。
混合 DR 下，latent 相对 vanilla 的五项误差下降为 3.39%–10.21%，全部五项下降的 95% 区间均大于 0。

**Any2Track 当前状态**

Any2Track 4600 的关节角 L2 为 nominal 1.62562 rad、混合 DR 1.72778 rad；成功数分别为 500/512、495/512。其五项误差均明显高于另外三组。相对 vanilla，成功率分别低 1.95、2.73 个百分点，对应 bootstrap 95% 区间为 [-3.71, -0.59]、[-4.88, -0.98] 个百分点。

核对该 checkpoint 对应的第 4601 次更新日志，训练关节角 L2 为 1.83648 rad，局部 body 位置为 0.08286 m，body 姿态为 0.38289 rad。训练日志也呈现较大 tracking 误差；日志包含探索噪声且采样不同，数值不能与确定性评测直接等同。本次只确认当前 checkpoint 的表现，未据此定位算法或实现原因，也没有改变任何训练配置。

**比较边界与复现**

四组训练更新次数不同，Any2Track 的更新数尤其少。Latent 还有预训练的 context encoder，并保留跨 episode 的长期物理历史；Any2Track 使用 episode 内历史。结果反映上述训练与部署协议下各自最新 checkpoint 的表现，不是同训练预算或完全相同历史条件的消融。

本次误差使用四组共同有效前缀；与之前三组对比的公共计分时段不同，不能把两张表均值直接相减当作纯训练进步。全部逐步 trace 均已保存，可重新做相同 checkpoint/计分时段的对比。

- [完整汇总、HDR/nominal 子集及所有置信区间](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_latest_four_20260922/summary.json)
- [CSV 结果表](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_latest_four_20260922/results.csv)
- [快照、完整命令、motion 与种子协议](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_latest_four_20260922/protocol.json)
- [冻结权重与训练设置审计](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_latest_four_20260922/checkpoint_contract_audit.json)
- [各 checkpoint 对应的训练指标](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_latest_four_20260922/training_metrics_at_checkpoint.json)
- [评测代码 SHA-256](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_latest_four_20260922/evaluation_source_sha256.json)
- [比较脚本](/data_zcy/wxy/intact-tracking/scripts/compare_heavy_teacher_tracking.py)

比较脚本已支持四组，并回放上一轮三组结果验证：原三组均值、成功率与 bootstrap 区间保持不变。

```bash
.venv/bin/python scripts/compare_heavy_teacher_tracking.py \
  runs/144000-exp-heavy/baselines/evaluation_latest_four_20260922
```
