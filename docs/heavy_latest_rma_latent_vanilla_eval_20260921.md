# 最新 RMA teacher / latent / vanilla 的 tracking 评测（2026-09-21）

本次使用各自最新保存的 checkpoint：RMA teacher 2600、latent 4300、vanilla 5100。RMA 与 latent 在 nominal 和混合 DR 的五项主要 tracking 误差上都优于 vanilla。Latent 在混合 DR 的四项误差均值上低于 RMA，其中关节角下降的 95% 区间不含 0，其余四项指标的差异区间均包含 0，尚不足以认定全面超过 RMA。

与[上一轮同为 2300 的评测](heavy_rma_latent_vanilla_eval_2300_20260921.md)相比，本次按用户要求比较各自最新模型，不再对齐训练更新数。快照在开始检查时锁定，评测过程中训练持续运行。

| 模型 | Checkpoint 编号 | 实际已完成 PPO 更新 | SHA-256 |
|---|---:|---:|---|
| RMA teacher | 2600 | 2601 | `3794f0820a9491519d65be5cc76fccfe18a42378c76ae5c2d7ba251fe4546857` |
| Latent | 4300 | 4301 | `891cb493e783677ca6d2af37c6b6cf6fac384a861f35a23c44ca70de5700e04e` |
| Vanilla（零 latent） | 5100 | 5101 | `8fa2c657781c35ab2037461ad50f32c5f4167f4935c0ad94293811c5162b57bd` |

RMA 从零开始全程 uniform，共 2601 次更新；latent 和 vanilla 前 2001 次更新使用 adaptive，之后分别进行 2300、3100 次 uniform 更新。这次比较各自最新模型，训练轮数和采样历史不同。

三组当前都使用相同 uniform 采样设置、4 × 8192 环境、seed=121、完整过滤后的 220,480 条 motion。所有 checkpoint 中的冻结 tracker 都已逐张量核对，与原始 144000 权重完全一致。Latent 继续使用冻结 HDR context encoder 和系数 0.5 的辅助监督；vanilla 输入全零 latent、辅助监督为 0；RMA 输入真实物理参数。

**评测协议**

- 沿用上一轮的 256 条随机可行训练 motion，每条 2 次，共 512 个环境/模型/物理设置。不是全数据集或未见 motion 泛化测试。
- Nominal：全部 512 个环境使用名义物理参数、无附加负载，关闭执行器 DR、encoder bias 和外力脉冲，保留本体感知观测噪声。
- 混合 DR：请求 10% nominal + 90% HDR，实际为 52 nominal + 460 HDR；HDR 包含原生 DR 与四肢负载，双手各 0–2.5 kg、双小腿各 0–4 kg，附加负载 COM xyz 独立采样于 ±5 cm，使用 256 个质量档位组合，并同步更新合成质量、质心和惯量。
- 评测 seed=20260921，确定性均值动作、policy FP32；冻结 context 沿用 BF16 推理。冻结 tracker 预热 500 步后，重置配对起点并保留填满的长期历史；正式查询最多 500 步（10 秒）。
- 三组 motion、查询初始 qpos/qvel、物理及控制器参数、奖励与终止定义、完整 500 步外力序列均通过配对检查。预热配置相同，独立仿真的预热轨迹并非逐位一致。
- 每个环境先在三组共同有效轨迹前缀上平均误差，再对环境等权平均；失败步计入，不计 padding。成功率及覆盖率使用各自完整查询。
- 按 motion 分组执行 10,000 次配对 bootstrap，两次重复保留在同一组。区间仅表示当前模型、种子和抽样协议下的不确定性，不含不同训练种子的变异。

**Nominal**

| 指标 | RMA teacher 2600 | Latent 4300 | Vanilla 5100 |
|---|---:|---:|---:|
| 成功率 ↑ | 99.61%（510/512） | 99.80%（511/512） | 99.61%（510/512） |
| 关节角整体 L2（rad） ↓ | 0.38884 | 0.37790 | 0.43595 |
| 局部 body 位置（m） ↓ | 0.02644 | 0.02608 | 0.03412 |
| body 姿态（rad） ↓ | 0.09397 | 0.09191 | 0.12690 |
| 全局 root 位置（m） ↓ | 0.11651 | 0.11723 | 0.14228 |
| 全局 body 位置（m） ↓ | 0.11901 | 0.11952 | 0.14693 |
| 轨迹覆盖率 ↑ | 99.84% | 99.92% | 99.84% |

三组共同有效轨迹覆盖率：99.84%。

**混合 DR：52 nominal + 460 HDR**

| 指标 | RMA teacher 2600 | Latent 4300 | Vanilla 5100 |
|---|---:|---:|---:|
| 成功率 ↑ | 99.80%（511/512） | 99.61%（510/512） | 99.41%（509/512） |
| 关节角整体 L2（rad） ↓ | 0.46831 | 0.46054 | 0.48073 |
| 局部 body 位置（m） ↓ | 0.03324 | 0.03348 | 0.03509 |
| body 姿态（rad） ↓ | 0.12545 | 0.12505 | 0.13412 |
| 全局 root 位置（m） ↓ | 0.17189 | 0.16677 | 0.18850 |
| 全局 body 位置（m） ↓ | 0.17628 | 0.17181 | 0.19395 |
| 轨迹覆盖率 ↑ | 99.91% | 99.83% | 99.74% |

三组共同有效轨迹覆盖率：99.74%。

**Latent 相对 RMA 的差距**

误差下降为 100 × (RMA − latent) / RMA；正值表示 latent 更好。

| 环境 | 指标 | 误差下降 | 95% 区间 |
|---|---|---:|---:|
| nominal | 关节角整体 L2（rad） | +2.81% | [+2.33%, +3.30%] |
| nominal | 局部 body 位置（m） | +1.38% | [+0.55%, +2.19%] |
| nominal | body 姿态（rad） | +2.19% | [+1.62%, +2.76%] |
| nominal | 全局 root 位置（m） | -0.62% | [-3.97%, +2.64%] |
| nominal | 全局 body 位置（m） | -0.43% | [-3.70%, +2.73%] |
| mixed | 关节角整体 L2（rad） | +1.66% | [+0.77%, +2.61%] |
| mixed | 局部 body 位置（m） | -0.75% | [-2.17%, +0.67%] |
| mixed | body 姿态（rad） | +0.32% | [-0.80%, +1.51%] |
| mixed | 全局 root 位置（m） | +2.97% | [-2.30%, +8.29%] |
| mixed | 全局 body 位置（m） | +2.53% | [-2.57%, +7.72%] |

Nominal 下，latent 的关节角、局部 body 位置与 body 姿态低于 RMA，下降分别为 2.81%、1.38%、2.19%；全局位置两项均值略高，但区间包含 0。

混合 DR 下，latent 的关节角误差比 RMA 低 1.66%，95% 区间为 [0.77%, 2.61%]；其余指标的差距仍不能排除抽样波动。相对 vanilla，latent 五项误差下降 4.20%–11.53%，RMA 下降 2.58%–9.11%，两者全部五项下降区间均大于 0。

成功率差仅来自 1–2 个 episode，所有配对成功率差区间都包含 0。混合环境的 460 个 HDR 世界中，三组均成功 459/460；52 个 nominal 世界中，RMA、latent、vanilla 分别成功 52/52、51/52、50/52。子集使用不同 motion，不能把混合环境内的 nominal/HDR 子集均值当作同一组动作切换物理条件的因果对比。

本次模型训练轮数和采样历史不同，因此结论描述的是各自最新模型的表现，不是相同训练预算下的优劣。

**原始结果与复现**

- [完整结果、子集与置信区间](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_latest_rma2600_latent4300_vanilla5100_20260921/summary.json)
- [快照、数据、种子和六次命令](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_latest_rma2600_latent4300_vanilla5100_20260921/protocol.json)
- [Checkpoint 权重与训练设置检查](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_latest_rma2600_latent4300_vanilla5100_20260921/checkpoint_contract_audit.json)
- [评测代码 SHA-256](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_latest_rma2600_latent4300_vanilla5100_20260921/evaluation_source_sha256.json)
- [比较脚本](/data_zcy/wxy/intact-tracking/scripts/compare_heavy_teacher_tracking.py)

各模型在 nominal/mixed 的 JSON 及同名前缀 `.traces.npz` 保留逐环境结果和逐步误差。

```bash
.venv/bin/python scripts/compare_heavy_teacher_tracking.py \
  runs/144000-exp-heavy/baselines/evaluation_latest_rma2600_latent4300_vanilla5100_20260921
```
