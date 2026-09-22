# RMA teacher 2300 与同轮 latent / vanilla 的 tracking 评测（2026-09-21）

本次抽样评测中，RMA teacher 和 latent 在 nominal、混合 DR 的五项主要 tracking 误差上都优于 vanilla。Nominal 下，latent 的关节角、局部 body 位置与 body 姿态更好，RMA 的全局位置更好；混合 DR 下 RMA 的五项误差均值最低，其中与 latent 的关节角差距很小，95% 区间包含 0。

使用开始检查时最新的 RMA `checkpoint_2300.pt`，匹配当前 uniform 分支中 latent、vanilla 的同名文件。三者内部 `completed_updates=2301`，文件编号从 0 开始；评测期间固定使用快照，后台训练持续运行。

| 模型 | Checkpoint 编号 | 已完成 PPO 更新 | SHA-256 |
|---|---:|---:|---|
| RMA teacher | 2300 | 2301 | `7ad89eebf4b30c4e614f906925053aca610482ff05b3aee611e56c30d798041d` |
| Latent | 2300 | 2301 | `c2feb9a468a16445e3d26cb825cfde1d2aa06e54fb73c6af14845444712b7bb6` |
| Vanilla（零 latent） | 2300 | 2301 | `e08ab1e4a396cf8880aee9eea92ca25555344b1daa9731d743ce2cfab54de178` |

**训练对照条件**

RMA 从零开始全程 uniform；latent 和 vanilla 从原 checkpoint_2000 恢复，前 2001 次更新使用 adaptive，之后又进行了 300 次 uniform 更新。因此本次匹配的是总 PPO 更新数，采样历史不同。三组使用相同冻结 tracker 144000、完整过滤数据集 220,480 条 motion 和 4 × 8192 训练环境；已逐张量验证三组的冻结 tracker 权重仍与原始 144000 完全一致。

RMA 的 actor 输入真实 108 维物理参数。Latent 使用本次 HDR 训练得到的冻结 context encoder，并保留系数 0.5 的辅助预测损失；vanilla 的 latent 恒为 0、辅助损失为 0。这不是仅改变一种输入、其他训练因素全部一致的消融。

**评测协议**

- 随机抽取可行训练目录中的 256 条 motion，每条 2 个环境，共 512 个环境/模型/物理设置。沿用之前初测的 motion manifest；不是完整数据集评测或未见 motion 测试。
- Nominal 为全部 512 个环境恢复名义物理参数、附加负载为 0；关闭执行器 DR、encoder bias 与外力脉冲，保留协议中的本体感知观测噪声。
- 混合 DR 与训练分布一致：请求 10% nominal + 90% HDR，实际按向上取整分为 52 nominal + 460 HDR。HDR 包含原生 DR、双手各 0–2.5 kg、双小腿各 0–4 kg 负载，以及附加负载 COM xyz 独立 ±5 cm。使用 256 个质量组合档位，并更新合成质量、质心和惯量。
- seed=20260921；确定性均值动作、policy FP32，冻结 context 沿用自身 BF16 推理。冻结 tracker 预热 500 步后，重置到配对查询起点；短历史清空，长期历史保留并已填满。正式查询最多 500 步（10 秒），motion 结束或失败时停止计分。
- 三组查询的 motion、起点、初始 qpos/qvel、逐环境物理与控制器参数、奖励/终止定义，以及完整 500 步外力序列均通过一致性检查。预热配置和随机种子相同，独立仿真的预热轨迹不是逐位一致；仅 latent 读取 context 历史。
- 五项误差先在每个环境的“三组共同有效轨迹前缀”求均值，再对环境等权平均。失败步计入，补零不计入。成功率、覆盖率按各自完整查询计算。
- 95% 区间使用 10,000 次按 motion 分组的配对 bootstrap，保留同一 motion 的两次重复；不包括训练随机种子的不确定性。

**Nominal**

| 指标 | RMA teacher | Latent | Vanilla |
|---|---:|---:|---:|
| 成功率 ↑ | 99.61%（510/512） | 99.80%（511/512） | 99.61%（510/512） |
| 关节角整体 L2（rad） ↓ | 0.39169 | 0.38670 | 0.43988 |
| 局部 body 位置（m） ↓ | 0.02677 | 0.02635 | 0.03480 |
| body 姿态（rad） ↓ | 0.09508 | 0.09311 | 0.12646 |
| 全局 root 位置（m） ↓ | 0.11588 | 0.12164 | 0.14446 |
| 全局 body 位置（m） ↓ | 0.11839 | 0.12451 | 0.14917 |
| 轨迹覆盖率 ↑ | 99.84% | 99.92% | 99.84% |

三组共同有效轨迹覆盖率为 99.84%。

**混合 DR（52 nominal + 460 HDR）**

| 指标 | RMA teacher | Latent | Vanilla |
|---|---:|---:|---:|
| 成功率 ↑ | 99.41%（509/512） | 99.80%（511/512） | 99.22%（508/512） |
| 关节角整体 L2（rad） ↓ | 0.47155 | 0.47262 | 0.48768 |
| 局部 body 位置（m） ↓ | 0.03314 | 0.03431 | 0.03552 |
| body 姿态（rad） ↓ | 0.12618 | 0.12804 | 0.13439 |
| 全局 root 位置（m） ↓ | 0.17322 | 0.18245 | 0.19857 |
| 全局 body 位置（m） ↓ | 0.17747 | 0.18733 | 0.20371 |
| 轨迹覆盖率 ↑ | 99.70% | 99.90% | 99.65% |

三组共同有效轨迹覆盖率为 99.51%。

**RMA 与 latent 的差距**

下表为 latent 相对 RMA 的误差下降比例，正值表示 latent 更好，负值表示 RMA 更好。

| 环境 | 指标 | Latent 相对 RMA 误差下降 | 95% 区间 |
|---|---|---:|---:|
| nominal | 关节角整体 L2（rad） | +1.27% | [+0.75%, +1.79%] |
| nominal | 局部 body 位置（m） | +1.57% | [+0.87%, +2.29%] |
| nominal | body 姿态（rad） | +2.07% | [+1.37%, +2.74%] |
| nominal | 全局 root 位置（m） | -4.97% | [-8.40%, -1.64%] |
| nominal | 全局 body 位置（m） | -5.17% | [-8.49%, -1.97%] |
| mixed | 关节角整体 L2（rad） | -0.23% | [-1.19%, +0.81%] |
| mixed | 局部 body 位置（m） | -3.53% | [-4.87%, -2.17%] |
| mixed | body 姿态（rad） | -1.47% | [-2.53%, -0.43%] |
| mixed | 全局 root 位置（m） | -5.33% | [-10.27%, -0.32%] |
| mixed | 全局 body 位置（m） | -5.56% | [-10.43%, -0.57%] |

相对 vanilla：nominal 的五项误差下降为 RMA 10.95%–24.82%、latent 12.09%–26.37%；混合 DR 中分别为 RMA 3.31%–12.88%、latent 3.09%–8.12%。这些下降比例的 motion-bootstrap 95% 区间均大于 0，结论仅针对当前 checkpoint 和抽样协议。

Latent 的失败数最少，但三组仅相差几个 episode，成功率差的 95% 区间均包含 0，不能据此认定稳定的成功率优势。混合 DR 的 460 个 HDR 世界中，RMA 与 latent 均成功 459/460，vanilla 为 458/460；52 个 nominal 世界中依次为 50/52、52/52、50/52。

混合评测中的 nominal/HDR 子集覆盖不同的 motion，子集均值不能直接当作同一批 motion 在两种物理条件下的因果对照。完整子集误差和置信区间保存在 summary.json。

**复现与原始数据**

- [完整结果、子集和 bootstrap 区间](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_rma2300_matched_nominal_mixed_20260921/summary.json)
- [Checkpoint 快照、motion、种子和六次评测命令](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_rma2300_matched_nominal_mixed_20260921/protocol.json)
- [冻结 tracker 与训练设置检查](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_rma2300_matched_nominal_mixed_20260921/checkpoint_contract_audit.json)
- [评测代码 SHA-256](/data_zcy/wxy/intact-tracking/runs/144000-exp-heavy/baselines/evaluation_rma2300_matched_nominal_mixed_20260921/evaluation_source_sha256.json)
- [三组比较与配对审计脚本](/data_zcy/wxy/intact-tracking/scripts/compare_heavy_teacher_tracking.py)

每个 `{rma_teacher,latent,vanilla}_{nominal,mixed}.json` 保留原始逐环境结果，同名前缀的 `.traces.npz` 保留全部逐步误差。

```bash
.venv/bin/python scripts/compare_heavy_teacher_tracking.py \
  runs/144000-exp-heavy/baselines/evaluation_rma2300_matched_nominal_mixed_20260921
```
