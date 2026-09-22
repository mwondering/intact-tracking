# Heavy residual：checkpoint 2000 续训与初测

2026-09-21，当前修订 `resume2000_mass5x_sampling_reset`。用户确认后实施：移除四肢负载 COM 的辅助监督、保留质量监督，将 latent 组辅助系数乘 5；两组分别从自己的 `checkpoint_2000.pt` 恢复，并重置 adaptive 统计。[上一阶段](heavy_residual_latent_baseline_plan_20260921.md) 的模型、日志和评测均保留。

## 当前训练配置

| 项目 | 144000-exp-heavy-residual-latent | 144000-exp-heavy-residual-baseline |
|---|---|---|
| GPU / 环境数 | GPU 0–3，各 8192，共 32768 | GPU 4–7，各 8192，共 32768 |
| 恢复来源 | 自己的 `ppo_4gpu8192_aux108/checkpoint_2000.pt` | 自己的同名 checkpoint |
| 文件内部 completed_updates | 2001，下一次完成更新为 2002 | 相同 |
| Actor / critic 的 latent | 5 × 64 维真实 latent | 全部 320 维恒为 0 |
| 辅助头 | 保留 108 维输出、原结构和权重 | 相同结构，辅助头不参与优化 |
| 实际有效监督 | torso COM xyz、摩擦、四肢附加质量，共 8 维 | 0 维 |
| 辅助损失总系数 | 0.1 → **0.5** | **0** |
| 模型 / optimizer / normalization / std | 完整恢复，不重新初始化 | 完整恢复，不重新初始化 |
| Adaptive 历史统计 | 重置到 visit=1、failure=0 的先验 | 相同，后续独立更新 |
| 训练更新上限 | 无 | 无 |

旧进程安全停止并保存于 latent 2227、baseline 2788 次更新；本次有意回到用户选择的 2000 号文件。该文件编号沿用 runner 的零起始 iteration，内部计数为 2001，不能将恢复后的第一轮重复标为 2001。

新目录均为 `runs/<实验名>/ppo_4gpu8192_resume2000_mass5x_sampling_reset/`。W&B 使用新的 run ID，保留 project `intact-preview-v2`、group `144000-exp-heavy` 和原实验名称，避免把回退后的更新写进旧曲线。最新进程由各实验目录的 `latest_training_process.json` 指向；旧监控归档为 `monitor_before_resume2000_mass5x`。

冻结 encoder 仍为 heavy context `update_008816.pt`，冻结 tracker 仍为 SPV5-2A `checkpoint_144000.pt`。每组覆盖完整过滤后的 **220480 条 motion**，每 rank 55120 条；每卡 820 个 nominal、7372 个 HDR 环境。四肢质量的 256 组分层采样、COM 每轴 ±5 cm、合成质量/COM/惯量计算及所有原生 DR 均保留。**去掉的是负载 COM 的预测损失，物理随机化没有关闭。** 具体范围见 [NDR/HDR 表](dr_ranges_ndr_hdr.md)。

Adaptive 继续采用原 144000 的 branch 模式、uniform branch=0.5、temperature=0.25、50 步 bin、EMA=1000、cap=200、failure rewind 概率 1/3、pre-failure window=100。恢复时丢弃累计访问、失败及 EMA 历史，按当前新环境重建正在进行的访问；之后两组各自积累统计。算法和超参数没有改动。

PPO 的 entropy=0.005、actor/critic LR=1e-4/5e-4、rollout=24、epochs=5、minibatches=4、FP32 和无界 residual scale=1 均保持。CLI 中 std=1.0 是从零初始化的默认值，本次随后由 checkpoint 中已学习的 std 覆盖。恢复源中 residual std 的均值分别为 latent **0.17962**、baseline **0.18837**。

## 辅助损失的精确定义

保留原 108 维参数顺序，新增 `dr_aux_payload_com_enabled=False`，在做误差运算前排除 12 个负载 COM 坐标；不缩小输出层，保证 checkpoint、optimizer 和部署兼容。torso 质量、Kp、Kd、armature 仍不监督，因此共有 **100 个输出不计入辅助损失**。

**归一化分母仍为 8**，不在移除 COM 后重新归一化。令 e 为物理范围归一化后的平方误差，w 为已有 350 步有效历史比例，则：

```text
L_aux = 0.5 × mean_batch[
  w / 8 × (
    e_torso_COM_x + e_torso_COM_y + e_torso_COM_z + e_friction
    + mean_four_limbs(e_payload_mass)
  )
]
```

| 保留目标 | 原有效系数（不含 w） | 新有效系数（不含 w） | 倍数 |
|---|---:|---:|---:|
| torso COM 每轴、摩擦各自 | 0.0125 | 0.0625 | 5 |
| 每个肢体的负载质量 | 0.003125 | 0.015625 | 5 |
| 负载 COM 每轴 | 0.003125 | 0 | 已移除 |

“乘 5”指保留目标的权重及其同一输入下的梯度，不表示包含已移除 COM 的旧总损失数值也会恰好乘 5。训练日志继续提供各质量的归一化 MSE / kg MAE、torso COM 的 m MAE、摩擦 MAE、有效历史比例及共享隐藏层辅助/PPO 梯度比例；不再产生负载 COM 的损失指标。baseline 只记录辅助项关闭，不生成伪造的零预测误差。

## 实现与验证

- [损失实现](../src/intact_tracking/residual_dr_aux.py)：保留分母、关闭 COM 坐标、先排除无效目标再计算误差。
- [恢复配置校验](../src/intact_tracking/memory350_heavy_policy.py)：显式 `--allow-dr-aux-change` 只允许本次 108 维 head 的 all→mass、0.1→0.5（baseline 0→0）转换，并要求 adaptive reset；其他模型、PPO、观测配置继续严格比较。
- [启动脚本](../scripts/resume_144000_heavy_residual.py)：两组新目录、新 W&B ID、原 4 卡分配、无限更新，配套健康监控和 CPU ONNX 导出。
- [核验脚本](../scripts/verify_heavy_mass_resume.py)：比较实际恢复的模型/optimizer 摘要，检查四个 rank 的采样先验、保留目标有效权重、完整数据和运行指标。

相关测试共 **48 项通过**；新增恢复测试的两处断言最初读到了尚未初始化的分布输出缓存，改为检查真实 std 参数后，两项重跑通过。两组各执行了实际 4 × 8192 的恢复检查，从 smoke u25 完成 u26–u28；四 rank 参数一致，恢复前后的模型、optimizer、normalizer、std 摘要完全相同，adaptive 在四 rank 均恢复到指定先验。两份 u28 ONNX 均通过数值验证。证据为 [preflight_verification.json](../runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/preflight_verification.json)。

恢复快照 `checkpoint_resume.pt` 在第一次新 PPO 更新前保存，包含冻结 tracker、context encoder 和新的损失配置。物理核验先统一 JSON 表示再比较：checkpoint 的 tuple、字符串枚举与 JSON 的列表、字符串等价，所有字段和值仍逐项检查。

复现启动顺序（已有目录会拒绝覆盖）：

```bash
.venv/bin/python scripts/resume_144000_heavy_residual.py --phase smoke
# 完成 u28 的 ONNX 导出并记录相关测试结果后：
.venv/bin/python scripts/verify_heavy_mass_resume.py --phase ready
.venv/bin/python scripts/resume_144000_heavy_residual.py --phase train
.venv/bin/python scripts/verify_heavy_mass_resume.py --phase launch
```

## 2000 号 checkpoint 初测

这里评估的是**修改损失之前、两组原始 `checkpoint_2000.pt`**，均已完成 2001 次更新。结果可作为本次续训的起点，不能用于判断新损失的收益。

从过滤后的训练 motion 目录随机抽取 256 条 motion，每条 2 个 query，共 512 个环境。物理 seed 与清单抽样 seed 均为 20260921。其中 460 个 HDR、52 个 nominal；冻结 tracker 先预热 500 步，再运行最多 500 步的确定性策略均值动作，FP32。每个环境先对时间求均值，再对环境等权平均。

两组逐环境的 motion、起始帧、时长、物理指纹、质量、COM、query 初始 qpos/qvel、完整 query 外力序列和 reward 合同均一致。独立仿真中的 tracker 预热轨迹并非逐位相同：首次动作相同，但第一步仿真后的 qpos/qvel 摘要不同，4 个环境的预热 reset 次数不同。两组 query 开始时长记忆均已满、短记忆均清空；baseline 丢弃所有 latent。因而本报告只确认查询条件配对，**不声称两次仿真或预热历史逐位一致**。

主表使用每对环境的**共同有效时间段**，截止两组有效长度的最小值，避免更早失败的一组因统计时段更短影响误差比较；失败步计入，停止后的零填充不计入。成功率与覆盖率仍根据各自完整 query 计算。原始逐 episode 截断统计同时保存在 JSON。

| 全部 512 个环境 | Baseline | Latent | 误差下降 |
|---|---:|---:|---:|
| 关节角 L2（rad） | 0.50545 | 0.48299 | 4.44% |
| 局部 body 位置（m） | 0.03705 | 0.03520 | 4.99% |
| Body 姿态（rad） | 0.14162 | 0.12893 | 8.96% |
| 全局 root 位置（m） | 0.20613 | 0.18948 | 8.08% |
| 全局 body 位置（m） | 0.21219 | 0.19507 | 8.07% |
| 成功率 | 99.22%（508/512） | 99.61%（510/512） | +0.39 个百分点 |
| 各自 query 覆盖率 | 99.59% | 99.75% | — |

单独看 HDR：

| 460 个 HDR 环境 | Baseline | Latent | 误差下降 |
|---|---:|---:|---:|
| 关节角 L2（rad） | 0.51044 | 0.49432 | 3.16% |
| 局部 body 位置（m） | 0.03692 | 0.03619 | 2.00% |
| Body 姿态（rad） | 0.14233 | 0.13295 | 6.59% |
| 全局 root 位置（m） | 0.21617 | 0.19840 | 8.22% |
| 全局 body 位置（m） | 0.22242 | 0.20441 | 8.10% |
| 成功率 | 99.57%（458/460） | 99.57%（458/460） | 相同 |

52 个 nominal 环境中，baseline 成功 50、latent 成功 52。关节角、局部位置、姿态误差分别下降 17.01%、30.64%、31.02%；全局 root/body 位置分别下降 5.67%、7.50%，但这一小子集的全局位置 bootstrap 区间跨过 0，不能据此认定稳定提升。总体减少的两次失败均来自同一 motion 的两个 nominal query。这只是混合环境中的 nominal 子集，不是全部 256 条 motion 的纯 nominal 评测，也不能直接比较两个子集的提升幅度来判断哪种 DR 更有利。

在该初测中 latent 的五项主要误差均更低，HDR 的改善幅度约 2%–8%。按 motion 聚类、保留两次重复的 10000 次配对 bootstrap 中，HDR 五项下降比例的 95% 区间依次为 [1.82,4.44]%、[0.50,3.47]%、[5.14,8.04]%、[3.03,13.40]%、[3.08,13.19]%。这些区间仅反映当前样本的 motion 差异，不覆盖训练 seed 或物理 seed 的不确定性。成功次数差异较小，不宣称成功率显著提高。

这仍是训练目录 motion 上、单个物理 seed 的初测，不是未见 motion 泛化评测。两组同时改变了 latent 输入和辅助监督，差异也不能全部归因于 latent 本身。

原始结果：[latent.json](../runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_u2001/latent.json)、[baseline.json](../runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_u2001/baseline.json)。汇总与配对检查：[summary.json](../runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/evaluation_u2001/summary.json)。目录内的 `protocol.json`、`motions_256.txt`、逐步 `.traces.npz` 和 `summarize.py` 可复现统计，无需重新运行仿真。

## 正式续训状态

**2026-09-21 07:33 UTC，正式恢复核验全部通过。** 每组完整载入 220480 条 motion；四 rank 的模型/optimizer/normalization/std 与各自源 checkpoint 完全一致，八个 sampler 均重置至先验。实际新更新从 2002 开始，latent 8 维 / 0.5、baseline 0 维 / 0 的日志正确，冻结来源、物理、数据、reward 和 adaptive 算法保持一致。源代码摘要与启动前通过验证的版本一致。见 [launch_verification.json](../runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/launch_verification.json)。

| 实验 | 已完成更新 | 本次新增更新 | 最近 10 次秒/更新 | 当前 W&B |
|---|---:|---:|---:|---|
| latent | 2023 | 22 | 7.64 | [在线记录](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/heavy-residual-latent-68b89eb65c) |
| baseline | 2038 | 37 | 4.66 | [在线记录](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/heavy-residual-baseline-0958d98bb8) |

记录时两组进程和各四个 worker 均健康，无错误或监控告警，训练无更新上限。latent 的有效历史权重已达 0.988，辅助/PPO 共享层梯度比约 **0.216%**；两组 mean episode length 分别约 490.5、483.6 步。新仿真初始化后 episode 日志先包含部分回合，不能将启动初期的短回合均值直接与旧 run 的稳定窗口比较；以上数据也不构成新损失改善效果的评测。完整快照为 [initial_training_status.json](../runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/initial_training_status.json)。

两组 `checkpoint_resume.pt` 均包含 encoder/tracker，并已在 CPU 导出 `deploy/checkpoint_resume/`，ONNX 数值校验通过，根目录 `policy.onnx` / `policy.json` 等别名已发布。该恢复快照仍是 u2001 的权重，带有新的损失配置；不能误记为新增更新后的模型。见 [resume_export_verification.json](../runs/144000-exp-heavy-residual-comparison/resume2000_mass5x_sampling_reset/resume_export_verification.json)。后台 CPU 导出器将继续跟进每 100 次 iteration 保存的数字编号 checkpoint，首次新周期快照为 `checkpoint_2100.pt`（completed_updates=2101）。健康监控和导出器均已验证存活。
# 后续状态

本页记录的 adaptive 分支已于 2026-09-21 正常保存并停止（latent 4528、baseline 5296 次更新）。当前两组重新从各自原始 `checkpoint_2000.pt` 恢复并改用 uniform，其他设置沿用本页；见 [Uniform 续训与预测头检查](heavy_residual_uniform_resume_20260921.md)。
