# Heavy residual：真实 latent 与全零 latent 对照（上一阶段）

**本页记录历史修订 `aux108_8192`。当前配置见 [checkpoint 2000 恢复、仅保留负载质量监督并将权重乘 5](heavy_residual_resume2000_mass5x_20260921.md)。** 本页的 latent 0.1 / 20 维监督已由 0.5 / 8 维监督替代。原正式进程在更新边界保存并停止：latent 2227 次、baseline 2788 次；新进程从各自 `checkpoint_2000.pt`（内部 completed_updates=2001）恢复，adaptive 统计清零至先验。旧模型和日志保留，旧监控归档为 `monitor_before_resume2000_mass5x`。

2026-09-21，历史修订：`aux108_8192`。**该配置通过检查后，两组正式进程于 03:18 UTC 启动，03:27 UTC 完成实际 PPO 更新并通过全量启动核验。** 更早的 `ppo_4gpu16384` 已在更新边界保存并停止：latent 85 次、baseline 118 次更新，最终 checkpoint 与已验证 ONNX 保留；旧监控归档为 `monitor_16384_aux92_v1`。

## 当前实验设置

| 项目 | `144000-exp-heavy-residual-latent` | `144000-exp-heavy-residual-baseline` |
|---|---|---|
| GPU | 0–3 | 4–7 |
| 环境数 | 4 × 8192 = 32768 | 4 × 8192 = 32768 |
| Actor / critic 的 latent 输入 | 五帧真实 latent，5 × 64 维 | 对应的全部 320 维恒为零 |
| 冻结 tracker | 原 SPV5-2A checkpoint 144000 | 相同 |
| 冻结 encoder 来源 | heavy context u8816 | 同一来源，跳过 encoder 前向并使用零输入 |
| 网络结构、参数量、初始化种子 | 相同，包括 108 维独立辅助头 | 相同，保留辅助头但不使用其损失 |
| 辅助预测总系数 | **0.1** | **0** |
| 实际监督坐标 | **20** | **0** |
| Residual / critic / optimizer | 新 run 从零初始化 | 新 run 从零初始化 |

此次按用户要求同时改变 latent 输入与辅助监督。因此比较的是“真实 latent + 参数监督”相对于“零 latent + 无辅助监督”的整体收益，不能将差异单独归因于 latent。两组按相同 completed updates 或环境交互量比较；baseline 省略 encoder 前向，墙钟速度会更快。

每轮 rollout 为 24 步，每组每次更新采集 **786432** 条 transition，随后 5 epochs × 4 minibatches。相比每卡 16384 的旧配置，每次更新的采样量也减半；单轮时间下降不等价于达到同等训练效果所需的总时间下降。

## 共享隐藏层的物理监督

独立线性辅助头连接 residual actor 最后一个 128 维共享隐藏层，输出 108 维、按固定物理范围归一化的参数。标签来自当前环境的实际 DR 值，按参数名对齐；标签不进入 actor/critic 输入，encoder 的 DR 距离权重不用于回归标签。Actor 输出动作的独立头仍为 29 维。

latent 组监督：

- torso COM x/y/z 与共享摩擦，共 4 维；
- 左右手、左右小腿的附加负载质量，共 4 维，分别覆盖 0–2.5 / 0–4 kg；
- 四个负载的局部 COM 偏移 xyz，共 12 维，每轴范围 ±0.05 m；标签为附加负载相对挂载点的质心偏移，不是连杆与负载合成后的 COM。

torso 质量、29 维 Kp、29 维 Kd、29 维 armature 仍不监督，共 88 个输出不计入损失；惯量没有加入预测目标。计算误差前排除这些坐标，避免无效值污染有效损失。

为了平衡原生参数与新增负载，原来的 4 个有效 group 各取原始权重 1，新增每个坐标取 0.25。归一化后原生坐标各为 1/8，新增坐标各为 1/32。将四肢视作每类的四个样本，则质量、COM x、COM y、COM z 四类各占 1/8。

以每个参数物理范围归一化的 MSE 记为 e，以四肢平均记为 mean，辅助项为：

```text
L_aux = 0.1 × mean_batch[
  history_weight / 8 × (
    e_torso_COM_x + e_torso_COM_y + e_torso_COM_z + e_friction
    + mean_limbs(e_payload_mass)
    + mean_limbs(e_payload_COM_x)
    + mean_limbs(e_payload_COM_y)
    + mean_limbs(e_payload_COM_z)
  )
]
```

`history_weight` 沿用已有 350 步有效历史比例。总系数从 0.05 增至 0.1：原 4 个参数各自的有效系数仍为 0.0125，新增负载监督合计增加 0.05 的系数预算。该公式只约束权重，实际梯度大小取决于预测误差和历史有效性。

baseline 的 `dr_aux_coef=0`，不采集辅助 rollout 标签、不计算辅助头前向和损失，辅助头参数不更新。两组 actor 参数数均为 **1,203,366**（包括辅助头），critic 均为 **7,632,385**；actor 输入 1994 维、critic 输入 6679 维。参数量相同不表示 baseline 的辅助头也参与训练。

W&B 延续原指标组织及 group `144000-exp-heavy`，project `intact-preview-v2`，两组 run name 使用上表名称。新目录对应新 run ID，避免混入旧配置曲线：latent 为 `heavy-residual-latent-d362629ddf`，baseline 为 `heavy-residual-baseline-8b0afd6dff`。latent 新增 16 个坐标各自的 `AuxDR/payload_*_mse_normalized` 与物理单位 MAE（质量 kg、COM m），共 32 条指标；继续记录加权损失、有效历史、辅助/PPO 共享层梯度比例。baseline 记录 `AuxDR/enabled=0` 和 `AuxDR/coefficient=0`，没有伪造的辅助损失值。

## 固定来源、环境与采样

冻结 encoder 为 `runs/144000-exp-heavy/stage1_proprio122_16384/update_008816.pt`，SHA-256 为 `decd72ae598603755d0a7df93250b2925574bd71bae3bc9a6d53eefe3d057931`。context 已安全停止于 8816 次更新 / 35264 optimizer steps，八卡模型一致。输入仍为 proprio122 和实际控制命令历史；本轮不更新 encoder，不使用 forward predictor 训练 PPO。

冻结 tracker 为 `/data_zcy/wxy/SP_Tracking/logs/rsl_rl/g1_tracking/2026-09-10_16-19-06_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_4gpu_8192env_motion_data_correct/checkpoint_144000.pt`，SHA-256 为 `717fa6e627f368f880bf71ecbc4021da02d8af9e069d5bf907fe45fdcfb79e49`。

两组使用 `/data_zcy/wxy/motion_data_correct` 完整数据，按原 checkpoint 的排除列表过滤。扫描 224651 条，排除 4171 条，保留 **220480 条 motion**。每组独立覆盖全量，按 `sorted_files[rank::4]` 分片，各 rank 55120 条。过滤后有序相对路径清单 SHA-256 为 `59b8e336c152e4773133bcdcd86cf86fd6545c3c0e6cc00ce67912d107c740ac`。

Heavy 环境保持 context 阶段的合同：原 tracker 的平地、reward、termination、动作 delay/smoothing，episode 最长 500 步。每 rank 820 个 nominal、7372 个 DR 环境；四肢各分 4 个负载质量档位，共 256 组，每组分配 28–29 个 DR 环境，余数分配随 rank 轮换。组内质量连续均匀采样。负载 COM 各轴独立 U(-5 cm,+5 cm)，同步计算连杆与负载的合成质量、质心、惯量。物理参数在 world 内跨 reset 固定。

Adaptive 完全沿用[144000 采样合同](adaptive_sampling_parity_20260920.md)：branch、uniform branch 0.5、temperature 0.25、50 步 / 1 秒 bin、访问/失败先验 1/0、EMA 1000、cap 200、失败后 1/3 概率回到当前 bin 起点、pre-failure window 100。两组新训练从初始先验开始，之后独立维护失败统计，不共享采样难度权重。

原 CUDA capped sampling 包含浮点归约，不能保证相同 seed 的百万级 bin 采样逐位相同。此前在真实 rank 0 的 1,138,426 个 bin 上复现了这种差异；保留原算法，不为了强制同一采样实现而修改 sampler。证据为 `runs/144000-exp-heavy-residual-comparison/sampler_cuda_repeatability.json`。网络初始权重一致；critic 的初始统计按相同规则、各自实际观测初始化。

其他 PPO 设置不变：actor/critic LR 1e-4 / 5e-4、FP32、entropy 0.005、初始 std 1.0、seed 121，rank seed 为 `121 + 1000003 * rank`；动作等于 tracker + residual，residual 输出零初始化、无 tanh 限幅、scale=1。更新和诊断后释放空闲 PyTorch 显存块的既有修复继续启用，防止与 Warp 图分配争用。

## 检查与运行

新产物目录为 `runs/<实验名>/ppo_4gpu8192_aux108/`；检查目录为 `smoke_4gpu8192_aux108/`，证据集中在 `runs/144000-exp-heavy-residual-comparison/aux108_8192/`。检查只验证功能，不用短程结果判断策略收益。

已通过 **52 项相关测试**，两组各完成 24 次八卡更新及恢复后的第 25 次更新，组内四个 rank 的参数摘要一致；两份新 checkpoint 各通过 7 个 ONNX 数值校验案例。启动前证据为 [preflight_verification.json](../runs/144000-exp-heavy-residual-comparison/aux108_8192/preflight_verification.json)。

本次检查涵盖：20 个有效坐标的物理归一化和梯度、88 个不监督坐标排除、baseline 无损失且头不更新、共享隐藏层学习、旧 92 维 checkpoint 兼容、108 维保存恢复、原 adaptive 对齐。八卡检查使用每卡 8192 环境及单个 motion，执行 24 次更新，再恢复到第 25 次，以验证实际 batch、长历史、分布式一致性和 ONNX 导出。完整数据集的过滤、分片和实际采样在正式启动时另行核对；旧完整数据集检查证据保留在父目录。

运行入口：

```bash
.venv/bin/python scripts/run_144000_heavy_residual.py --phase smoke
.venv/bin/python scripts/run_144000_heavy_residual.py --phase smoke --resume-final --attempt aux108_8192_resume
# 将各组第 25 次 checkpoint 导出至其 smoke 目录的 export_u25/
.venv/bin/python scripts/verify_144000_heavy_residual.py --phase ready
.venv/bin/python scripts/run_144000_heavy_residual.py --phase train
.venv/bin/python scripts/verify_144000_heavy_residual.py --phase launch
```

启动器拒绝覆盖已有训练；正式启动要求源码与通过验证的版本一致。两组正式训练从头初始化，不继承 smoke 的权重或采样状态，默认 **无更新上限**。checkpoint 包含冻结 tracker 与 context encoder，保存的 `latent_input_mode` 同时约束训练、恢复、评测、部署。编号沿用 runner 规则，`checkpoint_0.pt` 内 `completed_updates=1`，按文件内计数判断进度。

健康记录为 `monitor/health.json`，保留短 episode、辅助梯度过大等警报；监控只记录和报告，不自动修改训练。CPU 导出器维护最新 `policy.onnx`、`policy.json`、`deploy_metadata.json` 和 `policy_runtime.py`。旧配置训练后期出现过短 episode，需要继续观察新配置的实际跟踪表现，数值有限不能作为训练效果良好的充分证据。

## 正式启动实测

八个训练 worker、完整数据过滤/分片、每卡 8192 环境、两组初始网络权重、heavy 物理参数、原 adaptive、20/0 维实际监督、冻结来源和在线 W&B 均已核对，见 [launch_verification.json](../runs/144000-exp-heavy-residual-comparison/aux108_8192/launch_verification.json)。首份 checkpoint 与 ONNX/JSON 已生成，CPU 导出数值验证通过。

| 实验 | 状态记录时完成更新 | 近期秒/更新 | W&B |
|---|---:|---:|---|
| latent | 17 | 7.58 | [在线记录](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/heavy-residual-latent-d362629ddf) |
| baseline | 26 | 4.86 | [在线记录](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/heavy-residual-baseline-8b0afd6dff) |

速度为启动后最近 10 次更新的短窗口墙钟测量，后续会受 episode 长度、reset 和保存开销影响。监控在记录时没有运行时错误。早期辅助梯度未超过 PPO，但这些检查不构成策略性能提升的评测结论。状态快照为 [initial_training_status.json](../runs/144000-exp-heavy-residual-comparison/aux108_8192/initial_training_status.json)。
