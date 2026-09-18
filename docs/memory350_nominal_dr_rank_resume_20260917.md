# Memory350：DR 中心排序与 nominal 响应距离续训

2026-09-17 按用户要求断点续训，新增 DR 标签损失系数 0.02，并将十步 A/B 的 DR→nominal 表征距离损失系数由 0.04 提高到 0.08。

## 断点和输出

- 原任务在 u7720 正常保存退出，累计 30880 个优化器步骤。
- 源 checkpoint：`runs/limb_context_20260917_memory350_nominal10_independent50_scratch_4x8192/stage1_8192/update_007720.pt`。
- 源文件 SHA256：`033a35e20f4d2e5c9ef3a7a44e92fb5650bfbed042686b056cf5a6f4f33d364d`。
- 新目录：`runs/limb_context_20260917_memory350_nominal_dr_rank_resume_4x8192/stage1_8192`。
- 入口：`intact_tracking.cli.forward_memory_nominal_dr_rank_train`，使用 `--resume-new-stage --resume-retune-weights`。
- [W&B](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/m350nomrank-63b0c5b341ce)。

恢复模型、AdamW 状态、学习率调度、归一化统计和冻结的 nominal 方向。初始学习率沿用断点的 `1e-5`；仿真及 replay 重建。新损失需要 DR 标签和跨 motion 样本，因此验证样本和验证历史重建，跨阶段比较指标时应使用同一评估集。

## 最终损失

| 项目 | 系数 |
| --- | ---: |
| 五步 teacher-forced prediction | 1 |
| 五步 recursive prediction | 0.5 |
| 同 world / episode / motion，偏移 ±5 步的局部正样本 | 0.01 |
| 十步 A/B 响应监督的 DR→nominal 单位 latent 距离 | **0.08** |
| 同 DR、不同 motion、完整且不重叠历史的正样本 | 0.008 |
| nominal 单位 latent 靠近冻结方向 | 0.01 |
| DR 参数距离监督环境 latent 中心距离排序 | **0.02** |

弱负样本仍关闭。A/B 系数通过保持 `representation_weight=0.01`、将 `representation_relation_weight=4→8` 实现，局部正样本系数不变。

DR 项比较参数距离的大小顺序，不指定绝对 latent 距离或固定 margin。环境中心由当前 encoder 编码的同 world / physics session、不同 motion 的完整且不重叠历史构成，先分别归一化 latent，再求均值。38 个物理参数坐标按十组加权；温度 0.1，最小参数距离差 0.01。详见 [实现说明](memory350_nominal_dr_rank_20260917.md)。

## 训练配置

- GPU 0–3，每卡 8192 个 A 环境；其中 8064 用于训练，128 为独立验证环境。
- 每卡 819 个全 nominal 环境，占 9.9976%；其中训练 806 个、验证 13 个。
- 其余 7373 个 DR 环境的每个固定 DR 参数独立以 50% 概率取 nominal，50% 在原 DR 范围采样。
- 手部负载最大 2.5 kg，小腿负载最大 4 kg；其余物理、观测噪声和外力采样沿用原配置。
- 全量 129827 条 motion；四卡分片覆盖完整目录。
- Memory350 short50 + long30×10，latent64，21086486 个参数；每次更新四个优化器步骤，每卡 batch 1024 / microbatch 256，全局 batch 4096。
- `--until-user-stop`，不设训练上限、不自动按平台期停止；`--updates 8000` 仅作为 cosine 调度时间尺度，学习率下限 `1e-5`。
- 暖启动至少采集 1000 步，并要求各卡训练与验证 replay 都有可用于 DR 排序的跨 motion 样本。

## 验证和运行记录

29 项相关测试通过；两卡真实仿真短程续训通过，确认新增项有有效比较样本和非零贡献、A/B 系数翻倍、七项损失加权和一致，并保留优化器、归一化及 nominal 方向。

正式四卡首个更新 u7721 已通过 22 项恢复检查：优化器步数 30884，归一化统计和 nominal 方向与源 checkpoint 完全一致，四卡模型参数一致。DR 排序原始损失 0.137372、加权贡献 0.00274745；A/B 原始损失 0.023405、加权贡献 0.00187240，分别符合 0.02 和 0.08 的系数。总损失 0.170819，七项加权和核验通过。四卡固定验证样本分别覆盖 107、104、106、105 个满足完整跨 motion 历史要求的 DR world。

12:58 UTC 检查时任务已持续推进到 u7741，四个 worker 正常运行，监控无告警。这是启动与恢复验证，尚不能据此判断新损失对聚类质量的最终改善。

新运行根目录保存 `train_launch_contract.json`、`train_process.json`、`source_stop_verified.json` 和 `smoke_verification.json`。正式训练首个 checkpoint 的恢复检查记录在 `train_verification.json`，持续健康状态记录在 `monitor/status.json`。监控只观察和记录，不停止或重启训练。
