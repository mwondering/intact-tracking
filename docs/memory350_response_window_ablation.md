# Memory350：响应窗口 5 / 10 步对照及后续 residual PPO

用户已授权两条表征分支训练至 u15000，比较后自动选择 encoder，随后启动两个各占 4 张 GPU、无总轮数上限的 residual PPO。用户进一步明确：predictor 仍为 5 步；obs 压缩结构为 `1645 → 512 → 256 → 128`，再拼接 64 维 latent。

实验目录：`runs/limb_context_20260912_memory350_response_window_ablation`。
实时状态：`state.json`；监督进程日志：`supervisor.log`。

## 表征训练

|分支|GPU|恢复点|停止点|A−B 标签|predictor|
|---|---|---:|---:|---:|---:|
|response10|0–3|11000|15000|10 步|5 步|
|response5|4–7|12000|15000|5 步|5 步|

共同祖先为 `limb_context_20260912_memory350_encoder2x_nominal50_weakpairs_tune008_m11_s03/stage1_8192/update_011000.pt`，SHA256 为 `ba45c2fd63ec5589c3a944fb50ca9574bd957d278c0b4d21365cf554c634c07a`。5 步组已按用户要求连续运行到 u12000，随后才恢复。0–3 的旧 nominal50 训练已收到 SIGTERM，在 u44055 保存退出；原文件保留。

两分支均保持 nominal50、Memory350 扩容 encoder（2/4/4）、64 维 latent、predictor 架构、全部训练 motion、原归一化和固定验证集。弱正/负权重均为 0.008，负样本 margin=1.1，response scale=0.3。恢复 AdamW 和原 cosine 状态，学习率沿用 1e-5 下限；每轮 4 次 optimizer step，8192 worlds/rank。

10 步标签的实现：每次 A 收集 5 步并保留一个 5 步前瞻块，B 从对应 A anchor 的同一状态重新开始，回放全部 10 个物理 PD targets。不会把两个独立恢复的 B 五步片段拼成十步。replay 仍每 5 步生成原 5 步预测样本，局部正样本仍相差 ±5 步，encoder/弱样本档案只看到 anchor 之前的交互。任意一步 reset、motion/phase 不连续或物理参数变化，都屏蔽该十步响应关系监督，但保留有效的五步预测和局部/弱样本损失。

响应仍使用归一化物理响应差的 RMS，映射 `2D/(D+0.3)`。延长窗口不保证距离变大；需要测量实际效果。新分支需要额外 B 步进和一个初始前瞻块；旧 warmup 按 A 步数计数，十步组 warmup 结束时已入 replay 的交互比五步组少 5 步。用户接受两组恢复时点不同；本实验不能被描述为模拟器/replay 状态完全配对的单因素试验。

每 1000 轮使用已有冻结 raw-history cache 检查。普通 DR 仍为 409 个有效中心/3152 测试窗口，DR＋负载仍为 292 个中心/2761 窗口。主指标是 DR＋负载、跨 motion family、原始历史不重叠的 known-environment Top1；普通 DR、Top5、簇内/簇间距离和固定五步预测误差共同报告。最终必须直接比较两个 u15000 checkpoint。

选型规则预先固定：DR＋负载严格 Top1 更高者；完全相等时依次比较普通 DR Top1、DR＋负载 Top5，仍相等则选 response5。`context_selection.json` 记录数值、SHA 和依据，不把这项识别率解释为未见环境泛化能力。

## 后续 PPO

|项目|baseline（0–3）|latent（4–7）|
|---|---|---|
|Actor obs|同一套冻结 tracker 的 1645 维处理后特征|相同|
|Actor 压缩|1645 → 512 → 256 → 128，ELU|相同|
|latent 槽|64 维全零|选中的冻结 encoder 输出 64 维|
|Actor head|192 → 256 → 128 → 29|相同|
|Critic 压缩|原 6330 维 obs → 1024 → 512 → 256 → 128|相同|
|Critic head|192 → 256 → 128 → 1，latent 槽全零|同结构，输入真实 latent|
|初始化|actor/critic 全新、相同种子；actor 最后层为零|相同|
|每卡环境数|8192|8192|
|总训练上限|无|无|

压缩器和 residual/value heads 随 PPO 学习。冻结 tracker、context encoder 及其 normalization；PPO 不运行 predictor。保留原 1645 维信息，不使用早期的 `71+71+29` compact-state 输入。两组 head 的 latent 列初始为零，但为可训练参数；baseline 每次进入 head 前强制给零，参数量和初始权重匹配。

训练使用原 tracker DR + 四肢独立 U(0,4kg) 负载与原 force pulses，不使用 stage1 的 nominal50 混合。全 motion 数据集共 129827 条。保留原奖励；训练终止项 `no_ee_body_pos`，评估使用原完整终止项。残差 scale=0.25，初始动作标准差 0.25，rollout24，actor LR=1e-4，critic LR=5e-4，5 epochs/4 minibatches，entropy=0.0002。

前 1000 个 PPO updates 为 uniform motion sampling，随后从准确 u1000 checkpoint 恢复为 adaptive；这只是课程阶段边界，不是训练上限。之后训练循环使用无界迭代器，直到用户停止。每 100 个 PPO updates 暂停本组，做固定 512-motion 冷/热历史 × 0/4kg 端点评估；两组按照相同 update 比较，结果累计在 `ppo_comparison.json`。指标不把两组不同墙钟速度混在一起。

## 验证和自动衔接

`startup_verification.json` 核对表征分支的模型、全部损失、归一化、8 个验证文件、恢复点、4 卡设置、优化器更新次数和弱正样本活动。W&B 远端核验在 `wandb_upload_verification.json`。

`smoke/` 为 128 env/rank 的开发检查，使用 u11000 encoder，不是正式 PPO 结果。检查两组四卡更新、相同初始权重/参数量、normalizer 数值一致、严格恢复 optimizer/model、无上限继续和信号后保存，以及四卡暂停期间独立冷/热评估不改变训练状态。正式 PPO 必须等待 u15000 选型，从头初始化。

全部检查通过后写入 `PPO_READY.json`，包含程序 SHA。表征监督进程完成两条训练、每千轮检查和最终对比后，自动运行 `scripts/run_memory350_compressed_ppo.py`。该程序验证选型和程序 SHA，再启动两组无界 PPO；任一组异常会记录错误并停止另一组供检查。

根目录 `STOP` 停止整个实验；`STOP_PPO` 停止后续两个 PPO 任务。不要通过重新运行启动脚本创建重复进程。

## 2026-09-13 恢复末端高度终止

原 PPO 已保存退出，baseline 保留至 u4283、latent 至 u1360。按用户最终指定，两组从零启动，直接 adaptive，第一轮即启用原 `ee_body_pos`，继续 GPU 0–3 / 4–7 各四卡、无训练总上限。旧目录 `next_phase.json` 指向新根目录；后续进度与规则说明见 [从零直接 adaptive 的训练记录](memory350_compressed_adaptive_scratch_20260913.md)。旧结果与 checkpoint 均保留。
