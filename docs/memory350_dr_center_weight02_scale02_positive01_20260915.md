# DR-center 续训：中心关系 0.2、scale 0.2、同 DR 跨 motion 正样本 0.1

用户明确正样本方向为“同 DR、不同 motion，拉近环境簇”，随后将正样本系数指定为 **0.1**。原八卡任务已协调停止在 **u10338 / 41352 optimizer steps**；新阶段于 2026-09-15 15:25:18 UTC 从该保存点启动，使用 GPU 0–7，无训练轮数上限。

新目录：`runs/limb_context_20260915_dr_center_tuned_weight02_scale02/stage1_8192`。

W&B：[dr-center-w02-s02-drpos01-from10338](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/drcenter8p-0f4bb0922970)。

## 完整损失

\[
L=L_{\mathrm{teacher},5}+0.5L_{\mathrm{recursive},5}
  +0.2L_{\mathrm{DRcenter}}+0.1L_{\mathrm{DRpositive}}.
\]

中心关系仍由固定物理范围归一化后的 DR 参数距离监督：

\[
t_{ij}=\frac{2d_{\mathrm{DR},ij}}{d_{\mathrm{DR},ij}+0.2},\qquad
L_{\mathrm{DRcenter}}=\operatorname{mean}_{i<j}
\operatorname{SmoothL1}_{\beta=0.25}(\|\mu_i-\mu_j\|_2,t_{ij}).
\]

\(\mu\) 为当前 encoder 对同 world/session 两段不同 motion 历史所输出的单位 latent 的均值；同 world/session 的重复行继续合并为一个中心，均值不再次归一化。中心关系在每卡每个 microbatch 内计算。

新增正样本项：

\[
L_{\mathrm{DRpositive}}=\operatorname{mean}_{(a,b)\in P_{\mathrm{DR}}}
\left\|\frac{z_a}{\|z_a\|_2}-\frac{z_b}{\|z_b\|_2}\right\|_2^2.
\]

- 配对必须来自同一非 nominal world、同一未改变物理参数的 session，但 motion ID 不同。
- 两侧各有完整 350 步历史；复用 raw-history archive 已有的不重叠、因果检查。物理参数变化会使旧 archive 失效。
- 两侧均由当前 encoder 编码并反向传播，复用中心关系的同一次编码；每次 optimizer 更新后重新编码。
- 对 64 个坐标求和，再对有效 DR 正样本求平均，nominal 行不稀释该平均值。无有效 DR 正样本时贡献有限的零损失。
- **0.1 是独立的最终系数**，不会再乘中心关系系数 0.2。本项采用距离平方；与旧弱样本版本的 `1-cosine` 形式相差固定因子 2，系数不能直接混比。

本阶段没有额外的负样本、局部 ±5 正样本或中心容许半径损失。不同环境间的几何关系继续由 DR-center 项提供；推理端仍只输入交互历史。

## 恢复与配置

源 checkpoint：`runs/limb_context_20260915_dr_center_8gpu_weight004/stage1_8192/update_010338.pt`。

SHA-256：`bf2968bfd33a288857d2a4a466142803d3a2736670d5562ef50c0fe3a1686410`。

源 checkpoint 的八 rank 参数摘要一致，所有 AdamW 状态均为第 41352 步。恢复模型、优化器、scheduler 和冻结归一化；学习率沿用已到达的下限 `1e-5`。`--updates 8000` 仅保留原 cosine 时间尺度，配合 `--until-user-stop`、`stop_after_updates=null`，不会在 8000 或后续某一轮自动停止。

每卡 8192 个环境、训练 batch 512、microbatch 256，每 update 4 个 optimizer steps；八卡全局 batch 4096。沿用 nominal50、encoder2x Memory350、uniform motion/replay、手部负载上限各 2.5 kg、小腿各 4 kg、原 tracker DR 和同一 motion 数据集。

新阶段重建仿真/replay 和内部固定验证历史，并建立新的 W&B run。后续比较表征时应复用此前固定的外部 raw-history 聚类评测缓存；不要把内部验证曲线的恢复前后差值直接解释为训练改进。

损失恢复增加显式调参开关；旧 checkpoint 缺少 `dr_positive_weight` 时按历史值 0 恢复，不能从当前 CLI 参数意外继承 0.1。旧目录与合同不被改写。

## 验证

58 项相关测试通过，覆盖配对有效性、排除错误 world/session/同 motion/不完整历史、物理重置、双侧梯度、空样本零损失、独立系数、同次编码复用以及恢复兼容性。

使用真实 **u10250** 和旧 rank 0 固定验证 batch 的前 64 条样本，在 GPU 6 上做 BF16 前向及反向，无 optimizer 更新。有效 DR 正样本 25 对，nominal 39 行：

| 配置 | Prediction | 加权中心关系 | 加权正样本 | 总损失 |
|---|---:|---:|---:|---:|
| 原 0.04 / scale 0.3 | 0.101949 | 0.000396 | 0 | 0.102345 |
| 0.2 / scale 0.2，正样本 0 | 0.101949 | 0.012544 | 0 | 0.114493 |
| 0.2 / scale 0.2，正样本 0.1 | 0.101949 | 0.012544 | 0.020484 | 0.134977 |

所有损失和参数梯度有限；加入正样本后 chunk/memory/final encoder 梯度范数分别为 0.31408 / 0.15083 / 0.14255。真实父 run 的恢复合同校验通过。

实际启动验证已通过。首个新 checkpoint **u10339 / 41356 optimizer steps** 的八 rank 参数摘要一致，231 个模型状态张量均已更新；AdamW 连续增加 4 步，scheduler 连续增加 4 步，冻结归一化和模型结构与 u10338 完全相同。首轮训练总损失 0.18052772 = prediction 0.14065226 + 加权中心关系 0.01385134 + 加权正样本 0.02602411，梯度范数 1.20265，均有限。

各卡新固定验证 batch 中有 215–265 对有效 DR 正样本，两侧历史查询点相隔 355–430 步。训练日志里的 `dr_positive_pairs=22.953125` 是跨 rank、optimizer step、microbatch 汇总后的平均每 microbatch 对数，不是全局配对总数。

截至 2026-09-15 15:32:01 UTC，训练已到 **u10402 / 41608 optimizer steps**，`unbounded=true`、`stop_after_updates=null`，无 completion 文件，八个 rank 持续占用 GPU 0–7。W&B 服务端已读取到新阶段的真实训练更新、0.1 正样本系数以及非零正样本损失，状态为 `running`。这证明新损失已实际参与训练；聚类改善需要后续固定评测确认。

原始证据位于新 run 根目录的 `preparation/`：`user_clarification.json`、`checks.json`、`three_loss_gradient_preflight.json`、`resume_metadata_preflight.json`、`graceful_stop_request.json`、`source_resume_audit.json`、`started_training_audit.json`、`wandb_training_online_audit.json` 和修改前后源码快照。实际启动命令与进程身份在 `stage1_process.json`、`rank_processes.json`。
