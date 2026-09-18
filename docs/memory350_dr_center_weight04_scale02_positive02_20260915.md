# 两项表征权重再次翻倍：中心关系 0.4、正样本 0.2

用户在 u10700 检查后要求将两个表征损失的权重再次翻倍。本阶段仅改变中心关系系数 **0.2 → 0.4**、同 DR 跨 motion 正样本系数 **0.1 → 0.2**；DR 距离映射 scale 保持 **0.2**。

旧阶段已协调停止并保存 **u11071 / 44284 optimizer steps**。新阶段于 2026-09-15 **15:51:54 UTC** 从该点恢复，使用 GPU 0–7，训练轮数不设上限。

新目录：`runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192`。

W&B：[dr-center-w04-s02-drpos02-from11071](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/drcenter8p2-5329434f4644)。

## 损失和训练设置

\[
L = L_{\mathrm{teacher},5} + 0.5 L_{\mathrm{recursive},5}
    + 0.4 L_{\mathrm{DRcenter}} + 0.2 L_{\mathrm{DRpositive}}.
\]

中心目标继续为 `2*d_DR/(d_DR+0.2)`，SmoothL1 beta 为 0.25。正样本为同一非 nominal world、同一 physics session、不同 motion、完整且不重叠的两段 350 步历史，对两侧单位 latent 的距离平方求有效 DR 对的均值。两个系数独立施加，不相乘。

两个表征系数的比例仍为 2:1；本次将它们相对 predictor 的整体强度翻倍，不改变 DR 目标距离或两项原始损失的定义。

沿用 nominal50、Memory350 encoder2x、每卡 8192 个环境、uniform motion/replay、手部负载上限各 2.5 kg、小腿各 4 kg。每卡 batch 512、microbatch 256，每 update 4 个 optimizer steps，八卡全局 batch 4096。

模型、AdamW、scheduler 和冻结归一化从 u11071 恢复；学习率继续为 `1e-5`。`--updates 8000` 保留原 cosine horizon，`--until-user-stop` 与 `stop_after_updates=null` 保证没有训练上限或自动平台期停止。仿真/replay、内部固定验证历史重建；后续聚类比较仍应使用固定外部 raw-history 缓存。

源 checkpoint：`runs/limb_context_20260915_dr_center_tuned_weight02_scale02/stage1_8192/update_011071.pt`。

SHA-256：`5cede7e8cdd865a0c437e4911d8362d1466f4726920a2cc2200c4342ce273f1c`。八 rank 参数摘要一致，所有 AdamW 状态均为第 44284 步。

## 启动前检查

复用已实现并验证过的调参恢复入口，没有修改模型或训练代码。新阶段显式设置 `--retune-representation-weight`、`--retune-dr-positive-weight`，不启用 scale 调整。真实父 run 的恢复合同校验通过。

使用 **u11000** 和旧阶段 rank 0 固定验证 batch 的前 64 条样本，在 GPU 6 上做 BF16 前向/反向，没有执行 optimizer step。有效 DR 正样本为 30 对：

| 指标 | 0.2 / 0.1 | 0.4 / 0.2 |
|---|---:|---:|
| Prediction loss | 0.08630029 | 0.08630029 |
| 加权中心关系损失 | 0.00474262 | 0.00948523 |
| 加权正样本损失 | 0.00634591 | 0.01269182 |
| 总损失 | 0.09738881 | 0.10847734 |

两项加权表征损失均准确翻倍，原始损失和 prediction loss 相同，全部模型参数梯度有限；新配置下 chunk/memory/final encoder 梯度范数为 0.29361 / 0.14133 / 0.07913。

实际启动验证已通过。首个新 checkpoint **u11072 / 44288 optimizer steps** 的八 rank 参数摘要一致，231 个模型状态张量均已更新且有限。AdamW 和 scheduler 均在原保存点基础上连续增加 4 步，模型结构及冻结归一化保持一致。

首轮训练总损失 **0.16118951 = prediction 0.13797361 + 加权中心关系 0.01142031 + 加权正样本 0.01179559**，精确符合 0.4 / 0.2 的新系数；梯度范数为 1.24929。真实有效正样本数大于零，八个 rank 持续运行，`unbounded=true`、`stop_after_updates=null`。

2026-09-15 15:57:01 UTC，W&B 服务端已读取到 **u11072** 的真实训练指标，以及中心关系 0.4、正样本 0.2、scale 0.2 的配置，run 状态为 `running`。

这些检查证明新权重已实际生效；进一步聚类收益需要后续固定样本评测。

原始证据位于新 run 根目录 `preparation/`：`weight_doubling_preflight.json`、`resume_metadata_preflight.json`、`graceful_stop_request.json`、`source_resume_audit.json`、`started_training_audit.json`、`wandb_training_online_audit.json`、父 run 停止时的配置与归一化，以及本次源码快照。实际命令及进程身份位于 `stage1_process.json`、`rank_processes.json`。
