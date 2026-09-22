# 定稿 Pipeline 续训：重置 adaptive sampling（2026-09-20）

用户要求从最终 checkpoint 断点续训，但 adaptive sampling 不继承旧统计。本次保持[定稿 Pipeline](final_pipeline_20260920.md)的模型结构、损失、环境分布与采样算法，单独重置采样器的历史统计。

**后续修正：** 该次运行训练至 7283 次更新后保存，并迁移到恢复原 tracker failure rewind 的版本；访问/失败计数及 EMA 继续保留。以下内容记录 6008 → 7283 阶段，最新状态见[采样对齐记录](adaptive_sampling_parity_20260920.md)。

**状态：正式续训已启动并通过恢复核对。** 原模型、优化器及归一化状态摘要完全一致，恢复 std 为 `0.12810746`，8 个 rank 的采样器均回到初始先验，DR 物理参数身份与父运行一致。首次新更新为 6009；记录见 [正式恢复核对](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_resume6008_sampling_reset_20260920/launch_verification.json)，在线指标见 [W&B 续训运行](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/144000proprio-residual-a23c35371f0e)。

## 恢复与重置范围

| 状态 | 本次处理 |
| --- | --- |
| 源模型 | 定稿运行 `checkpoint_final.pt`，已完成 6008 次更新 |
| Actor / critic / DR 辅助头 | 恢复 checkpoint 权重 |
| Optimizer / 学习率 / normalization | 恢复 checkpoint 状态，并校验全部 RSL 训练状态摘要相等 |
| Gaussian std | 恢复已学习的值，约 0.1281；不会重新设为 1.0 |
| 更新计数 | 从 6008 继续；首个新更新为 6009 |
| Adaptive visits / failures | 有效 bin 恢复 prior：visit=1、failure=0；无效 bin 为 0 |
| Pending visits / failures | 清零 |
| Adaptive EMA 时间基准 | 置空，由第一个新 PPO iteration 建立；不衰减或继承旧统计 |
| 仿真、episode 与 context 历史 | 重新建立；新 episode 的开放访问记录由新环境管理 |
| 原 checkpoint 与定稿部署文件 | 保留在父运行目录中 |

重置不改变 adaptive 的概率公式、均匀分支比例、温度、EMA 衰减参数或 DR 分布。原采样统计文件不会被读取；从本次新 rollout 重新累计访问与失败。每个 rank 的重置核对写入 `sampling_resume_audits`，操作记录写入 `resume_history`。

## 入口与文件

新增开关为 `--reset-adaptive-sampling`，必须与 `--resume` 及活动的 adaptive sampling 一起使用。省略该开关时仍按原逻辑继承采样统计。

本次续训目录：

```text
runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_resume6008_sampling_reset_20260920/
```

启动命令如下；同一 attempt 不可重复启动：

```bash
.venv/bin/python -B scripts/run_144000_residual.py --phase train \
  --run-root runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_resume6008_sampling_reset_20260920 \
  --attempt resume6008 \
  --resume runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/ppo_8gpu8192/checkpoint_final.pt \
  --reset-adaptive-sampling \
  --wandb-name 144000_exp-residual-resume6008-sampling-reset
```

正式运行仍为 8 GPU × 8192 environments，无新的更新上限。W&B 沿用 project `intact-preview-v2` 和 group `144000_exp`，在独立运行中继续记录全局更新计数。

正式训练前重新生成与当前源码匹配的 `preflight_verification.json`，保留父运行原预检记录。恢复完成、开始新更新之前保存 `ppo_8gpu8192/checkpoint_resume.pt`，其中模型与优化器对应原 6008 次状态，采样统计为新先验。后续 checkpoint 直接内嵌冻结 tracker/context 依赖，并继续自动导出 ONNX 与 JSON。

当前状态与核对结果以续训目录中的 `latest_training_process.json`、`launch_verification.json`、`ppo_8gpu8192/progress.json` 和 `monitor/health.json` 为准。
