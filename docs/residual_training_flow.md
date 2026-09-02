# Context-conditioned Forward-only 训练流程

当前入口只验证一个问题：历史 interaction context 能否帮助 Forward Predictor 在固定物理环境中预测未来状态变化。Backward Predictor、Residual Policy 和 Tracking loss 已全部从活动模型、优化器与 checkpoint 中移除。

## 活动计算图

```text
16-token history context -> Context Encoder -> world latent z
                                                    |
current 71-D state + previous action + five actions v
                                      Forward Predictor
                                             |
                           five non-chained pose deltas
                                             |
                          true simulator states t+1...t+5
                                             |
                                      Forward pose loss
```

仿真动作始终由冻结的 tracker 产生，训练期间不存在 residual action，也不存在 policy optimizer。因此 Forward 更新不会改变数据采集策略。

每个 context token 仍为：

```text
[proprio_before, five executed tracker actions, proprio_after]
```

每条 query 使用同一物理 world 中、严格早于 query 的 16 个 token。Reset boundary 不进入五步 query；reset 后的真实状态可以作为新 query 起点，context 不因 motion reset 清空。

## 当前 Forward 目标

为了先隔离训练链路，本版暂时保留原 Forward 定义：

- 输入为当前完整 71-D state、previous action、五步 tracker action 和 context latent；
- 输出为相对同一个当前状态的五个非链式 pose delta；
- loss 包含 root position、root orientation 和 joint position；
- qvel 仍作为输入，但尚未作为输出目标。

因此这一步验证的是“五步 pose trajectory predictor”，还不是能够递归步进的 full-state simulator。单步 full-state delta 会在确认该基线后单独修改，避免同时改变训练链路和预测目标。

## 两种实验模式

### 1. Nominal-only 基线

```bash
./scripts/run_residual_training.sh \
  /path/to/tracker.pt \
  /path/to/motions \
  /path/to/runs/forward_nominal \
  --num-envs 4096 \
  --batch-size 512 \
  --nominal-rollout-fraction 1.0 \
  --nominal-pair-batch-size 0 \
  --wandb-name forward-nominal
```

这一步只回答 nominal 动力学能否被模型拟合。`nominal-rollout-fraction=1.0` 时，若未显式指定 pair batch，程序也会自动关闭 nominal counterfactual simulator。

### 2. Nominal/DR 混合与反事实配对

```bash
./scripts/run_residual_training.sh \
  /path/to/tracker.pt \
  /path/to/motions \
  /path/to/runs/forward_mixed \
  --num-envs 4096 \
  --batch-size 512 \
  --nominal-rollout-fraction 0.5 \
  --nominal-pair-batch-size 512 \
  --wandb-name forward-mixed
```

一半在线 world 为 nominal，一半为固定 startup DR。每个 model batch 还可把抽样起点恢复到独立 nominal simulator，重放相同五步动作，继续使用 nominal target、DR-minus-nominal effect 与 nominal consistency 三项 Forward 监督。

## 重点指标

- `forward_source_nominal_loss/nmse`：在线 nominal world 的真实预测能力；
- `forward_source_dr_loss/nmse`：在线 DR world 的真实预测能力；
- `forward_horizon_1...5_loss/nmse`：逐预测步误差，尤其先看 horizon 1；
- `forward_nominal/dr_zero_context_ratio`：把 context token 置零后的损失比；
- `forward_nominal_context_shuffle_ratio`：nominal history 互换后的损失比；
- `forward_dr_context_shuffle_ratio`：不同 DR history 互换后的损失比；
- `nominal_effect_nmse`：DR effect 预测相对“预测无 DR effect”的 NMSE；
- `nominal_context_swap_ratio`：DR/nominal context 对调后的配对损失比。

`NMSE < 1` 表示优于保持当前 pose 不变的 baseline。W&B 默认项目为 `intact-forward-world-model`，并只上传 Forward、context、优化器和 replay 指标。

多卡启动方式不变：

```bash
GPUS=0,1 ./scripts/run_residual_training.sh ...
```
