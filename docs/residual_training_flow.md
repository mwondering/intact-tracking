# Context-conditioned Forward-only 训练流程

当前入口只验证一个问题：一个统一 Transformer 能否根据历史 interaction、当前完整状态和
未来 action，预测固定 nominal 环境中的未来状态变化。Backward Predictor、Residual Policy
和 Tracking loss 已全部从活动模型、优化器与 checkpoint 中移除。

## 活动计算图

```text
S0,A0,...,S160,CURRENT,a1,...,a5
                 |
                 v
    Unified causal Transformer
      400 dim / 6 layers / 8 heads
                 |
                 v
       Δpose1,...,Δpose5
                 |
                 v
   true simulator states t+1...t+5
                 |
                 v
         Forward pose loss
```

仿真动作始终由冻结的 tracker 产生，训练期间不存在 residual action，也不存在 policy optimizer。因此 Forward 更新不会改变数据采集策略。

每个历史控制步保留一个 state 和一个 action，并与当前条件、五步未来 action 组成同一条序列：

```text
S0, A0, S1, A1, ..., S160, CURRENT, a1, a2, a3, a4, a5
```

历史 state、action、CURRENT 和未来 query action 使用各自的 token-type embedding。其中
`CURRENT = [当前71维完整状态, previous action]`。完整序列为 327 个 token，统一 Transformer
为 400 维、6 层、8 头，主干约 11.55M 参数，包含投影和输出头后约 11.8M。模型中不再存在
`[ENV]`、context latent、独立 Context Encoder 或独立 Forward Predictor。

每条 query 使用同一物理 world 中、严格早于 query 的历史。Reset boundary 不进入五步 query；
reset 后的真实状态可以作为新 query 起点，context 不因 motion reset 清空。跨 episode 的第一个
state 带有 boundary embedding，防止把 reset 跳变解释成 action response。

## 当前 Forward 目标

本版 Forward 定义为：

- 第 k 个 query action token 可以访问完整历史、CURRENT 与前 k 个未来 action；
- causal mask 保证它无法访问第 k+1 步以后的未来 action；
- 五个 query action token 分别输出相对同一个当前状态的五个非链式 pose delta；
- loss 包含 root position、root orientation 和 joint position；
- qvel 仍作为输入，但尚未作为输出目标。

因此这一步验证的是“五步 pose trajectory predictor”，还不是能够递归步进的 full-state simulator。单步 full-state delta 会在确认该基线后单独修改，避免同时改变训练链路和预测目标。

## 当前唯一实验模式：Nominal-only Forward

```bash
GPUS=0,1 ./scripts/run_forward_nominal_training.sh \
  /path/to/tracker.pt \
  /path/to/motions \
  /path/to/runs/forward_nominal \
  --wandb-name forward-nominal
```

该脚本固定 100% nominal physics、关闭 DR、nominal counterfactual simulator 和所有 pair/effect
loss，只保留 nominal Forward loss。生产默认值为每卡 4096 个环境、batch 768、replay 8192、
100000 次更新；普通参数仍可在命令末尾覆盖。统一 Transformer 架构与 nominal-only 契约由
脚本锁定，传入 DR、pair 或 Transformer 架构参数会直接报错，避免无意中切换实验。

## 重点指标

- `forward_source_nominal_loss/nmse`：在线 nominal world 的真实预测能力；
- `forward_horizon_1...5_loss/nmse`：逐预测步误差，尤其先看 horizon 1；
- `forward_nominal_zero_context_ratio`：把 context state/action 置零后的损失比；
- `forward_nominal_context_shuffle_ratio`：nominal history 互换后的损失比；

`NMSE < 1` 表示优于保持当前 pose 不变的 baseline。W&B 默认项目为 `intact-forward-world-model`，并只上传 Forward、context、优化器和 replay 指标。

多卡启动方式不变：

```bash
GPUS=0,1 ./scripts/run_forward_nominal_training.sh ...
```
