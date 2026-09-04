# Frozen-tracker residual PPO

这一阶段直接回答一个问题：Forward Predictor 学到的 dynamics latent 能否帮助冻结的
SPV5-2A tracker 获得更高的实际环境 reward。实现包含两个只相差 latent 输入的 residual PPO
基线。

## 两个基线

`latent` 基线的动作均值为：

```text
tracker observation ─→ frozen SPV5-2A tracker ─→ base action
          │
          └─→ frozen tracker feature ────────────────┐

100-frame (state, applied target, next state)
          └─→ frozen Context Encoder ─→ z ──────────┼─→ residual MLP
                                                    │
final mean = base action + 0.25 × tanh(residual)  ←─┘
```

`no-latent` 基线完全不加载也不执行 Context Encoder，residual MLP 只读取同一个 frozen
tracker feature。两者使用相同的 SPV5-2A task、reward、termination、critic observation、PPO
超参数、startup DR、motion sampling 和 residual action bound。

Forward Predictor 的 transition Transformer、prediction heads 和 simulator 参数 θ 都不会在
residual PPO 中执行或暴露。latent 路径只严格加载 checkpoint 中的 `context_encoder.*` 以及
`state/action` normalization，并检查 Forward checkpoint 记录的 tracker SHA-256（如果存在）。

## 与原 SPV5-2A 的兼容契约

- 基础任务固定为 `SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward`。
- frozen tracker 使用原 checkpoint 保存的 actor 配置、observation groups、normalizer、estimator、
  reference encoder、actor MLP 和 action distribution 参数，并严格加载权重。
- residual actor 的非 latent 信息是 frozen tracker 实际用于动作预测的 1645 维处理后特征；它只由
  原 tracker observation 计算，不增加额外特权信息。
- HEFT critic 使用原 `policy + priv` observation，结构和权重都从 tracker checkpoint 严格恢复，
  随 residual PPO 继续训练。
- PPO 保留原 SPV5-2A 的 clipped objective、GAE、entropy、epoch/minibatch 数、actor/critic 分离
  learning rate、adaptive actor LR 和 gradient clipping。
- action history 继续记录最终 residual policy 的 Gaussian mean；adaptive motion sampler 的逐轮
  hook 也保留。
- 默认移除 checkpoint 的 step/interval event（包括随机推力），保留 startup DR。只有显式传入
  `--include-disturbances` 才恢复这些扰动。

residual 最后一层以零初始化，Gaussian 标准差从 frozen tracker 精确复制。因此两个新策略在第一个
PPO update 前都与原 tracker 具有相同的动作均值和标准差，不会从随机补偿开始破坏已有能力。

## Context Encoder 在线语义

每个 vector world 维护独立的 100 帧 causal ring：

```text
(s_t, simulator joint_pos_target_t, s_t+1)
```

这里的 action 是 action manager 处理后真正写入 simulator 的 29 维 PD target，不是 raw policy
action。Context Encoder 输入只有 71 维 robot state 和 applied target，不包含 foot/contact 或 θ。
episode done 或 motion resample 时只清空对应 world 的历史。未填满的 context 仍可产生 latent，和
Forward Predictor 训练时的 padding 语义一致。

## 正式启动

latent 基线：

```bash
GPUS=0,1,2,3 ./scripts/run_residual_policy_latent.sh \
  /path/to/SPV5-2A/checkpoint_72000.pt \
  /path/to/forward_predictor_v12/last.pt \
  /path/to/motion_directory \
  ./runs/residual_policy_latent \
  --seed 42
```

无 latent 基线：

```bash
GPUS=0,1,2,3 ./scripts/run_residual_policy_no_latent.sh \
  /path/to/SPV5-2A/checkpoint_72000.pt \
  /path/to/motion_directory \
  ./runs/residual_policy_no_latent \
  --seed 42
```

launcher 固定全局 4096 个环境并均分到 `GPUS`，默认每个 rollout iteration 采集 24 步、训练
100000 iterations、每 1000 轮保存一次。输出目录必须为空；续训时传
`--resume /path/to/checkpoint.pt`。单卡可用 `DEVICE=cuda:0` 代替 `GPUS`。

为了让 A/B 对比有效，两次运行应使用相同 tracker、motion 集、seed、GPU 数量和所有 PPO 参数。
同一 seed 下，Context Encoder 构造造成的 RNG 消耗会在创建环境前被重置，因此两条路径从相同的
startup DR 随机流开始；不同 distributed rank 使用 `seed + rank`，避免各卡重复同一批 world。

## 判断 latent 是否被 residual policy 使用

除原 reward/episode 指标外，runner 记录：

- `residual_action_rms`、`residual_action_abs_max`：补偿幅度；
- `latent_shuffle_action_delta_rms`：保持 tracker feature 不变、打乱 batch 内 latent 后动作变化；
- `latent_zero_action_delta_rms`：把 latent 置零后的动作变化；
- `context_valid_fraction`、`context_full_fraction`：在线 context 填充状态。

只有 latent 基线相对 no-latent 在独立 DR seeds/motions 上稳定提高 episode reward，并且 shuffle/zero
指标显著非零，才能认为 latent 为控制提供了增量信息。训练集 reward 上升本身不能排除 residual
MLP 仅靠 tracker feature 学会通用补偿。

checkpoint 使用 SP/RSL-RL 熟悉的顶层 `actor_state_dict`、`critic_state_dict`、
`optimizer_state_dict`、`iter` 和 `cfg` 字段，同时保存 residual/context 来源与 SHA-256。当前导出
仍需要在线 context ring，因此不支持把 latent 版本直接当作无状态 ONNX actor 导出。
