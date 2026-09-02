# Nominal causal-Transformer Forward Predictor v3

这个独立任务只验证 Predictor：固定 nominal dynamics 和冻结 tracker，不使用 Context
Encoder、Residual Policy 或 Backward Predictor。

## 计算图

```text
(x_t-10,a_t-10,valid) ... (x_t-1,a_t-1,valid) (x_t,a_t,valid=1)
             |                       |                    |
             +------ state/action projections + validity + position ------+
                                         |
                                         v
                     11-token causal Transformer (6 layers, 8 heads)
                                         |
                                         v
                         final token -> normalized 70-D delta
                                         |
                                         v
                                    x_hat_t+1
                                         |
                   shift (x_t,a_t) into the 10-frame history
                                         |
                                  repeat five times
```

每个时刻的 71 维 state 和 29 维 action 分别线性投影后相加，形成一个融合 token；validity
embedding 区分真实历史与 reset 后缺失的历史，learned position embedding 保留时间顺序。
上三角 causal mask 保证第 k 个 token 只能读取自身和更早 token，模型只用最后一个当前 token
预测下一状态的 delta。缺失历史的 state/action 在投影前、投影后都被 mask，填充值不会泄漏。

默认 Transformer 宽度 512、6 层、8 头、FFN expansion 4，精确参数量为 19,010,118。
71 维状态包含 root position/quaternion、root linear/angular velocity、29 维 joint position
和 29 维 joint velocity。70 维 delta 使用 3 维 root rotation vector，因此不存在冗余的
quaternion delta。五步状态误差统一换算成 70 维物理 state-delta error，并按 warmup 得到的
one-step delta standard deviation 逐维缩放，避免绝对 root 轨迹范围掩盖局部预测误差。

## 固定权重的双分支损失

每个 batch 同时构造两条共享参数的计算图：

```text
teacher-forced：每一步使用真实 x_t 预测 x_t+1
recursive：     从真实 x_0 出发，把预测状态递推五步

L = L_teacher + 0.5 * L_recursive
```

两项均使用 normalized state-delta Huber loss，并从第一个 optimizer step 起共同优化。teacher
权重固定为 1.0，recursive 权重默认固定为 0.5；可以通过 `--recursive-weight` 显式修改，
训练过程不使用 warmup 或线性爬坡课程。

## 数据、batch 与归一化

Replay 保存目标五步 `x_t...x_t+5`、`a_t...a_t+4`，以及目标起点之前最多 10 帧历史。
目标五步不能跨 reset；reset 后状态可以立即成为目标起点，此时缺失历史被置零并通过 valid
mask 明确标记。默认每卡 replay 为 262144，并按 motion ID 进行平衡抽样；在默认
`num_envs=2048` 下，replay 与 rolling history 的静态估算约为 1.55 GiB/卡。

默认 `--batch-size 4096` 表示每卡每次 optimizer step 的有效 batch，不是单次前向 batch。
训练按 `--micro-batch-size 256` 切成最多 16 个 micro-batch，loss 按各 chunk 的样本比例缩放并
累积梯度，最后执行一次 AdamW step。DDP 只在最后一个 micro-batch 同步梯度。Forward Predictor
不进行梯度裁剪；非有限 loss/gradient 会立即报错，原始 gradient norm 继续写入日志。若显存
不足，应先降低 `--micro-batch-size`，无需降低有效 batch。

warmup 期间累计 state、action 和 one-step delta 的均值与标准差；训练开始后立即冻结统计量。
模型输入使用 `(value - mean) / std`，输出的 normalized delta 在更新物理状态前还原为
`delta * std + mean`。冻结可防止同一个 replay 样本的数值定义随训练时间漂移。

普通训练：

```bash
GPUS=0,1 ./scripts/run_forward_predictor_training.sh \
  /path/to/tracker.pt \
  /path/to/motions \
  /path/to/runs/forward_predictor \
  --wandb-name forward-predictor-nominal
```

固定批次容量测试：

```bash
GPUS=0,1 ./scripts/run_forward_predictor_training.sh \
  /path/to/tracker.pt \
  /path/to/motions \
  /path/to/runs/forward_predictor_overfit \
  --fixed-batch-overfit \
  --updates 5000 \
  --wandb-name forward-predictor-fixed-batch
```

固定批次 loss 若不能接近零，先检查 Predictor、状态定义和损失；只有该测试通过后才判断在线
数据覆盖率或泛化能力。普通训练同时记录一个 normalization 固定、内容不变的
`fixed_probe/*` batch，便于区分真实收敛和在线 batch 波动。

## 重点指标

- `train/rollout_loss`、`train/rollout_nmse`：五步完整状态误差及相对 no-change baseline；
- `train/teacher_loss`、`train/teacher_nmse`：使用真实输入状态的一步动力学质量；
- `train/recursive_weight`：固定的五步递推权重，默认 0.5；
- `train/horizon_1...5_loss/nmse`：误差是否随递推累积；
- `train/root_position_error_m`、`root_orientation_error_rad`：root 物理误差；
- `train/*_p95_*`、`train/*_p99_*`：长尾误差是否主导训练；
- `train/joint_position_error_rad`：逐关节绝对位置误差；
- `train/*velocity_error*`：递推所需的速度预测误差；
- `fixed_probe/*`：同一固定 batch 上的稳定曲线；
- `optimization/gradient_norm`：未裁剪的有效 batch 梯度范数。
