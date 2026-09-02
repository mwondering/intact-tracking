# Nominal flat-history Forward Predictor v2

这个独立任务只验证 Predictor：固定 nominal dynamics 和冻结 tracker，不使用 Context
Encoder、Transformer、Residual Policy 或 Backward Predictor。

## 计算图

```text
flatten([x_t-5,a_t-5,valid], ... [x_t-1,a_t-1,valid])
                         + x_t (71) + a_t (29)
                                      |
                                      v
                  shared residual MLP (~20.14M)
                                      |
                                      v
                     normalized full-state delta (70)
                                      |
                                      v
                                  x_hat_t+1
                                      |
             shift (x_t,a_t) into the flat history window
                                      |
                               repeat five times
```

历史帧不会先经过 Transformer、GRU 或独立 token encoder，而是按时间顺序直接展平后送入
同一个 MLP。默认 MLP 宽度 1100、8 个 pre-norm residual block，精确参数量为
20,141,070。

71 维状态包含 root position/quaternion、root linear/angular velocity、29 维 joint position
和 29 维 joint velocity。70 维 delta 使用 3 维 root rotation vector，因此不存在冗余的
quaternion delta。五步 loss 同时监督 pose 和 velocity；后续 tracking objective 仍可只使用
pose。五步状态误差统一换算成 70 维物理 state-delta error，并按 warmup 得到的 one-step
delta standard deviation 逐维缩放，避免绝对 root 轨迹范围掩盖局部预测误差。

## 固定权重的双分支损失

每个 batch 同时构造两条共享参数的计算图：

```text
teacher-forced：每一步使用真实 x_t 预测 x_t+1
recursive：     从真实 x_0 出发，把预测状态递推五步

L = L_teacher + 0.5 * L_recursive
```

两项均使用 normalized state-delta Huber loss，并从第一个 optimizer step 起共同优化。teacher
权重固定为 1.0，recursive 权重默认固定为 0.5；可以通过 `--recursive-weight` 显式修改，
训练过程不再使用 warmup 或线性爬坡课程。

## 数据与归一化

Replay 保存目标五步 `x_t...x_t+5`、`a_t...a_t+4`，以及目标起点之前最多 5 帧历史。
目标五步不能跨 reset；reset 后状态可以立即成为目标起点，此时缺失历史被置零并通过 valid
mask 明确标记。默认每卡 replay 为 262144、batch 为 2048，并按 motion ID 进行平衡抽样。
`num_envs=4096` 时 replay 与 rolling history 的静态估算约为 1.07 GiB/卡。warmup 期间统计
state、action 和 one-step delta，训练开始后立即冻结统计量。

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
- `optimization/gradient_norm`：梯度是否爆炸或消失。
