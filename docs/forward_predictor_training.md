# Nominal recursive Forward Predictor

这个独立任务只验证 Predictor：固定 nominal dynamics 和冻结 tracker，不使用 Context
Encoder、Transformer、Residual Policy 或 Backward Predictor。

## 计算图

```text
normalized x_t (71) + normalized a_t (29)
                    |
                    v
       shared residual MLP (~10.4M)
                    |
                    v
       normalized full-state delta (70)
                    |
                    v
   compose translation / rotation / velocity
                    |
                    v
                  x_hat_t+1
                    |
        feed back with action a_t+1
                    |
              repeat five times
```

71 维状态包含 root position/quaternion、root linear/angular velocity、29 维 joint position
和 29 维 joint velocity。70 维 delta 使用 3 维 root rotation vector，因此不存在冗余的
quaternion delta。五步 loss 同时监督 pose 和 velocity；后续 tracking objective 仍可只使用
pose。五步状态误差统一换算成 70 维物理 state-delta error，并按 warmup 得到的 one-step
delta standard deviation 逐维缩放，避免绝对 root 轨迹范围掩盖局部预测误差。

## 数据与归一化

Replay 只保存 reset-free 的五步窗口：`x_t...x_t+5` 和 `a_t...a_t+4`。Reset transition
不进入窗口，但 reset 后状态可以成为新窗口起点。warmup 期间统计 state、action 和 one-step
delta，训练开始后立即冻结这些统计量，避免训练目标随在线数据持续漂移。

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
- `train/horizon_1...5_loss/nmse`：误差是否随递推累积；
- `train/root_position_error_m`、`root_orientation_error_rad`：root 物理误差；
- `train/joint_position_error_rad`：逐关节绝对位置误差；
- `train/*velocity_error*`：递推所需的速度预测误差；
- `fixed_probe/*`：同一固定 batch 上的稳定曲线；
- `optimization/gradient_norm`：梯度是否爆炸或消失。
