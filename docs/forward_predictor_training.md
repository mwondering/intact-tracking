# Privileged-contact causal-Transformer Forward Predictor v4

这个独立任务只验证 Predictor：固定 nominal dynamics 和冻结 tracker，不使用 Context
Encoder、Residual Policy 或 Backward Predictor。

## 计算图

```text
71-D robot state -- differentiable G1 FK --> 8-D foot height/velocity
       |                                             |
       + 6-D contact force + 2-D contact state ------+--> 87-D state feature
                                                               |
29-D physical PD target ---------------------------------------+--> one token

10 historical tokens + 1 current token --> causal Transformer
                                             |       |       |
                                      70-D delta  6-D force  2-D logits
                                             |       |       |
                                             +-- next state --+
                                                      |
                         recompute foot features from predicted q/qdot
                                                      |
                                              repeat five times
```

71 维 robot state 包含 root position/quaternion、root linear/angular velocity、29 维 joint
position 和 29 维 joint velocity。每个时刻再加入 8 维足端派生量：左右脚底 site 各自的离地
高度与三维世界速度；它们由固定 G1 运动链的纯 PyTorch FK/Jacobian 运算得到。teacher-forced
分支从真实状态计算，recursive 分支必须从刚预测出的 root pose、q 和 qdot 重新计算，因此没有
把真实未来足端信息泄漏给模型，梯度也能沿 FK 回传到预测状态。

动态 contact state 由左右脚各自的三维世界接触力（共 6 维）和接触标记（共 2 维）组成。
Transformer 输出下一时刻的 normalized contact force 与 contact logits；递推时接触力原样回灌，
logits 经 sigmoid 形成可微分的接触概率后回灌。接触标记并不由 FK 硬阈值替代，因为碰撞切换还
取决于接触求解器。

29 维 action 不是 raw policy action，而是采样器在 Predictor 外部完成 scale、default offset、
joint offset、clip、encoder bias 和关节顺序映射后，真正写入 PD controller 的物理关节位置
target。若动作链包含 delay、alpha smoothing 或 boot-target override，独立训练入口会拒绝启动；
这类有状态链路需要显式建模其内部状态，不能伪装成静态动作变换。

validity embedding 区分真实历史与 reset 后缺失的历史，learned position embedding 保留时间
顺序；缺失历史的全部 87 维 state feature 与 action 在投影前、后均被 mask。默认 Transformer
宽度 512、6 层、8 头、FFN expansion 4，精确参数量为 19,022,414。70 维 robot delta 使用
3 维 root rotation vector，不预测冗余 quaternion delta。

## 固定权重的双分支损失

每个 batch 同时构造两条共享参数的计算图：

```text
teacher-forced：每一步使用真实 x_t 预测 x_t+1
recursive：     从真实 x_0 出发，把预测状态递推五步

L = L_teacher + 0.5 * L_recursive
```

两项均包含 normalized robot state-delta Huber、normalized contact-force Huber 和
contact BCE，并从第一个 optimizer step 起共同优化。teacher 权重固定为 1.0，recursive 权重
默认固定为 0.5；可以通过 `--recursive-weight` 显式修改，训练过程不使用 warmup 或线性爬坡
课程。接触两项的权重可由 `--contact-force-weight` 和 `--contact-binary-weight` 修改。

## 数据、batch 与归一化

Replay 保存目标五步的 robot/contact state、五个 physical PD target，以及目标起点之前最多
10 帧相同字段的历史。足端特征不单独保存，而是在使用样本时由 robot state 确定性重算。
目标五步不能跨 reset；reset 后状态可以立即成为目标起点，此时缺失历史被置零并通过 valid
mask 明确标记。默认每卡 replay 为 262144，并按 motion ID 进行平衡抽样；在默认
`num_envs=2048` 下，replay 与 rolling history 的静态估算约为 1.66 GiB/卡。

默认 `--batch-size 4096` 表示每卡每次 optimizer step 的有效 batch，不是单次前向 batch。
训练按 `--micro-batch-size 256` 切成最多 16 个 micro-batch，loss 按各 chunk 的样本比例缩放并
累积梯度，最后执行一次 AdamW step。DDP 只在最后一个 micro-batch 同步梯度。Forward Predictor
不进行梯度裁剪；非有限 loss/gradient 会立即报错，原始 gradient norm 继续写入日志。若显存
不足，应先降低 `--micro-batch-size`，无需降低有效 batch。

warmup 期间累计 robot state、physical PD target、FK foot feature、contact force 和 one-step
robot delta 的均值与标准差；二值接触标记不做 normalization。训练开始后立即冻结统计量。
冻结可防止同一个 replay 样本的数值定义随训练时间漂移。

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
- `train/foot_height_error_m`、`foot_velocity_error_mps`：由预测状态重新 FK 后的足端误差；
- `train/contact_force_error_n`、`contact_force_nmse`：接触力的物理误差和 no-change 相对误差；
- `train/contact_binary_accuracy`、`contact_binary_brier`：接触切换分类与概率校准；
- `fixed_probe/*`：同一固定 batch 上的稳定曲线；
- `optimization/gradient_norm`：未裁剪的有效 batch 梯度范数。
