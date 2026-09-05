# Frozen-predictor model-gradient residual baseline

这个 baseline 不使用 PPO 的 score-function 梯度，而是把 residual action 送入冻结的可微
Forward Predictor，通过五步 tracking surrogate 直接反向传播到 residual MLP。它用于回答：
当前 predictor 和 dynamics latent 是否已经足以提供有用的控制梯度。

## 梯度和数据流

每次 update 先在真实 SPV5-2A 环境中连续采集五帧。每个环境保留独立的 100 帧 causal history，
Context Encoder 只从 robot state 与实际 applied PD target 计算 latent。随后在优化阶段重新计算：

```text
frozen tracker feature_t + frozen z_t
                  -> trainable bounded residual policy
                  -> frozen tracker action_t + residual_t
                  -> exact raw-action-to-applied-PD-target transform
                  -> frozen latent-conditioned Forward Predictor
                  -> five recursively predicted robot states
                  -> reference tracking surrogate
                  -> residual policy parameters only
```

Forward Predictor 的 robot、foot 和 contact state 在五步内递推，起始 latent 同时作为 predictor
的 dynamics condition。tracker、Context Encoder、Forward Predictor 和真实 simulator 都不接收
梯度。residual 最后一层零初始化，输出限制为默认 `0.25 * tanh(.)`；幅度与时间平滑 penalty
限制策略离开 predictor 数据分布的速度。没有梯度裁剪。

SPV5-2A 的 1645 维 tracker feature 包含由长历史、estimator 和 reference encoder 得到的信息，
无法从 predictor 的 71 维 robot state 精确重建。因此五个未来 residual action 分别使用刚采集的
真实 tracker feature 和 latent，只有 simulator state 在 predictor 内递推。这个 teacher-forced
条件明确保存在 checkpoint 的 `run_config.rollout_contract` 中。

## 启动

```bash
GPUS=0,1 ./scripts/run_model_gradient_residual_training.sh \
  /path/to/SPV5-2A/checkpoint_72000.pt \
  /path/to/forward_predictor_v12/update_007000.pt \
  /path/to/motion_directory \
  ./runs/model_gradient_residual \
  --wandb-name model-gradient-residual-v1
```

脚本固定全局 4096 个环境，默认 `nominal_fraction=0`，即全部使用与 Forward Predictor 采样契约
一致的固定 startup DR；step/interval 随机扰动被移除。默认 model-gradient batch 为 1024，
micro-batch 为 128。续训使用 `--resume /path/to/last.pt`，并沿用原输出目录。

Forward checkpoint 会被严格恢复，包括完整 v12 模型、normalization 和 tracker SHA-256。若它与
传入 tracker 不匹配，训练会在创建 optimizer 前失败。只有无 action delay、无 alpha smoothing、
无 boot delay 的 memoryless action chain 可微；否则入口会明确拒绝训练。

## 最核心的诊断

- `train/real_reward_mean`：当前 residual 在真实 simulator 中得到的平均 step reward；这是最终判据。
- `train/real_root_position_error_m`、`real_root_orientation_error_rad`：root 位置与姿态误差。
- `train/real_root_linear_velocity_error_mps`、`real_root_angular_velocity_error_radps`：root
  线速度与角速度误差。
- `train/real_joint_position_error_l2_rad`、`real_joint_velocity_error_l2_radps`：29 个关节误差
  向量的 L2 范数；不是逐关节 MAE。
- `train/predicted_improvement`：相对零 residual，predictor 认为 tracking surrogate 改善了多少。
- `train/latent_shuffle_loss_ratio`：打乱环境间 latent 后，预测 tracking loss 的倍率；大于 1 才说明
  正确 latent 对 model-based control 有帮助。
- `train/latent_policy_sensitivity_rms`：固定 tracker feature、只换 latent 时 residual action 的变化。
- `train/gradient_norm`：五步 predictor 是否确实向 residual MLP 提供非零梯度。
- `train/valid_horizon_fraction`：未跨越 reset 的有效五步样本比例。

如果 `predicted_improvement` 持续上升但真实 reward 不升或下降，说明 policy 正在利用 predictor
误差，应减小 `--residual-scale`、提高 `--residual-weight`，或缩短每批数据复用次数。只有真实
reward 提升，同时 shuffle ratio 大于 1，才说明 predictor 与 latent 被有效用于控制。
