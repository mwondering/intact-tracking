# 最新定稿 Pipeline（2026-09-20）

**状态：已由用户确认的最新定稿版本。** Method 撰写、配置复现与部署以本文及其引用的实际运行记录为准。定稿对象为 **冻结 SPV5-2A tracker + proprio122 Memory350 context encoder + 五帧 latent / tracker-action 条件 residual PPO + COM / 摩擦辅助监督**。

对应实验：`stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919`。此前的 INTACT 四槽 actor、统一 Forward-only Transformer、100 帧 context、限幅 residual、MoE 等文档属于历史方案或独立实验。

**后续运行记录（2026-09-20）：** 用户要求基于定稿的 6008 次 checkpoint 继续训练，并将 adaptive sampling 统计重置为初始先验。模型与损失配置沿用本定稿；独立续训目录、恢复范围和启动参数见 [重置采样统计的续训记录](residual_resume_sampling_reset_20260920.md)。下文定稿产物与停止计数描述父运行。

## 定稿计算流程

1. **阶段一：学习 context。** 冻结 tracker `checkpoint_144000.pt` 采集在线交互，训练 proprio122 Memory350 context encoder 和 forward predictor；阶段二选用 `stage1_proprio122_8192/update_007179.pt`。阶段一的输入与训练设置见 [proprio122 记录](../runs/144000_exp/proprio122_training.md)。
2. **因果交互记忆。** 每条交互为 `(proprio122_before, raw_command29, proprio122_after)`，共 273 维。使用最近 50 步短历史及 30 个、每个 10 步的长期块，输出 64 维 latent。历史按 world 独立维护；排除跨 reset 的交互，motion/episode 边界保留完整长期块，物理 session 改变时清空记忆。
3. **阶段二：训练 residual PPO。** tracker 与 context encoder 及其归一化冻结；residual actor、critic 和 PPO optimizer 从头初始化。阶段一 forward predictor 不参与 PPO。actor 将 tracker 的 1645 维处理后特征、最近五帧 latent（320 维）和当前 tracker 动作（29 维）拼接为 1994 维输入；critic 使用原有特权特征及同样的 latent/action 条件。
4. **共享隐藏层辅助监督。** residual actor 的最后一个 128 维隐藏层连接 29 维动作头和独立的 92 维 DR 线性头。辅助梯度进入共享层；仅监督 COM x/y/z 与摩擦。DR 真值只作训练标签，不进入 actor 推理输入。
5. **部署。** 动作取 `tracker mean + residual mean`，residual scale 为 1，无 tanh 或 residual/action clip。ONNX 包含 tracker、context encoder、归一化和 residual；配套 runtime 维护交互及 latent 历史。DR 辅助头只用于训练，不参与部署动作计算。

阶段二的动作分布与辅助目标可写为：

\[
\mu_t=a_t^{\mathrm{tracker}}+\Delta a_\theta(h_t^{\mathrm{tracker}},z_{t-4:t},a_t^{\mathrm{tracker}}),
\qquad a_t\sim\mathcal N(\mu_t,\operatorname{diag}(\sigma^2)).
\]

\[
L=L_{\mathrm{PPO}}+0.05L_{\mathrm{DR}},
\]

其中 PPO 包含 `-0.005 × entropy`；DR 标签按固定物理范围归一化，损失由有效历史比例加权，四个受监督坐标等权。完整损失、标签范围和梯度路径见 [DR 辅助监督](residual_ppo_dr_aux_plan_20260919.md)。

## 定稿配置

| 项目 | 已采用的设置 |
| --- | --- |
| 环境与数据 | MJLab，50 Hz，平地完整筛选集 220480 条 motion，原 reward 和完整 termination，episode 最长 500 步 |
| 并行规模 | 8 GPU × 8192 environments，共 65536 |
| 物理分布 | `checkpoint_native_flat_v1`；每卡 820 个 nominal，其余为原始范围的随机 DR；无额外肢体负载 |
| 物理生命周期 | world 的物理参数跨 episode/motion 固定；DR 动作 delay/smoothing 沿原 reset 规则处理 |
| Motion sampling | 从首轮启用原 tracker 144000 adaptive sampling；已修复旧入口关闭 failure rewind 的差异，按原配置在失败后以 1/3 概率回到当前 bin 起点；采样规则与验证见[对齐记录](adaptive_sampling_parity_20260920.md)。使用已修复的 SPV5-3 motion-boundary GAE 语义 |
| Actor / critic 初始化 | 从头初始化；residual 动作输出层为零初始化 |
| Actor / critic 学习率 | `1e-4` / `5e-4` |
| PPO rollout / epochs / minibatches | `24` / `5` / `4` |
| Entropy / 初始 std | `0.005` / `1.0`；std 随训练学习 |
| Residual | `unbounded`，scale `1.0`，无额外限幅 |
| DR 辅助系数 | `0.05` |
| DR 辅助组权重 | COM x/y/z、摩擦各 `1`；质量、Kp、Kd、armature 均 `0`，这些环境 DR 仍保留 |
| 精度 | PPO actor/critic 为 FP32；冻结 context 的训练期 CUDA 推理使用 BF16；部署 ONNX 为 FP32 |
| W&B | online；project `intact-preview-v2`，group `144000_exp`，横轴 `completed_updates` |

配置依据为该次运行的 [run_config.json](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/ppo_8gpu8192/run_config.json)，训练入口为 [run_144000_residual.py](../scripts/run_144000_residual.py)。该入口默认启动无总更新上限的训练；本次运行后来按用户指令结束。**6000 次是本次运行的停止目标，不是模型结构或通用训练预算。**

## 定稿产物与运行状态

训练已结束。用户要求完成 6000 次更新后停止；停止控制脚本的接口兼容问题导致实际完成 **6008 次更新**。最终 checkpoint 的 `iter=6007`、`completed_updates=6008`，记录保留真实计数。8 个 rank 的最终参数一致，训练进程、导出进程和监控进程均已退出，W&B 状态为 `finished`。

- [最终 checkpoint](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/ppo_8gpu8192/checkpoint_final.pt)：包含 tracker、context encoder、归一化、residual、DR 辅助头和 PPO 训练状态。
- [最终部署目录](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/ppo_8gpu8192/deploy/checkpoint_final/)：`policy.onnx`、`policy.json`、`deploy_metadata.json`、`policy_runtime.py`。
- [最终保存与停止核对](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/monitor/final_stop_verification.json)：模型依赖校验、文件哈希、进程退出和 ONNX 七组数值校验结果。
- [checkpoint 格式说明](residual_checkpoint_bundle_20260920.md)与 [ONNX 部署说明](residual_onnx_deployment.md)。

最终 checkpoint SHA256：`0a526134928fe24b6e48bcf47f1053af10d362a2ad7194ad204b994e4da0bb84`。

现有冻结 tracker 配对评测使用的是 **4901 次更新**的中间模型，见 [评测摘要](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_20260919/evaluations/update_004901_warm_native_512_20260920/SUMMARY.md)。这些结果不能标成最终 6008 次模型的评测，也不能单独证明辅助损失的因果收益；最新定稿表示采用的 pipeline 已确认。
