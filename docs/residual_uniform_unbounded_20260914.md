# Residual：不限幅输出、uniform 采样、腕部 10 N·m

用户最终确认：取消 residual 网络输出限幅；腕部 pitch/yaw 扭矩上限提高到 10 N·m；手部负载最高 2.5 kg，小腿最高 4 kg；后续训练全程 uniform；旧八组保存退出，新八组从零开始。

正式运行目录：`runs/limb_context_20260914_fixed_dr_specialists8_uniform_unbounded_wrist10`。当前状态读该目录 `state.json`，全局指针为 `.runtime/limb_context/current_ppo.json`。

新训练入口是 `intact_tracking.cli.residual_uniform_train`，评测入口是 `intact_tracking.cli.residual_uniform_eval`。配置记录在 `configs/experiments/residual_uniform_wrist10_hand2p5.json`。历史训练入口保留原协议，用于复现旧 checkpoint，后续新实验使用上述新入口。

## 动作与物理

原 residual 均值为 `0.25 * tanh(head)`。现在为直接的 `residual_mlp(inputs)`，无 tanh，无幅度缩放，无输出 clamp；策略均值等于冻结 tracker 的原始 action 加 residual。Gaussian 探索方式保持原设置。

冻结 tracker 的动作均值来自 MLP/Gaussian mean，本身没有这层 tanh。当前 wrapper 的 `clip_actions=None`，joint-position action 的目标位置 clip 也为 None。网络输出不限幅不等于执行器不限力矩：MuJoCo 的 actuator force range 仍生效；仅左右腕 pitch/yaw 从 ±5 改为 ±10 N·m，其余关节保留原限制。

保留冻结 tracker 的 action scale、PD gains、完整原始奖励、末端高度 termination。负载继续用相同位置和刚体复合质量/质心/惯量计算，手部范围为 [0,2.5] kg，小腿范围为 [0,4] kg。随机环境训练使用逐肢体独立均匀分布；本次八个独立 B 在各自固定完整 DR 上训练。

## 八卡布局

质量顺序为左手、右手、左小腿、右小腿；各组仍带有独立预抽的其他静态 DR，零负载并非 nominal。

| GPU | 四肢负载 kg | 专家 |
|---|---|---|
| 0 | 0,0,0,0 | B00 |
| 1 | 1,1,1,1 | B01 |
| 2 | 2,2,2,2 | B02 |
| 3 | 2.5,2.5,4,4 | B03 |
| 4 | 2.5,2.5,0,0 | B04 |
| 5 | 0,0,4,4 | B05 |
| 6 | 2.5,0,4,0 | B06 |
| 7 | 0,2.5,0,4 | B07 |

每卡一个独立 MLP residual policy，无 latent、无 router；actor/critic 独立，八组间没有共享可训练参数或统计。每卡 8192 环境、完整 motion 数据集、seed 121、rollout 24、5 epochs、4 minibatches、actor LR 1e-4、critic LR 5e-4、FP32。从零启动，全程 uniform，不设总训练轮数上限。

每 100 个完成的 update 保存 checkpoint，每 1000 轮在同卡独立模拟器中评测固定的 512 条 motion，horizon 至多 1000 步。评测继承 checkpoint 的物理设置、完整 DR 和不限幅 actor，并检查训练 RNG、模拟器计数和 critic normalizer 未被改变。

旧共享 A 使用了 5 N·m 腕部、手部 0–4 kg、adaptive 和 ±0.25 residual，与新 B 的训练设置不同。旧 A 保留作历史参考；严格比较新共享/独立 policy 需要用新配置训练共享 A。

## 验证

22 项相关单元检查通过。真实 PPO 完成两轮并从 checkpoint 恢复继续至第三轮，随后完成独立评测。新八组完整 DR 已通过两种环境数量、不同随机种子和 episode reset 前后不变的物理审计。

完整动作链另用绝对值 20–48 的网络输出验证：残差和总 raw action 未被截断，位置目标等于 raw action 经原 action scale/offset 映射，执行器扭矩仍在 force range 内。

证据分别位于运行目录的 `physics_preflight.json`、`READY.json`、`previous_training_saved.json`、`startup_verification.json`，以及 `runs/residual_uniform_wrist10_hand2p5_preflight_20260914`。

2026-09-14 04:53 UTC 启动核验完成：旧八组均已保存退出；新八组在 GPU 0–7 独立运行，已各完成 18–21 轮 PPO 更新，每轮约 4.4–4.8 秒。完整数据集为每卡 129827 条 motion、48085337 帧。运行配置、固定 DR 副本、实际腕部力矩范围、uniform 采样、不限幅输出和有限 loss 均通过检查；八组 W&B 服务端均确认收到非零更新。服务器上传证据见运行目录 `wandb_startup_verification.json`。上述轮数是启动检查时快照，后续进度以 `state.json` 为准。

2026-09-14 推送归档检查发现汇总监控曾因读取 `progress.json` 时的短暂 `FileNotFoundError` 退出，八个训练进程持续运行。已修复状态文件的并发读取，并增加 `--attach-existing`，在校验原进程启动时间、命令和记录后接管监控，无需重新启动训练。新监控 PID 为 50761，原八个训练 PID 保持不变；恢复时各组约 909–973 轮，状态心跳重新更新且无异常。

此次仅修改 supervisor，并增加专项测试；186 份已有其他固定文件逐一验证未变化。原 READY 和 supervisor 源码在运行目录 `supervisor_recovery` 留存，更新后的代码哈希写入 READY。归档及恢复后状态见 [实验记录索引](experiment_records/20260914/README.md)。
