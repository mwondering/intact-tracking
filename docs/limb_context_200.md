# Context 200-step 阶段一重训

2026-09-09：新目录 `runs/limb_context_20260909_context200`，从头训练 context encoder 和
forward predictor。参照 `runs/limb_context_20260907/stage1/run_config.json`，训练参数逐项
核对后，唯一实验变量是 `context_history_steps: 100 → 200`；输出目录和 resume 状态随新实验改变。

运行更新：按用户要求，GPU 0–3 上的旧阶段一已在 **24988 updates** 保存并正常退出，
保留 `last.pt`、`best.pt` 和 `update_024988.pt`。W&B run 为 `limb-ae3fc9feaba5`。
`baseline_123` 的旧自动接卡任务已继续挂起，给[新长短期方案](limb_context_memory350.md)留出 GPU。
另已完成 [LocoFormer 长 context 调研](locoformer_context_scaling.md)；其中的新边界/缓存设计
尚未应用到此实验，下方启动记录保留当时的排队状态。

保留 nominal counterfactual 监督、完整 129827-motion 数据集、四肢各自独立 U(0,4) kg、
小腿中部负载、其余 DR/观测噪声关闭。阶段一仍使用原终止条件，包括 `ee_body_pos`；
此前去掉这一项的变更仅应用于阶段二 PPO。

保留 4 GPU、每卡 1024 训练 world + 128 独立验证 world、global batch 4096、microbatch 256、
BF16、seed 717、motion-balanced replay、warmup 500 steps、AdamW 及原损失权重。
Predictor 的短历史仍为 10 步，预测 horizon 和正样本偏移仍为 5 步；latent 仍为 64 维。
context transformer 仍为 128 宽 / 2 层 / 4 heads，predictor 仍为 512 宽 / 6 层 / 8 heads。
位置编码随窗口增加，总参数从 19500310 增至 19513110。

本次训练原先不设 update 上限；8000 是 cosine 学习率衰减时标，尾段 LR 为 1e-5。
每 100 updates 验证、每 250 updates 定期保存并保留最佳 checkpoint。新队列只有阶段一及
其启动检查，没有 PPO 作业。旧 encoder/checkpoint 和正在训练的 PPO 使用关系不变。

## 200 步和 500 步 episode 的关系

实际环境为 `10 s / (4 × 0.005 s) = 500` 个控制步；200 步相当于 4 s。
历史以环境 reset 或 motion resample 为边界，较短的历史使用 padding mask，满 200 步后
滑动更新。在没有提前终止、没有动作切换的理想 500-step episode 中，约 60% 的决策时刻
能使用完整 200 步历史；实际比例会更低。训练日志保留
`training_context_full_fraction` 和 `validation_context_full_fraction`。

因此 200 步无需改 episode length，但 context 不能无限延长。表征正样本要求同一 world /
episode / motion 内，两个相隔 5 步的完整历史窗口及各自 5 步预测目标。从段首开始，首对
完整样本最早需要约 210 个连续有效交互步，采集块未对齐时可能再晚几步。窗口接近 500 时，
完整正样本会趋于稀少；窗口达到 500 时，500-step episode 已不能提供这类样本。
当前 warmup 必须凑齐训练和验证正样本，所以这种配置可能最终耗尽 warmup 并报错。
延长窗口也增加 Transformer 的计算成本；这里保留 500-step episode，只验证 200-step 改动。

## 启动及验证

- 调度采用暂定的“当前组完成后腾卡”：等待现有 baseline seed 122 完成 5000 updates，
  GPU 0–3 优先运行新阶段一。baseline seed 123 延后，待新 encoder 按用户决定停止后再启动。
  GPU 4–7 上的既有 FiLM 队列继续。调度器已支持启动等待条件，无需重启训练进程。
- 新 W&B 分组为 `intact-forward-predictor / limb_context_20260909_context200-stage1`；
  账号 API 核验为 `2486344338@qq.com`。训练日志沿用实时 JSONL 同步器，run 在训练初始化
  生成配置后出现；排队阶段没有训练曲线。启动短测在独立的 stage1-smoke 分组。
- 42 项相关测试通过，覆盖 200-step 完整正样本、reset 隔离、短历史反向传播、
  500-step context 无完整正样本，以及调度/日志兼容性。
- 全尺寸 19513110 参数模型在 CPU 上完成 200-step forward/backward，context 梯度有限且
  非零；真实 encoder 推理输出 `[4,64]`，一个 world reset 后有效历史为 `[1,200,200,200]`。
  结果位于新目录的 `preflight.json`。四卡 BF16 和实际模拟器短测将在 GPU 到位后执行，
  成功后自动进入完整数据集训练。

调度器更换前后，原两个 PPO 主进程及 8 个训练 rank 的身份保持一致，记录在
`gpu_handoff.json`。本次未暂停、resume 或修改现有 PPO 的训练设置。
