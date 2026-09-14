这次检查没有发现 latent 的轨迹构建、归一化、缓存数学计算、actor/critic 拼接或 PPO 存储错位，但复现了一项策略前向的 TF32 数值一致性问题。

使用当前 tracker-action 实验的 latent 组 update 1000 checkpoint，冻结权重，在与训练相同的 256 个静态 DR 配置上运行随机策略。诊断使用 16 条 motion 的子集、每个 DR 一个 world，共 816 步、208896 个 transition，发生 842 次 reset。物理 bank 的哈希与正式训练一致。这个运行检查输入通路，不代表全训练数据上的表征识别率评估。

| 检查 | 结果 |
|---|---|
| 预训练动作变换与 simulator 实际 joint target | 最大差 0 |
| 当前 (state, target, next-state) 与历史最后一项 | 最大差 0 |
| reset 后短期历史清理 | 842 次边界，无失败 |
| actor、critic 实际接收的 64 维 latent | 与观测 latent 最大差 0 |
| actor 实际接收的 29 维 tracker action | 与当前原始 tracker 输出最大差 0 |
| PPO 采样时 latent 到存储中的 latent | 最大差 0 |
| 采样后所有观测组被原地改写 | 额外 96 步检查，所有组最大差 0 |
| 真正 FP32 下缓存与原始 encoder | 68 份真实历史，最大逐元素差 1.43e-6 |
| 当前 BF16 缓存与原始 BF16 推理 | 抽检最大相对 RMS 差约 0.475%，包含 batch 形状差异 |
| 固定历史重复读取缓存 | 差 0 |
| Actor / critic 的 latent 输入权重 | 两者 64 列全部非零 |
| 已有相关测试 | 34 项通过 |
| 正式训练固定的源码 | 163 个文件哈希全部一致 |

策略概率的一致性需要关注。把相同的实际观测按正式 rollout 的 8192 和 minibatch 的 49152 两种 batch 尺寸计算，权重保持不变，TF32 下动作均值 RMS 差为 0.000349，主要来自冻结 tracker 的输出。PPO 概率比相对 1 的绝对偏差，中位数约 0.91%，95 分位 3.31%，99 分位 4.89%，最大 11.68%。该批测试没有仅因该误差越过 [0.8, 1.2]，但产生了与参数更新无关的概率比偏差。改为 IEEE FP32 后，该 batch 对照中的动作和 log-prob 差消失。49152 条输入由 256 份实际观测复用构造，用于隔离 batch 尺寸，不能当作 49152 个独立状态。

PyTorch 官方说明了不同 batch 计算的数值结果可能不同，以及 TF32 的精度限制：[Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)。这里的具体幅度来自本地复现。

这项数值问题值得在后续控制实验中处理。可考虑统一策略前向为 FP32，或把采样时冻结 tracker 的 action 随观测保存，供 PPO 更新复用；后一方案还需验证剩余 residual 前向误差。尚未证明这个问题造成了真实 DR / latent 输入的收益有限，也没有测量修正后的速度与学习收益。

现有闭环使用评估提供了另一条证据：update 1000 的同 motion/phase 跨 DR latent 交换，在有效交换覆盖 85.9% 的条件下，使共同前缀 body pos 误差上升 2.10%（按 motion bootstrap 的 95% 区间：1.38%～2.98%）。这支持当前 policy 从正确 latent 中获得了一点收益，仍不证明 latent 已充分表达控制所需的信息。

结合用户报告的真实 DR 参数仅带来小幅收益，目前更合适的工作假设是：这套控制训练从环境条件输入中获得的增量收益有限。数值输入通路大体正确、latent 信息是否足够、以及 PPO 能否充分利用信息，是需要分别判断的三件事。

本次只生成诊断文件，没有修改正式训练源码、配置、checkpoint 或进程。详细可复核数据见同目录 audit_report.json、summary.json、precision.json、formal_batch_precision.json 和 logprob_focus/summary.json。
