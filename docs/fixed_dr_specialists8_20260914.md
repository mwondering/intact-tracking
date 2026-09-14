# 八个固定 DR 专家与现有通用策略的比较

用户确认使用八张卡分别启动八个独立 B，选取八组有代表性的 DR。A 复用现有不带 latent 的 MLP checkpoint，所有 B 都与同一个固定 A 比较。

运行目录：`runs/limb_context_20260914_fixed_dr_specialists8`。正式状态以该目录 `state.json` 为准。

## 环境与训练

质量顺序：左手、右手、左小腿、右小腿。负载保留原来的位置、尺寸及复合惯量计算。

| GPU / B | 四肢负载 kg | 描述 |
|---|---|---|
| 0 | 0, 0, 0, 0 | 零负载，其他 DR 仍存在，因此不是 nominal |
| 1 | 1, 1, 1, 1 | 对称轻载 |
| 2 | 2, 2, 2, 2 | 对称中载 |
| 3 | 4, 4, 4, 4 | 对称重载 |
| 4 | 4, 4, 0, 0 | 双臂重载 |
| 5 | 0, 0, 4, 4 | 双小腿重载 |
| 6 | 4, 0, 4, 0 | 左侧重载 |
| 7 | 0, 4, 0, 4 | 右侧重载 |

其他静态 DR：在 A 的原始范围内，为每组独立预抽一次躯干质量、质心、足部摩擦、每关节 armature 和 encoder bias。精确物理张量及 SHA256 记录在 `protocols/dr_bank.json`；每个 B 的全部 8192 个并行环境重复同一套完整物理参数。每个 event 始终只采样一个原型再复制，因此训练/评测环境数量和动作噪声随机种子不会改变物理环境。所有 episode 重置都保留该环境。

motion、phase、初始状态噪声、观测噪声、原始随机躯干推力仍逐环境变化。每个 B 都加载完整 129,827 条 motion / 48,085,337 帧，不分配互不相交的 motion 子集。

每张卡运行一个直接 Python 进程，不使用 torchrun、DDP 或跨 B 梯度/统计同步。每个 B 的 actor、critic、观测编码器、优化器及 critic normalization 完全独立。从和 A 一样的 scratch 初始化训练，不从 A 微调。冻结 tracker 使用相同权重；B 不执行 context encoder，也没有 latent/router。

结构与 A 相同：actor `1645→512→256→128`，拼接当前冻结 tracker 的 29 维 raw action 后，`157→256→128→29`；critic 独立使用 `6330→1024→512→256→128`，拼接同一个当前 raw action 后，`157→256→128→1`。residual 均值仍为 `0.25*tanh(head)`。保留完整原始奖励和末端高度 termination，从第 0 轮开始 adaptive sampling。初始化同样有 500 步冻结 tracker 交互。

每卡 8192 环境、rollout 24、5 epochs、4 minibatches、actor LR 1e-4、critic LR 5e-4、entropy 0.0002、FP32，训练不设总轮数上限。每 100 个完整 PPO update 保存一次，每 1000 轮暂停该卡训练并在同卡的独立模拟器里做一次配对评测。

## 比较协议

A 的不可变副本及原始来源、update、SHA256 固定在运行目录；具体选择见 `protocols/evaluation.json` 和 `reference_A/selection.json`。预先固定同一份 512-motion 评测列表及起点种子，1000 步 horizon。两边均不需要长期记忆，使用 cold 评测，不重复做没有额外信息的 warm 评测。

启动时先把 A 放入八组完整固定 DR 各测一次。随后每个 B 到达 1000、2000……轮，分别在自己的 DR 上评测，并复用该环境的 A 结果。配对检查涵盖完整物理指纹（包含 encoder bias 和 motor privileged observation）、motion、phase、起始 qpos/qvel、噪声/推力种子和奖励配置。

主要报告共同存活时间内 body pos / joint pos 误差、失败率和覆盖率；同时报告 root pos、root rot 等原始指标及 2000 次 motion 配对 bootstrap 区间。原始 root 等误差按各自 episode 截断，需结合失败率一起解释。单个 B 的统计区间衡量 motion 差异，不代表多个训练 seed 的不确定性。

每个 B 的结果保存在 `ppo/B*/evaluation` 和 `specialist_eval_metrics.jsonl`；八个 B 到达相同轮数后，汇入 `comparison.json`。训练 RNG、模拟器步数及 critic normalization 在评测前后核对，确保评测不改变训练状态。

## 结论边界

这个实验检验：针对固定环境的独立 residual policy，是否可以超过当前通用 A。A 原先在连续 DR 分布上训练，B 在单个固定 DR 上训练，所以这并不是只改变参数共享的严格消融。

A 每轮采集 32768×24 = 786432 个 transition，单个 B 每轮采集 8192×24 = 196608 个 transition；相同 update 下 A 的总样本量是单个 B 的四倍。报告必须同时展示 update 和 transition，不能把短期 B 未超过 A 等同于专门策略无价值。B 若稳定改善对应 DR 的误差和失败率，可支持继续检验环境条件化路由；这还不证明共享 obs encoder 的 MoE 一定能复现独立专家收益。

物理 preflight 会在两种环境数量和不同种子下测试全部八组，并验证重置前后不变。正式启动前还须通过短程 PPO、checkpoint/optimizer/normalization 续训恢复和实际 A/B 配对评测。

## 正式启动验证

2026-09-14 02:20 UTC：八组均已完成真实 PPO 更新并通过启动审计。每组实际加载 129827 motions / 48085337 frames；8192 个 world 的完整静态 DR 与预检及 A 评测一致。模型、优化器及初始化与 A 对齐，训练进程没有 RANK/WORLD_SIZE，actor/critic 和八个 B 之间均无参数共享。

训练当前约 4.7–5.2 秒/update，显存约 43 GiB/卡。W&B 服务端确认八组都在运行并收到非零 update 和 transition。此前 A/MoE 已分别在 4020/3300 轮完整保存退出；比较始终使用预选的 A 4000 轮副本。

| GPU | B | W&B |
|---|---|---|
| 0 | all_0kg | [运行链接](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/dr-specialist-db3922625121) |
| 1 | all_1kg | [运行链接](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/dr-specialist-ab223a3f994e) |
| 2 | all_2kg | [运行链接](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/dr-specialist-4807a028f1cc) |
| 3 | all_4kg | [运行链接](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/dr-specialist-2c3373f33b0a) |
| 4 | arms_4kg | [运行链接](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/dr-specialist-f59ea52d1451) |
| 5 | shins_4kg | [运行链接](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/dr-specialist-0a6e501403dc) |
| 6 | left_4kg | [运行链接](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/dr-specialist-26289d79e79d) |
| 7 | right_4kg | [运行链接](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/dr-specialist-e6d58fa5fe5a) |

证据：运行目录的 `READY.json`、`physics_preflight.json`、`reference_A/verification.json`、`startup_verification.json`、`wandb_startup_verification.json`。最新进度读 `state.json`；八组同轮配对结果读 `comparison.json`。
