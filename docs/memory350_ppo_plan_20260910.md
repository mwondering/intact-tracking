# Memory350 latent 与无 latent residual PPO 对照

目标：在冻结 tracker 原始 DR 加四肢负载下，检验 Memory350 latent 是否使 residual policy 的 tracking 优于无 latent baseline。本文在正式 PPO 结果出现前固定协议。

| 项目 | 两组共同设置 |
|---|---|
| GPU | baseline 用 0–1，Memory350 FiLM 用 2–3；每卡 8192 环境 |
| 数据集 | `/data_zcy/wxy/motion_data_correct/motion_data_full`，129827 motions；每组两 rank 完整覆盖 |
| DR | 冻结 tracker checkpoint 的全部原始 DR，另加双手与两侧小腿中部各独立 U(0,4 kg) |
| 负载采样 | 每个 world、每条肢体独立采样，启动时固定；没有专门 nominal 环境 |
| Episode | 1000 个控制 step；reset 保留完整长期片段，清空短期历史 |
| Termination | 训练去掉 `ee_body_pos`，其余不变；测试保留原 tracker 完整失败标准 |
| PPO | seed 121；每组恰好 5000 completed updates，每 update 每环境 24 step |
| Motion sampling | 前 1000 updates uniform；从精确第 1000 轮 checkpoint 恢复至 adaptive，failure rewind 关闭 |
| 初始化 | residual actor、critic 都从头初始化；公共 trunk 使用相同专用 seed，初始 residual 均值为零，action std 0.25 |
| 优化 | actor LR 1e-4，critic LR 5e-4，5 epochs，4 minibatches，entropy 0.0002，residual scale 0.25 |
| 冻结部分 | tracker、Memory350 context encoder、context 归一化统计均冻结；PPO 不执行 predictor |

context 固定为 stage1 的 **22700 update** checkpoint，选择依据为阶段一保存的固定验证集最优五步 NMSE，先于 PPO 结果。不可变副本、来源和 SHA256 见 `runs/limb_context_20260910_memory350_ppo/context_selection.json`。

baseline 仅使用原 tracker 输入，actor 的原始特征为 1645 维；本次运行实际 critic 输入审计为 6330 维。Memory350 输出 64 维 latent，经 actor、critic 各自的 `64 → 256 → 各隐藏层 gain/bias` FiLM 使用。公共网络宽度不变，调制头初始化为零。这里只比较是否带来性能提升，不拆分 actor 与 critic 的贡献。

Memory350 保持 short50 与 long300 不重叠、每个长期片段内 10 步连续、最多 30 个完整片段的原始协议。冻结推理缓存已提交片段的编码和长期 summary，短期最终 attention 每个控制 step 都执行；没有改变已训练权重或上下文长度。

每 100 completed updates 保存不可覆盖的编号 checkpoint，暂停两 rank 的 PPO，在各自两卡上运行 512 个固定配对 motion 的 0/4 kg 测试，分 cold 与 warm 两种起始状态。测试结束在终端打印结果，并写入对应训练 W&B run。测试不改变训练 RNG、环境状态或归一化统计；1000 轮 curriculum 切换之外无需为了测试重启 PPO。

最终固定使用第 5000 轮，测试 **4096 个固定 motion/start/physics**，覆盖 0、2、4 kg 和四肢独立 U(0,4 kg)，每个场景都有 cold/warm 两种起始状态。另测冻结 tracker。最终 manifest 包含 periodic 的 512 motions，均在正式 PPO 开始前抽取，具体列表和哈希见实验目录 `protocols/`。

- cold：query reset 时短期与长期历史均为空。
- warm：先用同一冻结 tracker、相同 seed 和 DR 收集 500 步历史，再 reset 到固定 query；短期清空、完整长期片段保留。baseline 和 frozen tracker 同样执行 warm-up。GPU 接触仿真不保证逐位确定性，因此保存实际轨迹摘要，同时审计 DR、query 初始状态、起始帧和 warm-up 策略一致。
- query 最长 1000 步，提前 motion 结束或原始 failure termination 时截止；存活世界的时间线、状态和 observation history 不得被其他世界的 reset 推进。

主要评估场景是 `uniform_warm` 和 `uniform_cold`。同时报告 body/joint 误差、失败率、有效追踪覆盖率，并利用逐步轨迹计算两组共同有效时段的误差，避免提前失败带来的截断偏差。按 motion 配对 bootstrap 给出 95% 区间。强改善判据要求两个主要场景 body/joint 降幅区间均大于零，且失败率差值区间上限不大于零；不满足时分别解释各项指标，不选择其他轮次替代最终结论。

测试 motion 属于训练数据集，固定起点和物理随机种子不同；不作未见动作泛化声明。单个配对训练 seed 的 bootstrap 只描述测试动作的不确定性，不描述训练 seed 间波动。FiLM 比 baseline 增加参数，因此 tracking 改善本身不单独证明全部收益来自环境语义。

所有正式训练进入用户 W&B 项目 `intact-preview-v2`，group 为 `memory350-trackerdr-residual-20260910`；冒烟测试另设 group。实验目录为 `runs/limb_context_20260910_memory350_ppo`。源码改动、缓存、日志、checkpoint、报告均限定在当前项目，GPU 4–7 不用于此任务。
