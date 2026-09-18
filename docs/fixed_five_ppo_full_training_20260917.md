# 完整数据集上的五套独立 PPO 训练

2026-09-17 07:23:50 UTC 按用户要求启动四卡版本。此前单卡任务在数据加载阶段停止，没有执行预热或 PPO 更新。数据集使用用户指定的 `/data_zcy/wxy/motion_data_correct/motion_data_full`，扫描得到 129,827 条 motion，无排除文件。GPU 4–7 执行 mjwarp；GPU 0–3 的既有 encoder 训练继续运行。

## 配置

| 项目 | 设置 |
|---|---|
| 专家环境池 | 每卡 4,096、全局 16,384 个固定 DR 环境，按预热 latent 距离固定分成四类 |
| baseline | 每卡 4,096、全局 16,384 个逐一配对的物理副本，覆盖全部四类，独立交互 |
| 仿真环境总数 | 32,768 |
| 预热 | 冻结 tracker 100 秒/环境，5000 控制步 |
| 分类 encoder | 冻结 response10/u15000，沿用原 nominal 中心与四档边界 |
| PPO | 四专家＋混合 baseline，共五套独立 actor/critic/optimizer/buffer |
| latent 输入 | actor、critic 均无 latent；训练期间固定归属，不再推理 encoder |
| 固定 DR | 每项标量独立 50% nominal＋50% 原范围均匀采样 |
| 每轮 rollout | 24 步；每分支全局合计 393,216 条 transition |
| 更新 | 5 epochs，4 minibatches；目标 1,000 轮 |
| 学习率 | actor 1e-4，critic 5e-4，fixed |
| entropy 系数 | 0.0002 |
| 初始 residual/std | 0 / 0.25 |
| seed | 121 |
| 保存 | 初始模型、每 100 轮、退出时 final |

复现实验入口和验证范围见[实现记录](fixed_five_ppo_implementation_20260917.md)。该实验比较的是四专家系统与单一混合策略在相同总交互预算下的表现。baseline 不使用专家行为轨迹进行 PPO 更新。

## 运行与监控

运行目录：[limb_context_20260917_fixed_five_ppo_full_4x4096](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096)。启动参数、PID 和源码快照分别在 `launch.json`、`train.pid`、`source/`。全局统计在 `train/`，每卡物理参数、固定分类和 checkpoint 在 `train/rank_00` 至 `rank_03`。

- [训练日志](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096/train.log)
- [监控当前状态](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096/monitor.json)
- [监控历史](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096/monitor_history.jsonl)
- [训练进度](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096/train/progress.json)
- [逐轮五 PPO 指标](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096/train/metrics.jsonl)

独立后台监控进程每 60 秒采样一次，持续至训练进程退出；记录数据加载/预热/PPO 阶段、各网络最近 20 轮奖励与损失、两分支交互预算、显存、速度与剩余时间估计，检测非有限指标、空类别、更新轮数不同步、长时间停滞和提前退出。异常写入 `monitor.json` 的 `alerts`。监控只记录与告警，不自动重启或重新分类。

监控中的四专家总奖励按环境数加权，可与 baseline 总奖励观察趋势；它不是留出评测结果，不能单独证明收益。

## 四卡与 W&B 核查

四卡加载分别为 32,457 / 32,457 / 32,457 / 32,456 条 motion，共 129,827 条、48,085,337 帧。各专家按 rank 上的实际样本数加权梯度，优势统计与 critic 归一化在同一个专家的四个 rank 间同步；五个专家/基线之间不共享这些状态。

[W&B 在线记录](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/c00dab6e)：项目 `intact-preview-v2`，run `c00dab6e`。后台 logger 每 30 秒补传所有新 PPO 更新，不丢弃中间轮数；五个角色分别使用 `train/expert_0` 至 `train/expert_3` 与 `train/baseline`。横轴为 `trainer/update`，另有按样本数加权的四专家总奖励及与 baseline 的差值。监控曲线横轴为 `monitor/elapsed_seconds`。

首次 W&B 接入使用系统默认账号被拒绝；已切换为仓库现有训练凭据，服务端确认 run 为 running 且已有监控历史。未改变训练进程或数据采样。

四卡 GPU 预检完成 100 秒预热及两轮 PPO 更新，并成功恢复至第三轮，全局四类数量为 80 / 35 / 241 / 156，baseline 512；模型和 critic 归一化跨卡 hash 检查通过。预检是独立小规模任务，其分类比例不代表正式 16,384 个环境的结果。

四卡预检审计：[verification.json](../runs/limb_context_20260917_fixed_five_ppo_4gpu_preflight/train/verification.json)。

## 正式启动完成核查

截至 2026-09-17 07:38:52 UTC，完成 18 轮正式 PPO 更新；最近 10 轮中位耗时约 6.50 秒/轮。四类固定数量为 1,771 / 955 / 8,534 / 5,124；baseline 为 16,384。两分支各累计 7,077,888 条 transition。四卡每环境完整 latent 查询数的最小值分别为 21 / 18 / 20 / 21。

四个实际 motion 文件清单无重复且并集恰为 129,827 条；首次正式更新后的五套模型及 critic 归一化跨卡 hash 一致；固定物理参数与标签、相等交互预算检查通过。详见 [startup_verified.json](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096/startup_verified.json)。后台监控与 W&B logger 继续独立运行。这些是运行正确性核查，尚不是收敛收益结论。

## Reward 一致性

四专家与 baseline 共用原始 tracker 的 20 项 reward。已核对四卡配置，并与 2026-09-16 tracker FiLM 的 baseline/latent、2026-09-14 独立 MoE16 及 uniform residual 的实际 run_config 比较，完整 reward contract（函数源码 hash、参数、权重、dt）完全相同：`abee73632d079f08c160584b04ace15532ecf6bd9dfa8139979af518d8a1553c`。本次未添加 residual 幅度、latent 或 DR 类别奖励。审计：[reward_consistency_audit.json](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096/reward_consistency_audit.json)。

## 第 226 轮后的 episode 计时修复

原五 PPO 入口遗漏了旧训练器 `init_at_random_ep_len=True` 的计时随机化。100 秒预热是 20 秒 episode 上限的整倍数，大多数环境在第 42、84、125、167、209 轮集中 reset；baseline 每次约 1.5 万个 episode endings，重置后的短暂跟踪误差造成 reward 尖峰。

按用户要求，在预热完成并固定分类后、或 checkpoint 恢复时，一次性将 episode 计时均匀随机化到 `[0,1000)` 控制步。同一 DR 的专家/baseline 副本使用相同初始计时，不同 rank 使用不同种子；专用随机数生成器避免改变策略采样 RNG。只设置首次 episode 的剩余时长，后续正常 reset 仍归零，20 秒上限不变。

新增每套 PPO 的 `timeouts`（只有超时）和 `failure_terminations`（失败优先，包括恰好同时超时），两者互斥且相加严格等于 `episode_endings`。原有 PPO 超时 bootstrap 逻辑未改。指标自动同步到原 W&B run `c00dab6e`。

2026-09-17 08:01:38 UTC 请求正常停止，保存至第 226 轮；08:02:35 UTC 从四卡完整 checkpoint 恢复。旧 checkpoint、源码、配置与启动记录保存在 [episode_phase_fix](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096/episode_phase_fix)。新入口恢复原模型、optimizer、分类与物理参数，跳过重新预热分类。11 项测试通过，包括连续两个多 episode 周期的重置分散检查、配对计时一致和超时/失败重叠处理。

恢复后观察第 227–272 轮，共 46 轮、每环境 1,104 个控制步，超过原来的 1,000 步超时周期。五套 PPO 的结束事件均已分散：

| PPO | 修复后每轮 episode_endings | 每轮 failure_terminations |
|---|---:|---:|
| expert_0 | 33–53 | 0–2 |
| expert_1 | 13–37 | 0–1 |
| expert_2 | 171–239 | 0–5 |
| expert_3 | 106–145 | 0–6 |
| baseline | 368–433 | 1–9 |

旧 baseline 五次集中结束分别为 15,344 / 15,212 / 15,045 / 14,888 / 14,777；本次观察未再出现这种同步重置尖峰。恢复仿真后的前五轮单独排除时，baseline 的 mean_step_reward 在 0.26856–0.27068，四专家也未再出现此前集中重置引起的同等幅度下跌。最近 20 轮中位耗时 6.48 秒/轮，监控无告警。

恢复入口的五套模型与 critic 归一化 hash 和 u226 checkpoint 完全相同，首次恢复更新的四卡同步检查通过；逐卡 reward、物理参数、固定分类、训练配置和 motion 清单均保持一致。五套 PPO 的超时与失败计数均满足相加等于总结束数。新增指标已确认出现在原 W&B run；该 run 的配置也已补记此次修复及恢复轮数。可重跑审计：[verify_fix.py](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096/episode_phase_fix/verify_fix.py)；结果：[verification.json](../runs/limb_context_20260917_fixed_five_ppo_full_4x4096/episode_phase_fix/verification.json)。这是修复和运行正确性验证，尚不代表专家相对 baseline 的收益结论。

## 第 300 轮配对评测

按用户要求抽取 512 个固定 DR 环境（每类 128 个），每环境 3 段 motion，对比同条件下的对应 expert 与 baseline。按训练比例加权、在双方共同存活时间内，body 误差降低 1.47%、root 位置误差降低 3.45%，joint 误差升高 2.90%；计入失败后的 reward 差异 +0.175%，区间覆盖零。C0/C1 的 body 改善较明显，C3 在 body、joint 和 reward 上落后，尚无全面胜出的结论。设置、各类结果和统计口径见[第 300 轮评测报告](fixed_five_ppo_u300_evaluation_20260917.md)。评测使用原训练 DR 与 motion 清单，不是留出泛化测试。

## 训练完成与第 1000 轮评测

四卡训练已正常完成 1000 轮，final checkpoint 同步和固定物理参数/分类检查通过。最新 checkpoint 复用第 300 轮完全相同的 512 个 DR 环境及 1536 组 motion/起点：四专家加权 body 误差降低 2.14%，joint 误差升高 2.37%，root 误差差异 +0.42%、reward 差异 -0.0026%（后两项区间覆盖零）。C1/C3 的 reward 低于各自类别 baseline，仍未形成整体胜出。详见[第 1000 轮与第 300 轮对照](fixed_five_ppo_u1000_evaluation_20260917.md)。
