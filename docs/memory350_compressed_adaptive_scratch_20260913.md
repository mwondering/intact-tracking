# Compressed residual PPO：从零训练，直接 adaptive

2026-09-13，baseline 和 latent 两组从零初始化 residual actor、critic、各自压缩 MLP、优化器和 critic 归一化统计。训练根目录为 `runs/limb_context_20260913_memory350_compressed_ee_adaptive_scratch`，实时进度见该目录 `state.json`。第一轮起使用 adaptive motion sampling，`adaptive_after_update=0`，没有 uniform 前缀或第 1000 轮课程切换。

两组从第一轮启用原 `ee_body_pos`：左右脚踝、左右手腕的 z 高度相对 pelvis 对齐后的参考高度，任意一项绝对误差超过 0.5 m 时终止。原 anchor 高度、姿态和 1000 步超时规则继续启用。奖励、DR、网络结构和其余 PPO 超参数沿用之前的压缩观测对照。

|配置|Baseline|Latent|
|---|---|---|
|GPU|0–3|4–7|
|每卡环境数|8192|8192|
|Actor 压缩|1645→512→256→128|相同|
|Critic 压缩|6330→1024→512→256→128|相同|
|Actor / critic 拼接槽|64 维全零|64 维冻结 encoder 输出|
|初始化种子|121|121|
|训练总上限|无|无|

冻结 tracker 和 response10 update 15000 的 Memory350 encoder；encoder SHA256 为 `db5d7bd7df026db163b9eedf588d37dcc81b55551d28afe3744f67a8ab2642ac`。Context encoder 及其归一化不随 PPO 更新，predictor 不执行。只更新 residual actor、critic 及各自的观测压缩 MLP。

在线短期/长期记忆在进程启动时为空，长期记忆在常规 episode reset 后继续保留。每个 world 的静态 DR 在进程内固定，仍保留原动态力扰动。Adaptive 统计从 priors 开始，不从旧训练恢复。

继续使用原固定 512-motion、0/4 kg、cold/warm 端点评估协议，初始化评估一次，此后每 100 updates 评估；评估始终启用原完整 termination。同轮次对比写入 `ppo_comparison.json`，新训练独立记录 W&B。根目录 `STOP_PPO` 请求两组停止。

原实验 baseline u4283、latent u1360 的最终和中间 checkpoint 均保留。曾启动的共同 u1200 恢复分支已按用户新要求在初始化期间取消，没有记录到 PPO update。它的源快照与取消记录保留；当前训练完全使用从零初始化的独立目录。

原 PPO 运行代码恢复为已验证版本，本次配置通过既有 `--training-terminations original --motion-sampling adaptive --adaptive-after-update 0` 实现。`PPO_READY.json` 记录测试、启动参数解析、源文件和固定输入哈希；实际初始化及训练推进另写启动审计。

启动核对已通过：26 项检查通过，实际 u0 checkpoint 的两组 actor 和 critic 可训练权重逐位相同，优化器状态为空，四 rank 的 adaptive 统计均从 u0 保存。两组 critic 各用首批 32768 个观测初始化归一化，计数相同，数值统计有差异：均值最大差为 0.00209 个标准差，标准差最大相对差约 1.38%；未宣称归一化缓冲逐位相同。前六轮都实际运行 adaptive 且数值有限。第 0 轮四项固定评测完成，双方 query 初始状态、物理指纹、motion/start 和负载配对一致，训练状态保护通过。完整结果见训练根目录 `startup_verification.json`、`initial_endpoint_pairing.json`。

W&B：[baseline](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/m350compressed-43cb8d622e96)、[latent](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/m350compressed-a71787775410)。

清理 GPU 4–7 的四个占卡进程后，前后各 30 个稳定 PPO updates 的实际每轮时间中位数为：

|组别|清理前|清理后|
|---|---:|---:|
|Baseline|3.89 s|3.90 s|
|Latent|16.49 s|4.81 s|

Latent 提速 3.43 倍，清理后比 baseline 每轮多约 23%。该测量来自旧训练同一运行进程，排除端点评估，不能当成新 termination 下的稳态速度。原始记录见旧根目录 `analysis_20260913_gpu_cleanup/speed_comparison.json`。

## 第 1000 轮固定评测

核对双方精确 u1000 checkpoint、四项评测与原始逐步轨迹，512 条固定 motion/start/query/physics 配对通过。共同有效时段 body 误差：0 kg 冷/热均降低约 19.6%，joint 降低 13.1% / 12.8%；4 kg 冷/热 body 则高 2.35% / 1.73%，joint 高 1.84% / 1.47%。这些误差差异在按 motion 配对的 95% 区间内方向一致，但不包含训练 seed 波动。

4 kg 失败次数 cold 为 baseline 44、latent 38，warm 为 41、35，均下降 1.17 个百分点；差值区间分别为 [-2.93, +0.39]、[-2.93, +0.59] 个百分点，尚不能确认失败率改善。按各自有效步数统计，4 kg 根部位置误差低 11.2% / 12.8%，body 线速度误差低约 5.6%。这轮存在不同指标之间的取舍，不能认定 latent 已全面优于 baseline；0/4 kg 端点也不代表完整的独立 U(0,4 kg) 负载分布。

训练继续。[第 1000 轮完整报告](../runs/limb_context_20260913_memory350_compressed_ee_adaptive_scratch/analysis_001000/report.md) · [全部指标 CSV](../runs/limb_context_20260913_memory350_compressed_ee_adaptive_scratch/analysis_001000/metrics.csv)。

## 第 2000 轮固定评测

两组精确 u2000 checkpoint 与四项固定 512-motion 评测、逐步轨迹核对通过，优化器 step 均为 40000，adaptive/EE termination/冻结 encoder 配置一致。共同有效时段 body 误差：0 kg 冷/热由 baseline 4.397/4.390 cm 降至 latent 3.546/3.535 cm，降幅 19.35%/19.48%；4 kg 则由 6.451/6.434 cm 升至 6.642/6.611 cm，高 2.95%/2.74%。关节位置误差在 0 kg 低 9.46%/9.31%，4 kg 高 2.78%/2.93%。上述误差差异的按 motion 配对 95% 区间均不跨零。

4 kg 失败次数 cold 为 baseline 38、latent 37，warm 为 39、35，差值分别为 -0.20、-0.78 个百分点；95% 区间为 [-1.76,+1.37]、[-2.54,+0.98]，仍不能确认失败率改善。相比 u1000，低负载 body 优势基本不变，高负载 body/joint 劣势略扩大，失败率差距缩小。按各自有效步数，4 kg 根部位置误差低 15.1%/13.3%，body 线速度误差低 6.6%/7.2%，仍存在指标取舍。

此处比较每个 checkpoint 内双方共同有效时段的组间差距；不同轮次的有效时段不保证相同，也不包含训练 seed 波动。训练继续。[第 2000 轮完整报告与 1000→2000 对照](../runs/limb_context_20260913_memory350_compressed_ee_adaptive_scratch/analysis_002000/report.md) · [全部指标 CSV](../runs/limb_context_20260913_memory350_compressed_ee_adaptive_scratch/analysis_002000/metrics.csv)。

用户随后要求暂停本实验，测试 256 种共享 DR。baseline 已于 4500 轮、latent 于 3701 轮保存并退出，最终 checkpoint、优化器和 motion sampling 状态均已核验保留。新的两组从零训练已启动并通过实际更新检查，配置及证据见 [grid256 实验记录](memory350_grid256_ppo_20260913.md)。
