# Uniform DR 在线 K-means16 MoE：八卡各 1024 环境，1000 轮

用户要求用与新共享 baseline 相同的预算训练现有 MoE，验证能否接近八个独立环境专家。运行目录：`runs/limb_context_20260914_uniform_moe16_8x1024_u1000`。

## 已固定的训练方案

从零训练一个共享的 MoE policy，8 卡 × 1024 环境，rollout24，1000 个已完成 PPO update，总计 196608000 个 PPO transition。相同完整 129827-motion 数据集按 rank 分片；全程 uniform motion，静态 DR 在每个 world 启动时独立均匀采样，此后跨 episode 保持。

Actor：1645→512→256→128，拼接当前冻结 tracker raw action29，然后使用 16 个独立的 157→256→128→29 head。Critic：6330→1024→512→256→128，拼接同一当前 tracker action29，然后使用 16 个独立的 157→256→128→1 head。Actor 与 critic 完全不共享参数；各自的 16 个 head 共用本侧 obs encoder。

使用先前选定的 response10 / stage1 / update15000 Memory350，SHA256 `db5d7bd7df026db163b9eedf588d37dcc81b55551d28afe3744f67a8ab2642ac`，在本次目录保留硬链接。Context encoder 及其归一化冻结；沿用缓存 BF16 推理，actor/critic 使用 FP32。64 维 latent 只用于归一化后的最近中心硬路由。

先进行相同的 500 步冻结 tracker 交互预热，在第 350/400/450/500 步收集 32768 个跨 rank latent，初始化 16 个中心。中心在每个完整 rollout 和全部 PPO epochs 内固定，PPO 完成后用跨 rank 汇总统计更新一次：center_rate=0.01，回溯限制本批重分配比例≤2%。Router 没有可训练参数；actor/critic 保存独立但相同的路由 buffer。长期完整 chunk 按原 Memory350 规则跨 episode 保留，reset 边界不作为交互。

其余均与 baseline 对齐：residual 直接线性输出，无 tanh/输出 clamp；wrist pitch/yaw ±10 Nm；双手各 [0,2.5] kg、双小腿各 [0,4] kg；原 tracker action scale/PD gains；原奖励和末端高度 termination；seed121；actor LR1e-4、critic LR5e-4、entropy0.0002、5 epochs、4 minibatches。每 100 轮保存，恰好 1000 轮停止。

## 验证与评测

33 项模型、路由、分布式统计和指标相关测试通过。真实八卡各1024环境、两轮 PPO 预检通过，16 个 expert 均被使用；八 rank 的 actor、critic、critic normalizer、全部 router buffer 哈希一致，中心更新次数为2。第二轮约4秒。预检首次因未带项目 W&B 凭据而在 PPO 前退出，已归档原日志；补齐凭据后重跑成功，未修改训练算法。

启动和完成时逐项核对 baseline 的训练参数、物理参数、共享 encoder 初始化、完整数据集、模型结构和 PPO 配置。额外将路由 buffer 一致性加入已有的训练结束核验，不改变梯度或路由更新算法。

1000 轮后，对比原共享 MLP、此次 MoE 和八个对应独立专家的第1000轮 checkpoint。使用相同八组完整固定 DR、512条 motion、起点、seed20001、最多1000步，保存全局和对齐后的全部20项指标逐步 trace。三方统一使用各 trial 中最短的共同存活窗口，并同时报告失败率和覆盖率。

分别评测 cold（空记忆）和 warm（同一冻结 tracker 预热500步后 reset 到相同测试起点）。Cold 的 MLP/专家复用刚完成的全局补测；warm 三方均实际执行相同预热协议。评测冻结 checkpoint 中的中心。重点看全局 body/root 位置、姿态、joint 误差，以及 MoE 弥补的 baseline→专家差距 `(baseline−MoE)/(baseline−expert)`；0%为baseline、100%为专家。

每个 policy 的 PPO 预算一致；八个专家合计八倍样本，且 MoE 比单头 MLP 参数更多。当前只使用一个训练seed；motion配对区间不代表跨训练seed的不确定性。

Supervisor：`scripts/run_residual_uniform_moe_comparison.py`；汇总：`scripts/report_residual_uniform_moe.py`。运行目录 `state.json` 为实时状态，`moe/metrics.jsonl` 保存每轮指标，结束生成 `comparison.json`、`comparison.csv` 和 `README.md`。

正式 supervisor 于 2026-09-14T08:07:47.300099+00:00 启动；完整数据和500步预热结束后已进入 PPO。启动配置核验通过，加载129827条motion、48085337帧；早期每轮约3.6秒。另有2项三方统计测试通过，共35项相关测试。W&B：[uniform-moe16-a2187efa3c7c](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/uniform-moe16-a2187efa3c7c)。

## 1000轮训练完成

正式训练已完成1000轮、196608000个PPO transition。首末update相隔3506.27秒（约58.44分钟），collect+learn平均3.4116秒/轮。所有1000轮均有16个expert被使用，单expert最高全局占比8.669%，中心更新单次重分配最高0.08596%。这些指标只说明路由占用没有坍塌，控制效果仍由后续配对评测判断。

结束核验通过：八rank的actor、critic、critic归一化和所有router buffers哈希一致，actor与critic路由一致，中心更新次数和PPO完成次数均为1000。当前进入cold/warm八组DR评测。

## 完整配对评测结果

各模式为8组DR×512条motion；三方使用同一trial共同存活窗口。

| 记忆 | 指标 | baseline | 共享MoE | 专家 | MoE相对baseline改善 |
|---|---|---:|---:|---:|---:|
| cold | 全局body pos(cm) | 20.7035 | 20.2815 | 17.1170 | +2.04% |
| cold | 全局root pos(cm) | 20.0992 | 19.6454 | 16.6169 | +2.26% |
| cold | 对齐body pos(cm) | 3.5034 | 3.5588 | 3.1847 | -1.58% |
| cold | joint pos(rad) | 0.5042 | 0.5186 | 0.5058 | -2.84% |
| cold | 全局root rot(rad) | 0.0589 | 0.0626 | 0.0543 | -6.30% |
| cold | 失败数/4096 | 22 | 15 | 12 | — |
| warm | 全局body pos(cm) | 20.7293 | 20.5642 | 17.1356 | +0.80% |
| warm | 全局root pos(cm) | 20.1251 | 19.9266 | 16.6362 | +0.99% |
| warm | 对齐body pos(cm) | 3.5098 | 3.5658 | 3.1892 | -1.60% |
| warm | joint pos(rad) | 0.5046 | 0.5197 | 0.5060 | -3.00% |
| warm | 全局root rot(rad) | 0.0589 | 0.0629 | 0.0544 | -6.63% |
| warm | 失败数/4096 | 21 | 23 | 12 | — |

全局body误差改善的motion配对95%区间：cold [0.10%,4.16%]，warm [-1.29%,2.80%]。弥补baseline→专家差距分别11.77%和4.60%。当前共享MoE尚未接近独立环境专家；对齐body、joint、姿态和速度指标总体还弱于baseline。Warm没有改善整体表现，因此不能仅用空记忆启动解释该差距。

Cold逐DR全局body：B4/B5/B7有显著改善，B6显著变差，其余区间包含零。Warm中B5/B7改善，B3/B6变差，其余区间包含零。只使用一个训练seed，不把这些区间解释为跨seed稳定结论。

W&B最终状态finished，completed_updates=1000。原实验完整结果保存在运行目录的`comparison.json`、`comparison.csv`、`README.md`；另有`comparison.png/svg`和`router_usage.png`。

[三方完整报告](../runs/limb_context_20260914_uniform_moe16_8x1024_u1000/README.md) · [位置误差图](../runs/limb_context_20260914_uniform_moe16_8x1024_u1000/comparison.png) · [路由占用图](../runs/limb_context_20260914_uniform_moe16_8x1024_u1000/router_usage.png)
