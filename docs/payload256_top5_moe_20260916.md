# 256 个负载原型的 Top-5 MoE 对照

本实验比较相同专家网络在两种路由下的 PPO 表现。baseline 的 actor 和 critic 均不读取 latent：actor 由观测产生路由，由 PPO 策略梯度更新；critic 使用相同的、分离梯度后的路由权重。latent 组使用冻结 Memory350 的 latent 对固定负载中心作最近邻软路由；actor 不直接拼接 latent，critic 在私有特征压缩后拼接归一化 latent。两组同时改变了路由信息与 critic 的 latent 输入，因此结果比较的是整套环境表征接入方式，不能单独归因于 router。

## 环境与训练约定

- 仅四肢负载变化，COM、躯干质量、摩擦、armature、encoder bias 等背景 DR 为 nominal；两组均关闭 pushes 和观测噪声。
- 每只手四档：0、2.5/3、5/3、2.5 kg；每条小腿四档：0、4/3、8/3、4 kg。笛卡尔积为 256 个固定负载原型，顺序为左手、右手、左小腿、右小腿。
- 每卡 8192 环境，其中 4096 使用平衡的固定档位，每原型 16 个；4096 使用各肢体独立连续均匀负载。四卡每原型合计 64 个固定档位环境，另有连续环境训练混合控制。
- 每个 world 的负载固定，motion 均匀采样；正式 PPO 使用完整 `motion_data_full` 数据集。两组相同 seed/rank seed 规则及物理分布。
- wrist pitch/yaw 上限 10 Nm，原 tracker action scale 与 PD gains 保留。无 residual 输出 clamp/tanh/额外缩放，最终物理 actuator 力矩限制保留。
- 原始 termination（含末端高度），1000 control steps/episode；不预热，从空 memory 开始 PPO。长期记忆保留同一物理 world 的完整历史 chunks，可跨 episode；重置边界不作为真实 transition。
- 每次 rollout 24 步，5 epochs、4 minibatches；actor LR 1e-4、critic LR 5e-4、entropy 0.0002。初始动作 std=0.25。FP32 policy collection/update/evaluation；冻结 context 沿用 BF16 CUDA 推理，两处一致。
- 两组从头训练、不设更新上限，每 250 次完整更新保存；SIGTERM/SIGINT 请求完成当前更新后保存退出。

## 网络和梯度

| 部分 | Actor | Critic |
|---|---|---|
| 各自共享观测层 | 1645→512 | 6330→1024→512 |
| 每个专家的私有观测层 | 512→256→128 | 512→256→128 |
| 私有特征之后拼接 | tracker 原始均值动作 29 | tracker 原始均值动作 29 + latent 64（baseline 为零） |
| 私有输出头 | 157→256→128→29 | 221→256→128→1 |

每组有 256 个 actor 专家、256 个 critic 专家。actor/critic 不共享参数；同一个网络内只有表中首行跨专家共享。隐层用 ELU，专家独立随机初始化；actor 最后一层为零，使初始动作均值等于冻结 tracker。

baseline router 为 `[actor 512 维共享特征, tracker 动作29]→128→256`，选 logits 最大的 5 项后 softmax。只由 PPO actor loss 更新，不加 DR 标签、latent、辅助分类或负载均衡损失。critic 路由与 tracker action 均 detach。每个 PPO minibatch 重算当前 actor 路由，bootstrap value 也先重算，避免旧路由与当前 value 混用。

latent router 为 `softmax(-||normalize(z)-μ_e||²/τ)`，仅保留最近五个原型。中心是多 motion、完整记忆下单位 latent 的算术均值，不再把均值强制归一化。中心、温度和 context encoder 都冻结。运行时 router 不读取真实质量。

五个 residual **均值**加权后加到 tracker 原始动作均值上；使用一套可学习 29 维 Gaussian std。PPO log-prob 对实际最终动作分布计算，未混合不同专家采样动作或对多个 log-prob 求平均。critic 加权五个 value，监督实际混合策略的 return。

专家分配先统一排序，再对各专家的连续样本块运算，不丢弃超容量样本。未被选中的专家保留零梯度图，保证不同 GPU 访问不同专家时梯度同步形状一致。

## 原型标定与文件

源 encoder：`runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/update_035857.pt`，SHA256 `3c489cf5457889bf88441175ec4f10f4448104f44f7914eaaf762a9c3659d8f7`。它在完整 tracker DR + 四肢负载上训练，本实验明确将它部署在仅负载变化的子集，不修改其训练来源元数据。

标定使用冻结 tracker、4096 个 world（每原型 16 个独立 world），从全数据集均匀抽取固定的 4096 个 motion 文件。按完整 world 分为 50% 拟合、25% 温度验证、25% 留出测试，不把同一 world 的相邻时刻分进不同集合。标定不是 PPO 预热，正式 PPO 使用全新的 simulator/history。测试是不同 world/历史，未强制 motion 文件集合互斥；因此不能将其称为严格的留出 motion 家族泛化结果。

运行目录：`runs/limb_context_20260916_payload256_top5_moe`。`calibration/report.json` 记录温度、Top-1/Top-5、权重分布与物理核验；`calibration/prototypes.pt` 固定原型与来源；`calibration/calibration_samples.pt` 保留离线复查所需 latent、world、motion 信息。

本次标定共取得 58023 个完整记忆样本。留出测试 14473 个样本：Top-1 **80.76%**、Top-5 **94.74%**；验证集选定温度 **0.00341010**。测试平均最大权重 **78.91%**，每样本有效专家数（1/Σw²）**1.64**；这不表示整个训练只使用 1.64 个专家。质量加权读出 MAE 左手/右手/左小腿/右小腿分别为 0.0029/0.0029/0.3152/0.1842 kg，主要混淆来自小腿档位。原型文件 SHA256：`e18cdecc94d2b2fd7fad51f36ad4e1194f67e9450900905e61922a130af23eab`。

训练入口为 `python -m intact_tracking.cli.payload_prototype_train`；评测入口为 `python -m intact_tracking.cli.payload_prototype_eval`，支持 fixed masses、冷/暖记忆及全局 tracking 指标。启动脚本 `scripts/run_payload_prototype_comparison.py` 在两组真实仿真、多卡 smoke 都通过后才允许正式启动，记录精确命令、PID、GPU、W&B run ID 和源文件 hash。

## 检查与后续比较

代码测试覆盖稀疏分配相对稠密参考的输出/梯度、未访问专家零梯度、baseline latent 隔离、critic 梯度隔离、最终 Gaussian log-prob、router 在零输出初始化之后学习、固定原型 state roundtrip、负载范围与平衡复用。启动前两组各跑两卡、8192 环境/卡、两次完整 PPO 更新，检查分布式参数及物理状态一致。

正式分配：GPU 0–3 baseline；GPU 4–7 latent。记录总体专家使用数、单样本有效专家数、平均最大权重、逐步 Top-1 切换率及每专家权重份额，避免把单样本稀疏度误当作总体专家坍塌。

启动前实际 PPO 轨迹温度检查（u2 checkpoint、1024 world、50% 固定档位/50% 连续负载、真实 Gaussian 探索、800 步）：完整记忆固定档位最大权重均值 76.18%，连续负载 76.02%；有效专家数分别 1.75、1.71。连续负载中最大权重≤30% 的样本占 0.97%，有效专家数≥4.5 的占 0.86%。完整记忆 Top-1 逐步切换率 4.04%；采样轨迹的 Top-5 合计覆盖全部 256 个专家。故 latent 组保留 τ=0.00341010。这些是早期 checkpoint 的路由检查，不能代替后续控制效果评测。细节在 `temperature_runtime.json`。

用户明确要求只关注 latent 组的温度；baseline 维持 logit 温度 1，让 PPO 更新 router，不做额外温度标定。两组单元/相关回归共 24 项通过；分别完成两卡8192环境的 u0→u2 smoke 和 u2→u3 严格恢复检查。模型、优化器和归一化恢复验证及多卡参数一致均通过；冻结 tracker 逐 tensor 完全未变；latent 中心/温度逐 tensor 完全未变，baseline router 确实更新。两组初始专家权重 hash 一致。

后续比较应按相同 PPO 更新数/总交互数，分别报告固定负载档位与连续负载下的 tracking、root、失败率；另作 latent 打乱/置零干预，检查策略是否使用路由信息。离线原型识别率不能代替 PPO 新轨迹的路由可靠性或控制收益。

旧 FiLM 两组已完成当前更新后退出：baseline 1722，latent 1797。最终 checkpoint 及四卡参数一致性检查保存在 `runs/limb_context_20260916_tracker_film_obs_latent_4x8192_cold/{baseline,latent}/`。

2026-09-16 正式两组均已完成至少 6 次 PPO 更新，loss 有限，运行正常；四 rank 的初始实际物理审计两组逐项相同。baseline 每更新约 16 秒、latent 约 19 秒（启动早期统计，长期 chunk 完全填满后可能变化）。W&B API 已确认两组状态为 running 且可读取已上传 history。八张卡各一个正式训练 worker，显存约 baseline 41.5 GB/卡、latent 43.9 GB/卡。

- baseline：[W&B](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/payload256-e766cace997e)
- latent：[W&B](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/payload256-ad35ba2a3646)

完整启动核验在运行目录 `launch_verification.json`，恢复检查在 `preflight.json`，在线上传确认在 `wandb_online_verification.json`。固定质量的评测入口另已实际运行通过（2 motions、10 steps、包含全局/root 指标）；它只检验评测流程，不作为控制收益结论。

## 后续检查：环境类别是否真正对应专家

使用同一个已保存的 **启动测试 u2 checkpoint**，固定 1024 world 的负载 seed 与 512-motion 目录，比较确定性 tracker、PPO action mean 和 PPO Gaussian sample；仅统计短50+长30chunks完整的固定档位轨迹。

| 控制动作 | 主专家与四肢负载档位完全对应 | 正确专家进入 Top-5 | 正确专家平均权重 |
|---|---:|---:|---:|
| 冻结 tracker 均值 | 77.55% | 92.73% | 67.80% |
| PPO u2 均值 | 78.00% | 92.61% | 68.02% |
| PPO u2 采样（训练探索） | 50.07% | 88.86% | 46.46% |

专家身份确实固定绑定负载中心，但路由的语义正确性显著受探索轨迹影响。不能从最大权重较高推出选中了正确专家，也不能通过降低温度改善 Top-1 排序。PPO 采样时，最大权重≥90%的样本只有60.93%选对了主专家。

手部档位在 PPO 采样完整记忆轨迹中的正确率为94.49%/95.19%，左/右小腿为65.55%/81.72%。四肢仅按轻/重分类时同时匹配率为84.14%。双手均0kg的16个腿部档位尤其薄弱：PPO均值下Top-1为16.80%，采样下14.67%。整体平均数会掩盖该子集。

详细逐类混淆、每专家接收的质量分布、原始路由 trace、连续负载指标与图在 `runs/limb_context_20260916_payload256_top5_moe/expert_mapping_u2/README.md`。报告生成时正式 latent 训练为u95，尚未到u250首个周期保存点；以上不能冒充最新在线策略的评测。两组正式训练继续运行。原型、温度与训练网络未改动。

## 正式 u250：最新 checkpoint 与同 obs 专家动作检查

使用正式 latent 训练首次周期保存的 `latent/checkpoint_update_000250.pt`，内置 `completed_updates=250`，SHA256 为 `b5fcdbfba9fec6ea9c363e45bba179a3c28c9387dba2657884d1392cca78e880`。等待自动保存后评测，训练没有中断或重启。context encoder、256 个中心与温度均与上述原标定一致。这里的 u250 来自全数据集正式训练；此前 u2 来自启动测试，不是同一条正式训练链的中间快照。

使用与 u2 相同的 512-motion 清单、seed 910616、1024 world、50% 平衡离散负载 / 50% 连续负载、800 步。按完整记忆筛选，取 steps 500–790，每10步统计；分别执行策略动作均值和带探索噪声的动作。

| 模式 | u2 Top-1 | u250 Top-1 | u250 Top-5 | u250 正确专家平均权重 | 主导专家正确的类别 |
|---|---:|---:|---:|---:|---:|
| 动作均值 | 78.00% | 67.43% | 90.66% | 59.20% | 207/256 |
| 探索动作 | 50.07% | 56.72% | 91.04% | 51.96% | 173/256 |

u250 探索 std 均值0.16575，跨29关节RMS为0.16801。均值轨迹路由正确率下降，探索轨迹小幅上升；encoder/router固定，但 residual 和探索 std 同时变化，不能把差异单独归因于某一因素。双手均零负载子集 Top-1 仍只有均值18.78% / 探索19.05%。探索轨迹各肢体档位正确率：左手98.58%、右手98.91%、左小腿68.40%、右小腿81.84%。

同 obs 测试取 steps 500/600/700，分别获得2259/2246份完整记忆观测。固定1645维特征与同一冻结tracker action，用每个私有expert前向一次，比较确定性residual均值；最终action均值的专家间差值相同。诊断确认不消耗rollout RNG，Top-5重新加权与原策略输出最大误差低于1e-6。

| 同 obs 动作指标 | 均值轨迹 | 探索轨迹 |
|---|---:|---:|
| 全部256专家两两差异RMS | 0.23087 | 0.23008 |
| 当前Top-5两两差异RMS | 0.23304 | 0.22659 |
| 实际混合residual RMS | 0.13411 | 0.11788 |
| Top-5差异 / 实际residual幅度 | 1.74 | 1.92 |
| Top-5折算目标关节角差RMS | 5.43° | 5.27° |
| 候选残差方向平均cosine | 0.0926 | 0.0959 |
| 固定obs、更换另一条观测的路由后动作差RMS | 0.17570 | 0.15516 |

RMS在观测、不同expert对和29关节上平均差平方后开方。目标角仅是action乘实际scale后的目标差异，不是执行位移或扭矩。Top-5差异明显；排除最大输出的expert 30后仍为0.21662/0.21107，不依赖单个异常expert。

输出已分化，但尚不能证明有效的负载分工：相邻档位专家动作差约0.220/0.219，远档位约0.227/0.227，归一化负载距离与动作差的Pearson仅0.048/0.050；排除expert 30后也只有0.067/0.069。这是同obs的输出诊断，较弱的全局距离相关性不单独证明专家无用，亦没有替代专家执行后的控制收益对照。该检查未修改训练网络、参数、温度或路由。

完整记录与图：`runs/limb_context_20260916_payload256_top5_moe/expert_mapping_u250/README.md`、`expert_actions.md`、`summary.json`、`expert_action_summary.json`、`expert_action_outlier_check.json`；原始路由和每个expert的输出均已保存。复用入口为 `scripts/watch_payload_checkpoint_audit.py`、`scripts/summarize_payload_checkpoint_audit.py`、`scripts/report_payload_expert_actions.py`。
