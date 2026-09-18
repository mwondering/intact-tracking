# 取消 expert 参数共享：16 个完整独立 actor/critic

用户要求先完成共享 encoder MoE 的1000轮训练，再训练不共享参数的MoE作为对照。原运行 `runs/limb_context_20260914_uniform_moe16_8x1024_u1000` 保持原结构训练并评测；新增代码使用独立模块，不修改原训练模型实现。

## 实验变量

共享版的16个actor head共用一个1645→512→256→128 encoder，16个critic head共用另一个6330→1024→512→256→128 encoder；actor与critic不共享。Actor另有一组所有路由共用的29维Gaussian探索std。

独立版每个expert拥有自己的actor encoder、157→256→128→29输出head、29维Gaussian std，以及自己的critic encoder和157→256→128→1输出head。所有可训练参数均只属于一个expert，actor/critic完全独立。为对齐初始控制行为，encoder初值逐份复制自共享版的对应初值，head保持相同初始化规则，随后各份独立更新。

共同使用冻结tracker、冻结Memory350 context encoder、非梯度在线KMeans路由机制，以及critic观测的全局运行归一化统计。后者只有统计buffer，没有可训练参数；保留以隔离网络参数共享这一变量。Latent只用于硬路由，actor和critic使用相同16个中心。

先路由再编码，仅对分到各expert的样本运行其完整网络。空expert仍执行空batch以生成零梯度tensor，保证不同GPU上的梯度同步布局一致。探索std使用同一次actor前向得到的路由ID，并正确保存逐样本Gaussian参数供PPO计算概率比和KL。

## 参数量与解释边界

| 网络 | 共享 encoder MoE | 独立 encoder MoE | 独立版单个 expert |
|---|---:|---:|---:|
| Actor，含探索std | 2240365 | 17345440 | 1084090 |
| Critic | 8347536 | 115927056 | 7245441 |

Critic第一层6330→1024含6482944个参数，独立版复制16份，仅此一层合计103727104，约占critic总参数的89.5%。这是保留单expert宽度、取消共享后的容量增加，不是额外加宽网络。每个样本仅执行一个expert，但优化器状态和全量梯度同步开销增加。此对照同时改变共享程度与总容量，不能解释为容量匹配的路由消融。

## 相同训练与评测

两版均从零训练8卡×1024环境、1000个已完成PPO update、每轮24步，总计196608000个transition。Uniform motion、完整129827条motion按rank分片、独立连续随机DR、手部[0,2.5]kg/小腿[0,4]kg、wrist pitch/yaw ±10Nm、residual线性输出不限幅、原tracker动作scale与PD、原奖励和末端高度termination保持一致。Seed121，actor LR1e-4、critic LR5e-4、entropy0.0002、5epochs、4minibatches。

冻结context沿用response10/stage1/update15000，SHA256 `db5d7bd7df026db163b9eedf588d37dcc81b55551d28afe3744f67a8ab2642ac`。相同500步冻结tracker预热和32768个latent初始化16中心；每轮PPO结束后更新中心，rate0.01，重分配上限2%。Memory350长期完整chunk跨episode保留规则不变。

原共享版全部训练和评测完成、释放八卡以后，先做真实八卡两轮预检，核验所有rank的模型、归一化与router一致性，再开始独立版正式1000轮。两轮预检不计入正式训练预算，也不恢复其权重。

评测同一8组完整DR、512条motion、起点、seed20001、FP32和最多1000步。Cold空记忆与warm冻结tracker500步预热各一次，评测期间冻结中心。复用共享版已经完成的MLP、共享MoE和环境专家评测，与新增独立MoE共同使用每个trial四方最短存活窗口，报告全局body/root、对齐body、joint等20项指标和失败数。置信区间按motion配对bootstrap，八DR等权。

新增实现：`src/intact_tracking/memory350_independent_moe_policy.py`；训练/评测入口：`residual_independent_moe_train`、`residual_independent_moe_eval`。顺序运行脚本：`scripts/run_residual_independent_moe_comparison.py`；四方报告：`scripts/report_residual_independent_moe.py`。

45项新增及相关测试通过。独立版运行目录为 `runs/limb_context_20260914_uniform_independent_moe16_8x1024_u1000`，346份源码已快照并固定哈希。顺序supervisor已启动等待共享版训练和评测完成；尚未启动独立版GPU训练。

## 八卡预检完成与正式启动

2026-09-14，原共享版1000轮及cold/warm三方评测全部完成后，独立版自动开始真实八卡两轮预检。预检通过：所有rank的actor、critic、归一化、router状态一致，输入、初始化、物理与PPO配置匹配共享版，冻结tracker未改变。两轮均使用全部16个expert；第二轮collect3.5265秒、learn0.9221秒，总计4.4487秒。

预检结束后从零启动正式8×1024、1000轮任务，torchrun PID36112，supervisor PID18688，W&B run ID `independent-moe16-7604684a7047`。完整motion数据加载和500步预热后进入PPO。训练结束将复用已完成的对照评测，生成cold/warm四方共同存活窗口报告。

正式启动核验通过，完整加载129827条motion、48085337帧，已经进入PPO更新，早期约4.1秒/轮。预检额外核查16份actor encoder与16份critic encoder的末层投影均独立更新并产生16种不同权重，16组探索std也全部更新。对应证据为运行目录的 `smoke_expert_learning_audit.json` 和 `training_startup_audit.json`。

[正式W&B](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/independent-moe16-7604684a7047)；实时状态 `state.json`，逐轮日志 `moe/metrics.jsonl`，第1000轮后自动生成 `comparison.json/csv` 与 `README.md`。

补充初始router核查：相同seed与预热协议下，接触仿真轨迹并非逐bit一致；两版初始中心元素RMS差为0.00291035，同编号中心最大距离0.0576545。新中心各自最近的旧中心仍为相同编号，16个expert身份未出现交换。此差异保存在 `initial_router_comparison.json`，属于该单seed对照的解释边界，未在训练中途替换中心。

## 正式1000轮及四方评测完成

正式训练、八rank一致性核验与cold/warm四方配对评测均已完成。每个trial使用四方共同存活窗口；完整指标及区间见运行目录comparison.json/csv。

| 记忆 | 指标 | MLP | 共享MoE | 独立MoE | 环境专家 | 独立相对共享改善 |
|---|---|---:|---:|---:|---:|---:|
| cold | 全局body pos(cm) | 20.5969 | 20.1871 | 23.4068 | 17.0399 | -15.95% |
| cold | 全局root pos(cm) | 19.9913 | 19.5501 | 22.7244 | 16.5384 | -16.24% |
| cold | 对齐body pos(cm) | 3.5059 | 3.5602 | 3.9105 | 3.1870 | -9.84% |
| cold | joint pos(rad) | 0.5044 | 0.5187 | 0.5824 | 0.5059 | -12.28% |

cold失败数（每组4096次）：{'baseline': 22, 'shared_moe': 15, 'independent_moe': 30, 'specialist': 12}。

| warm | 全局body pos(cm) | 20.6269 | 20.4583 | 23.5566 | 17.0474 | -15.14% |
| warm | 全局root pos(cm) | 20.0211 | 19.8195 | 22.8354 | 16.5459 | -15.22% |
| warm | 对齐body pos(cm) | 3.5114 | 3.5677 | 3.9515 | 3.1925 | -10.76% |
| warm | joint pos(rad) | 0.5047 | 0.5199 | 0.5889 | 0.5063 | -13.27% |

warm失败数（每组4096次）：{'baseline': 21, 'shared_moe': 23, 'independent_moe': 34, 'specialist': 12}。


[完整报告](../runs/limb_context_20260914_uniform_independent_moe16_8x1024_u1000/README.md) · [位置误差图](../runs/limb_context_20260914_uniform_independent_moe16_8x1024_u1000/comparison.png)

## 用户请求后的评测复核

重新核验两个MoE的实际u1000文件SHA与报告一致，并核对64份四方评测JSON中的DR、motion、起点、物理指纹、记忆启动方式、FP32精度和1000轮checkpoint条件，全部通过。未重复启动相同GPU评测。

无共享版在cold/warm的全局body误差分别比共享版高15.95%/15.14%，相对增长的motion配对95%区间分别为[13.64%,18.58%]/[12.95%,17.51%]。两种模式都有6/8组DR的全局body显著差于共享版，没有一组显著优于共享版。Cold失败15→30，warm失败23→34（各4096次）。当前1000轮预算下，取消共享没有改善控制效果。

训练全部1000轮都使用了16个expert，最低有效expert数15.60，单expert全局最高占比8.77%，没有全局占用坍塌。独立expert各自累计获得约950万至1590万PPO样本，共享版的encoder则使用全部1.96608亿样本；数据量分摊与是否充分收敛仍是解释边界，不能把一次1000轮对照推广为独立expert永远更差。

无共享版collect+learn均值3.9085秒/轮，首末update相隔4018.71秒（66.98分钟）。训练已按计划在1000轮停止，W&B最终状态finished。核验记录为运行目录 `evaluation_review.json`。
