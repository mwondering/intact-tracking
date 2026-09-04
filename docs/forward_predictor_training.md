# Nominal-counterfactual Context Forward Predictor v12

该任务训练一个由历史推断 dynamics latent 的可微分五步仿真器。控制器始终是冻结的 tracker；
Residual Policy、Backward Predictor、theta encoder/decoder 和梯度裁剪均不参与训练。

## 数据契约

每次数据采集包含两个批次：

- 批次 A：全局 4096 个环境，其中一半恢复为 compiled nominal physics，另一半保留 checkpoint
  定义的 startup DR。每个 vector slot 的动力学在整个训练期间固定，但 motion、phase 和 reset
  相互独立。
- 批次 B：与 A 数量相同的纯 nominal simulator。每收集一段 A 的五步轨迹，就把各 A 环境的
  起始 71 维物理状态恢复到 B，并直接重放 A 实际进入 simulator 的五个 29 维 PD joint target。
  B 不运行 policy，也不重复 action scale、offset、delay 或 smoothing。

因此，同一样本的 `A trajectory - B trajectory` 是在相同起点和相同控制输入下由 DR 产生的
可观察响应。B 仅提供表征监督，不进入 predictor 的预测目标；predictor 始终预测 A 的真实后续
状态，且 nominal 与 DR 数据共用同一个网络。

checkpoint 中的 step/interval 随机推力不会复现。startup DR 只在构造 A 时采样一次，之后 reset
不会重采样动力学。

## 模型与表征目标

Context Encoder 读取过去 100 个完整本体交互：

~~~text
(robot state_t, applied PD target_t, robot state_t+1) x 100 -> latent z
~~~

它不读取 foot、contact 或真实 simulator 参数。reset 后尚未积累 100 帧时，valid mask 会屏蔽
缺失历史；这种样本仍可训练 predictor，但不参与任何表征约束。

表征监督只有两项：

1. 同一个 A world、episode 和 motion 中，起点精确相差 5 帧且都具有完整 100 帧 context 的
   两个窗口作为局部不变视图，使其 latent 接近。
2. 对来自不同 A world 的样本，计算各自五步 A-B 响应之间的连续距离，并令归一化 latent
   距离匹配它。没有环境类别、正负阈值或 theta 距离筛选；很小的可观察差异对应很小但非零的
   目标距离，较大的响应差异对应较大的目标距离。

Transition Transformer 使用最近 10 帧 robot/foot/contact 状态、当前状态、applied target 和 z，
预测下一帧 70 维 robot delta、8 维足端状态、6 维接触力和 2 维接触 logits，并递推五步。
foot/contact 是 predictor 可使用并递推预测的 simulator 特权状态，但不会进入 Context Encoder。

默认损失为：

~~~text
L = L_teacher + 0.5 * L_recursive + 0.01 * L_representation
~~~

## Replay 与优化

Replay 保存普通、广泛的 A 五步片段及对应 B 轨迹，不再构造同步 motion family 或 128 类动力学
batch。predictor 直接从普通 motion-balanced replay 采样，因此能持续看到多种 motion 和 phase。
100 帧历史保存在时间归档中，采样时重建；默认每卡 replay capacity 为 262144。

模型使用 BF16 autocast、fused AdamW 和 micro-batch 梯度累积，不做梯度裁剪。robot state、
applied target、foot、contact force 与 state delta 的 normalization 在 warmup 后冻结。

## 启动

~~~bash
GPUS=0,1 ./scripts/run_forward_predictor_training.sh \
  /path/to/tracker.pt \
  /path/to/motions \
  /path/to/runs/forward_predictor_v12 \
  --wandb-name forward-predictor-v12
~~~

启动脚本固定全局 A 为 4096 个环境，并按 rank 均分；每个 rank 内严格保持 50% nominal / 50% DR。
Context Encoder 默认 100 帧，局部视图固定偏移 5 帧。

## 唯一保留的核心诊断

- `one_step_nmse`：predictor 一步误差相对“不改变状态”基线的比例，越低越好。
- `nominal_five_step_nmse` / `dr_five_step_nmse`：分别判断 nominal 与 DR 的递推五步精度。
- `latent_positive_cosine`：同环境相邻窗口的 latent 一致性，越接近 1 越好。
- `latent_response_correlation`：latent 距离与 A-B 响应距离的相关性，越高说明表征越符合可观察动力学差异。
- `latent_shuffle_dr_error_ratio`：在同一输入上换成其他环境的 latent 后，DR 五步 MSE 与原 MSE 的比值；
  明显大于 1 才能直接证明 predictor 在利用 latent，接近 1 表示基本忽略 latent。
- `dr_counterfactual_rms`：A-B 中实际存在的 DR 响应信号强度。
- `nominal_counterfactual_rms`：nominal A 与 nominal B 的配对误差，应接近 0；它用于检查反事实数据链路。

训练日志除此之外只保留三个优化 loss、学习率、update、样本数和 replay 大小，不再输出组件误差、
分位数、梯度范数或滚动窗口统计。
