# Memory350：仅用 DR 中心距离监督的表征变种

实现日期：2026-09-15。独立入口：
`python -m intact_tracking.cli.forward_memory_dr_center_train`。

后续启动配置已按用户指令更新为双手各 U(0,2.5) kg、双小腿各 U(0,4) kg，物理采样和 DR 距离归一化使用同一组上限。正式训练记录见 `docs/memory350_dr_center_hand2p5_shin4_20260915.md`；本页末尾首次实现检查的两个旧 smoke 仍使用当时的四肢 U(0,4) 配置。

此次实现针对“先只加第一个损失”的消融：用固定物理参数的归一化距离监督跨 motion 的 latent 中心距离，**不加中心容许半径损失**。局部正样本、弱正样本、异 world 弱负样本、A−B 响应关系损失也全部关闭。五步 predictor 继续训练，encoder 继续接收 predictor 和中心关系损失的梯度。

## 唯一启用的 representation loss

先按预先确定的物理采样范围归一化参数，不使用 batch 的均值或方差：

\[
x_{e,k}=\frac{\theta_{e,k}-l_k}{u_k-l_k},\qquad
d_{\mathrm{DR}}(i,j)=
\sqrt{\frac1{10}\sum_{g=1}^{10}\frac1{|g|}\sum_{k\in g}(x_{i,k}-x_{j,k})^2}.
\]

默认包含原 context 训练的全部 38 个已记录、直接影响物理动力学的坐标，组成十个等权因素：

| 参数 | 固定范围 | 在距离平方中的权重 |
|---|---|---|
| 左手、右手的附加质量 | 每项 [0, 2.5] kg | 每项 1/10 |
| 左小腿、右小腿的附加质量 | 每项 [0, 4] kg | 每项 1/10 |
| 躯干 COM x、y、z 偏移 | 每项 [−0.075, 0.075] m | 每项 1/10 |
| 躯干附加质量 | [−1, 1] kg | 1/10 |
| 共享脚部滑动摩擦系数 | [0.3, 2.0] | 1/10 |
| 29 个关节的 armature 比例 | 每项 [0.8, 1.2] | 整组 1/10，组内每项 1/29 |

COM、躯干质量、摩擦和 armature 范围从运行中的事件配置读取，表中是当前 tracker 的配置。躯干质量标签实际保存的是相对质量变化，归一化上下界会除以编译模型的躯干质量，保持与 kg 表示等价。四肢负载上限由 `--limb-max-masses-kg 2.5 2.5 4 4` 同时传给仿真采样和 DR 标签归一化；其余 tracker 物理配置保持不变。

不把 encoder bias 加入距离：当前 predictor 已输入经过动作链转换后的物理 PD target；保持既有 causal-label 口径。时变 force pulse 是扰动，不作为固定环境参数。负载引起的惯量、COM 变化已由四项质量决定，不重复计数。

每个世界维护四份 float32 **原始交互历史**候选，最短捕获间隔 200 个控制步，覆盖 nominal 和 DR 世界。采样时选择一份满足条件的旧历史，与当前 anchor 组成两个视图：

- 同一 world、同一物理参数 session；motion ID 不同。
- 两个视图各有完整的 short50 和 long30×10。
- 历史在各自查询时刻之前，两个视图不重复使用交互。
- 使用本次前向的 encoder 参数重新编码；不缓存旧 latent。

对两个视图的 latent 先做 L2 归一化，再取均值；同一 microbatch 内同一 world/session 的多行继续合并平均：

\[
\mu_e=\frac{1}{2n_e}\sum_{b:e_b=e}
\left(\frac{z_b}{\|z_b\|}+\frac{z_b^{\mathrm{archive}}}{\|z_b^{\mathrm{archive}}\|}\right).
\]

**不再归一化平均后的中心**。这是当前小批量的跨 motion 中心估计，不是全量 motion 的精确中心，也不是 EMA 中心。只有一个视图可用时，该样本仍参与预测任务，但不参与中心关系损失。

\[
t_{ij}=\frac{2d_{\mathrm{DR}}(i,j)}{d_{\mathrm{DR}}(i,j)+0.3},\qquad
L_{\mathrm{center}}=\underset{i<j}{\operatorname{mean}}
\operatorname{SmoothL1}_{\beta=0.25}
\big(\|\mu_i-\mu_j\|_2-t_{ij}\big).
\]

每个 rank、每个 microbatch 内，对所有合格中心的无序对等权计算。不是跨卡聚合中心；microbatch 大小会影响中心对数量及中心估计。不同 world ID 若 DR 参数完全相同，目标距离为零，包括不同 nominal 副本。没有按“异 world 就是负样本”的规则施加额外 margin。

可调参数：`--representation-weight 0.02`、`--dr-distance-scale 0.3`、`--dr-relation-beta 0.25`。固定范围内 $d_{\mathrm{DR}}\le1$，因此当前默认目标距离最大约 1.538；映射函数的渐近上限仍为 2。

## 全部 representation loss 对照

旧版权重取自已使用的 response10 / u15000 配置：
`runs/limb_context_20260912_memory350_response_window_ablation/response10/stage1_8192/run_config.json`。
下表权重均是**乘到总损失上的最终系数**，不是嵌套系数。

| 损失 | 定义 | 旧版 response10/u15000 | 本变种 |
|---|---|---:|---:|
| 局部正样本 | 同 world/episode/motion、相差 ±5 步，平均 $1-\cos(z,z^+)$ | 0.01 | 关闭 |
| A−B 响应关系 | 单条 latent 的单位 L2 距离拟合响应 RMS 映射的目标距离，SmoothL1 β=0.25 | 0.02 = 0.01×2 | 关闭 |
| 跨 motion 弱正样本 | 同 DR world/session、不同 motion 的完整且不重叠历史，平均 $1-\cos(z,z^+)$ | 0.008 | 关闭 |
| 异 world 弱负样本 | 不同非 nominal DR world 的完整历史，平均 $[\max(0,1.1-\lVert\hat z_i-\hat z_j\rVert)]^2$ | 0.008 | 关闭 |
| DR 中心关系 | 上述 $L_{\mathrm{center}}$ | 无 | **0.02** |
| 中心容许半径 | $[\max(0,\lVert z-\mu_e\rVert-r)]^2$ | 无 | **未加入** |
| nominal 原点锚定、方差/协方差正则、DR 参数回归头 | 无 | 无 | 无 |

原版响应距离是“两个样本各自的 A−B 标准化响应之差”的 RMS，原映射为 $2d/(d+0.3)$。新变种只沿用这个映射形式，输入已改为固定物理 DR 距离。旧 `response_distance_scale` 字段只是兼容元数据，**不控制新目标**；新参数为 `dr_distance_scale`。

本变种总损失：

\[
L_{\mathrm{total}}=L_{\mathrm{teacher,5step}}+0.5L_{\mathrm{recursive,5step}}
+0.02L_{\mathrm{center}}.
\]

两个 prediction 分支仍监督 root 位置/姿态/线速度/角速度、关节位置/速度、脚部特征、接触力和接触状态，默认各分项权重为 1。它们不计入 representation loss 清单，但会更新 encoder。A−B 的 nominal B 仿真暂时保留作数值和响应幅度诊断，其状态标签不进入新变种的优化损失。

## 训练、验证与兼容性

- 网络保持 nominal50 Memory350 encoder2x：short50 + long300，chunk/memory/context 深度 2/4/4，latent64，predictor 只预测五步。
- motion 采样和 replay 均默认 uniform。默认更新上限 5000，余弦调度跨度保持 8000。后续已按用户指令启动正式训练，详见单独的启动记录。
- 固定验证使用独立 world 的、含 DR 标签和跨 motion 历史的新 probe。正式 warmup 必须等到中心历史就绪。不会把缺少这些字段的旧 probe 当成中心验证，也不会静默把缺失标签解释成零损失。
- `--resume` 支持本变种在原输出目录中以相同损失和 schema 续训；不支持把旧响应版 checkpoint 直接当作同配置 resume。
- checkpoint 明确标记 `representation_supervision=dr_parameter_center_distance_v1`，保存 DR schema，并关闭旧的 counterfactual-supervision 标志。已有 history-only encoder 加载接口保持可用，policy 侧的 DR-profile 验证也识别此类型。
- 物理参数改变后，历史 session 失效；短历史也按查询时刻的有效长度裁剪，避免同 episode 中换参数时带入旧物理历史。

日志分别报告中心间距离、DR 目标距离、两者相关性、有效中心/中心对数量、DR–DR 和 nominal–nominal 子集距离，以及 `dr_within_center_distance_mean`、`dr_cross_motion_distance_mean`。后两项**只做监控，不反向传播**。中心对数量为零时，其余距离指标为零，不能作为已经聚类的证据。

仅约束平均中心并不保证每条 motion 的 latent 都紧凑；这次消融正是要测这一点。效果应再用跨 motion 环境 Top-1/Top-5、簇内距离和 predictor 误差判断，不能用训练的中心关系损失独自代替识别效果。

实现文件：

- `src/intact_tracking/memory350_dr_centers.py`：中心估计、唯一表征损失、原始历史 replay。
- `src/intact_tracking/memory350_dr_center_rollout.py`：固定物理参数 schema 与运行时审计。
- `src/intact_tracking/cli/forward_memory_dr_center_train.py`：独立训练入口、日志与 checkpoint 口径。
- `tests/test_memory350_dr_centers.py`：距离、因果历史、参数切换、损失隔离和梯度测试。

正式启动命令模板（实际命令保存在新 run 的 `launch_contract.json`）：

```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  --module intact_tracking.cli.forward_memory_dr_center_train \
  --checkpoint-file <frozen_tracker_checkpoint> \
  --motion-path <motion_dataset_directory> \
  --output-dir <new_dr_center_run_directory> \
  --num-envs 8192 --validation-worlds 128 \
  --limb-max-masses-kg 2.5 2.5 4 4 \
  --representation-weight 0.02 --dr-distance-scale 0.3 \
  --until-user-stop
```

`--checkpoint-file` 是冻结 tracker 的 checkpoint，不是 context checkpoint。每卡为全部 world 保存四份完整 raw history，archive 本身约占 7.3 GiB；启动时还需预留 predictor、replay 和仿真占用。

按用户后续要求，当前默认不设训练轮数上限。`--updates` 在此模式下只控制学习率衰减周期；只有显式传入 `--stop-after-updates` 才启用固定轮数停止。当前八卡、表征权重 0.04 的实际续训命令见对应 run 的 `unbounded_resume_contract.json`。

## 实现验证

56 项相关 CPU 测试通过，覆盖新变种以及原 Memory350、弱样本版、response-window 版、nominal50 扩容版和 policy DR-profile 校验。另通过 `git diff --check` 和 Python 编译检查。

单 motion、16 个仿真环境、1 次更新的检查已完成：
`runs/memory350_dr_center_20260915_smoke`。完整规模网络（21,086,486 参数）执行 BF16 前后向、日志和 checkpoint 保存；GPU 峰值分配约 1.20 GiB。该检查没有跨 motion 中心对，因此只验证训练链路，不用于判断表征效果。

两条 motion、24 个仿真环境、900 步 warmup、1 次更新的检查也已完成：
`runs/memory350_dr_center_20260915_cross_motion_smoke`。GPU 峰值分配约 1.23 GiB。
固定验证的八条历史全部符合跨 motion 条件，形成两个 DR 中心、一个中心对；中心距离约 0.9855，目标距离约 1.1567，中心关系损失 0.05860。

这次极小 smoke 的普通训练 batch 只有一条合格中心样本，拆成四条一组后没有训练中心对，所以这一次更新的 representation loss 为零。没有把它当作表征已经学好的证据。为直接验证真实数据的梯度通路，随后使用保存的验证历史单独反传中心关系损失，**不执行 optimizer 更新、不保存模型**：

- BF16 encoder 前向、FP32 中心距离，唯一表征损失非零；总损失公式核对通过。
- chunk / memory / final encoder 的表征梯度范数分别约 0.1494 / 0.1264 / 2.8866。
- anchor / archive 两个视图的梯度范数分别约 0.02800 / 0.02659。
- predictor 参数没有从这一项获得梯度，证明这里检查的是独立的表征梯度。

记录：`runs/memory350_dr_center_20260915_cross_motion_smoke/center_gradient_check.json`。
正式训练需关注 `dr_center_pairs` 和 `dr_center_eligible_fraction`，特别是 microbatch 很小时。

生成的 checkpoint 通过既有 history-only 加载与 DR-profile 校验，能够输出有限的 `[2,64]` latent，加载为推理模型后 encoder 正确冻结。首次实现阶段的短测均已结束，没有替换已有实验 checkpoint；正式训练是后续收到用户授权后单独启动的任务。
