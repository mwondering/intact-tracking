# Local-contrastive Context Forward Predictor v11

该任务训练一个可微分 Forward Predictor，控制器是冻结 tracker。启动时只采样 128 个
dynamics domain-randomization prototype，并将它们重复铺到所有 motion family；这些 prototype
在整个训练过程中不更新、不重采样。Residual Policy 和 Backward Predictor 不参与这个独立实验。

## 计算图

~~~text
100 个已完成的 (71-D robot state, applied target, next 71-D robot state) 交互 H
                              |
                       Context Encoder
                              |
                    dynamics latent z -------------------> Forward Predictor
                              |                                  |
                              +--> local contrastive loss         |
                                                                 v
last-10 state/foot/contact + current state/applied target --> next state/foot/contact

训练期匹配信息：world / dynamics ID / cohort / motion / phase / episode step
                              |
                              +--> 正负样本选择（不进入模型）
~~~

Context Encoder 的模型输入严格限定为 100 帧历史 71 维 robot state、历史 applied target 和
当前 71 维 robot state。它不读取足端高度/速度、接触力、接触标记、当前尚未执行的 action
或真实 theta。历史前缀因 reset 尚未填充时由 valid mask 屏蔽，但剩余有效历史仍可产生 z 并
用于 Forward Predictor；只有表征损失要求 100 帧全部有效。足端与接触特权特征只进入
Transition Transformer；真实 theta 不参与网络或正负样本选择。模型没有 theta encoder 或
theta decoder，部署时不需要 theta。

Transition Transformer 仍然使用 10 个历史 token 和 1 个当前 token，并在所有 token 上加入由 z
投影得到的 context condition。默认 transition 为 512 维、6 层、8 头；Context Encoder 为
128 维、2 层、4 头，输出 64 维 z。默认总参数量为 19,500,310，且参数量不随 theta
维度变化。

## 固定动力学族与同步 motion

每个连续的 128 个 vector slot 构成一个 motion family，family 内第 0～127 个 slot 分别对应
同一组固定 dynamics prototype。每个 family 在完整 reset 或 motion 结束时只采样一个 motion
和 phase，再广播给全部 128 个成员。因此同一时刻可直接比较相同运动条件下 128 种动力学的
不同响应。相同 prototype 虽会跨多个 family 出现，但跨 family、跨 motion 的 context 不再强制
映射到同一个 latent。

启动 phase 在 family 内相同、family 间随机错开，避免所有环境同时 timeout。正常 motion-end
和 timeout reset 保持整族同步。单个机器人因跌倒等原因提前失败时只 reset 该 slot，并让它
重新加入本族当前 motion/phase；不把一次局部失败扩散成 128 次 reset。该 slot 新产生但尚未
补满 100 帧的 context 可以训练 predictor，但不能作为任何正样本或负样本。多卡训练使用相同
dynamics seed，并在启动时校验 128 个 prototype 的跨 rank 哈希一致性。

## 状态、action 与递推

71 维 robot state 包含 root position/quaternion、root linear/angular velocity、29 维 joint
position 和 29 维 joint velocity。每个时刻另有：

- 8 维左右足端离地高度与三维世界速度；
- 6 维左右脚世界接触力；
- 2 维接触标记。

这些量在数据采集时直接从 simulator 读取。模型同时预测下一时刻的 70 维 robot delta、
8 维足端状态、6 维接触力与 2 个接触 logits。Recursive 分支只回灌自己的预测，
不使用真实未来特征，也不在模型热路径中执行 articulated FK。

29 维 action 是 Predictor 外部计算好的物理 PD joint target，不是 raw policy action。无状态的
scale、default offset、joint offset、clip、encoder bias 和关节顺序映射均在模型外完成。
如果 action 链包含 delay、alpha smoothing 或 boot-target override，训练入口会拒绝把它伪装成
无状态变换。

## DR provenance metadata

采集器仍根据 checkpoint 中真正启用的 startup event 构建紧凑参数向量，用于记录与核验固定
DR prototype，而不是把整个 MuJoCo model 复制进 replay。目前支持 COM offset、relative body
mass、friction、motor kp/kd/armature/frictionloss 和 gravity。使用 0903_ckpts_71000 的实际
schema 为：

- torso COM offset 3 维；
- torso relative mass 1 维；
- shared foot friction 1 维；
- 29 个 joint armature scale。

共 34 维。encoder_bias 不是 conditional physics；由于 Predictor 已经接收 bias 作用后的真实
physical PD target，它不再影响给定 (state, applied target) 之后的 transition，所以被显式排除。
checkpoint 只保存字段名、忽略的 startup event 与 prototype hash 作为实验 provenance。theta
不复制进 replay、不做 normalization、不进入模型、不作为预测目标，也不用于筛选或排序对比样本。

## 局部正样本与严格同相位负样本

默认 micro-batch 为 512，Replay 为一个 block 放入同一 motion family 的 4 个局部时间 cohort；
每个 cohort 对应一个 collector step，并按 dynamics ID 排序。默认 20 步局部范围下，4 个窗口
取相对偏移 0/5/10/20，因此 pairwise offset 覆盖 5/10/15/20 步。只有 100 帧 context 全部有效
的样本才进入 supervised contrastive loss：

- 同一个实际 world、同一 episode、同一 motion，且窗口偏移不超过 20 步：正样本；
- 同一 cohort 内除自身外的其余 127 个 world：全部作为负样本；
- 不同 cohort、不同 motion 或不同 phase 的跨环境 pair：忽略；
- 不读取 theta，不按参数距离、state/action 距离或接触模式设置阈值、排序或截断；
- context 未填满的样本既不能作为 anchor，也不能作为别人的正/负样本，但其动力学预测损失
  照常计算。

这里的 20 步只约束“同一 world 的两个窗口是否仍是局部正样本”，不是 dynamics 差异阈值。
系统不会预先裁定多小的动力学差异不值得学习：只要不同 world 位于严格相同的 motion/phase
cohort，它们就进入对比分母；encoder 能从历史中观察到多少响应差异，就学习多少。

对比学习在每个 rank 的 micro-batch 内并行计算；默认只产生 512 × 512 的余弦相似度矩阵。
没有正样本、没有有效负样本或 context 未满的 anchor 不贡献对比 loss，并由
`contrastive_valid_anchor_fraction` 显式记录。

## 损失

~~~text
L = L_teacher + 0.5 * L_recursive + 0.01 * L_contrastive
~~~

L_teacher 使用真实当前状态做五个向量化的一步预测；L_recursive 从真实起点开始递推五步。
两者包含 normalized robot-delta/foot/contact-force Huber 和 contact BCE。
L_contrastive 使用 temperature 0.1 的余弦 supervised contrastive loss。各项可分别通过
--recursive-weight、--contrastive-weight、--contrastive-temperature 和
--contrastive-positive-max-offset-steps 覆盖，不使用课程。

## 数据与优化

Replay 对每个样本保存五步 robot/foot/contact trajectory、physical PD targets、dynamics ID、
motion/episode/cohort metadata 和 compact DR provenance。100 帧 state/action context 不在
每个 sample 内重复物化，而是保存在按 collector step 索引的时间归档中，采样时再重建；
predictor 所需的足端/接触历史只取最后 10 帧。目标五步不能跨 reset boundary，reset 后缺失的
历史由 valid mask 显式标记。

启用对比损失时，batch size 和 fixed-probe batch size 必须能被 micro-batch size 整除；
micro-batch size 必须是 128 的整数倍且至少包含两个完整 cohort。Warmup 会分别等待足够的
局部 cohort run 和至少一个完整 100 帧 context 的固定 probe；普通训练 batch 仍可包含 reset 后
未填满的 context。设置 `--contrastive-weight 0` 可恢复普通采样，但固定动力学族布局保持不变。

默认每卡 replay 262144，按 motion ID 平衡采样。--batch-size 4096 是每卡每个
optimizer step 的有效 batch，按 --micro-batch-size 512 做 BF16 梯度累积，最后只执行一次
fused AdamW step。DDP 仅在最后一个 micro-batch 同步梯度，不执行梯度裁剪。

## 启动

~~~bash
GPUS=0,1 ./scripts/run_forward_predictor_training.sh \
  /path/to/tracker.pt \
  /path/to/motions \
  /path/to/runs/forward_predictor_context \
  --wandb-name forward-predictor-context
~~~

脚本默认每卡 2048 个环境，即 16 个 motion family、每族 128 类固定动力学。该布局要求
`--nominal-fraction 0`；prototype 不会在训练中刷新。第一次实验建议加
`--fixed-batch-overfit --updates 5000` 先验证容量。

## 重点指标

- train/rollout_nmse 和 horizon_1...5_nmse：递推误差相对实际变化的比例；
- train/teacher_nmse：真实状态条件下的一步动力学误差；
- train/contrastive_loss：local exact-cohort supervised contrastive loss；
- train/contrastive_positive_pair_count：同 world 局部窗口的有向正样本数；
- train/contrastive_negative_pair_count：严格同 cohort 跨 world 的有向负样本数；
- train/contrastive_candidate_pair_fraction：实际进入对比分母的 pair 占完整 context pair 的比例；
- train/contrastive_exact_cohort_negative_fraction：负样本中 motion 与 phase 精确匹配的比例，应为 1；
- train/contrastive_positive_episode_step_gap 和 positive_motion_step_gap：正窗口平均偏移；
- train/contrastive_positive_cosine 和 contrastive_negative_cosine：正负样本相似度；
- train/contrastive_valid_anchor_fraction：同时拥有正样本和有效负样本的 anchor 比例；
- train/contrastive_full_context_sample_fraction：可参与表征损失的完整 context 样本比例；
- train/context_full_history_fraction：具有完整 100 帧历史、可参与表征学习的样本比例；
- train/dynamics_latent_rms：z 的尺度诊断；
- fixed_probe/*：内容不变的 batch 上的真实收敛曲线；
- optimization/gradient_norm：未裁剪的有效 batch 梯度范数。
