# Grouped-dynamics Context Forward Predictor v10

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
                              +--> matched contrastive loss       |
                                                                 v
last-10 state/foot/contact + current state/applied target --> next state/foot/contact

训练期匹配信息：dynamics ID / cohort / motion / phase / contact / state-action + true theta
                              |
                              +--> 正负样本选择（不进入模型）
~~~

Context Encoder 的模型输入严格限定为 100 帧历史 71 维 robot state、历史 applied target 和
当前 71 维 robot state。它不读取足端高度/速度、接触力、接触标记、当前尚未执行的 action
或真实 theta。历史前缀因 reset 尚未填充时由 valid mask 屏蔽，但剩余有效历史仍可产生 z 并
用于 Forward Predictor；只有表征损失要求 100 帧全部有效。足端与接触特权特征只进入
Transition Transformer；真实 theta 只作为训练期 theta-far 负样本挖掘标签，不进入任何网络。
模型没有 theta encoder 或 theta decoder，部署时不需要 theta。

Transition Transformer 仍然使用 10 个历史 token 和 1 个当前 token，并在所有 token 上加入由 z
投影得到的 context condition。默认 transition 为 512 维、6 层、8 头；Context Encoder 为
128 维、2 层、4 头，输出 64 维 z。默认总参数量为 19,500,310，且参数量不随 theta
维度变化。

## 固定动力学族与同步 motion

每个连续的 128 个 vector slot 构成一个 motion family，family 内第 0～127 个 slot 分别对应
同一组固定 dynamics prototype。每个 family 在完整 reset 或 motion 结束时只采样一个 motion
和 phase，再广播给全部 128 个成员。因此同一时刻可直接比较相同运动条件下 128 种动力学的
不同响应；相同 prototype 又会跨多个 family 观察不同 motion 和 phase。

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

## θ mining metadata

采集器根据 checkpoint 中真正启用的 startup event 构建紧凑参数向量，而不是把整个 MuJoCo
model 复制进 replay。目前支持 COM offset、relative body mass、friction、motor kp/kd/
armature/frictionloss 和 gravity。使用 0903_ckpts_71000 的实际 schema 为：

- torso COM offset 3 维；
- torso relative mass 1 维；
- shared foot friction 1 维；
- 29 个 joint armature scale。

共 34 维。encoder_bias 不是 conditional physics；由于 Predictor 已经接收 bias 作用后的真实
physical PD target，它不再影响给定 (state, applied target) 之后的 transition，所以被显式排除。
每个 theta 维度在 warmup 期间单独建立 mean/std，训练开始后冻结。标准化 theta 只用于
负样本距离计算，不存在 theta 回归头或对应 loss。

## Cross-motion 正样本与 matched hard negatives

默认 micro-batch 为 512，Replay 每次放入 4 个完整的 128-class cohort；每个 cohort 对应一个
motion family 在同一个 collector step 生成的样本，并按 dynamics ID 排序。只有 100 帧 context
全部有效的样本才进入 supervised contrastive loss：

- 同一 dynamics_id、不同 context window：正样本；motion、phase 和 episode 可以不同；
- 不同 dynamics_id 且标准化 θ 的逐维 RMS 距离至少为阈值：theta-far 负样本候选；
- 同一 cohort 的候选具有严格相同的 motion 和 phase，始终优先于其他 cohort；
- 候选内部按 `normalized state/action RMS + |motion_step gap| / phase scale + contact mismatch`
  排序，每个 anchor 最多保留 255 个困难负样本；
- 若同 cohort 的 θ-far 候选不足，才由其他 motion/phase 中最接近的 θ-far context 补足；
- θ 接近的不同环境不进入该 anchor 的对比分母；context 未填满的样本既不能作为 anchor，也
  不能作为别人的正/负样本，但其动力学预测损失照常计算。

默认 θ 距离阈值为 1.25。对 0903 checkpoint 的 128 个随机 world 实测，距离 P10、P25、P50
分别为 1.227、1.318、1.413；阈值 1.25 会排除最近约 10%–15% 的不同环境。该定义不会把参数
非常接近、只是 vector slot 不同的环境误当成负样本。默认 phase scale 为 50 simulator steps。
对比学习在每个 rank 的 micro-batch 内并行计算；默认 512 micro-batch 只产生 512 × 512 的
距离与相似度矩阵。没有正样本、没有有效负样本或 context 未满的 anchor 不贡献对比 loss，
并由 `contrastive_valid_anchor_fraction` 显式记录。

## 损失

~~~text
L = L_teacher + 0.5 * L_recursive + 0.01 * L_contrastive
~~~

L_teacher 使用真实当前状态做五个向量化的一步预测；L_recursive 从真实起点开始递推五步。
两者包含 normalized robot-delta/foot/contact-force Huber 和 contact BCE。
L_contrastive 使用 temperature 0.1 的余弦 supervised contrastive loss。各项可分别通过
--recursive-weight、--contrastive-weight、--contrastive-temperature、
--contrastive-negative-distance、--contrastive-hard-negative-count 和
--contrastive-phase-distance-scale 覆盖，不使用课程。

## 数据与优化

Replay 对每个样本保存五步 robot/foot/contact trajectory、physical PD targets、dynamics ID、
motion/episode/cohort metadata 和 compact theta mining label。100 帧 state/action context 不在
每个 sample 内重复物化，而是保存在按 collector step 索引的时间归档中，采样时再重建；
predictor 所需的足端/接触历史只取最后 10 帧。目标五步不能跨 reset boundary，reset 后缺失的
历史由 valid mask 显式标记。

启用对比损失时，batch size 和 fixed-probe batch size 必须能被 micro-batch size 整除；
micro-batch size 必须是 128 的整数倍且至少包含两个完整 cohort。Warmup 会等待 Replay 中存在
足够的完整 cohort；设置 `--contrastive-weight 0` 可恢复普通采样，但固定动力学族布局保持不变。

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
- train/contrastive_loss：matched supervised contrastive loss；
- train/contrastive_positive_pair_count：同 dynamics prototype 正样本的有向配对数；
- train/contrastive_cross_motion_positive_fraction：正样本中跨 motion 的比例；
- train/contrastive_nonoverlap_positive_fraction：跨 motion、跨 episode 或窗口不重叠的正样本比例；
- train/contrastive_theta_far_pair_fraction：不同 dynamics 中通过 θ 距离阈值的比例；
- train/contrastive_hard_candidate_pair_fraction：theta-far 对中属于同 cohort 的比例；
- train/contrastive_matched_negative_fraction：最终负样本中来自严格同 motion/phase cohort 的比例；
- train/contrastive_negative_pair_fraction：最终被选为困难负样本的不同 dynamics 对比例；
- train/contrastive_ignored_near_pair_fraction：因 θ 接近而被排除的比例；
- train/contrastive_positive_cosine 和 contrastive_negative_cosine：正负样本相似度；
- train/contrastive_hard_negative_phase_gap、state_action_rms、theta_rms：最终困难负样本质量；
- train/contrastive_valid_anchor_fraction：同时拥有正样本和有效负样本的 anchor 比例；
- train/context_full_history_fraction：具有完整 100 帧历史、可参与表征学习的样本比例；
- train/dynamics_latent_rms：z 的尺度诊断；
- fixed_probe/*：内容不变的 batch 上的真实收敛曲线；
- optimization/gradient_norm：未裁剪的有效 batch 梯度范数。
