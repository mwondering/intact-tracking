# θ-aware contrastive dynamics-context Forward Predictor v7

该任务训练一个可微分 Forward Predictor，控制器是冻结 tracker，每个 vector world 在创建时
采样一次 dynamics domain randomization，之后跨 episode reset 保持不变。Residual Policy 和
Backward Predictor 不参与这个独立实验。

## 计算图

~~~text
10 个已完成的 (state, applied target, next state) 交互 H
                              |
                       Context Encoder
                              |
                    dynamics latent z -------------------+
                         /    |                           |
                        /     +--> contrastive loss       |
                       /      |                           v
       privileged head（仅训练）                      Forward Predictor
                    |
   normalized theta_hat <-> true theta
                                                          |
current state + current applied target -------------------+--> next state
~~~

Context Encoder 不读取当前尚未执行的 action，也不读取真实 theta。真实 theta 只用于
privileged regression 和训练期正负样本筛选，不会作为 Forward Predictor 的输入。普通
forward 不执行 privileged head；部署时也不需要 theta。

Transition Transformer 仍然使用 10 个历史 token 和 1 个当前 token，并在所有 token 上加入由 z
投影得到的 context condition。默认 transition 为 512 维、6 层、8 头；Context Encoder 为
128 维、2 层、4 头，输出 64 维 z。当 privileged target 为 34 维时，总参数量为
19,505,720。

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

## Privileged target

采集器根据 checkpoint 中真正启用的 startup event 构建紧凑参数向量，而不是把整个 MuJoCo
model 复制进 replay。目前支持 COM offset、relative body mass、friction、motor kp/kd/
armature/frictionloss 和 gravity。使用 0903_ckpts_71000 的实际 schema 为：

- torso COM offset 3 维；
- torso relative mass 1 维；
- shared foot friction 1 维；
- 29 个 joint armature scale。

共 34 维。encoder_bias 不是 conditional physics；由于 Predictor 已经接收 bias 作用后的真实
physical PD target，它不再影响给定 (state, applied target) 之后的 transition，所以被显式排除。
每个 target 维度在 warmup 期间单独建立 mean/std，训练开始后冻结。

## θ-aware 正负样本

Replay 为每个 anchor 采样同一 dynamics_id 下另一个不同 replay window，并把两者相邻排列，
因此任意 micro-batch 边界都不会拆开正对。一个样本内部五个 teacher-forced context latent
先按有效历史做 masked mean，再归一化后参与 supervised contrastive loss：

- 同一 dynamics_id、不同 replay window：正样本；
- 不同 dynamics_id，且标准化 θ 的逐维 RMS 距离至少为阈值：负样本；
- 不同 dynamics_id，但 θ 距离小于阈值：从分子和分母中完全排除。

默认 θ 距离阈值为 1.25。对 0903 checkpoint 的 128 个随机 world 实测，距离 P10、P25、P50
分别为 1.227、1.318、1.413；阈值 1.25 会排除最近约 10%–15% 的不同环境。该定义不会把参数
非常接近、只是 vector slot 不同的环境误当成负样本。对比学习在每个 rank 的 micro-batch 内
并行计算；默认 512 micro-batch 只产生 512 × 512 的相似度矩阵。

## 损失

~~~text
L = L_teacher + 0.5 * L_recursive + 0.1 * L_privileged + 0.01 * L_contrastive
~~~

L_teacher 使用真实当前状态做五个向量化的一步预测；L_recursive 从真实起点开始递推五步。
两者包含 normalized robot-delta/foot/contact-force Huber 和 contact BCE。
L_privileged 仅在 10 个历史交互都有效时，对标准化后的 theta_hat 与 theta 计算 Huber。
L_contrastive 使用 temperature 0.1 的余弦 supervised contrastive loss。各项可分别通过
--recursive-weight、--privileged-dynamics-weight、--contrastive-weight、
--contrastive-temperature 和 --contrastive-negative-distance 覆盖，不使用课程。

## 数据与优化

Replay 保存五步 robot/foot/contact trajectory、physical PD targets、10 帧历史，以及该
world 的 dynamics_id 和 compact theta。启动 DR 在同一 world 的 episode reset 后不重采样，
因而历史和 target 不会错配。目标五步不能跨 reset boundary；reset 后缺少的历史由 valid mask
显式标记。

启用对比损失时，batch size、micro-batch size 和 fixed-probe batch size 必须为偶数。Warmup
会等待 replay 中存在足够的同 world 不同窗口正对；设置 --contrastive-weight 0 可恢复普通
非成对采样。

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

脚本默认 0% nominal，即所有 world 都使用各自固定的 startup DR。要保留一部分 nominal 世界可传
--nominal-fraction；例如 2048 个环境时传 --nominal-fraction 0.125。第一次实验建议加
--fixed-batch-overfit --updates 5000 先验证容量。

## 重点指标

- train/rollout_nmse 和 horizon_1...5_nmse：递推误差相对实际变化的比例；
- train/teacher_nmse：真实状态条件下的一步动力学误差；
- train/privileged_dynamics_loss/mse/mae/nmse：z 对真实动力学参数的可识别性；
- train/contrastive_loss：θ-aware supervised contrastive loss；
- train/contrastive_positive_pair_count：同 world 正样本的有向配对数；
- train/contrastive_negative_pair_fraction：不同 world 中通过 θ 距离阈值的比例；
- train/contrastive_ignored_near_pair_fraction：因 θ 接近而被排除的比例；
- train/contrastive_positive_cosine 和 contrastive_negative_cosine：正负样本相似度；
- train/contrastive_valid_anchor_fraction：同时拥有正样本和有效负样本的 anchor 比例；
- train/context_full_history_fraction：privileged loss 实际覆盖的样本比例；
- train/dynamics_latent_rms：z 的尺度诊断；
- fixed_probe/*：内容不变的 batch 上的真实收敛曲线；
- optimization/gradient_norm：未裁剪的有效 batch 梯度范数。
