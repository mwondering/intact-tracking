**v12 与原版 Memory350 的训练差异及证据审计，2026-09-11**

最值得优先验证的是表征监督与训练分布的变化。现有证据支持长期记忆在历史充分时提供预测收益，也能降低新版匹配验证集上的跨 motion 漂移；不能把 v12 到 Memory350 的全部退化归因于长期模块。当前目标没有直接约束同一 DR 在不同 motion 下输出相同的环境表征，关系目标本身又显著随状态、动作和 motion 改变。

这里的 v12 是 `forward_predictor_transformer_v12_8192_run3/update_008000.pt`；Memory350 是原版 `limb_context_20260909_memory350/stage1_8192/update_022700.pt`，不是 encoder2x，也不是旧 context200。两个 checkpoint 的在线聚类对照见 [完整报告](README.md)。本次额外检查仅加载已有数据和 checkpoint，没有启动或修改训练。

**实际训练差异**

| 项目 | v12 | 原版 Memory350 |
|---|---|---|
| A 侧 nominal 环境比例 | 50% | 0% |
| B 侧 nominal 反事实 | 每条 A 的同初始状态、同物理 PD target | 相同协议，仍有 nominal B |
| motion 数据 | AMASS_LAFAN_Qingtong，42 条 | motion_data_full，129,827 条 |
| DR 与临时外力 | tracker startup DR；移除 push_robot | tracker startup DR + 四肢各独立 U(0,4 kg)；保留 push_robot |
| context | 同 episode 最近 100 步交互 | 最近 50 步 + 不重叠的 30×10 步长期交互；长期跨 reset 保留 |
| encoder 参数量 | 440,512 | 1,035,328 |
| predictor | 512 宽、6 层、8 heads；历史 10 步、预测 5 步；19,059,798 参数 | 相同 |
| 最终 latent | 64 维，LayerNorm；表征损失再做逐向量 L2 归一化 | 相同 |
| 同环境正样本 | 同 world/episode/motion 的精确 ±5 步 | 相同；并未增加跨 motion 正样本 |
| 表征损失有效性门槛 | anchor 和 positive 均有完整 100 步历史 | 有任意短期或长期历史即可 |
| 有效 positive / relation 权重 | 0.01 / 0.01 | 0.01 / 0.02 |
| response_distance_scale | 1.0 | 0.75 |
| 每 rank A 环境数 | 1,024 | 8,192，其中 128 个固定验证 world |
| 全局优化 batch | 8,192 | 4,096 |
| 归一化预热采集步数 | 100 | 500 |
| checkpoint update / optimizer steps | 8,000 / 32,000 | 22,700 / 90,800 |
| checkpoint 实际 LR | 2.9528747e-4 | 1e-5 |
| cosine horizon | 100,000 updates | 8,000 updates，随后最低 LR 继续训练 |
| 累计采集 transitions | 164,270,080 | 3,735,388,160 |
| seed | 0 + rank | 717 + rank |

配置来源：[v12 run_config](../forward_predictor_transformer_v12_8192_run3/run_config.json)、[Memory350 run_config](../limb_context_20260909_memory350/stage1_8192/run_config.json)。学习率和优化步数同时核对了 checkpoint 的 optimizer/scheduler 与对应 update 的 metrics.jsonl。总参数量减去两者相同的 predictor 参数量得到 encoder 参数量。归一化统计、数据难度和采样也不同，不能直接用不同运行的 loss 数值判断拟合质量或梯度强弱。

**为什么 nominal 的变化尤其大**

当 A 本身也是 nominal 时，A−B 响应通常接近零，且对 motion 的依赖很小。v12 有大量 nominal A，因此来自不同 nominal world、不同 motion 的样本经关系损失获得接近零的目标距离，形成一个隐式的共同簇。它约束的是相互距离，不要求簇中心位于原点。

Memory350 的 nominal_fraction=0，虽然仍计算 nominal B 作为监督目标，却没有用这些 B 轨迹作为 context encoder 的输入。上述 nominal 跨 motion 的隐式约束消失了；同时我们的干净 nominal/无额外负载测试也偏离了新版的主要训练分布。这是 nominal 0.070→0.479 的强候选解释，但尚未通过单独恢复 nominal A 的训练消融证明因果，也不能独自解释全部 DR 跨 motion 漂移。

**“同探针”在代码里只成立于每条 A–B 配对**

令 q_i 表示第 i 个窗口的初始状态与未来 5 步物理 PD targets，r(e,q) 为环境 e 相对 nominal 的归一化状态响应。当前代码计算：

`D_ij = RMS[r(e_i, q_i) - r(e_j, q_j)]`。

每个 r 内部的 A 与 B 使用相同 q；跨两个 A world 比较 r 时，q_i 和 q_j 没有匹配。`_cross_world_partner` 只把 batch 索引平移半个 batch，并检查 world_id 不同；不匹配状态、动作、motion 或 phase。因此它不等价于对共同 Q 计算 `E_q ||r(e_i,q)-r(e_j,q)||²`。这个限制在 v12 已存在，不是 Memory350 新引入的 bug。更广的 motion、负载和临时推扰可能使它的影响加重，需要消融验证。

来源：[关系损失与配对](../../src/intact_tracking/forward_predictor_objective.py)、[正样本筛选](../../src/intact_tracking/data/predictor_online.py)、[nominal B 状态与动作恢复](../../src/intact_tracking/rollout/nominal.py)。

本次直接重算新版四 rank 固定 broad validation 的真实响应。输入共 2,048 个窗口，有可用 context 的 2,001 个；以下沿用训练的 5×70 响应及映射 `target=2D/(D+0.75)`，统计的是每对的映射目标的平均值：

| 响应对 | 对数 | 响应距离 D 的平均值 | 映射目标平均值 |
|---|---:|---:|---:|
| 同一物理 world，不同 motion | 3,509 | 1.7230 | 1.2745 |
| 不同 world，未匹配 motion/phase | 495,168 | 1.7807 | 1.3295 |
| 同 world、不同 motion，双方长期均满 30 段 | 416 | 1.7845 | 1.2697 |

这直接说明当前响应特征不是跨 motion 不变的环境描述。同 world 对在实际 relation loss 中被排除，本表不是在声称训练直接把同 world 的这些对推开。不同 world 组未共用状态/动作，仅作为目标尺度参照；这些成对观察相互依赖，不能把对数当作独立样本数。检查也没有分离 motion、状态动作激励、临时推扰和数值误差各自的因果贡献。原始结果、公式和文件哈希见 [training_response_audit.json](training_response_audit.json)。

Memory350 保留的 push_robot 是 torso_link 上每隔 3–6 秒、持续 0.3–0.5 秒、幅值范围 0–10 N 的随机力脉冲；B 移除了全部事件。固定 DR/负载不意味着此刻外力也固定，A−B 响应可能因此包含临时扰动。关推扰是可以隔离这一因素的对照，不代表推扰对最终控制鲁棒性无用。

**为什么提高 relation 权重不保证聚类更好**

两版目标分别为 `L_pred + 0.01 L_pos + 0.01 L_rel` 和 `L_pred + 0.01 L_pos + 0.02 L_rel`。新版没有同步增加跨 motion 的拉近约束；现有 L_pos 只约束同 motion 的局部 ±5 步平滑。response scale 从 1 降至 0.75 也增大了相同正响应距离的目标值，例如 D=1 时目标从 1 变为 1.1429。

如果 relation 目标包含运动状态差异，提高它的权重就可能更强地保留这些差异。预测损失也会反传到 encoder，并允许 latent 携带有助于预测的当前运动/历史信息。因此局部时间平滑、较低预测误差和跨 motion 的环境聚类不是同一个目标。现有证据没有测量各损失对 encoder 的梯度范数或冲突，不能把具体损失系数直接解释成实际梯度贡献。

**已经能够判断的用途：同轮次、匹配训练证据**

已有从头训练的 Short50 对照：仅移除长期 encoder/token，保留公共 predictor/短期层的初始权重；数据、DR、seed、采样、归一化、全局 batch 和 LR 日程匹配。总参数量随移除长期模块减少，不是等总参数量对照。以下两者均为 update 7,500 / 30,000 optimizer steps，评估同一组 2,048 个固定窗口、489 个验证 world。

| 分组 | 窗口数 | Memory350 相对 Short50 的五步 NMSE 变化 | 按 world 配对 bootstrap 95% 区间 |
|---|---:|---:|---:|
| 长期已满 30 段 | 707 | -27.31% | [-36.15%, -21.85%] |
| 短期不足 50 步、已有长期 | 931 | -18.15% | [-28.98%, -5.74%] |
| 长期为空 | 297 | +105.29% | [+49.93%, +206.11%] |
| 全部窗口 | 2,048 | +6.27% | [-2.40%, +16.87%] |

这些分组有重叠，区间不包含训练种子之间的不确定性，整体差异的区间包含零。长期为空时 memory token 本来就被 mask；退化不能解释为某段旧历史直接干扰，较可能涉及训练中长期输入几乎总可用所形成的依赖。原版 update 22,700 日志中的训练 batch 长期可用比例与在线长期填满比例均为 100%。

来源：[匹配训练设计](../../docs/short50_matched_control.md)、[update 7500 原始预测对照](../limb_context_20260910_short50/comparison/update_007500/result.json)。这比在同一训练好的模型上突然关掉 memory 更能判断“新增模块是否有收益”。

本次进一步加载同一对 update 7,500 checkpoint，用 FP32 在上述完全相同的已保存窗口编码 latent。checkpoint 和四份验证数据的 SHA256 与原配对预测实验一致。长记忆填满的 707 个窗口中，同 world 跨 motion 共 416 对、涉及 117 个 world：

| 指标 | Short50 | Memory350 |
|---|---:|---:|
| 同 DR，不同 motion 的单位 latent RMS 距离 | 1.0663 | 0.8502 |
| 不同 world、未匹配 motion 的单位 latent RMS 距离 | 1.2531 | 1.2743 |
| 上述同环境 / 不同环境距离比 | 0.8509 | 0.6672 |

长期可用的全部分组也同向：同环境距离 1.1012→0.9174。这个匹配检查支持长期模块可以减小新版中的跨 motion 漂移，不能直接把整体退化归咎于它。它是启动验证集上的探索性检查，不是前面的 3,200 步在线 probe：不同 world 组没有匹配 phase/状态/动作，长期可能重叠，不能直接与 v12 在线表格的 0.319/0.669 比值横向比较。短期和长期都满的子集只有 16 个跨 motion 对、13 个 world，不作为主要结论。

原始结果：[matched_training_latent_audit.json](matched_training_latent_audit.json)；latent：[matched_training_latents_u7500.npz](matched_training_latents_u7500.npz)；复现脚本：[audit_matched_training_latents.py](audit_matched_training_latents.py)。

**当前不能判定“有用”的改动**

取消 nominal A、relation 权重翻倍、减小 response scale、放宽历史门槛，以及更大的 motion/DR 范围，都还没有各自独立的匹配消融。不能把这些改动的共同结果当作各自有效或无效的证据。更大的数据和物理范围改变了任务要求，本身不是无效组件。

同一 Memory350 checkpoint 推理时把长期从 300 截到 200 步，已有结果显示误差增加约 2.54%，提示最后 100 步的边际收益较小；这不替代分别按长度从头训练。已有 PPO 对照也未证明整体 tracking 优势：uniform 负载下 cold body/joint 误差更高，anchor 和速度指标改善，存在指标取舍。预测收益不能自动等价为 policy 收益。来源：[记忆干预报告](../../docs/memory350_effect_analysis_20260910.md)、[PPO 完整结果](../../docs/memory350_ppo_results_20260910.md)。

**下一轮最有辨识力的实验**

1. 以新版设置为基线，先只增加同一固定物理 session、不同 motion、历史不重叠且信息充分的正样本项；保留当前 relation 项，其余条件固定。直接检验缺失跨 motion 约束的影响，避免仅加大现有 ±5 步正样本权重。
2. 独立改变关系目标：为各物理环境复用覆盖多个运动状态的共同 Q，匹配初始状态与物理动作序列；先固定 loss 系数，再检验目标定义是否改善。在这之前用单独的 relation 2→1 对照检查当前目标权重，不同时改变 scale。
3. 单独恢复 nominal A 样本，检验隐式 nominal 簇的贡献；报告 nominal 与 DR 的有效训练样本预算。它不等价于只把 nominal latent 固定到原点，也不保证 DR 跨 motion 稳定。推扰开关及空 memory 训练采样可分别针对临时外力、冷启动退化做后续对照。

各对照使用同一训练预算、验证物理参数、motion 和 query，同时看同 DR 跨 motion / 跨 DR 匹配 query 的距离比、跨 motion 的环境读出、分组预测 NMSE。候选表征再进入匹配 PPO；不以簇图好看或整体放大 latent 作为通过标准。
