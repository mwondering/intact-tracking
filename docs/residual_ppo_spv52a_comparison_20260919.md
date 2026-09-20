# Residual PPO 与 SPV5-2A 训练差异审计

审计日期：2026-09-19。重点对象是当前 `scripts/run_144000_residual.py` 启动的 `memory350_proprio_native_policy_train`，同时检查通用 `residual_policy_train` 的共享实现。

**结论：存在可能影响训练效果的实质差异，但目前没有证据证明当前 native residual 存在一个会普遍、显著拖垮训练的环境构建错误。最值得优先验证的是探索强度与学习率调度，其次是失败回放和优化目标。确实发现了一处原 critic 权重复用的输入兼容性问题，但当前从零训练的 critic 不属于该问题的直接受害范围。**

本次只新增审计文档；没有修改训练实现、调整参数或重启训练。审查过程中其他工作在更新 motion 边界代码，因此下文区分磁盘实现与运行记录。

**比较基准**

- 原仓库：`/data_zcy/wxy/SP_Tracking`，提交 `2fd6aa20efc346f0b9dcf552a0a27b3c9594d592`，含现有工作区改动。
- 当前 tracker：`2026-09-10_16-19-06_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_4gpu_8192env_motion_data_correct/checkpoint_144000.pt`。训练目录的 `git/SP_Tracking.diff` 记录同一原仓库提交。
- 当前 residual：`runs/144000_exp/stage2_proprio122_history5_tracker_action/ppo_8gpu8192`，五帧 latent 与当前冻结 tracker action 输入 actor/critic。
- 以 checkpoint 内保存的配置和实际 `run_config.json/config.yaml` 为主要比较依据；原仓库今天的默认 YAML 另作对照。二者不能混用。

**已核实的环境一致性**

把同一份 tracker task 配置分别送入两个仓库的 builder，观测配置、奖励项及有效权重、终止项、动作配置、仿真配置和 metrics 配置一致。冻结 SPV5/5-1/5-2 actor 的函数实现经 AST 对比一致。

控制步长为 0.02 s（物理步长 0.005 s，decimation=4），episode 上限均为 500 步。延迟上限为 2 个物理子步，alpha 范围为 [0.8,1.0]；mean action 历史、sampled action rate 的约定相同。当前 native 入口保留原始 reset 噪声和全部原始终止项，包括 `ee_body_pos`。

进一步分别构建并编译机器人 MuJoCo model：质量、惯量、关节范围、armature、PD gain/bias、力矩限制及初始姿态一致。腕部 visual/collision geom 的数组顺序不同；按 body、名称、类型和碰撞属性匹配后，几何及接触参数也一致。未把数组编号差异误判为物理差异。

本地 `FlatTerrainHeightOffset` 在平地中为空操作；原版对应事件在没有 terrain plan 时也为空操作，不是当前平地实验的有效动力学差异。

**PPO 和策略训练差异**

| 项目 | 原 SPV5-2A / 144000 tracker | 当前 native residual | 判断 |
|---|---|---|---|
| 优化对象 | 完整 actor；继续训练 height/contact estimator、reference encoder | 冻结 tracker 和 context encoder，仅训练 residual、critic、探索分布 | 有意限制可优化空间，不等价于继续训练原策略 |
| actor 结构 | [2048,2048,1024,1024,512,256,128] | residual [512,256,128]，额外读 320D latent 和 29D tracker action | 容量和可修正范围不同 |
| 动作均值 | 原 actor 输出 | tracker + 0.25 × tanh(residual MLP) | 单关节均值补偿有界；探索噪声不受该 tanh 限制 |
| actor LR | 初始配置 1e-3，按 KL 自适应；144000 保存值约 2.25e-5 | 固定 1e-4 | 比保存时实际 LR 约大 4.44 倍；不能仅按初始默认值说“降了十倍” |
| entropy coefficient | 0.005 | 0.0002 | 降低 25 倍 |
| 初始 std | 从头训练默认 1.0；144000 各关节已学习，均值 0.23645 | 所有关节重新设为 0.25 | 重新初始化，未继承按关节学习的探索尺度 |
| critic | 已训练 critic | 相同主干尺寸，但权重和归一化从零开始 | 初期 value/advantage 质量更差，不能直接当作 PPO resume |
| KL 控制 | adaptive 分支计算 KL 并调 LR | fixed 分支不计算 KL，也未记录 KL/clip fraction | 缺少判断更新是否过大或过小的关键证据 |
| PPO 核心参数 | rollout=24，epochs=5，minibatches=4，gamma=.99，lambda=.95，clip=.2 | 相同 | 未发现这些参数被错误更改 |
| 并行规模 | tracker 目录记录 4×8192 | 8×8192 | 单次更新采样量翻倍；不能只按 update 数比较收敛 |
| 归一化 | actor 在线更新，critic 使用 DecayVecNorm | tracker normalizer 冻结；critic 在线更新且跨 rank 同步 | 冻结符合 residual 设计；当前同步实现未见梯度缩放错误 |

代码依据：[训练参数覆盖](../src/intact_tracking/cli/memory350_policy_train.py)、[PPO 更新与动作组成](../src/intact_tracking/residual_policy.py)、[scratch 初始化](../src/intact_tracking/limb_context_policy.py)、[分布式统计](../src/intact_tracking/limb_context_distributed.py)。

**优先验证：探索快速衰减与固定 LR**

实际日志显示，u1901–2000 的平均 std 已为 0.03623；u3271–3370 为 0.03355。后一个窗口中 residual RMS 为 0.06446，达到 95% tanh 上界的比例仅约 0.0278%。

这说明探索明显减弱，而均值残差并未普遍饱和。低 entropy、较强 sampled-action 平滑惩罚会给收缩 std 很强的激励；std 缩小时相同均值改变量可能产生更大的 KL。这里是有数据支持的机制风险，尚无 KL 日志或对照训练证明它是效果下降的根因。

建议先记录 KL、clip fraction、逐关节 std，再在同一数据与评测协议下比较 adaptive LR / fixed LR，以及更高 entropy 的小规模消融。不要把恢复原始从头训练 std=1.0 当作默认修复。

**环境分布与困难样本差异**

| 项目 | 原版 | 当前 native residual | 可能影响 |
|---|---|---|---|
| 失败 rewind | 开启；失败时约 1/3 概率 rewind | 显式关闭 | 难片段得到的针对性重试减少，可能影响恢复与困难动作 |
| 其余 motion 采样 | uniform/adaptive 分支，checkpoint temperature=.25 | 同样从 update 0 开始 adaptive，temperature=.25 | 并未关闭全部自适应采样 |
| Kp/Kd | 每次 env reset 重新采样 | 覆盖 motor.reset，整次训练对每个 world 固定 | 有利于跨 episode context 一致；减少控制参数的动态变化覆盖 |
| nominal 比例 | 原生随机 DR | 每 rank ceil(10%) 强制 nominal，其余原生 DR | 有意改变训练分布 |
| nominal delay / alpha / push | 随 DR | 0 / 1 / 禁用 | nominal 子集更容易 |
| DR delay / alpha | reset 时重采样 | 仍然 reset 时重采样 | 长期 context 跨 reset 保留，旧记忆可能混合不同控制器扰动 |
| 额外 limb payload | 无 | 当前 native 无 | 不能把旧 payload 实验的负载当成当前配置 |

代码依据：[native 环境](../src/intact_tracking/memory350_native_policy.py)、[采样配置](../src/intact_tracking/limb_context_sampling.py)、原仓库 `mdp/randomizations.py:416`。

这些差异主要改变学习目标、难度和泛化范围，并非自动意味着训练更差。关闭 rewind 值得做单独对照；固定 Kp/Kd 若改回逐 episode 采样，还必须同步处理长期记忆和 stage-1 训练协议。

**确定的兼容性差异：critic 的动作历史顺序**

原仓库 `SpTrackingJointPositionAction.get_recent_action_obs()` 会把动作历史从物理 target 顺序重排到 `robot.joint_name_order`；当前本地实现没有这一步。该变更在原仓库提交 `d69eb56`（2026-09-08）已存在，144000 训练保存的 Git 快照没有撤销它。

使用 checkpoint 的机器人配置进行 CPU 复现，29 个通道中有 27 个位置不同。例如 critic 期待的第二个通道是 right_hip_pitch，当前实际是 left_hip_roll。原 critic 的 `policy.prev_actions` 使用 8×29 维历史。

必须限定影响范围：

- 冻结 actor 的 `estimator_history.last_action` 调用的是 MJLab 的 `last_action`，两个仓库该实现相同。不能把这处差异说成冻结 actor 的 27 个 action 通道错位。
- 当前 native critic 从零训练，固定排列仍保留全部信息；不能据此认定当前训练被严重破坏。
- 通用 `residual_policy_train` 会直接加载 source critic 权重及 normalizer。这种用法若接入原版 144000 critic，就会产生真实的输入语义不兼容，应优先修复或加入显式转换。
- 对已经在本地顺序下训练的 residual checkpoint，不应直接改观测顺序后无转换续训。

代码依据：[本地动作历史](../src/intact_tracking/environment/mdp/actions.py)、[critic 加载](../src/intact_tracking/residual_policy.py)、[通用入口](../src/intact_tracking/cli/residual_policy_train.py)。当前 original/scratch actor 和 critic 的观察组已逐项核对。

**Motion 边界：共享风险与审查期间的修复**

原版 SPV5-2A 没有把自然 motion resample 纳入 PPO 的 done mask，旧 residual 继承了这一行为。它会允许新 motion 的 value/advantage 进入旧 motion 的 GAE。CPU 复现中，下一段 reward=10 会给上一段带来 9.405 的未标准化 advantage；rollout 最后一个 transition 会从下一 motion 的 value=7 bootstrap 出 6.93。

这不是 residual 相比 spv5-2a 独有的退化。是否以及多大程度影响当前训练，仍取决于自然切换频率，不能由该复现量化。

审查时磁盘上的 `TrackerActionPPO.process_env_step` 已加入 SPV5-3 式处理：切断跨 motion GAE，并使用当前 transition 的 value 做 timeout 式 bootstrap。它与 SPV5-4 的“完全取消边界 bootstrap”不同。审查读取的运行 `run_config.json` 尚无 `motion_boundary_contract`，因此不能把磁盘修复当作正在运行的旧进程已启用。这个改动并非本次审计写入。

代码依据：[TrackerActionPPO](../src/intact_tracking/memory350_tracker_action_policy.py)、[边界信号](../src/intact_tracking/memory350_policy_env.py)、原仓库 `rl/ppo.py:1278,1364,1387`。

**奖励目标与评测解释**

当前奖励与 144000 checkpoint 一致，但与原仓库未经启动参数覆盖的默认 YAML 不完全一致。以下是实际 builder 输出的有效权重：

| 奖励 | 原仓库默认 YAML | 144000 / 当前 residual |
|---|---:|---:|
| root_pos_tracking | 1.0 | 1.5 |
| action_rate_l2 | 0.2 | 0.3 |
| waist_action_rate_l2 | -0.5 | -5.0 |
| joint_pos_limits | 1.0 | 1.5 |
| joint_torque_limits | 0.01 | 0.015 |
| joint_vel_l2 | 0.00025 | 0.00035 |

另外，当前原仓库 YAML 的 adaptive temperature 是 .0625，而 source checkpoint 和 residual 都是 .25。这属于 checkpoint/启动配置差异，不能归咎于移植时丢失配置。当前全局 root position reward 存在，且 sliding_root_xy_reward=False。

现有 u2001 配对评测：关节角 L2 误差改善 2.43%，局部 body 位置误差变差 1.97%，全局 root 位置误差变差 1.89%，后者 95% 区间 [-0.17%,4.00%] 跨零；失败数 0→1/1024。它支持“不同指标发生权衡”，不支持“训练整体显著崩坏”的结论。

同样，nominal latent 靠近锚点不等于 residual 输出为零。已有 u2501 在线 probe 中 nominal residual RMS 与 DR 接近；当前 loss 没有 nominal-action 保持约束。若目标包含“保留 nominal tracker 能力”，需要明确加入评测约束或目标，而不是假定 encoder 的 nominal 锚定已经实现该目标。

**建议顺序与验证边界**

1. 确认后续运行是否启用并记录 motion boundary contract；与原 spv5-2a 比较时注明这已是算法变更。
2. 为当前 PPO 增加 KL、clip fraction 和逐关节 std 观测，再对 LR 调度与 entropy 做对照。
3. 恢复 rewind 做单独消融，按相同环境交互量比较，避免把并行规模差异混入结论。
4. 若要复用 source critic，先解决动作历史排列兼容性；当前 scratch checkpoint 应保留其输入契约。
5. 用相同 warm history / DR / motions 比较正确、置零、打乱 latent 的闭环性能，并单独报告 nominal 和 DR 指标。现有动作敏感度不等于 latent 带来控制收益。

本次完成：源代码语义对比、两套 builder 的实际配置对比、两份 MuJoCo model 的物理参数对比、动作顺序与 GAE 的 CPU 复现、4 个相关测试模块（31 项）通过；另有 2 项 motion-boundary 定向测试通过，并读取已有配对评测和在线 probe。没有运行新的 GPU 性能评测或新的训练消融；“显著降低训练效果”的因果判断仍需上述对照。
