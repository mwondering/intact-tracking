# 共享环境池的四套独立 residual PPO：100 秒预热后固定归属

用户最后确定的初版：共享 DR 环境采样和交互循环，冻结 tracker 预热每个环境 100 秒，按全程 latent 中心到 nominal 的距离分为四类；随后**整个训练期间固定环境所属专家，不再更新归属**。每类一套完全独立且不输入 latent 的 residual actor/critic 和 PPO。第一阶段仅比较四专家系统与全部环境混合训练的单 residual，不蒸馏。

已实现 `intact_tracking.cli.fixed_five_ppo_train`：四专家与一个跨类别 baseline，共五个独立 PPO。完成 GPU mjwarp 的 100 秒预热、两轮联合训练与断点恢复后的第三轮更新；尚未进行收敛训练和收益评价。实现及验证记录见[五 PPO 预检](fixed_five_ppo_implementation_20260917.md)。现有独立 MoE16 入口不能直接作为该实验：它的网络参数独立，但仍在一个 PPO 实例中联合更新。

## 环境与初始分类

- 保留当前全部固定 DR 参数各自 50% nominal 的混合采样。四肢负载、躯干质量、COM、摩擦、armature、encoder bias 在一个物理环境生命周期内固定；正常 episode/motion reset 不重采样。
- 力脉冲、观测噪声保持当前设置。四专家采用同一个环境池，motion 采样不按专家类别过滤，各专家均可遇到不同 motion。
- 使用固定版本的 response10/u15000 encoder 和原 nominal 参考中心。不能在 PPO 期间追随另一条 encoder 训练任务更新权重，否则距离和类别的定义会变化。
- 每个环境由冻结 tracker 交互 100 秒，即 dt=0.02 时的 5,000 控制步。此阶段 residual 不执行，不计入 PPO 训练样本。
- 建议沿用每 2 秒一次的 latent 查询，仅纳入 short50、long30 均完整的历史。每次 latent 单位归一化；每环境取有效 latent 的算术平均作为初始中心，平均后不再归一化。
- 四档沿用现有内部边界 0.47875843875890006、0.9575168775178001、1.4362753162767001。最后一类接收所有超过第三个边界的有限距离；之前 1.915 是标定区间上端，不应将在线超出的环境丢弃。
- 100 秒初始分类的实际数量应重新统计，不能把之前 50 秒得到的 13.09%/8.11%/44.32%/34.47% 当作新结果。

## 训练期间固定专家归属

预热结束保存每环境的有效查询数、latent 中心、中心距离和 expert ID。专家 ID 随 checkpoint 保存；恢复训练不能重新预热并重新分组，从而静默改变专家任务。物理参数及其环境 ID 映射也需保存或严格复现。

整个 PPO 训练期间，episode 结束、motion 改变、力脉冲和 latent 波动均不改变 expert ID。每个专家保持相同的物理环境子集，但在该子集中持续遇到不同 motion。不存在 rollout 中途或 PPO 更新边界的专家交接。

若继续计算 EMA，仅用于诊断控制策略改变后的 latent 中心漂移，不参与动作选择、PPO 样本归属或网络输入。EMA 的更新频率、半衰期不会影响初版控制行为；在线重新分配及切换缓冲留待后续独立实验。

rollout 截断不当作物理终止。最后一个 next-state 必须由对应环境唯一归属专家的 critic 计算 bootstrap；GAE、更新和动作概率比均只涉及该专家自己的轨迹。

若以后真正重采样物理参数，需要明确结束旧环境身份、清理中心/历史，并重新校准；当前固定 DR 设置不发生此事。

## 四套 PPO 的独立性

每个专家独立拥有：

- residual actor、critic 及其可训练参数；actor 的 Gaussian 探索参数；
- optimizer 状态、KL 统计与自适应学习率状态；
- rollout storage、return/GAE、优势归一化与 mini-batch；
- 自身需更新的观测归一化统计。

共同部分仅为：冻结 tracker、用于预热分类及可选诊断的冻结 context encoder、环境采样和批量仿真、固定的 environment→expert 分配表，以及相同的奖励/动作/终止协议。expert ID 不作为网络特征输入；latent 只用于预热分类及可选诊断。

一个环境在每个控制步只由其固定所属专家产生动作。该 transition 只进入这个专家的 on-policy buffer，不能作为另外三个专家的 PPO 采样。共享环境采样不等于共享同一条 PPO 经验。

各专家只在自己的样本上训练，实际样本量取决于类别占比。本轮先保留用户指定的共享环境分布，不额外把四类重采样成等频；报告每专家累计 transition、有效 batch 大小和更新次数。空类别不能产生 NaN 或更新一个没有样本的 optimizer。

## 混合训练对照与评价

四专家系统和单 residual 基线采用相同总环境数、DR 分布、物理参数配对、motion 数据与采样规则、100 秒冻结 tracker 预热、每轮采样长度和后续总交互预算。单 residual 的 actor/critic 也不使用 latent，单网络结构与单个专家一致。

基线必须用自己的动作重新交互，不能直接拿四专家的轨迹做普通 PPO 更新。四专家的总可训练容量约为单专家的四倍，应单独报告；相同总交互预算下，混合基线每个网络获得的数据更多。

实现使用同一个向量仿真中的两组环境：前 N 个归四专家，后 N 个归 baseline，逐个复制固定物理参数及 encoder bias。baseline 覆盖全部四类；双方动作、motion/reset 和噪声独立演化，采样规则相同。每轮四专家合计与 baseline 各采集 N×T 条 transition。五个 PPO 的 actor、critic、optimizer、buffer 和归一化统计均独立，只有冻结 tracker 共享。

留出独立物理环境和评测 motion。测试环境也先由冻结 tracker 预热 100 秒，再固定选择专家进行后续评测；不重新训练专家。主要比较原 tracker、单 residual、固定归属四专家系统；报告总体 tracking/body/root/joint 误差、失败率以及按四个预热类别分组的结果。

额外记录：预热类占比、每专家样本量和 PPO/KL/critic 拟合指标；审计实际专家归属始终不变。若采集 EMA 诊断，可报告潜在类别漂移，但不能把它作为实际换类。先回答“这套共享环境池中的独立专家训练是否优于混合训练”，再决定动态路由或蒸馏。

## 已核查的现有接口

- `memory350_independent_moe_policy.py` 的旧实现有独立网络，但 `residual_independent_moe_train.py` 仍使用单个 PPO 流程，且为在线 KMeans16，不能只把专家数改成 4。
- `memory350_moe_training.py` 的旧预热默认 500 步，属于约 10 秒而非本方案的 100 秒；其中心按 PPO 更新迭代，并非逐环境 latent 中心 EMA。
- `memory350_policy_env.py` 可在统一批量仿真中记录实际动作后的历史，冻结 encoder 可用于维护每个环境中心。
- `TrackerActionMoEPPO.compute_returns` 已明确最后 next-state 的 tracker 动作须重新计算；新的四 PPO 调度需要保留这一语义，并由对应专家完成 bootstrap。

此前分类数据：[四档统计](../runs/limb_context_20260917_all_fixed_dr_half_nominal_50s/analysis4/README.md)。
