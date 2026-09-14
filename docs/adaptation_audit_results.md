# DR / nominal 对照审计结果

## 新 teacher 的 nominal 退化定位（2026-09-07 06:30 UTC）

同一 nominal 测试环境中，新 DR teacher（oracle-wide-smalllr77，checkpoint400）
确实弱于原冻结 tracker 的 body/joint 指标，不能以失败少一次来宣称全面改善。
补测同一 teacher 的零 residual 初始化，发现特权前端本身有收益，退化发生在
DR 学习后的 residual 分支（包括其特权编码器/归一化），不是冻结 tracker 权重
被意外更新：初始到400的53个 tracker 参数及缓冲张量逐一比对，全部相同。

先在开发种子10001定位，再预先固定模型、独立测试98001/98002/98003：
每模型1008片段，相同起始帧、物理指纹、500步上限和v2隔离重置协议。

| nominal 测试策略 | Body error | Joint error | 失败/1008 |
| --- | ---: | ---: | ---: |
| 原冻结 tracker | 0.03079445 | 0.48399478 | 1 |
| 特权前端，零 residual 初始化 | 0.02788254 | 0.45737074 | 0 |
| DR 训练后 teacher400 | 0.03223682 | 0.51179765 | 0 |

teacher400 相对冻结 tracker 高4.68%/5.74%；相对自己的零 residual 初始化
高15.62%/11.90%。三个测试种子方向一致；按动作文件与测试种子聚类 bootstrap，
teacher400/冻结 tracker 的95%比值区间为body[1.01775,1.08156]、
joint[1.04336,1.07323]。仅一个训练种子，不声称跨训练重复性或失败率相等。

这仍不能解释 DR 学习退化的全部机制。一个已核对的覆盖差异是：DR 训练始终
包含1–3 kg负载，而 nominal 为零负载。冻结底座并未约束 residual 在 nominal
下归零，因此不提供行为非退化保证。

开发种子下，teacher400在DR中相对自身零residual初始化改善body11.77%、joint1.16%；
相对原始冻结DR tracker的15.6%/4.7%包含特权前端和学习修正的共同收益，不能全部
归因于物理latent。此前同架构物理输入的受控训练对照仍是独立证据。

结果目录：`runs/adaptation_goal/eval_v2/teacher_nominal_diagnosis_980xx/`，
比较报告`teacher400_vs_frozen.json`、`initial_vs_frozen.json`、
`teacher400_vs_initial.json`。这些是同nominal诊断，不是第一阶段DR追平nominal的验收。
新路线保持奖励不变：保留原nominal动作分支，仅在新修正分支使用特权，并用
nominal中心化物理码约束修正归零。完整训练已启动，尚无质量改善结论。

## 导出闭环与单种子失败计数敏感性（2026-09-07 05:43 UTC）

同一lowrank context94 checkpoint1500的CPU导出文件，已直接完成950xx
三个完整闭环。相对Python/GPU版本，10项均值最大比值1.00212，body/joint
1.001195/1.000581；失败12对10（分别2/5/5对1/4/5）。因此单步数值一致性
不能替代闭环可靠性，当前不声称部署执行方式的失败数非劣。
导出模型相对冻结DR仍为12对15失败，但相对固定nominal为12对2，目标未达。

最佳oracle300在开发种子做部位诊断时，误差均值与原评测相差不到约1%，
但失败从先前2变为4。SHA、物理世界、动作片段起点相同；保留两份记录，
不把先前2次当稳定保证，正在补无诊断复跑。右臂占与nominal的body位置/
朝向净差距约74%/88%，提示后续结构设计可关注载荷影响，不改变奖励权重。

## Context student的绝对基线差距（2026-09-07 05:29 UTC）

为已经固定SHA并完成950xx确认的lowrank context94 checkpoint1500补齐了
完全相同95001/95002/95003的冻结DR与固定nominal参考，没有重选student。
336片段×3、原v2协议；student/冻结DR物理世界指纹逐项匹配。
student/冻结DR body/joint比值0.917775/0.978894，95%区间
[0.900525,0.934723]/[0.970696,0.987086]，失败10对15，新增3、挽回8。
student/固定nominal body/joint比值1.313325/1.256845，失败10对2。
因此已实现的latent替换并不等于最终第二阶段完成，teacher水平仍是瓶颈。
报告：`runs/adaptation_goal/eval_v2/context_replacement_950xx/student_vs_frozen_dr.json`
以及同目录`student_vs_fixed_nominal.json`。950xx不是最终600xx确认种子。

## 64维负结果与最佳teacher标签依赖复核（2026-09-07 05:24 UTC）

逐关节64维静态物理编码、共享rank16+条件rank64的输入开关对照完成。
固定训练seed77/checkpoint500、相同118 actor/17 critic初始张量、仅输入
normal/zero不同，新97001/97002/97003评测：真实输入body/joint比值
1.026536/1.009772，失败18对17，新增8、挽回7。所有10项均值均未改善。
因此该架构的额外物理维度没有显示收益，不能只报告7维方案的正结果。
报告：`runs/adaptation_goal/eval_v2/input_control_970xx/actuator64_shared16_seed77/comparison.json`。
两臂已停止后续训练，保留500及已有后续检查点，未删除文件。

当前最佳原奖励oracle_large checkpoint300，在开发seed10001做全路径
高度/接触替换：高度单独替换body/joint比值1.00290/1.00175，接触单独
替换1.00478/1.00234，合并替换1.00497/1.00322；失败分别2、2、3，对照2。
同时替换了oracle特征入口和state_physics latent入口，非仅遮挡一条路。
这是训练后依赖诊断，不是重新训练的因果消融；不能据此否认标签在训练中的
作用。此前两组7维输入因果对照从训练起就没有真高度/接触，结论作用域不同。

## 7维物理输入收益在两组独立训练中复现（2026-09-07 05:00 UTC）

训练seed76、77的同低秩架构真实7参数/无信息常量输入对照均已完成，每组
固定500 checkpoint、新94001/94002/94003，共2016片段。逐对102 actor/17
critic初始张量逐位相同，训练设置仅actor物理输入/输出目录不同；均无真高度、
接触、clean proprio或nominal prior，奖励保持原样。常量输入保留每个低秩
分量的学习能力，不是关闭residual。每对的真实参数组body/joint均有改善：

| 训练seed | Body比值（真实/常量） | Joint L2比值 | 失败（真实/常量） |
| --- | ---: | ---: | ---: |
| 76 | 0.97198 | 0.95555 | 14 / 18 |
| 77 | 0.97139 | 0.95052 | 10 / 17 |
| 合并均值 | 0.97169 | 0.95304 | 24 / 35 |

合并body/joint改善2.83%/4.70%。交叉配对bootstrap同时重采样训练对、共享
评测世界种子和共享动作簇，95%区间分别[0.95801,0.98348]、
[0.94333,0.96184]。独立训练单位仍只有2对，不能把2016片段当2016次独立
训练，区间不能被解释为精确的训练总体泛化保证。Joint velocity比值1.00028、
anchor angular velocity比值1.00331，并非所有指标严格改善。失败率差区间
[-1.68651,+0.29762]百分点仍包含零，也存在新增失败片段。

这比“新方案超过普通DR residual”更直接地支持了DR物理输入在该架构内
确实有帮助，但不意味着已经追平固定nominal目标或完成最终第二阶段。
结果：`eval_v2/input_control_940xx/compact_two_training_pairs_summary.json`；
逐对文件在`compact76/comparison.json`、`compact77/comparison.json`。

## 同架构输入消融与 context 替换复核（2026-09-07 04:36 UTC）

FiLM 输入开关对照的固定500 checkpoint、新94001/94002/94003评测完成。
两组初始86 actor/27 critic张量逐位相同，critic两组均能读真实物理，训练
设置仅actor真实物理/零输入不同；零输入组仍有可学习的状态条件residual。
真实参数/零输入的body比值0.99342（改善0.66%，95%区间[0.98520,1.00224]）；
joint比值0.99218（改善0.78%，区间[0.98686,0.99855]）。失败14对16/1008，
分种子2/2、4/7、8/7，新增5/恢复7；失败率差区间[-1.28968,+1.19048]百分点。
这给出同架构下物理输入改善joint的直接证据，但只有一个训练种子，body与
失败风险仍有不确定性，不能声称整体稳定胜出或把此前3.5%全归因于信息。
文件：`eval_v2/input_control_940xx/full77/comparison.json`。

纯7维teacher500的context替换也已用新95001/95002/95003复核；student为
seed94的固定1500 checkpoint。102项非encoder的控制器/预处理张量与teacher
逐位相同，actor评测输入仅可部署组。Student/teacher的body/joint比值
0.99704/0.99613，95%区间[0.99075,1.00205]/[0.99019,0.99976]；10项均值
最大比值1.00245（anchor linear velocity）。失败10对14/1008，分种子1/4、
4/5、5/5，新增2/恢复6；失败率差区间[-1.38889,+0.39683]百分点。
这是已有teacher的物理输入替换证据，不是固定nominal最终目标完成：该teacher
本身仍不达标，失败风险区间也没有证明严格零增量。
文件：`eval_v2/context_replacement_950xx/comparison.json`。
Student SHA `461017433cd4696066f69f622eb87a86d3eb1ec105d37a1de8b46eab6544306a`；
CPU导出8个独立进程最大动作差5.72205e-6，artifact SHA
`d104b5bc9f4b371922c50ca9307e700a0a9b58a887ab819abfa7837499bd5559`。

7维低秩的normal/constant同架构控制，独立训练seed76和77仍在等预定500
checkpoint；新940xx评测路径已预先记录，不能依据中间checkpoint更换预算。

## DR residual 新种子复核完成（2026-09-07 04:02 UTC）

冻结上一节同一对500 checkpoint，用新93001/93002/93003评测，共1008片段。
低秩物理方案/普通DR residual：body比值0.98941（改善1.06%，95%区间
[0.97732,1.00015]，仍略跨1），joint比值0.96496（改善3.50%，区间
[0.95748,0.97273]）。三个种子的body/joint点估计都改善；只有一组训练
种子，不代表训练可重复性或同架构输入的因果收益。Body angular velocity
比值1.00203，不能声称所有tracking指标都变好。

失败总数9对9/1008，分种子candidate/reference为2/1、2/3、5/5；共5个新增、
5个恢复片段。失败率差95%区间[-0.89286,+0.89286]个百分点。不能把总数持平
写成“没有新增失败”或零风险保证。这是均值收益的中间证据，不是完整胜出。

结果：`eval_v2/dr_residual_confirmation_930xx/comparison.json`，含冻结计划、
同物理指纹与reset审计、逐种子失败。普通DR residual SHA
`543c43348d0d1e9c9a00ca18ae46c0a10c4bafcd6c912f62a6f479be01af8ee4`。

来源限制：这个普通DR residual训练早于运行时奖励签名机制，汇总初次因此
被拦。已单独核对保存的reward/decimation/sim配置与签名candidate完全相同，
奖励参数全为原始值、reward_changes为空、相同奖励源码hash、同原始tracker，
没有适配初始化或蒸馏teacher。其旧配置支持这一对照，但不能补造历史运行时
签名。报告保留这个限制；旧baseline仍不允许作为新fixed-reward训练初始化，
没有修改checkpoint或放宽全局训练奖励保护。六项评测未重跑或换种子。

新增低秩专属消融`fixed_reward_constant_lowrank_seed77`：仅将7维参数替换成
每个环境共用的全1常量。所有秩均可训练且不接收世界信息；若置零会结构性
关闭低秩residual，不是合理的容量对照。该组与真实参数组初始actor/critic
逐张量相同、所有训练配置相同，仅输入模式/输出目录不同，计划固定500比较。

## 新增中间目标：与 DR-trained residual 比较（2026-09-07 03:52 UTC）

用户明确：最终仍以固定 nominal 为目标，但优于同样在 DR 中训练的普通
residual 本身也是有意义的进展。这与“优于冻结 tracker”是不同的比较。
还需区分“物理输入+新架构”的联合收益与物理输入本身的因果收益。

训练 seed77、评测 seed10001、同 DR 物理指纹、同 checkpoint500，原奖励、
1024环境、scale1、固定 actor LR2e-5、critic LR5e-4、warmup50、std.1、
entropy.0002、uniform/reference 相同；均无 nominal prior、无真高度/接触。
普通 DR residual 对比纯7维物理参数低秩适配器：

| 指标 | 普通 DR-trained residual | 7维物理低秩方案 | 后者/前者 |
| --- | ---: | ---: | ---: |
| Body position (m) | 0.04156518 | 0.04095393 | 0.98529 |
| Joint position L2 | 0.63686089 | 0.61486494 | 0.96546 |
| 失败/336片段 | 4 | 3 | — |

Body/joint 分别改善1.47%/3.45%；配对动作簇 bootstrap95%区间为
[0.97091,0.99740]、[0.95620,0.97337]。Body angular velocity 比值1.00035，
并非10项严格都改善。这里只是一组训练/评测种子，架构与输入同时不同，
不能把这个结果写成“已经单独证明物理信息的贡献”或最终目标达成。
计划总训练长度与存盘频率不同，但这里冻结同500 checkpoint且LR恒定。

来源：`eval_v2/fixed_reward/dr_residual_vs_lowrank77_500_development.json`。
新93001/93002/93003复核已冻结同一对checkpoint SHA并启动，计划及输出在
`eval_v2/dr_residual_confirmation_930xx/`；这只是新评测世界/起点，不是新的
独立训练种子。最终第二阶段600xx仍未使用。

更严格的同架构输入消融也已启动：`fixed_reward_physics_critic_prior_seed77`
对`fixed_reward_zero_actor_physics_seed77`，两者真实初始86 actor/27 critic
张量逐位一致，配置仅actor_physics_input与输出目录不同。两者critic都能
读真实物理；zero组在encoder和normalizer之前将actor物理输入置零，仍保留
可训练全局latent及状态条件residual，不是把控制器结构性关闭。
这是另一种 full495→64 FiLM 架构上的输入贡献实验，不能替代上述7维低秩
架构的专属消融。原奖励SHA保持不变，完整273项测试通过。

## 新的严格配对对照已完成（2026-09-07 03:02 UTC）

训练种子76、77、78，各自 nominal/DR 两分支；真实更新前 actor62项、
critic17项全部张量逐位相同，agent 配置相同。唯一处理差异是后续训练 DR
开关；原始奖励、观测、网络、优化设置与1000更新预算相同。预先指定最终
checkpoint，不根据测试表现选模型。每个模型测试 nominal 与 DR，各用
92001/92002/92003，共36项 v2 评测，全部成功并通过同物理指纹/reset审计。

结果文件：`runs/adaptation_goal/eval_v2/matched_controls_920xx/three_training_pairs_summary.json`。
置信区间同时重采样独立训练配对、配对内评测种子及共享动作簇，10000次。
每个测试条件共3024片段，但独立训练单位只有3对，不能夸大精度。

同 nominal 测试下，DR训练 / nominal训练的平均误差比：

| 指标 | 比值 | 配对分层 bootstrap 95% 区间 |
| --- | ---: | ---: |
| Body position | 1.13493 | [1.10418, 1.17202] |
| Joint position L2 | 1.09196 | [1.07552, 1.10928] |
| Anchor position | 1.19510 | [1.14876, 1.24416] |
| Anchor rotation | 1.10463 | [1.07572, 1.13618] |
| Body rotation | 1.21870 | [1.18319, 1.26230] |
| Joint velocity | 1.02231 | [1.01637, 1.02793] |
| Anchor linear velocity | 1.07773 | [1.06710, 1.08856] |
| Anchor angular velocity | 1.04109 | [1.02746, 1.05428] |
| Body linear velocity | 1.06030 | [1.05191, 1.06955] |
| Body angular velocity | 1.04269 | [1.03195, 1.05337] |

三对各自10项均为 nominal训练更好。各对 body/joint比值：seed76
1.14998/1.09979，seed77 1.11920/1.09253，seed78 1.13555/1.08357。
汇总 body 均值0.0343375对0.0389706m，joint L2 0.566115对0.618174。
失败为 nominal训练7/3024、DR训练14/3024；DR减nominal失败率差
95%区间[-0.1323,+0.7275]个百分点，包含零，不能宣称已证明失败风险差异。

同 DR 测试下，DR训练 body改善10.09%（比值0.89912，区间
[0.87666,0.92140]），joint无明确改善（1.00232，[0.99245,1.01201]）；
失败39对65/3024，差区间[-2.1825,+0.1653]个百分点，也包含零。

范围限制：共享 tracker 本身经过DR预训练，结论仅针对后续训练开关，
不是终生无DR的从头 nominal-only 对照。DR右手负载1–3kg，而nominal为0，
存在训练支持范围差异；未改变这个范围。新对照不是最优策略的证明，
也没有把它们替换掉原来固定SHA的强nominal目标来降低验收标准。

评测等待脚本曾把“controller正常退出但最终文件尚未超过15秒”误报为
训练失败；两个最终模型完整存在。修复只涉及等待逻辑，有回归测试，
之后恢复剩余评测；未重训、换模型或使用最终第二阶段600xx种子。

以下为此前模型的历史审计，reward-shaped teacher/student 仍不符合当前约束。

固定奖励约束更新（2026-09-07 01:23 UTC）：下述旧 teacher/student 训练包含
奖励调整，不能证明仅靠输入/架构达标。nominal/DR 控制未改奖励，对照仍有效。
高度/接触消融只说明旧模型的依赖性，不排除奖励调整的贡献。

2026-09-07 00:57 UTC。54 项评测全部成功完成；这不是第二阶段完成报告。
冻结模型、源码、任务列表见
`runs/adaptation_goal/eval_v2/audit_910xx/freeze.json`；全部统计、置信区间与
逐片段失败明细见同目录 `summary.json`。所有同环境对照的物理参数指纹一致。
每个条件均为 42 动作 × 8 起点 × 3 评测种子 = 1008 片段，v2 隔离 reset。

## 仅后续训练是否使用 DR：同 nominal 测试

旧 seed42 配对训练：同架构、同冻结 tracker、相同保存参数（仅 physics/
output_dir 不同），共同 update1750，其他训练超参数及观测均相同。

| 指标 | DR 训练 / nominal 训练，在 nominal 测试 |
| --- | ---: |
| Body position | 1.28257 |
| Joint position L2 | 1.34924 |
| Anchor position | 1.20290 |
| Anchor rotation | 1.13454 |
| Body rotation | 1.50187 |
| Joint velocity | 1.08441 |
| Anchor linear velocity | 1.12138 |
| Anchor angular velocity | 1.09393 |
| Body linear velocity | 1.11072 |
| Body angular velocity | 1.12097 |

全部 10 项平均误差 nominal 训练更好；失败为 DR 训练 7/1008、nominal
训练 4/1008。只能对这组训练配置下的模型作此结论：3 个评测种子不能冒充
3 次独立训练，旧对照本身也不是目前最强模型。其 nominal 分支与冻结
tracker 相比 body/joint 仍差 5.3%/7.7%，因此旧对照不是安全改进的成功方案。
新的 seed76/77/78 配对训练已安排；seed76 已实际核对全部更新前 actor/
critic 张量 bitwise 相同，预算、架构、观测、奖励、学习率相同。

这比较的是同一个已有 DR 预训练 tracker 之上的后续训练，不是从头训练。
原 DR 定义包含 1–3 kg 右手负载，而 nominal 无负载；nominal 不是该负载
维度训练范围的内部点。结果不能单独证明纯随机化的代价，更不能忽略这种
训练/测试分布关系。没有为了改善结果偷偷将 DR 改为含零负载。

## 高度 / 接触特权消融

两条 actor 路径（tracking 前端和特权 latent）均用现有可部署估计替代
对应真值。不是只改前端、不是将特权置零。物理参数和其他真值保持不变。

DR 下同时去掉高度/接触真值后：body +0.611%，joint +0.343%，anchor
position +0.523%；全部 10 项平均误差变化均不到 1%。失败 9/1008，对照
teacher 8/1008。单独去掉高度为 8 次，单独去掉接触为 7 次。
这些结果不支持“teacher 的主要平均误差收益来自这两个标签”。这是测试时
依赖性消融，不等价于去掉它们重新训练的因果效果，也没有排除干净本体状态、
参考特征、物理参数和奖励/网络优化带来的收益。

## 特权 teacher 复核与失败审计

Teacher DR 相对固定 nominal 基线：body 0.93005、joint 0.98762、anchor
position 0.94872；全部 10 项平均误差仍低于 nominal 基线。

| 模型 | nominal 测试失败数 / 1008 | DR 测试失败数 / 1008 |
| --- | ---: | ---: |
| 冻结 tracker | 2 | 9 |
| 固定 nominal 残差基线 | 1 | 11 |
| 旧配对 nominal1750 | 4 | 16 |
| 旧配对 DR1750 | 7 | 6 |
| Clean teacher500 | 8 | 8 |
| Rotation350 student | 2 | 4 |

Teacher 在 DR 下总失败比同环境冻结 tracker 少 1，但有 7 个新失败片段、
修复 8 个旧失败片段；不能宣称逐片段不退化。在 nominal 下 teacher 的
8 次失败也明显多于冻结 tracker 的 2 次。许多 teacher 失败涉及原数据中的
fallAndGetUp 动作；保留这些动作和原终止阈值，不能删掉难例。

基础动作加残差的网络形式本身不保证优化后性能单调提高；此前仅允许失败差
+1pp 的验收保护不满足用户的新要求。当前 teacher 仍只是平均 tracking
达标，不是完整可靠性达标。正在测试训练用真实失败惩罚，排除正常 timeout，
不改变评测终止、物理或传感器。

## 第二阶段

Rotation350 的额外三种子审计：最差项 anchor position 高 8.60%，body
rotation 高 6.38%；失败 4/1008 对 nominal 1/1008。未达标。
GPU2 正在做新的动作+特征 DAgger：从三种子已验证的 Linear1050 student
精确 warm start，冻结 context / 原控制器，仅学习零初始化的可部署特征
修正网络。初始动作误差 0.0，短训只有新模块更新；CPU 导出经过 8 个独立
进程校验，最大动作差 1.067e-5，无特权推理字段。实验结果尚待验证。
最终第二阶段 60001/60002/60003 确认种子仍未使用。
# 固定奖励约束说明（2026-09-07 01:23 UTC）

下述旧 teacher/student 的训练包含奖励调整，其收益不能证明“原始奖励不变、
仅改变输入或架构”的目标已达成。配对 nominal/DR 控制未改奖励，仍可用于
相应对照问题。高度/接触消融只说明旧模型的依赖性，不排除奖励调整的贡献。
