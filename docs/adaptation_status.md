# DR tracking 研究状态

## 最新：2026-09-07 06:30 UTC

第一、二阶段的最终目标均未完成；原奖励及固定nominal目标不变。
新teacher400在同nominal下弱于冻结tracker已在三个独立测试种子复现：
body/joint高4.68%/5.74%。零residual的特权前端初始化反而低9.46%/5.50%，
说明本次nominal退化发生在DR学习后的修正分支。详见`adaptation_audit_results.md`。

正在GPU2训练保留nominal动作分支、物理码归零时修正严格归零的teacher，
GPU1运行冻结旧teacher后加可训练修正分支的试验。后者200步尚未改善。
新架构的context替换、冻结权重与CPU导出检查已通过小规模测试，但不是性能验收。
当前任务与自动checkpoint评测继续，运行身份见`adaptation_live.md`。

## 补充：2026-09-07 05:24 UTC

新64维同架构输入对照是负结果：真实输入body/joint反而高2.65%/0.98%，
失败18对17（3个新测试种子，单训练种子）。已停止该方案，保留完整记录。
最佳原奖励teacher的全路径高度+接触估计替换，开发误差增加0.50%/0.32%，
失败2变3；只说明推理依赖，不等于训练因果消融。

正在对最佳oracle300 teacher做context+可部署特征蒸馏，而不仅是此前较弱
7维teacher。另测试更大残差范围/较小固定学习率，以及分离的nominal冻结
动作分支与特权残差输入路径。均保持原奖励、原真实电机限制和原DR范围。
最终两阶段目标仍未达成，不将开发集选择结果当作新的独立确认。

## 最新结论：2026-09-07 05:00 UTC

“DR物理输入有帮助”的中间目标已有更直接的证据：相同低秩架构、相同初始化/
预算、原奖励、无真高度/接触，在独立训练seed76、77及各3个新评测种子中，
真实7参数都优于无信息常量输入。合并body/joint改善2.83%/4.70%，失败24对35。
仍有新增失败，只有2组独立训练，不能声称所有指标/失败风险均严格占优。

另外，已有teacher的context替换在新3种子上基本保持tracking均值，失败10对14。
这是两项独立的中间进展；teacher与固定nominal仍有差距，最终两阶段目标未完成。
当前继续测试逐关节物理输入、共享+条件低秩适配，以及检查oracle分支的逐关节
residual幅度约束。奖励、物理参数范围与真实电机力矩限制均未改变。

## 最新复核：2026-09-07 04:36 UTC

同架构FiLM、仅actor物理输入开关的固定500对照完成：新3评测种子下，真实
物理输入使body/joint均值降低0.66%/0.78%；joint95%区间支持小幅改善，body
区间仍跨无改善。失败14对16，但有5个新增片段，仅一个训练种子，不能夸大。

第二阶段已有一个明确的中间结果：纯7维teacher的context student在新3种子
上，body/joint与teacher基本持平（比值0.99704/0.99613），10项均值最大差
约0.25%，失败10对14。控制器/预处理102项张量逐位相同；真实物理参数只被
因果、可部署历史观测估计的latent替代。仍有2新增失败、失败差区间包含零。
该teacher尚未追平固定nominal，所以不把此结果写成最终第二阶段达标。
完整审计见 adaptation_audit_results.md 顶部；最终600xx未用。

## 最新进展：2026-09-07 04:08 UTC

固定checkpoint的新3评测种子复核完成：纯7维物理低秩方案相比普通DR-trained
residual，body/joint均值改善1.06%/3.50%。Body区间略跨无改善，joint区间
不跨；失败9对9但有5新增/5恢复片段。仍是单训练种子且架构不同，不宣称
已经单独证明物理输入贡献。旧baseline缺运行时奖励签名，配置/源码核对结果
及限制均保留在报告中，未补签或允许它初始化新训练。

同低秩架构的真实7参数/无信息常量输入消融在seed77和新增独立seed76进行；
每对actor/critic初始张量逐位一致，固定checkpoint500比较。另有FiLM的
真实参数/零输入配对。原奖励和DR物理分布均保持不变。

第二阶段新增encoder-only迁移：在当前开发种子上，用历史观测估计latent
代替真实7参数，初始student body/joint仅高于该teacher0.38%/0.17%，10项
均值最大差0.38%；但失败4对3，仍未合格。控制器保留新teacher、旧prior未
带入，正式蒸馏继续。此teacher本身仍远于固定nominal目标，不能把上述
替换精度说成最终第二阶段完成。全部276项测试通过，最终600xx未使用。

## 最新研究口径：2026-09-07 03:52 UTC

最终 nominal 目标保持不变；新增独立中间结论：是否优于 DR-trained
residual，而不是仅优于冻结 tracker。原奖励、同500更新、同开发种子下，
纯7维物理低秩方案相比普通DR residual，body/joint改善1.47%/3.45%，失败
3对4/336。仅一组训练/评测种子，且架构不同；这是联合方案的初步正向证据，
不能单独归因于物理参数。固定SHA的新930xx复核与同架构actor输入开关训练
正在进行；完整设计/结果见 adaptation_audit_results.md 顶部。

原奖励第一、二阶段最终目标仍未达成。扩大幅度FiLM分支在600后失败9次，
已停止并保留模型。新增非线性物理低秩teacher、physics critic对照继续；
新低秩context短跑的输入隔离、完整336评测、8独立CPU导出审计通过，但
tracking不合格。完整273项测试通过。没有使用最终600xx种子。

## 最新结论：2026-09-07 03:02 UTC

三组独立的原奖励、同初始化/配置/预算配对训练及36项评测全部完成。
同nominal测试，DR训练 body/joint平均误差分别高13.49%/9.20%；
三组各自10项均nominal训练更好。分层95%区间：body比值
[1.10418,1.17202]，joint[1.07552,1.10928]。失败14对7/3024，
失败差区间包含零。完整结果与范围限制见 adaptation_audit_results.md 顶部。

固定奖励第一、二阶段仍未达标。三条较大批量 teacher 与一个保守配置
teacher 在训练；7维 context encoder 预训练/直接参数估计/latent消融
也在运行。预训练teacher尚不合格，不把它当作第二阶段完成证明。
旧修改奖励模型继续排除；最终验收600xx未使用。

## 当前运行：2026-09-07 02:44 UTC

奖励及其来源已锁定；185 项回归测试通过。第一、二阶段在这个约束下均未
达标。当前对比纯物理输入的双线性与低秩适配，并新启用一个完全冻结的
nominal 控制器作为先验。该先验使用原奖励，有固定 SHA、训练时干净代码
快照及逐张量来源审计；不是此前的改奖励模型。新分支不输入特权高度或
接触标签，初始动作与原 nominal 控制器严格一致，但不因此宣称 DR 安全。

严格配对训练 seed76、77 均完成：同 nominal 测试中 DR 训练的 body/joint
误差分别高15.00%/9.98%、11.92%/9.25%，各自10项均更差，失败6对2、
5对3次/1008片段。第三个独立训练种子78仍在运行。上述是同一预训练
tracker 后续训练的 DR 开关对比，不能写成完全从头训练的 nominal-only 对比。
详细进程、停止原因、开发集成绩见 adaptation_live.md 顶部。

## 当前运行：2026-09-07 01:53 UTC

固定原始奖励已强制执行；180 项回归测试通过，改奖励 CLI 实际调用会在
创建环境之前被拒绝。GPU0/2 上四个全新 teacher 分支对比特权输入、latent
中间层调节与基础控制网络是否可训练；均从原始 tracker 开始，没有继承
旧改奖励模型。GPU1/3 的 nominal/DR 严格配对控制继续。

新 seed76 严格配对已完成：同 nominal 测试下 DR 训练 body/joint 误差高
15.00%/9.98%，10 项均更差，失败 6 对 2 次/1008。仍需另外两个独立训练种子。
固定奖励 teacher 目前尚未达标；最佳已评测 oracle200 的最大误差比约1.376，
失败2次对 nominal1次。第二阶段只完成小规模蒸馏/输入隔离/导出链路验证，
没有 tracking 达标证据。当前进度与全部 PID 见 adaptation_live.md 顶部。

## 最高优先级更新：2026-09-07 01:23 UTC

用户明确要求：**原始奖励不变，只通过输入、架构及蒸馏改进**。
已停止 GPU0 的失败惩罚 teacher 与 GPU2 的改奖励静态修正器，两者保留至
checkpoint300；GPU1/3 的原始奖励 nominal/DR 严格配对实验继续。
此前 teacher/student 的训练包含奖励调整，下文平均 tracking 收益仅为
历史结果，不能证明新的固定奖励目标已完成；第一、二阶段均需重新验证。
新实验从原始 tracker 开始，训练入口锁定奖励项、函数、权重、参数与 dt，
拒绝静默继承改奖励模型。失败率仅作为评测门槛，不增加失败奖励惩罚。

## 最新：2026-09-07 01:03 UTC

第二阶段仍未完成；按用户要求持续实验，直到完成或用户询问进度。
新的完整审计见 [对照审计结果](adaptation_audit_results.md)。54 项 v2
评测全部完成，同环境物理指纹一致，没有使用最终 student 600xx 种子。

- 同 nominal 测试、旧同预算 seed42 配对模型：DR 训练 body/joint 误差
  分别高 28.26%/34.92%，10 项均更差。补充 3 个独立训练种子的严格配对
  实验正在进行；真实更新前 actor/critic 全部张量已核对完全一致。
- 高度+接触的全部 actor 通路消融：DR 下 10 项平均误差变化均不到 1%，
  不支持它们是主要收益来源。没有排除其他特权或训练方法的作用。
- Teacher DR 的 10 项平均误差仍低于 nominal 目标，但失败 8/1008 对
  nominal 目标 1/1008，配对失败差 95% 区间为 [+0.099, +1.488] 个百分点。
  同 DR 冻结 tracker 失败 9/1008；同 nominal 环境 teacher 却有 8 次
  失败，而 tracker 为 2 次。可靠性未达标，不能把平均 tracking 当整体成功。
- Rotation350 student 新审计最差 anchor position 高 8.60%，失败 4/1008
  对 nominal 1/1008。零初始化特征蒸馏新分支也未改善，已于 800 轮停止。
- 当前 GPU0：teacher 真实失败惩罚优化；GPU1/3：严格配对 nominal/DR
  训练；GPU2：只有 3 个静态物理特权的小修正器及 context 替换验证。
  所有物理、噪声、终止阈值、动作数据范围保持原样。

下面为 2026-09-06 23:22 的历史快照；当前实验分配以 [实时记录](adaptation_live.md) 为准。

更新时间：2026-09-06 23:22 UTC。研究仍在运行，尚未完成第二阶段。

## 当前结论

第一阶段的平均 tracking 目标已有独立评测支持：一个共享的特权 DR
policy，在 42 个动作、3 个独立随机种子、1008 个固定起点片段上，10 项
平均误差均低于固定 nominal 基线，各项误差比的 95% 置信区间上界均小于
1.05。不能据此宣称失败风险相等：失败数为 5/1008，对照为 1/1008，
失败率差的 95% 区间为 [0, +1.091] 个百分点，略跨过 +1 个百分点的保护线。

第二阶段尚未达标。额外 3 个开发种子上，线性 PPO student 相比上一版
慢速 PPO 改善了全部 10 项平均误差，但最大误差比仍为 1.07378，高于 1.02
标准。线性奖励分支在单个开发种子上的最大误差比约为 1.065，尚无
最终确认。不能只凭 body position 接近 nominal 或 latent 能辨识部分
物理参数就宣布成功。

| 指标 | 第一阶段 teacher / nominal，独立确认 | Linear1050 student / nominal，3 个开发种子 |
| --- | ---: | ---: |
| Body position | 0.91981 | 1.01386 |
| Joint position L2 | 0.98776 | 1.03840 |
| Anchor position | 0.93688 | 1.05164 |
| Anchor rotation | 0.86645 | 0.99891 |
| Body rotation | 0.98068 | 1.07378 |
| Joint velocity | 0.94951 | 1.01886 |
| Anchor linear velocity | 0.96628 | 1.05069 |
| Anchor angular velocity | 0.89359 | 1.04789 |
| Body linear velocity | 0.95942 | 1.04645 |
| Body angular velocity | 0.93149 | 1.05197 |

两列使用不同评测种子，分别与同种子 nominal 配对；不能把两列当作直接
配对的 teacher–student 统计比较。第二阶段最终确认种子 60001/60002/60003
仍未使用。

## 固定模型与证据

- Nominal：`runs/residual_policy_no_latent_nominal_v13_run2/checkpoint_1000.pt`。
  SHA256：`800da8c40016bba3263e685e9694e51e82b83a5e1c3df3e2e80543bd47ad8f1f`。
- 第一阶段最佳 teacher：
  `runs/adaptation_goal/oracle_clean_payload_seed58/checkpoint_500.pt`。
  SHA256：`5a91d70ef4e8756796d074a7fbff75d8c5f9d5b325dc5121b8daf2267af31daf`。
- 已完成 3 种子开发验证的改进 student：
  `runs/adaptation_goal/context_physics_linear_all_ppo_seed69/checkpoint_1050.pt`。
  SHA256：`2b21aa60913373acb8a64872d7f9c5891e09318eb756617280080284e58ccc3f`。
- 第一阶段确认报告：
  `runs/adaptation_goal/eval_v2/teacher_clean_confirmation.json`。
  冻结时间和模型身份见同目录 `teacher_clean_confirmation_freeze.json`。
- Student 开发评测：
  `runs/adaptation_goal/eval_v2/development_validation_900xx/linear1050_comparison.json`。

Teacher 使用真实物理参数、真实状态，以及独立的无噪声 tracking 特征。
它没有更改原来的带噪声观测通道。Student 的动作通路仅使用可部署观测与
context latent；物理参数、真实速度等只能作为训练损失标签。当前最佳
student 的 context 使用 50 帧测量历史，以及机器人 XML 中已有的 pelvis
加速度计。没有增加外部定位等传感器。

## 评测约束

数据集：`/data_zcy/wxy/motion_data_correct/motion_data_full/AMASS_LAFAN_Qingtong`。
全部 42 个 NPZ、453512 帧参与训练范围，不筛掉困难动作或失败片段。

有效协议为 `balanced_fixed_starts_v2_isolated_resets`：每动作 8 个片段，
每片段最多 500 个控制步，同种子 nominal/DR 使用相同动作与起点，按动作
等权汇总，失败末步计入误差，并单独报告失败率及覆盖率。置信区间采用
配对动作/种子 bootstrap。评测中部分环境 reset 不得改变存活环境的
状态、参考游标或观测历史，代码有逐次断言。

早期 v1 评测存在部分 reset 污染其他环境的问题，全部只作为探索记录，
不能用于验收。没有为了通过评测降低 DR 范围、观测噪声、负载、力矩上限
或更换较弱 nominal 基线。

DR 包括 torso COM ±0.075 m、torso 质量 ±1 kg、编码器偏差 ±0.01 rad、
足底摩擦 0.3–2、armature 比例 0.8–1.2，以及右手固定 1–3 kg 刚性负载。
负载包括完整的质量、COM 和惯量变化；没有 interval push。

## 当前方案

- GPU 0 / 2：`context_trainable_aux_ppo_seed74` /
  `context_trainable_plain_ppo_seed74`。从 Linear1050 出发，以 PPO 同时
  更新 context encoder 和控制网络，基础策略固定；对比有／无辅助物理
  与速度预测损失。额外监督仅更新 encoder 和训练用读出头，标签不进入动作。
- GPU 1：`context_physics_linear_all_ppo_seed69`，主指标及附加指标奖励
  改为分段线性。仅附加指标线性化的对照在 750 轮后停止，检查点全部保留。
- GPU 3：`context_causal_mean_linear_ppo_seed72`，测试因果累计 context
  的 PPO 优化。只复用已有测量，不增加未来信息或真值输入。相同记忆结构
  的模仿训练已在 1600 轮后停止，最佳开发误差比约 1.077，仍未达标。
- 慢速无记忆 PPO 已完成；特权特征替换 teacher 未保持目标水平，停止于
  1200 轮。编码器偏差辅助监督及其配对对照也已停止，留出环境中预测误差
  不优于零先验，tracking 未改善。所有检查点与失败记录保留。
- 参考去噪单独微调 `context_reference_warm_seed70` 已完成 1500 轮，
  没有改善综合误差；其去噪模块与 PPO 控制模块的组合也未超过已有结果。

完整过程、失败方案、PID 和日志索引见 [实时记录](adaptation_live.md)。
历史结果及决策依据见 [研究记录](adaptation_goal.md)。

## 复现与软件检查

所有代码修改只在本仓库，使用仓库 `.venv/bin/python`。只分配物理 GPU 0–3，
未改动或停止 GPU 4–7 的任务。评测命令示例：

```sh
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python -m intact_tracking.cli.adaptation_eval \
  --physics dr \
  --checkpoint runs/adaptation_goal/context_payload_imu_seed59/checkpoint_2500.pt \
  --seed 10001 \
  --output runs/adaptation_goal/eval_v2/student_reproduction_seed10001.json
```

CPU TorchScript 已有结构性导出验证：部署输入无特权字段，要求单线程 CPU，
8 个独立 Python 进程核对动作误差不超过 2e-4。此前发现多线程 CPU 偶发
数值偏差，不能宣称任意线程配置均通过。最终选定的成功模型仍需重新导出，
再进行完整闭环评测；当前导出 smoke 通过不代表 tracking 或实机安全通过。
因果记忆模型的新导出显式传入/返回均值和计数状态，8 个独立进程已通过，
包括顺序状态传递、选择性重置、均值到 EMA 的边界。GPU 数值参考显式使用
FP32 卷积；首次启用 TF32 时有少数 latent 分量超出数值核对阈值，未放宽
阈值。该导出的完整闭环结构检查已通过，但未达到 tracking 验收标准。
