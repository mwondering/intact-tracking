**Router 利用 latent 信息的探索记录，2026-09-14**

本次发现：原来的 16 类硬路由丢掉了 latent 中大量可读出的四肢负载信息。最有依据的改法是先均衡环境信息的度量，再输出连续专家权重。保留在线 KMeans 的候选中，四组、每组四个中心、组内连续混合的结果最好。它在两组新的 DR 世界上，主要八项参数的平均预测 R² 都约为 0.92；原硬路由约为 0.32。

这里完成的是路由方案探索、独立环境复核和运行原型验证。没有训练新的 PPO expert，也没有将新路由直接接到旧 expert 上执行控制。因此，这些结果证明的是路由输出保留信息的能力，不能当成 tracking 收益或环境分类准确率。

使用的 context 仍是经过四肢负载训练的 nominal50、Memory350 扩容版本，响应标签 10 步、predictor 5 步，update 15000。没有换 encoder，也没有用无负载训练的 encoder 替代它。

| 输入与检查对象 | 固定身份 |
|---|---|
| 原共享 MoE | `runs/limb_context_20260914_uniform_moe16_8x1024_u1000/moe/checkpoint_update_001000.pt` |
| policy SHA256 | `5239793a01de2c95045a0bb39365a2d539360fd5e0fa5a6df115b2acc835c0d1` |
| context | 上述实验的 `references/context/update_015000.pt` |
| context SHA256 | `db5d7bd7df026db163b9eedf588d37dcc81b55551d28afe3744f67a8ab2642ac` |
| context 原始路径 | `runs/limb_context_20260912_memory350_response_window_ablation/response10/stage1_8192/update_015000.pt` |
| 当前交互 DR | 原始连续随机 DR，加四肢独立负载；手 0–2.5 kg，胫部 0–4 kg |
| motion | 完整 129827 条目录，uniform 采样 |
| 精度 | 原 policy FP32；冻结 context 按原实现使用 BF16 autocast；新路由度量和权重 FP32 |

表中的分数通过“只读取 router 输出，预测真实 DR 参数”计算。原路由输出是 16 维 one-hot，新路由输出是 16 个权重，解码器均为带截距的线性 ridge。R² 定义为 `1 - MSE / Var(target)`，1 是完美预测，0 接近只预测总体均值，负数表示比该参照更差。主要八项是四肢负载、三轴 torso COM 和足底摩擦，按参数完整范围缩放。全部 67 项参数的结果也保留在原始 JSON 中，没有只保存表现好的维度。

训练、选参和测试按完整 DR world 分开，防止同一个环境的相邻帧分散到训练集和测试集。seed 30402 中的 768 个 world 用于拟合投影、中心和解码器，其余 256 个用于选择设置。每个 world 取 32 个长期记忆充分、policy step 至少 500 的样本。seed 30401 用于探索性泛化检查。温度、top-k、EMA 等最终候选在读取 seed 30403 的结果前封存；30403 和随后新采集的 30404 都没有用于拟合或修改这些设置。

选择记录的 SHA256 为 `91854b4021adc2a6f24c961e612e94527b1d39d8e6ff513208c555f8567a5a20`，封存时间为 `2026-09-14T16:56:43.593305+00:00`。共考察了 153 个初始度量/路由组合，补充分组与插值候选，再对 16 个候选考察 5 个 EMA 时间常数。最终同时封存了分组 top-3 和 top-4 等候选；初始推荐槽位记录为 top-3，本报告根据后续完整比较推荐 top-4，原始选择文件没有被改写。

| 路由方法 | 同时激活的 head 数 | seed 30403 平均八项 R² | seed 30404 平均八项 R² |
|---|---:|---:|---:|
| 原距离，硬选 1/16 | 1 | 0.324 | 0.321 |
| 原距离，软 top-4 | 4 | 0.399 | 0.392 |
| 白化度量，连续 16 个 | 16 | 0.882 | 0.881 |
| DR 标定度量，联合 top-8 | 8 | 0.803 | 0.800 |
| DR 标定度量，联合连续 16 个 | 16 | 0.912 | 0.912 |
| 四组 KMeans，每组 top-2 | 8 | 0.824 | 0.825 |
| 四组 KMeans，每组 top-3 | 12 | 0.884 | 0.886 |
| **四组 KMeans，每组连续 4 个** | **16** | **0.920** | **0.921** |
| 四组固定三角插值 | 最多 12 | 0.923 | 0.924 |

这些行使用各自仅在验证集上选出的温度与 EMA，不是完全相同超参数的单因素比较。分组连续四个的 EMA 为 2 秒，联合 top-8 为 1 秒。另一个可分辨的因素是时间平滑：在探索性 seed 30401 上，分组连续四个的无 EMA 分数为 0.888，EMA 2 秒后为 0.920。不能把时间平均带来的去噪解释成 router 凭空产生了新的环境信息。

| seed 30403，负载预测 R² | 左手 | 右手 | 左胫 | 右胫 |
|---|---:|---:|---:|---:|
| 原硬路由 | 0.117 | 0.353 | -0.032 | 0.010 |
| 分组连续四个 | **0.955** | **0.950** | **0.861** | **0.848** |

分组连续方案的平均八项 R² 的 95% world-bootstrap 区间为 **[0.9168, 0.9228]**。bootstrap 每次整体重采样 1024 个 world，把同一 world 的 32 个时刻一起保留；它反映这次冻结 policy 下的环境抽样不确定性，不反映重新训练 policy 的随机性。额外将预测对应的 world 打乱后，平均 R² 降到 -0.929，确认高分依赖于正确的 latent–环境对应关系。

![独立 DR 世界的信息保留对比](../runs/limb_context_20260914_router_information_exploration/router_information_heldout.png)

推荐的路由先用一个冻结的线性映射，把单位长度的 64 维 latent 投影为八个按范围归一化的 DR 估计值，并裁剪到 [0,1]。**这个映射用了仿真 DR 标签进行前期标定。运行时映射、KMeans 和 expert 都不读取真实 DR 参数，映射也不接收 PPO 梯度。** 四组分别为左右手负载、左右胫负载、COM x/y，以及 COM z/摩擦；每组各有四个 KMeans 中心。

对每个二维分组 `q_g`，权重为 `w_gk = softmax(-||q_g-c_gk||² / T_g) / 4`。四组共同组成 16 个非负权重，总和为 1。实际温度依次为 `[0.21369466, 0.19152939, 0.23421663, 0.18728098]`，完整矩阵、中心和配置已导出为 `exports/product_dr8_gk4_t1.pt`。

这样可以继续在线更新 KMeans，同时避免把多个连续因素压缩为一个离散编号。数学上，每组的 `T log(w_j/w_0)` 等于输入二维坐标的一个仿射函数。只要三个中心差向量张成二维空间，且权重没有数值下溢，就能由权重恢复这两个投影坐标。本次实际中心满足该条件，37120 个投影、边界及随机点的 FP32 权重经解析反演后，最大坐标误差为 **1.73e-7**，不需要拟合另一个 DR 解码器。检查保存在 `group_geometry_audit.json`。保留的是**投影后的八维**，不能称为完整保留原来的 64 维 latent。

在 seed 30404 上，16 个 head 的平均权重在 0.053–0.071 之间，但每个环境的权重并不相同：平均最大单个权重约 0.163，平均有效 head 数约 12.16。因此，这不是把所有环境都恒定平均分给同一组 head。固定三角插值也可以保留这八维信息，而且最多激活 12 个 head，但它放弃了用户此前指定的在线 KMeans 中心更新，所以作为参照方法保留。

分组会改变 expert 的含义：它们学习的是若干环境因素对应的修正分量，最终相加，不再是 16 个互斥“完整环境”的控制器。当前观测下，这种对不同因素的可加组合是否足以表达所需的控制修正，尤其跨组因素之间的耦合，仍需新的 PPO 实验验证。若要保留“每个 expert 对应完整环境”的结构，联合连续 16 个或联合 top-8 更接近原架构；前者的信息分数较高，但权重更接近均匀分配，后者的信息损失更明显。

仅看预测分数还不够。高温度可以把信息编码成非常小的权重差异，要求 expert 输出很大的差异才能利用它。为此记录了线性解码器在权重单纯形顶点间的预测跨度，即 `prototype_span_in_DR_ranges`：白化连续 16 个约为 17.35，DR 度量联合连续 16 个约为 13.55，分组连续四个约为 4.52。它是输出放大需求的诊断量，不是实际扭矩或 policy 输出上限；分组权重本身有 1/4 的平均系数，也应结合这一尺度解释。所有设置的数值在 `comparison.csv`，不能据此直接推导控制精度。

如果路由标定阶段也不使用数值 DR 标签，白化后连续使用 16 个中心是已有证据支持的替代方案，独立环境 R² 为 0.882。这里“白化不用 DR 标签”只描述度量拟合；解码评估和超参数选择仍用到了验证集 DR 标签，不能把整个研究流程称为完全无监督。

50 Hz 的复核使用 seed 30404 的 1024 个全新 DR world，500 步 tracker 预热和 1500 步原 MoE 交互，每个控制步保存 latent。记忆充分、step 至少 500、连续同一 motion 的有效相邻区间共有 714454 个。所有新路由都在**同一原 policy 轨迹上因果回放**，并不是分别用新路由控制机器人产生的轨迹。

| 每 20 ms 的变化，seed 30404 | 原硬路由 | 联合 top-8 | 分组连续四个 |
|---|---:|---:|---:|
| 平均权重总变差 `0.5 sum(abs(Δw))` | 0.010728 | 0.001128 | **0.000478** |
| 99 分位权重总变差 | 1.000000 | 0.003001 | **0.001475** |
| 解码后八项 DR 变化的平均 RMS，按完整 DR 范围归一化 | 0.001936 | 0.000972 | **0.000470** |

分组方案的平均权重变化约为原来的 1/22；通过同一冻结解码器衡量有物理含义的变化后，约为原来的 1/4。这两项要一起读，避免把“接近恒定的均匀权重”误判为有效稳定。原硬路由的权重 TV 等于换 expert 的比例；软路由同时混合多个 head，不能再把全局 argmax 换号率解释成控制器切换率。

![50 Hz 权重及解码后的变化](../runs/limb_context_20260914_router_information_exploration/router_stability_50hz.png)

跨 motion 的试验均值权重 TV，在更长的 seed 30403 轨迹中，从原硬路由的 0.186 降至分组连续路由的 0.068。这也包含持有固定 DR 的历史信息带来的稳定作用。示例图固定取 world 0，没有按结果挑选 world；这个 world 的原硬 ID 本身就不变，所以稳定性结论应看全部 world 的统计，而不是只看例图。灰区内短期 context 不完整，路由保留此前估计。

![固定 world 的逐步轨迹](../runs/limb_context_20260914_router_information_exploration/router_fixed_world_timeline.png)

在线中心更新的附加检查沿用了原训练每 24 步更新一次、更新率 0.01 的设置，新加权重平均 TV 上限 0.02。更新仅使用当时已经收集的 latent 投影特征，不使用该测试集的 DR 标签。在 1500 步内完成 62 次更新，中心总 RMS 位移为 0.00558，单次更新导致的平均权重 TV 最大为 0.000580。冻结原解码器后，R² 为 **0.920988**，与完全冻结中心时的 **0.920976** 接近。该检查覆盖 30 秒固定 policy 轨迹，尚不证明长时间 PPO 与路由共同变化时也没有漂移。

运行原型在 `src/intact_tracking/memory350_metric_router.py`。它没有 trainable parameter，投影、中心和温度均为 buffer。`advance()` 在环境步进时更新 EMA；普通 `forward()` 是纯函数，不会在 PPO minibatch 中推进历史。现有训练入口没有切换到这个原型。正式接入时应遵循以下已经验证的约定：

1. 每个新的环境观测计算一次 gate，并将 16 个权重存进 rollout observation。actor、critic 和 bootstrap value 使用该观测对应的同一份权重；PPO 重放时读取缓存，不能对打乱的 minibatch 重新推进 EMA。
2. actor 与 critic 各自保留独立的 obs encoder 和 expert 参数。共享的是只读路由权重，不共享 actor/critic 的神经网络参数。若延续共享 encoder 的实验，各自一套 encoder 下有 16 个 head。
3. actor 混合 residual mean：`mean = tracker_action + sum(w_k * residual_k)`；critic 混合 value：`V = sum(w_k * V_k)`。继续使用原来的单个 Gaussian action distribution 计算 PPO log-prob，不能把它误写成多个 Gaussian 概率的混合。连续权重混合专家输出属于常见 MoE 形式，见 [Shazeer 等人的原论文](https://arxiv.org/abs/1701.06538)。
4. 只有一轮 PPO 更新结束以后才更新 KMeans 中心。多卡累加各组计数和特征和，即使某个 rank 没有有效样本也参与 collective，中心编号保持原位。单次平均 TV 的约束是软路由下的变化限制，不能解释为每个环境的最大变化保证。
5. DR 在当前训练中启动后固定，所以跨 episode/motion 保留 EMA 和长期 DR 记忆，短期记忆不足时暂停估计更新。真实物理参数改变时需清空相应环境的 DR 历史，包括 context 长记忆；原型提供 EMA 的 `parameters_changed` 标记。2 秒 EMA 针对静态 DR 选出，突变响应速度和冷启动控制效果不在本次结论中。

数值一致性没有被笼统记成“全部通过”。在 CPU/NumPy 的 5 Hz 回放中，三个稀疏 top-k 候选出现极少数大于阈值的权重差异；逐点检查发现截断位置的距离差仅约 1e-8–1e-7，浮点舍入改变了选入集合。50 Hz CUDA 回放中也有七个稀疏候选未通过最大误差阈值，均保留在结果中。推荐的分组连续四个方案，完整 50 Hz 回放最大权重误差为 **3.13e-7**；固定三角插值为 **3.28e-7**，均通过。没有用放宽阈值来掩盖稀疏候选的差异。

六项原型测试通过，覆盖 PPO 梯度隔离、纯前向与缓存不变、EMA 跨 motion 保持和真实参数变化清空、中心更新 TV 限制、固定插值坐标恢复，以及包含空 rank 的两进程 Gloo 汇总。保存的预测值也由独立脚本重新计算了每项 R² 和置信区间。

速度方面只做了 MLP 微基准：FP32、1024 个输入、实际 checkpoint 权重、缓存 gate，不包含 tracker、context、仿真、DDP 或 optimizer。在两张有其他任务共享的 GPU 上，连续计算全部 16 个小 head 的 actor+critic 前向约为 2.2–2.3 ms，前向加反向约 10.1–10.3 ms；原逐 expert 索引的硬路由约为 21.9–25.9 ms 和 29.5–35.4 ms。将相同的 one-hot 硬路由也改为全 head 批量计算后同样变快，且 actor、critic 的输出一致性检查通过。这说明加速主要来自执行方式，不能把微基准的倍率当成 PPO 端到端提速。

对剩余 DR 的检查同样保留：64→256→128→67 的非线性解码器，按六个 DR family 等权训练，三个初始化、完整 world 验证选 epoch，再测 seed 30403。torso mass 的 R² 约 -0.005–0.014，armature 和 encoder bias 的组均值仍为负。它没有证明这些信息数学上完全不存在，但目前没有足够证据说 router 可以从该 latent 中充分利用这几类 DR。最强分组方案主要针对已经确认可读的八项。

后续控制实验应重新训练适应该权重语义的 expert，不能只替换旧 checkpoint 的路由来判断方案是否有效。与无 latent MLP 的比较需要保持相同的 motion、DR、物理设置、环境数量和训练预算，并在对应负载上看 body/root tracking 指标；正确、打乱和固定 gate 的控制评估可进一步检查 policy 是否实际依赖路由信息。分组可加结构是否足够、从头训练的收敛速度以及长期中心漂移，都是控制训练需回答的问题，不是本次 R² 实验已经回答的问题。

所有数据在 `runs/limb_context_20260914_router_information_exploration/`。主要入口如下，导出的 `.pt` 只含冻结路由，不含为新权重训练过的 expert。

| 记录 | 用途 |
|---|---|
| `protocol.json`、`selection_seal.json`、`final_selection.json` | world 划分、选参规则与封存配置 |
| `comparison.csv`、`comparison.json` | 全部 16 个最终候选、信息、稳定性与放大需求 |
| `heldout_confirmation/summary.json` | 独立 seed 30403 结果 |
| `runtime50hz_confirmation/summary.json` | seed 30404 每步回放及数值核对 |
| `online_update_audit.json` | 62 次因果中心更新、缓存检查及 dense one-hot 等价检查 |
| `group_geometry_audit.json`、`statistical_checks.json` | 八维可恢复性与独立 R² 重算、world bootstrap |
| `remaining_dr_nonlinear_probe.json` | 剩余 67 维 DR 的非线性解码检查 |
| `exports/product_dr8_gk4_t1.pt` | 推荐的冻结度量和四组在线 KMeans 初始状态 |
| `verification.json`、`execution_manifest.json`、`source_snapshot/` | 完成检查及对应源码身份 |

复核可以使用项目 `.venv/bin/python`；不要解析 Python 符号链接后再启动，否则可能离开项目虚拟环境。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/report_moe_router_information.py --root runs/limb_context_20260914_router_information_exploration
.venv/bin/python -m pytest tests/test_memory350_metric_router.py -q
```

本次没有修改生产 PPO 路由、更新已有 policy/context checkpoint 或启动新的 policy 训练任务。所有新增 `.py` 的执行入口和源码哈希在实验 manifest 中，可以沿用已有原始轨迹复算结果。
