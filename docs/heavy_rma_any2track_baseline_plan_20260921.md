# 144000-exp-heavy：RMA teacher 与 Any2Track baseline 实现方案

2026-09-21。**状态：已实施；验证记录与运行入口见[实施文档](heavy_baselines_implementation_20260921.md)。** 用户随后将两种 baseline 改为 **uniform 连续训练，不再在 2000 轮重启**；最新协议和运行状态见 [uniform 训练记录](heavy_baselines_uniform_20260921.md)。下文保留原始设计依据，其中 adaptive 与两阶段调度部分已被最新要求取代，网络、环境和 PPO 设置不变。

建议新增两个实验，W&B group 均为 `144000-exp-heavy`：

- `144000-exp-heavy-rma-teacher`：真实物理参数条件化的 residual teacher。
- `144000-exp-heavy-any2track`：在冻结 SPV5-2A 上移植 AnyAdapter，在线训练预测型 history encoder。

实现目标是**统一 SPV5-2A / HDR 平台下的适应方法对照**。论文中分别注明 `RMA-teacher (residual implementation)`、`Any2Track / AnyAdapter (SPV5-2A backbone)`，避免把移植后的结果写成原论文整套系统的原样复现。

## 1. 对两种方法的理解

RMA 第一阶段读取真实环境因素，但会先用可训练的环境编码器压缩，再将 embedding 输入策略；编码器和策略共同通过 PPO 学习。第二阶段才训练从交互历史估计该 embedding 的 adaptation module。本次只做用户要求的 teacher 阶段。[RMA §III-A](https://arxiv.org/html/2107.04034v1#S3.SS1)

Any2Track 的 AnyAdapter 通过未来状态预测学习 history embedding，然后将 embedding 注入冻结 tracker 的各层。其预测目标是机器人动态状态，并非真实 DR 参数。论文使用 79 步历史、20 步自回归预测，并交替训练世界模型和策略适配器。[Any2Track §III-B](https://arxiv.org/html/2509.13833v3#S3.SS2)

OpenTrack 默认配置明确设置 `train_history_encoder_in_policy=False`、`train_world_model=True`、`supervised_loss_weight=0`：encoder 由世界模型目标训练，PPO 不更新 encoder，也不启用额外的世界模型引导动作损失。[官方配置](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/track_mj/envs/g1_tracking_adapter/train/g1_env_tracking_general_dr.py#L228-L258)

| 方法 | 策略获得的 dynamics 信息 | 表征如何训练 | 动作调整位置 |
|---|---|---|---|
| 现有零 latent baseline | 全零 | 无 context 学习 | tracker 输出之外的 residual |
| 现有 latent 方法 | 冻结 Memory350 的五帧 latent | 已完成的 context 预训练；PPO 时 encoder 冻结 | 输出 residual，并对其共享隐藏层加 DR 辅助监督 |
| 新 RMA teacher | 真实 108 维 DR 参数的可学习 embedding | 与策略联合 PPO | 沿用当前输出 residual |
| 新 Any2Track | 79 步交互历史的 128 维 embedding | 在线世界模型预测损失 | 冻结 tracker 动作 MLP 内的逐层 adapter |

两种新方法都不加载 u8816 Memory350 encoder 的权重，也不加入现有的 8 维 DR 辅助预测损失。RMA 使用真实物理标签作为输入；Any2Track 不读取物理标签。模型结构、历史长度和新增表征优化器属于方法差异，共同的环境与 PPO 设置保持一致。

## 2. 固定的实验条件

以[当前恢复配置](../runs/144000-exp-heavy-residual-latent/ppo_4gpu8192_resume2000_mass5x_sampling_reset/run_config.json)与[上一阶段协议](heavy_residual_latent_baseline_plan_20260921.md)为基准，启动时逐字段核对，而非另建近似配置。

| 项目 | 两个新 baseline 的共同设置 |
|---|---|
| 基础 tracker | 同一个冻结的 SPV5-2A `checkpoint_144000.pt` |
| 数据集 | `/data_zcy/wxy/motion_data_correct`；扫描 224651，过滤 4171，保留 220480 条 |
| motion 清单摘要 | `59b8e336c152e4773133bcdcd86cf86fd6545c3c0e6cc00ce67912d107c740ac` |
| 并行规模 | 每组 4 GPU × 8192 环境；每组独立覆盖完整数据 |
| HDR / nominal | 每 rank 7372 HDR、820 nominal；沿用全部原生 DR 和四肢负载随机化 |
| 负载采样 | 四肢质量各 4 档，共 256 组；组内连续均匀；负载 COM 每轴 ±5 cm |
| 控制、reward、termination | 原 144000；50 Hz 控制、episode 最长 500 步 |
| Adaptive | 原 144000 branch；uniform=0.5、temperature=0.25、bin=50、EMA=1000、cap=200、rewind=1/3、pre-failure=100 |
| Rollout / PPO | 24 步；5 epochs × 4 minibatches；每组每次更新 786432 transitions |
| 学习率 | actor 1e-4、critic 5e-4，固定 schedule |
| 其他 PPO | entropy=0.005、初始 std=1.0、clip=0.2、gamma=0.99、lambda=0.95、max grad norm=1、value coef=1、clipped value loss |
| 精度 / 随机种子 | FP32；seed=121，rank seed=`121 + 1000003 * rank` |
| 记录 | project `intact-preview-v2`，group `144000-exp-heavy`，沿用 tracker 的日志分组 |

物理范围及跨 reset 的采样语义见 [NDR/HDR 表](dr_ranges_ndr_hdr.md)。尤其注意：关闭物理参数回归损失不会关闭任何 DR；Kp、Kd、armature、torso 质量与负载 COM 均继续随机化。

## 3. RMA teacher 的实现

### 输入与网络

采用当前 HDR 的完整 108 维物理 schema：torso COM 3、torso 质量 1、摩擦 1、Kp/Kd/armature 各 29、四肢质量 4、四肢负载 COM 12。真实值从 nominal 恢复与负载惯性合成之后的环境采集，按各维物理范围映射到 [-1,1]。不使用 context 距离函数的坐标权重，也不因已有辅助头只监督 8 维而删减 teacher 输入。

建议的本仓库网络规格：

```text
实际 θ[108] → 固定范围归一化 → MLP[256,128,64] → e[64]
concat(冻结 tracker 特征[1645], tracker 动作[29], e[64])
  → residual MLP[512,256,128] → Δa[29]
a_mean = a_tracker + Δa
```

参数编码器从零初始化、ELU；residual 末层权重/偏置置零。初始动作均值与 tracker 一致，探索 std 使用共同的 1.0。策略始终读取当前真实参数，不需要构造五帧伪历史；因此 actor 输入为 1738 维。64 维 embedding 和 residual 外壳是本实验的移植选择，不声称是原 RMA 的机器人或网络规格。

Critic 沿用当前 6330 维原始 privileged observations、tracker 动作及 `[1024,512,512]` 隐藏层，将 context 条件替换为自己的 64 维 DR embedding，输入共 6423 维。actor / critic 的参数编码器独立，分别属于 actor / critic optimizer，避免一个参数被两个 Adam 状态重复更新。

不配置 DR 回归头、world model、history adaptation student 或 SIGReg 等损失。actor 的 DR 编码器由策略 PPO 梯度更新，critic 的编码器由 value loss 更新。

### 必须处理的梯度边界

现有 `ConditionedMLP` 和 latent 读取逻辑有显式 `detach()`，用于冻结 Memory350。直接把可训练 DR 编码器接到这条路径，会使它收不到 PPO 梯度。

新 actor 必须在每个 PPO minibatch 中从存储的原始 θ 重新计算 embedding，保持编码器到 residual 的计算图；不能在 rollout 时算出 embedding 后永久存为一个 detached 输入。物理标签本身可以 detach，编码器输出不能 detach。零初始化动作末层可能令第一批次的编码器梯度为零，测试应在动作末层获得非零权重后核验梯度。

### Teacher 的解释范围

这是“已记录 108 维物理因素已知”的 oracle 参考。108 维 schema 不包括外力脉冲、编码器偏置、动作 delay/alpha 等额外随机变量；本版不隐式扩展这些特权输入，也不能把有限步数训练的 teacher 宣称为所有扰动下的严格性能上界。评测与导出必须显式提供 θ，不能用零向量代替后仍称为 teacher。

## 4. Any2Track / AnyAdapter 的实现

### 历史编码器与预测任务

参考版本固定为 OpenTrack commit `cb9b751993a2483e5d1805a2565ddbfe950c04c9`。网络结构以该版本的 [ConvMLP 与世界模型](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/track_mj/learning/policy/model_based_ppo/brax_networks.py)为依据，在 PyTorch 中实现。

- 每帧历史采用可观测状态 64 维：gyro 3、gravity 3、joint position 29、joint velocity 29；另附实际总控制命令 29 维，共 **93 维**。状态从当前缓存的含噪本体感知提取，避免额外抽取一次观测噪声。
- 历史长度 **79**，不叠加我们的方法的五帧 latent。Conv1D：93→64，kernel=9/stride=5；64→64，kernel=6/stride=3；展平 256→128，使用 SiLU。
- 世界模型输入为当前状态 65 维（上述状态再加 root height）、当前动作 29 维和 embedding 128 维。隐藏层 `[512,512,256,256,256,128]`，输出 33 维的角速度增量、关节速度增量和高度增量，结合 0.02 s 积分恢复状态。
- 自回归 **20 步**，监督 gyro、gravity、joint position、joint velocity、root height。root height 和干净状态只用于世界模型训练，不进入 history encoder 或 actor 的新输入。
- 保留官方世界模型分项权重 `500 / 500 / 1 / 0.5 / 500` 及其速度缩放语义；gravity 使用方向误差，其余使用 L1。训练前用固定张量核对缩放、聚合方式与分项数值，避免只复制权重却改变单位。
- 不使用 θ、DR 参数回归、SIGReg 或对比损失。world model 与 encoder 从零开始，PPO 训练期间持续在线学习。

上述 93 维历史沿用本仓库的总 raw action 作为命令坐标。官方环境历史使用 PD motor target；两者坐标不同。移植时统一使用 SPV5-2A action contract，并在训练、递推、部署中保持一致，不混用“policy 输出”“PD 目标”“delay 后的动作”。

### 逐层适配与动作输出

实际 144000 的动作 MLP 为：

```text
1645 → 2048 → 2048 → 1024 → 1024 → 512 → 256 → 128 → 29
```

保留该网络全部权重，按官方 `MLPWithAdapter` 的连接方式添加可训练的、全零初始化的线性层：

```text
h1 = activation(W1·tracker_features + b1 + A1·embedding + c1)
hℓ = activation(Wℓ·hℓ₋₁ + bℓ + Aℓ·hℓ₋₁ + cℓ)
a_mean = WM·hM₋₁ + bM + AM·hM₋₁ + cM
```

第一层 adapter 接收 dynamics embedding，后续 adapter 接收融合后的上一层隐藏状态。输出层也有 adapter。它不是每一层都独立拼接 latent，也不是再在最终动作之外添加一套当前 residual MLP。[官方 adapter 实现](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/track_mj/learning/policy/model_based_ppo/brax_networks.py#L48-L89)

零初始化时只要求动作**均值**精确恢复 tracker；探索分布采用共同设置的 trainable std=1.0。冻结 tracker 的 reference encoder、estimator、normalizer 和所有原 MLP 权重。原 MLP 运算仍须保留对输入的梯度，否则早期 adapter 收不到梯度：使用 `requires_grad_(False)` 冻结 W，不能用 `no_grad()` 包住整个适配前向。

最终输出直接进入当前高斯动作分布和 SP 控制链，不再叠加一次 tracker 动作。继续使用原 `default_q + joint_scale × action`、delay/smoothing、PD 与 reward 定义。原 AnyTracker 的 reference-relative / tanh 动作定义依赖其自身基础 tracker；本次保留现有 SPV5-2A 的动作语义，并在报告中列为移植差异。[AnyTracker 动作定义](https://arxiv.org/html/2509.13833v3#S3.SS1)

Critic 保留当前 privileged observations、tracker 原动作和 `[1024,512,512]` 隐藏层，附加 detached 的当前 128 维 embedding，输入为 6487 维；不通过 value loss 更新 history encoder。

### 在线训练与 PPO 的一致性

参考官方的独立优化器与更新顺序：[训练循环](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/track_mj/learning/policy/model_based_ppo/train_model_based_ppo.py#L416-L463)。本仓库建议落实为：

1. 当前 encoder + adapter 采集 24 步 rollout，保留真实行为动作、old log-prob、value 和因果历史索引。
2. 每个环境从这 24 步中选择一个 20 步预测窗口，使用独立 RNG；为世界模型执行 5 epochs × 4 minibatches，独立 Adam，LR=1e-4。该新增训练预算单独计数，不扩大环境采集量。
3. 固定更新后的 encoder，按原 PPO 的 5 epochs × 4 minibatches 更新 adapter/std 和 critic。新的 embedding 从对应原始历史重新计算，并 detach；old log-prob 必须保留采集时的值。
4. 记录世界模型更新引起的 embedding 和策略均值漂移、PPO 更新前 KL，检查表征更新是否使策略突然跳变。

不能只存旧 latent 后更新 encoder，再让 PPO 与下一轮采样分别使用不同版本的表征。保留原始历史可使策略概率计算对应实际的新 encoder。建议按每 rank 的 `79 + 24 + 1` 帧连续序列加索引保存，FP32 约 **0.295 GiB**；给每个 transition 复制完整 79 帧需约 **5.38 GiB**，不采用这种重复存储。

对 reset / motion boundary 明确标记，预测损失排除不连续边；边界处重新锚定真实状态和合法历史，不跨边界积分。定义历史为 `(s_j,a_j)` 后，预测循环也按同一时序移入完整 state-action 对，防止状态与动作错一拍。padding 不当作真实历史完整度。

### 公开代码中需要核对的实现细节

静态检查发现：官方环境构造的历史每帧是 **64 维状态 + 29 维 motor target = 93 维**，但当前 autoregressive loss 用 `predicted.shape[-1]-1 = 64` 平移扁平历史，并只追加预测状态。按该公开配置推导，后续历史帧会发生通道错位。这是基于代码维度关系的检查结论，未通过运行官方完整训练来判断其性能影响。[环境历史构造](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/track_mj/envs/g1_tracking_adapter/train/g1_env_tracking_general_dr.py#L500-L509)、[自回归损失](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/track_mj/learning/policy/model_based_ppo/model_based_ppo_losses.py#L87-L99)

移植版应依据明确的 `(state, action)` 合同按 **93 维完整帧**更新，并增加时序/维度验证；将修正记录为与所参考 commit 的差异，不默默复制该不一致。

## 5. 参数量和计算量的对比边界

按照实际 tracker 层宽，原版全秩 adapter 含 **8,301,085** 个参数，加 29 维 std 后为 **8,301,114**；这是根据各线性层尺寸计算的设计值。history CNN 另有约 111,168 个参数，world model 和 critic 另计。实现后以 `numel()` 审计最终值。

当前 residual actor 约 120 万参数，因此只能宣称环境交互量、PPO 超参数和基础 tracker 一致，**不能宣称策略参数量相同**。同时报告 actor/adapter、encoder、world model、critic 的参数量，秒/更新、峰值显存以及各自 optimizer steps。

现有 Memory350 方法使用过 u8816 的 context 预训练；两个新 baseline 的新增模块从零开始。主表比较匹配的 PPO 更新数，附表同时列出 encoder 预训练与在线预测训练开销，避免把不同总计算预算描述成完全相同。

## 6. 2000 边界后的自动 resume

沿用现有两组的对比边界：**`checkpoint_2000.pt` 内实际 `completed_updates=2001`**，stage A 完成到该边界；stage B 第一条新增更新为 2002。调度器以文件内部计数核验，不根据文件名推断。如果后续改用严格 completed_updates=2000，需要对所有比较组统一说明；本方案先与已完成的两组 2000 号初测保持一致。

stage A 正常保存并退出后，启动新的 stage B 进程，执行**一次** adaptive reset：

- 保留模型、actor/critic optimizer、normalizer、std、更新计数；RMA 还保留两套 DR encoder，Any2Track 还保留 history encoder、world model 及其 Adam 状态。
- adaptive 访问/失败统计恢复 visit=1、failure=0，清除 EMA 和待结算历史，按新环境重建当前访问；算法和 cap=200 等设置不变。
- 新仿真 episode 与 rollout/history 缓存重新建立，模型权重不重置；Any2Track 的预测训练不跨进程拼接物理轨迹。
- 这次 resume **只改变采样统计**；不复制现有 latent 组在恢复时的 DR 系数乘 5 操作。两种新方法没有那一项 DR loss。
- stage B 无更新上限；后续正常断点恢复默认继承当前 sampler，避免每次意外重启都反复刷新采样。保存 reset generation/来源摘要防止重复执行阶段转换。

新增正式阶段控制参数，不能借用 `bounded_smoke` 来实现 stage A 的训练上限，因为该分支会禁用在线 W&B。阶段切换使用独立目录和 W&B ID，同一个 group，记录 parent checkpoint 与 sampler reset 审计。

建议产物目录：

```text
runs/144000-exp-heavy/baselines/rma_teacher/{stage1,resume_sampling_reset}/
runs/144000-exp-heavy/baselines/any2track/{stage1,resume_sampling_reset}/
```

## 7. 工程落地顺序与验收

1. **固定共同协议**：复用 heavy physics、完整 motion 过滤、reward/termination、144000 sampler 和日志；把方法条件提供器与当前冻结 Memory350 依赖解耦。
2. **实现 RMA teacher**：新增 DR 条件 actor/critic，验证 θ 的名称、维度、归一化和非零 PPO 梯度。
3. **实现 AnyAdapter**：先验证全零 adapter 与原 tracker 的均值一致，再接历史 CNN、20 步世界模型和独立优化器，最后接入在线交替更新。
4. **实现自动阶段切换**：保存 stage A 完成快照，恢复后比较模型/全部优化器摘要、检查四 rank 的 adaptive 先验及无重复 update。
5. **运行实际 batch 检查**：每组 4 × 8192 验证内存、有限梯度、四 rank 一致、日志与保存/恢复；通过后按同样规模启动正式阶段。
6. **评测和部署产物**：在相同 completed updates 对比；使用相同 motions、初始状态和扰动，并报告 nominal、HDR 两套完整评测。nominal 也运行整份 motion 清单，不仅使用混合环境中约 10% 的子集。

建议新增模块（名称为实施规划）：`heavy_rma_teacher.py`、`heavy_anyadapter.py`、`anyadapter_world_model.py`、`anyadapter_training.py`，以及配套 baseline train/eval/export CLI 和阶段启动器。现有 `memory350_heavy_policy.py` 的物理构建与 sampler 合同继续复用，不将新方法硬塞入“冻结 context encoder”的旧类约束。

关键验收项：

- RMA 的 actor DR encoder 可由 PPO 更新；Any2Track 的 encoder 在 PPO backward 中无梯度、在预测 backward 中有梯度；冻结 tracker 权重与归一化摘要始终不变。
- Any2Track 每层 adapter 可获得梯度，初始均值等于 tracker；概率分布对应最终执行的总 action；action rate reward 仍计算最终 sampled action。
- 历史严格因果，保存/恢复前后 index 含义不变；每帧 93 维、20 步递推、边界 mask 和状态/动作单位均通过测试。
- 记录原有 `Train/`、`Loss/`、`Policy/`、`Metrics/`、`Episode_Reward/` 指标，并新增 `WorldModel/*`、`Adapter/*` 或 `Teacher/*`；不能把“参数预测关闭”写成预测误差为零。
- AnyAdapter 的有效动作改变量统一记录为 `adapted_mean - frozen_tracker_mean`，便于和现有 residual 幅度比较，但执行时不重复加回 tracker。
- checkpoint 保存所有推理依赖和续训优化器。Any2Track 的部署包保留 history encoder 与 adapter，world model 仅续训需要；teacher 导出包明确 θ 输入 schema。两者都支持 ONNX/JSON 数值核验。

执行新训练需要每组独占四张卡。实施阶段保留现有训练进程；正式启动器先检查资源并执行 4 × 8192 的功能预检，再按上述阶段运行。
