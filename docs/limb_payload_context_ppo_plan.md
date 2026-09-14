# 四肢负载下 nominal 监督表征与 residual PPO 实验草案

2026-09-07。用户已授权实施本方案并推进至最终结果。参考最新 LaFAN ABC 结果后修订。
实现、短测及正式队列状态见 [limb_context_live.md](limb_context_live.md)。

## 已确认的条件

- 所有项目改动仅限 `/data_zcy/wxy/intact-tracking`，不修改其他目录。
- 2026-09-08 最新资源约定：本实验仅使用 GPU 0–3；GPU 4–7 留给用户其他任务。
- 数据目录：`/data_zcy/wxy/motion_data_correct/motion_data_full`。本次只读清点得到
  129,827 个 NPZ 和 129,827 个 JSON；motion 数据采用全部 NPZ。
- 仅保留负载随机化，关闭其他 DR。
- 双手与左右小腿中部各增加 0–4 kg，覆盖 nominal 至最大负载，总新增质量 0–16 kg。
  用户已确定全部真实训练环境采用纯独立均匀采样：每个 world 的四处质量分别
  独立 U(0,4)，不同 world 之间也独立；不加入固定比例的全零或全四端点环境。
- 第一阶段使用 nominal counterfactual 监督表征的双 Transformer，训练至收敛，
  取消步数上限与自动早停，持续到用户确认收敛。只有用户明确允许进入第二阶段才放行 PPO。
- 第二阶段将 context latent 同时提供给 residual actor 和 critic；baseline 的
  actor/critic 仅使用原 tracker checkpoint 的观测信息。正式 PPO 每组至少完成
  5,000 个实际训练 iteration。
- 2026-09-08 新增采样安排：前 1000 个 PPO updates 使用 uniform motion sampling，
  然后从准确的 `checkpoint_update_001000.pt` 恢复，用 adaptive motion/bin sampling
  继续到总计至少 5000 updates。所有 baseline/FiLM seed 及 concat/constant 对照采用
  相同安排。只改变 motion 起点采样，四肢质量仍独立 U(0,4)，failure rewind 保持关闭。
  adaptive 使用源配置的 branch 策略、uniform 分支概率 0.5、temperature 0.25，
  概率上限及失败前窗口沿用源配置，并保存各 rank 的采样统计以支持后续恢复。
- 用户明确只关心有 latent 是否提高整体性能，不安排 actor-only / critic-only 训练消融。
- 用户于 2026-09-08 明确要求所有训练均可在 W&B 监控；阶段一和每组 PPO 都须在线同步，并补传已保存的历史指标。
- W&B 账号邮箱须为 `2486344338@qq.com`；stage 1 在 `intact-forward-predictor`，stage 2 在 `intact-preview-v2`，两阶段使用独立分组，训练短测另分组。

## 最新 ABC 结果对本实验的意义

来源：`docs/lafan_abc_results.md` 及 `docs/lafan_abc_finetune.md`。
在同一预训练 tracker、40 条 LaFAN motion、1000 PPO update 和单一训练 seed 下，
DR 策略 C 相对 nominal 专用策略 A，nominal body/joint 误差高 50.86%/64.01%；
相对最大负载专用策略 B，最大负载 body/joint 误差高 6.53%/11.69%，失败 40 -> 73。
C 相对冻结 tracker 在最大负载仍明显改善，但 nominal 精度下降。

这些结果为“用环境表征缓解单策略的端点折中”提供了直接实验动机；本次要验证
预训练 latent 能否带来这个收益。ABC 使用原 actor 控制 MLP fine-tune，没有 residual
action head 或 latent，不能把 C checkpoint 直接作为本次 residual 主对照。
本次重新训练共同协议下的无 latent / 有 latent residual 策略，并同时评估两端和中间负载。
ABC 的区间来自 motion bootstrap，不代表训练种子的波动；其 actor LR 1e-5 是原
控制头 fine-tune 的设置，不能据此直接断言新建 residual MLP 的最优学习率。

## 负载及环境的具体约定（建议）

复用已存在的 hands-shins 安装几何，将它接入两阶段的共同配置。只随机附加质量，
安装位置与形状固定。采用复合刚体惯性更新质量、质心、主惯量和惯性坐标系。

| 位置 | body | 局部质心位置 / m | 长方体尺寸 / m |
| --- | --- | --- | --- |
| 左手 | left_wrist_yaw_link | (0.12, 0, 0) | (0.10, 0.08, 0.08) |
| 右手 | right_wrist_yaw_link | (0.12, 0, 0) | (0.10, 0.08, 0.08) |
| 左小腿中部 | left_knee_link | (0, 0, -0.15) | (0.10, 0.08, 0.10) |
| 右小腿中部 | right_knee_link | (0, 0, -0.15) | (0.10, 0.08, 0.10) |

建议每个 world 在构造时采样一次，episode reset 不重采样，不做 episode 中途切换。
两阶段保持一致，评估使用独立的负载样本。实际加载后核对四肢新增质量和其余 nominal
物理参数；不叠加旧的右手 1–3 kg 事件。不额外增加碰撞几何。

负载分布已经确定为纯独立均匀采样。对每个真实训练 world i：

```text
(m_left_hand, m_right_hand, m_left_shin, m_right_shin)_i ~ U(0,4)^4 kg
```

第一阶段 A 和第二阶段 PPO 的全部真实训练 world 使用这个分布。baseline 与 latent
组采用相同采样规则，并在配对训练 seed 下使用相同的负载样本。第一阶段 B 保持
全部 nominal，只承担反事实表征监督，不构成额外的 nominal A 训练子集。
全零、全四和固定不对称组合只用于物理审计及额外评估，不混入训练采样或 replay。

关闭其余质量、质心、摩擦、armature、增益、随机 joint offset、随机 action
delay/smoothing、推力等 DR。建议同时关闭随机观测噪声，保留 checkpoint 的观测项、
特征处理和归一化语义。motion/phase/reset 采样仍正常运行，两组规则完全一致。
原任务的 reward、执行器限制和 residual action bound 保持共同约定。
2026-09-08 用户修订：后续新批次的 residual PPO 统一关闭 `ee_body_pos` termination，
baseline、FiLM 和其他 residual 对照采用 `no_ee_body_pos` 训练配置；保留 `time_out`、
`anchor_pos` 和 `anchor_ori`。当前该项在四个手腕/脚踝任意一个的参考高度误差超过
0.5 m 时直接终止，训练中去掉它以检验策略能否学习偏离后的恢复。
默认独立评估继续使用原 tracker 的完整终止标准，保证与历史及冻结 tracker 对照可比；
不能从旧失败结果中直接扣除该项来推断取消终止后的成功率。
历史 `limb_context_20260907` / `limb_context_20260907_adaptive1000` 实验矩阵
通过根目录 `training_terminations.json` 固定为 `original`；现按用户要求归档并终止，
未完成种子不参与完整配对结论。新实验由调度器统一传入
`--training-terminations no_ee_body_pos`；独立新训练也默认此配置，普通续训继承
checkpoint 的配置，拒绝在同一实验中静默改动。实际终止项写入 config/checkpoint、
每轮 JSONL/W&B 和完成记录，启动时核对各 rank 的 termination manager。

2026-09-08 后续明确授权：新批次 `limb_context_20260908_no_ee_4gpu` 使用完整 8 GPU，
baseline 占 0–3、FiLM 占 4–7，每卡 8192 env。两组 residual actor/critic 从头训练，
保留已冻结的 stage1 encoder；仍为配对 seeds 121/122/123、每组 5000 PPO updates，
uniform 1000 后 adaptive 4000。新批次只运行 baseline/FiLM。卡数翻倍导致每轮样本量
翻倍，旧/新 latent 优势的差异是描述性对照，不能全部归因于取消终止。

## 代码中需要先处理的比较偏差

1. 当前 `residual_policy_train.py` 的 latent 组使用 compact 输入
   `64 + 71 + 71 + 29 = 235`，no-latent 组使用 1645 维 tracker feature。
   本次统一两组的非 latent 输入为同一套 tracker feature。
2. 当前标准 residual PPO 的 critic 没有 context latent 路径，必须补上。
3. 标准 forward/residual 入口的 payload 仍是单 body 配置；其他 preview 路径已有
   hands-shins profile。需要复用共同物理实现，避免两阶段各定义一套随机化。
4. 现有 nominal-physics 开关禁止与 payload 组合，不能直接把两个 CLI flag 拼起来。
   需要显式表达“nominal 基底物理 + 仅四肢 payload”。
5. 现有 forward fixed probe 来自 replay，不足以单独作为独立收敛验证。
6. payload 参数校验需要接受 0 kg。全零负载 world 的物理字段应与 compiled nominal
   一致，不能保留非零负载的质心或惯量变化。
7. ABC 的逐 tensor 审计记录原 critic 输入为 6330 维；本次按实际 checkpoint 读取
   输入维度，不沿用前期讨论中的 6331 维估计，也不额外添加一个标量观测。

## 第一阶段（建议）

- batch A 全部 world 的四处负载独立 U(0,4)，其余参数全部 nominal，与 PPO 同分布。
  A 不保留旧实现中的额外 nominal 子集；需让 nominal_fraction=0，并调整旧入口
  固定 nominal_fraction=0.5 的校验及依赖该混合比例的诊断代码。
- batch B 为 nominal 对照，复制 A 的起点并重放完全相同的五步实际 PD targets。
- nominal B 仅监督 A-B 响应表征；predictor 的预测目标仍然是 A 的实际轨迹。
- 延续 100 帧 causal context 和 5 步预测，latent 维度保持 64，不加入真实质量监督。
- 先短测重载下有效历史覆盖、reset 边界和 nominal A-B 配对误差；避免大量早期跌倒
  导致完整 context 样本不足而没有被发现。
- 使用独立的负载、起始 phase 和轨迹窗口做固定验证，不参与 replay 优化及统计量拟合。
  每 100 update 验证一次；建议连续 1,000 update 验证预测误差改善不足 1% 时视为
  平台期，同时检查表征诊断稳定；仅作诊断，不自动停训。学习率沿原曲线衰减至 1e-5 后保持。
- 主要收敛指标在独立 U(0,4)^4 验证环境上计算。nominal 预测误差和 nominal A-B
  配对误差来自独立的审计/评估 rollout，不向训练 A 或 replay 增加 nominal 样本。
  训练 A 不含专设 nominal world 时，不能继续对空 nominal 子集取均值并输出 NaN。
- 同时报告 nominal/DR 五步 NMSE、latent-response correlation、打乱 latent 的预测
  误差比、有效历史比例及 nominal A-B 误差。预测误差低和 latent 被利用需要分别验证。
- 固定每个 update 的 optimizer step 数，并保持有效全局 batch 不随卡数扩大。
- 按验证结果选择 checkpoint，第二阶段共同使用该 checkpoint 并冻结 encoder 及其
  输入归一化。PPO 只训练控制网络、价值网络和条件分支。

## latent 融合方案（待讨论，推荐 FiLM）

64 维不自动意味着信息容量不足。现有 encoder 输出已经经过 LayerNorm；主要需要
验证的是控制网络是否愿意、是否能够有效使用这些信息。

推荐保留 64 维，通过 actor/critic 各自独立的 `64 -> 256` 条件分支，产生各隐藏层
的缩放与偏置。沿用 encoder 自身已经训练好的输出 LayerNorm，不将 latent 与高维
观测合在一起重新做按样本归一化。当前建议允许隐藏特征最多 ±50% 的条件缩放，
条件头零初始化使实际初始调制为零；该边界是待实测的实现选择。

```text
z = stop_gradient(context_encoder(history))
h_l = 原观测分支的第 l 层隐藏特征
h'_l = (1 + 0.5 * tanh(gamma_l(z))) * h_l + 0.5 * tanh(beta_l(z))
```

critic 的调制置于原 LayerNorm 之后。条件输出以恒等变换初始化；actor residual
输出仍零初始化，两组初始 action mean/std 一致；残差 actor 与 critic 主干均从头训练，配对 seed 使用相同随机权重。
调制固定系数不初始化成与条件头同时为零的可学习门，避免分支梯度被锁死。
同一个 z 提供给两者，但条件分支不共享参数。
actor 的条件化作用于可训练的 residual MLP；原 tracker 及其 1645 维特征处理继续冻结。
critic 保留 checkpoint 的原非 latent 输入结构，权重重新随机初始化，不加载旧 critic 参数或统计。
critic 归一化从当前 DR 初始观测的跨卡统计开始累计；残差 actor 的 action std 从统一的 0.25 开始。
当前 ABC 审计报告该输入为 6330 维，最终以 checkpoint 实测为准。
新条件分支以恒等变换接入；所有组一起重置可训练 actor/critic，初始公共主干和归一化须通过配对哈希检查。

“独立”指 actor 与 critic 各有自己的条件 MLP 参数；它们读取同一个冻结 encoder
输出的 z。critic 条件 MLP 只产生隐藏层的缩放和偏置，不单独估计另一个 value。
例如第一个 1024 维隐藏层由 z 产生 1024 个 gain 和 1024 个 bias，逐通道调整后
继续经过原 critic 的后续层，最终仍输出一个 V(o,z)。critic 条件 MLP 和主干一起
由正常 PPO value loss 更新；actor 的条件 MLP 由 actor 目标更新。不同 world 共用
整套参数，只是输入 z 不同。

这一结构给 latent 更直接的作用路径，不能保证 PPO 必然使用它。FiLM 的原始方法
是条件信息驱动的逐特征仿射变换；在此任务上的收益仍需实验检验：
https://arxiv.org/abs/1709.07871

## 八卡实验矩阵（建议）

| 组别 | actor/critic 额外输入方式 | PPO seeds | 用途 |
| --- | --- | --- | --- |
| B | 无 latent，原 checkpoint 信息 | 3 | 主 baseline |
| F | 64 维 learned latent，多层 FiLM | 3 | 主实验 |
| C | 64 维 learned latent，直接输入拼接 | 1 | 融合方式诊断 |
| K | 与 F 相同结构，latent 固定为常量 | 1 | 无环境信息的结构对照 |

B/F 使用三个配对 seed；C/K 首轮单 seed 只用于诊断，不用于宣称跨 seed 稳健优势。
K 是同结构对照，不应称为有效表达能力完全相等的对照。

实验重点已经确定为 latent 对整体性能的增益。用户不要求分离 actor/critic 的贡献，
因此不再安排仅 actor / 仅 critic 的训练分组。当前推荐保留上述 3+3+1+1 主比较矩阵。

2026-09-08 最新修订：第一阶段持续使用 GPU 0–3，不设步数上限。只有用户明确说可以进入
第二阶段，且已保存、停止第一阶段并锁定选用的 context checkpoint，才放行第二阶段。
第二阶段每个 B/F/C/K 任务使用两 GPU，每 GPU 8192 environments，全局 16384 environments、
24 steps/update，即每轮 393216 transitions。B/K 固定用 GPU 0、1，F/C 固定用 GPU 2、3；
训练与评估都禁止使用 GPU 4–7。先完成 2×8192 / 3-update 短测，再启动正式配对训练。
新结果目录为 `runs/limb_context_20260907/ppo_2gpu8192_scratch`；旧单卡、旧四卡 warm-critic
试跑单独保留，不进入正式比较。梯度、critic VecNorm 和 rollout advantage 统计跨 rank 同步；
环境种子为 `seed + 1000003 * rank`，全目录按 `[rank::2]` 分片。
用短测估计结束时间，以至少 5000 次完成的 PPO update 为结束条件，不用 checkpoint
文件名代替实际计数。所有 checkpoint 记录配置、数据清单及来源模型标识。

## 主结论与辅助诊断

1. **表征质量**：独立负载上的预测精度和依赖 latent 的预测收益。真实质量只可用于
   离线 probe/分桶诊断，不作为新增 actor/critic 输入或 encoder 监督目标。
2. **策略收益**：B/F 在相同 motion/start/load 条件下比较学习曲线、最终固定评估
   reward、tracking error、失败率和有效时长覆盖；保留三个训练 seed 的单独结果。
   同时报告全零 nominal、最大负载和均匀分布样本，避免总体均值隐藏 nominal 退化。
   与 ABC 一样同时报告新失败和救回片段，不单独用失败截断后的平均误差判断优劣。
3. **整体 latent 利用的辅助诊断**：同一策略使用正确、跨负载替换、常量 latent 的闭环评估；
   检查负载匹配错误是否损害实际表现。替换尽量匹配 motion/phase/历史有效长度，
   避免把纯粹的输入分布变化解释为识别负载的收益。
4. **可选的低成本 critic 诊断**：固定 actor 和同批 trajectory/return target，仅替换 critic
   的 latent，比较 value prediction 误差、explained variance 和 GAE 变化。
   critic 不直接决定推理动作，不能用“仅替换 critic 后 rollout reward”证明其作用。
5. **结论范围**：主结论是相同协议下加入 latent 的整体性能增益，不分别归因 actor
   与 critic 的训练贡献；辅助诊断不增加单独的 PPO 训练分组。

额外固定评估包括四处全 0、1、2、3、4 kg 和不对称组合。按左手、右手、
左小腿、右小腿排序，重点配对 `(4,4,2,2)` 与 `(2,2,4,4)`，以及 `(4,2,4,2)`
与 `(2,4,2,4)`。每组总新增质量均为 12 kg，有助于区分平均重载补偿与针对负载分布
的补偿；仍需结合 latent 替换测试，不能单凭这些组的分数证明表征被使用。
另加入同总重 8 kg 的 `(4,4,0,0)` 与 `(0,0,4,4)`，检查跨 nominal/重载部位的补偿。
主训练使用全数据目录，因此
这些评估证明的是新负载/新起点上的表现，不等同于未见 motion 泛化。
记录 residual 饱和率与有效 context 覆盖，并检查 PPO 访问状态相对第一阶段的变化。

## 执行权限状态与先前诊断

用户现已将本会话切换为 Full access / approval never。普通命令已成功执行，
无需逐次升级审批；本轮也已直接读回本文档。项目工作不再被先前的 bwrap 故障阻挡。
这表示当前会话不再使用失败的沙箱路径，不代表底层容器的挂载隔离问题已修复。
用户限定只改当前项目的约束继续有效，未修改全局 Codex 配置或容器设置。

以下保留先前诊断，供追溯：

本次默认执行命令在启动前报 `bwrap: Failed to make / slave: Permission denied`。
容器使用 `cri-containerd.apparmor.d (enforce)`；独立 unshare 测试也报 root filesystem
propagation permission denied。项目被配置为 trusted，并不能解决该挂载限制。

本机版本为 codex-cli 0.153.4。已检查 `use_legacy_landlock` 兼容后端；对显式 workspace
权限 profile 的试验失败，报：`permission profiles requiring direct runtime enforcement
are incompatible with --use-legacy-landlock`。该试验在执行探针前退出。

故障发生时工具收到的是托管的 workspace-write / restricted 配置。普通项目文件
修改不能刷新运行中的托管执行策略；本次由用户更改会话权限后恢复正常执行。

官方参考：
- https://learn.chatgpt.com/docs/sandboxing
- https://learn.chatgpt.com/docs/permissions
- https://learn.chatgpt.com/docs/config-file/config-reference
