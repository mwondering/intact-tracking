# 多源时序 Transformer 与原始观测 MLP

2026-09-16。实现用户确认的 29-token residual actor，并按后续要求让 latent 组 critic
也采用独立 Transformer。Baseline 使用原有 1645 维观测的普通 MLP，不输入 latent、
额外 tracker 动作或新增的短历史。此对照比较完整控制方案，不能单独归因于 latent。

## 网络与输入

| 输入 | 每帧维度 | 帧数 | 编码器 |
|---|---:|---:|---|
| 本体感觉、身体状态、高度/接触估计 P | 320 | 5 | 320→256→128 |
| 当时的当前参考 R | 269 | 5 | 269→256→128 |
| 当前跟踪误差 E | 260 | 5 | 260→256→128 |
| 当时的环境 latent Z | 64 | 5 | 单位归一化，64→128 |
| 冻结 tracker 原始确定性动作 A | 29 | 5 | 29→128 |
| 当前已知的未来参考 F | 77 | 4 | 77→128 |

序列为 `[P,R,E,Z,A] × 5 + [F(t+1),...,F(t+4)]`，共 29 个 token。
同类型投影跨帧共享，类型间不共享。时间嵌入覆盖 -4…+4，另有 token 类型嵌入。
每个样本只有过去/当前真实信息及当前已知的未来参考，窗口内允许联合 self-attention。
两层 Pre-LayerNorm Transformer，宽度 128，4 heads，FFN 256，GELU，dropout=0。

现有 tracker 的 577 维参考包恰好拆成 `269 + 4×77`。源数组是按观测项排列的
term-major 历史，必须按字段取当前帧，不能直接 reshape 1645。当前身体参考有 195 维，
保留在 R；四个 F 使用原输入已提供的目标量，没有加入额外未来实际状态/身体观测。
特征保留冻结 tracker 的原归一化和空间坐标约定；A 保留策略动作顺序和原始单位。

Actor 从更新后的当前 P token 读出 128 维，拼接当前 raw tracker action 29 维，
经 `157→256→128→29` 输出 residual。输出层零初始化，线性无限幅。
最终动作均值为 `frozen_tracker_mean + residual`，Gaussian std 初始 0.25。

Latent critic 的 token 编码器、attention、head 均与 actor 完全独立。
默认 `--critic-observation privileged`：原 6330 维 critic 观测经
`6330→1024→512→256→128` 压成一个 critic 专用 token，形成 30-token 序列。
该设置保留原价值网络可用的特权信息；`--critic-observation common` 则使用与 actor
完全相同的 29 个 token。读出头为 `157→256→128→1`。

Baseline actor 为 `1645→512→256→128→29`，critic 为
`6330→1024→512→256→128→1`，均为普通 ELU MLP；不加载 context encoder。
两组 actor 都保留冻结 tracker 的基础动作，并从零 residual 开始。

## 时间、冻结与 PPO 约定

- 环境 wrapper 在决策前计算一次冻结 tracker 特征及动作，保存该时刻真实在线 latent。
- 存储的是原始冻结特征/latent/action，不缓存可训练投影或 Transformer 的激活。
- 5 帧窗口、4 帧目标、mask 与样本一起进入 RolloutStorage。PPO 打乱 minibatch 时仍
  使用对应窗口；每次更新重新运行所有可训练投影与 attention。
- 同一时刻反复读取只替换当前帧，不推进窗口。旧观测快照不会被后续缓存写入覆盖。
- Episode、motion 切换或不连续 motion 时间清空该 world 的短窗口，并 mask 不足帧。
  外部显式重置可调用 `clear_policy_history(env_ids)`。
- 冻结 encoder 的 memory350 长记忆仍遵循原协议：固定 DR world 内跨 episode 保留。
  控制器的 5 帧短窗口与 encoder 的长期记忆是两个独立层次。
- 训练从冷记忆直接开始，没有 tracker/PPO 预热。Policy FP32，冻结 context 沿用 BF16。
- checkpoint 恢复模型、优化器、normalizer、计数；模拟器及控制短窗口重新开始。
- 评测 zero/paired-swap 干预整段 5 帧 latent，不能只替换当前 `dynamics_latent` 字段。

## 训练条件与入口

正式启动时用户要求所有环境随机采样，因此默认 `--anchor-fraction 0`：nominal 背景，
每个 world 的四肢负载全部独立连续均匀采样，不保留固定网格环境。手部最大 2.5 kg，
小腿最大 4 kg，wrist pitch/yaw 10 Nm；同一 world 的物理参数跨 episode 保持固定。
可通过 `--dr-profile tracker_dr_plus_limb_payload` 选择原 tracker DR 加均匀负载；
该模式忽略负载网格比例。
两组 motion 始终 uniform，保留原末端 termination，默认每卡 8192 环境、4 卡、无限轮数。
数据源默认使用现有完整 motion_data_full；实现验证只使用一条 motion。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node 4 -m intact_tracking.cli.memory350_token_policy_train \
  --architecture mlp --output-dir runs/limb_context_20260916_temporal29_transformer/baseline \
  --anchor-fraction 0 --training-ranks 4 --num-envs 8192 --until-user-stop

CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node 4 -m intact_tracking.cli.memory350_token_policy_train \
  --architecture transformer --output-dir runs/limb_context_20260916_temporal29_transformer/latent \
  --context-checkpoint runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/update_035857.pt \
  --critic-observation privileged --anchor-fraction 0 \
  --training-ranks 4 --num-envs 8192 --until-user-stop
```

正式启动脚本为 `scripts/run_memory350_temporal29_comparison.py`，包含 W&B、缓存目录和
完整训练参数；实际命令与源码摘要记录在运行根目录的 `baseline_process.json` 和
`latent_process.json`。

按用户要求，旧 `limb_context_20260916_payload256_top5_moe` 八卡训练已通过 SIGTERM
在完整更新边界保存退出：baseline u563、latent u542；各自 `checkpoint_interrupted.pt`
可恢复，四卡参数一致性审计均通过。新实验从零 residual、全新 critic 和优化器启动，
baseline 占 GPU 0–3，latent 占 GPU 4–7，各 32768 环境，不设更新轮数上限。
每轮每卡 24 步，5 epochs、4 minibatches，actor LR 1e-4、critic LR 5e-4，entropy 2e-4；
policy FP32，每 250 轮保存一次。两组 seed 121，rank seed 为 `121 + 1000003 * rank`，
物理分布、motion 数据/采样、reward、termination 与动作无限幅约定一致。

运行目录：`runs/limb_context_20260916_temporal29_transformer/`。
W&B：[baseline](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/temporal29-a537727fe983)、
[latent](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/temporal29-7a2e032aed62-retry1)。

首次正式运行中 latent 完成 u1 后，在下一轮 Warp CUDA graph launch 发生 OOM，
FP32 PPO 的完整前向/反向本身成功。失败记录归档在
`failed_latent_attempt01_warp_oom/`，其 W&B ID 为 `temporal29-7a2e032aed62`。
处理方式是 Transformer 每次完整 PPO 更新及诊断结束后调用 `torch.cuda.empty_cache()`，
把闲置 attention workspace 交还给 CUDA，以供独立的 Warp allocator 申请；同时记录
释放前后 reserved/allocated/free 显存及释放耗时。没有改变 env 数、精度、网络、batch、
loss、梯度、优化器步数或学习率。Latent 以独立 W&B retry1 从零重启，baseline 连续运行。

正式规模启动验证通过：记录时 baseline 已完成 u117、latent u17，所有已记录 loss
均有限。Latent 的 long-memory 30 个 chunk 已在全部 world 填满，跨 PPO/仿真边界继续
正常运行，重启日志没有 OOM。最近 5 轮平均实际墙钟耗时 baseline 4.77 秒、latent
12.50 秒；latent 每轮平均约 5.63 秒采样、6.67 秒 PPO 更新。
u17 的 PyTorch reserved 峰值约 73.37 GiB，更新后释放约 50.62 GiB，释放耗时约
0.099 秒；释放后设备空闲约 53.30 GiB。这一处理解决的是两个 allocator 间的显存
交接，未引入 PPO microbatch 或梯度累计变更。

四个 rank 的运行时物理审计均确认 `anchor_worlds=0`、`continuous_worlds=8192`，
手部/小腿负载上限 2.5/4 kg、四个 wrist pitch/yaw 扭矩范围均为 [-10,10] Nm。
两组全部物理运行时审计、数据 manifest、reward、termination 和 PPO 超参数一致。
完整数据包含 AMASS_LAFAN_Qingtong 42 条、sonic_filtered 129785 条，总 129827 条，
48085337 帧；数据清单 SHA256：
`df31b45ef2a32dbda7107c57f55bb90b2d300a182bcc05ef62c19e1959d85978`。

机器可读验证位于运行根目录：`launch_verification.json`、
`configuration_verification.json`、`wandb_upload_verification.json`。
其中说明 `motion_sampling.scope` 继承了旧版 U(0,4) 的描述字符串；实际负载范围以
`physics.residual_physics_contract` 和逐 rank 运行时审计为准，手部确实是 2.5 kg 上限。

恢复时保留架构、critic 输入、DR 分布及训练规模，增加
`--resume <checkpoint_update_XXXXXX.pt>`；`--iterations` 表示目标累计完成轮数。

评测入口为 `python -m intact_tracking.cli.memory350_token_policy_eval`，支持
`--checkpoint`、`--motion-manifest`、`--fixed-masses`、`--global-metrics`，以及
`--latent-mode correct|zero|paired-swap`。paired-swap 使用 `--repeats 2 --paired-starts`。

## 实现验证

验证产物：`runs/limb_context_20260916_temporal29_implementation_check/`。

- 新增测试覆盖 577 维参考无丢失拆分、term-major 当前帧提取、时间/重置与不可变快照、
  无效帧屏蔽、PPO storage 与 minibatch、六类投影及 Q/K/V 梯度、完整 latent 历史干预、
  tracker 冻结、初始动作一致、无限幅输出、独立 actor/critic、baseline 无 latent 信息路径。
- 初步真实训练：baseline 单卡 ×64 环境；latent 双卡 ×64 环境；各 2 轮，rollout 8、
  epochs 2、mini-batches 2。双卡梯度/参数一致性通过，全部 loss 有限。
- 两组从 u2 恢复至 u3，模型/优化器/normalizer 摘要恢复一致；双卡参数一致性再次通过。
- 四条闭环路径分别完成 4 个 episode ×80 步：baseline、正确 latent、整段 latent 置零、
  跨环境交换整段 latent。初始物理指纹、状态、motion 与起始帧全部一致；参考时间线和
  历史审计通过，paired-swap 实际执行 120 个 world-step。短测试覆盖率均为 100%。
- 47 个相关测试通过：13 个新增 token 测试，以及 34 个既有模型/PPO 测试。
- 机器可读验证与实现源码摘要见验证目录中的 `validation_summary.json`。
- Actor 可训练参数：baseline 1,010,746；latent 683,194。
  Critic：baseline 7,172,097；latent（含特权 token）7,851,649。
- 这些短测试验证实现与训练链路，不用于评价表征收益或收敛情况；每卡 8192 的容量与速度
  需要在正式训练规模上测量。
