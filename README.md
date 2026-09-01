# INTACT Tracking

本仓库的正式路径是纯在线训练：使用仓库内的 MJLab 环境与冻结的 SPV5-2 tracker
持续执行 rollout，transition 直接进入内存 causal replay；一旦凑齐完整训练 batch，立即
更新 context-conditioned INTACT。过程中不生成或读取 `manifest.json`。环境构造、MDP
terms、策略结构、G1 MJCF 与 mesh 均随本包安装；运行时不需要另一个源码仓库或额外的
`PYTHONPATH`。

实现保留 INTACT 的训练骨架：一个 observation Encoder、一个 LeWM-style Forward
Predictor、一个同时服务 physical/goal 两条分支的四槽 INTACT Predictor，以及 Forward
MSE、SIGReg、physical NLL、goal NLL 四项联合目标。

## 安装

需要 Python 3.10–3.13、支持当前 MJLab 的 NVIDIA 驱动，以及 `uv`：

```bash
uv sync --extra dev
```

`mjlab==1.5.0`、MuJoCo、RSL-RL、Torch 和 TensorDict 都是本项目声明的依赖。
checkpoint 与 motion 数据是运行输入，不随仓库分发。

## Notation 与固定 context

控制频率为 50 Hz。环境动作记为 `u_t ∈ R^29`。actor 始终只预测当前一步
`u_t`；`E=5` 步动作序列只作为 Forward 的条件：

```text
U_t = [u_t, ..., u_t+E-1] ∈ R^(5×29)
```

默认训练 window 包含 `H=5` 个相邻的 effect-span transition。核心变量为：

```text
z_t           = Encoder(o_t, w)
z_robot,t+E   = Encoder(o_robot,t+E, w)
z_ref,t+E     = Encoder(o_ref,t+E, w)

m_physical = z_robot,t+E - z_t
m_goal     = stopgrad(z_ref,t+E) - z_t
```

两种 intent 进入同一个 actor，四槽 grammar 不变：

```text
[z_t, m_t, z_t ⊙ m_t, A_prev(u_t-1)] → p(u_t | z_t, m_t, u_t-1)
```

Forward 则显式读取完整 `U_t`，预测 `z_robot,t+E`。因此未来第 5 步 endpoint 由 5 个
真实动作共同解释，而 physical/goal NLL 的 label 都只有当前 `u_t`。默认
`sample_stride=1`，相邻 window 每个环境步平移一次，长期来看每个动作都会成为单步 label。

`w_t` 由固定 16 个 interaction token 编码，并通过共享 FiLM 调制 `z`，不会成为
第五个 actor slot：

```text
κ_i = [p_i, u_i:i+4, p_i+5]
C_t = [κ_t^1, ..., κ_t^16]
w_t = ContextEncoder(C_t)
```

每个 token 默认覆盖 5 个控制步，即 100 ms；16 个有效 token 对应 1.6 s 的历史交互证据。
正式训练不允许填充缺失 context。

完整训练框图、时间索引与所有可调节点见
[docs/training_flow.md](docs/training_flow.md)。

## Rollout 字段语义

核心 JEPA observation 是机器人侧和 reference 侧语义一致的 64 维量：

```text
o = [joint_pos(29), joint_vel(29), projected_gravity(3), base_ang_vel(3)]
```

机器人侧来自部署 observation stream；reference 侧由已知参考轨迹构造。122 维
`proprio` 只用于 interaction context：

```text
p = [joint_pos(29), joint_vel(29), projected_gravity(3), base_ang_vel(3),
     previous_action(29), joint_torque(29)]
```

每条 transition 同时记录动作前后两端。`robot_state/reference_state` 各 71 维，是
rollout 时可用的 simulator/reference 完整状态：

```text
[root_pos(3), root_quat(4), root_lin_vel(3), root_ang_vel(3),
 joint_pos(29), joint_vel(29)]
```

这两个 raw state 不要求部署可得，也不进入当前核心 INTACT 训练；它们仅供数据审计和
后续 probe 使用。完整字段定义见 `src/intact_tracking/data/schema.py`。

## 纯在线 rollout 契约

正式训练启动时，每个 vector slot 对应一个固定 physics world：

- 仅保留 checkpoint 中 `mode=startup` 的 DR event；它们在首条 rollout 前确定每个 slot 的
  DR 参数。
- 整个训练进程不再调用 startup DR，因此 reset 和 motion 切换都不会改变该 slot 的物理参数。
- MJLab class-based event 的 reset callback 在初始化 reset 后被禁用，避免其绕过 event mode
  在后续 episode reset 中重采样参数。
- motion command 在初始化和 episode reset 时随机采样 motion；机器人同步重置到 reference。
- `auto_reset=True`。终止的 slot 独立 reset，其他 slot 的 causal history 不受影响；boundary
  transition 不进入训练 window，因此在线训练不依赖 terminal observation。
- 各 slot 的初始 timeout phase 默认独立随机化；只排除 teleport boundary，reset 后状态仍可
  作为新 query 的起点，同一 physics world 的 context 不清空。
- tracker 权重严格加载后执行 `requires_grad_(False)` 和 `eval()`；优化器只持有 INTACT 参数。

每个 rank 的 replay、rolling transition history、context ring 和 running sufficient
statistics 都常驻该 rank 的 GPU；rollout 主路径不再把 vector batch 搬到 CPU，也不再逐环境
执行 Python sample construction。query 必须位于一个连续 episode 内；16 个 context token
只从同一固定 physics world 的 query 之前选取，可以跨越早先 episode。默认
`effect_steps=5`、`query_transitions=5`、`context_chunk_steps=5` 时，首个合法样本最早出现在
每个 world 的 `16 × 5 + 5 × 5 = 105` 个环境步之后；不使用 padded context。达到该条件后，
`sample_stride=1` 会使每个有效环境每推进一步就产生一个新 window。

## 正式在线训练

```bash
./scripts/run_training.sh \
  /path/to/checkpoint.pt \
  /path/to/motion_directory \
  /path/to/runs/intact_online \
  --num-envs 4096 \
  --warmup-steps 120 \
  --updates 100000 \
  --rollout-steps-per-update 1 \
  --gradient-steps-per-update 8 \
  --batch-size 512 \
  --replay-capacity 8192 \
  --effect-steps 5 \
  --query-transitions 5 \
  --context-chunk-steps 5 \
  --sample-stride 1
```

第二个位置参数也可以是单个 motion `.npz`。脚本默认使用仓库 `.venv` 和 `cuda:0`；
可通过 `PYTHON_BIN=/path/to/python` 与 `DEVICE=cuda:1` 指定解释器和单张 GPU。

单机多卡使用 `GPUS` 指定物理 GPU：

```bash
GPUS=0,2,3 ./scripts/run_training.sh \
  /path/to/checkpoint.pt \
  /path/to/motion_directory \
  /path/to/runs/intact_online_ddp \
  --num-envs 16 \
  --batch-size 64 \
  --updates 10000
```

脚本会通过 `torchrun` 为每张可见 GPU 启动一个进程。每个 rank 在自己的 GPU 上运行冻结
tracker、固定 DR vector worlds 和本地 causal replay；完整 INTACT 模型经 DDP 同步全部可训练
参数的梯度。16 个 context token 始终来自同一个 rank 内的同一个固定 physics world，不会跨
world 或跨 rank 拼接。在线 observation/proprio/action 的 sufficient statistics 每轮经
all-reduce 合并，因此所有 rank 使用同一份全局 normalization。

`--replay-capacity` 同时决定每张卡上 GPU sample ring 的容量。默认维度下，
`num-envs=4096, replay-capacity=8192` 的 rolling history、context 和 replay 合计约占
489 MiB/卡；`run_config.json` 会记录启动前估算值，训练日志记录实际分配字节数。

`--num-envs`、`--batch-size`、`--replay-capacity` 都是**每个 rank**的值。例如
`GPUS=0,1 --num-envs 16 --batch-size 64` 表示全局 32 个环境、每个 optimizer step 的全局
batch 为 128。DDP 对各 rank 梯度取平均，learning rate 不会随卡数自动放大。motion 目录由
各 rank 分片加载；如果传入单个 motion 文件，则每个 rank 使用相同 motion、不同固定 DR
world。全局 `world_id = rank × num_envs + local_env_id`，不会冲突。

训练调度如下：每个 rank 先至少采集 `warmup-steps`；如果任一 rank 的合法 replay sample
尚不足本地 `batch-size`，所有 rank 就继续 rollout，直到全部就绪或触及
`max-warmup-steps`。第一次同步优化随即发生。
此后每轮先新增 `rollout-steps-per-update` 个 vector-environment step，再执行
`gradient-steps-per-update` 个 mini-batch 更新。因而：

- `--batch-size` 控制每个 rank 每次梯度更新抽取的 causal window 数；
- `--updates` 控制 rollout/update 轮数；
- 总 optimizer step 数为 `updates × gradient-steps-per-update`；
- `W` 张卡的一次环境步产生 `W × num-envs` 条 transition。

忽略 reset，在默认 `sample_stride=1` 下，每张卡一个 rollout step 也会产生约
`num-envs` 个新 window。因而 `num-envs=4096`、`rollout-steps-per-update=1`、
`batch-size=512`、`gradient-steps-per-update=8` 时，每卡每轮约新增 4096 个 window，同时
抽取 `512 × 8 = 4096` 个 window。这个调度比与 Forward 的 `effect_steps=5` 无关。

默认 logger 不依赖 W&B。终端按 `log-interval` 打印一条 JSON，包含 `loss`、
`forward_loss`、`sigreg_loss`、`action_loss`、`physical_nll`、`goal_nll`、两项 MAE、
Forward NMSE、相对 copy-state baseline 的误差比例、预测/目标 latent cosine similarity、
预测 endpoint 与真实 endpoint 经同一 action actor 解码后的 action consistency error、
原始环境 action 单位下的 MAE/RMSE、latent scale/collapse、四项加权 loss contribution、Gaussian log-std、
gradient mean/max/p95/clip fraction、learning rate、env/optimizer step、replay size、reset 与
motion 计数。每条记录还在 `window` 中保存最近 `metric-window` 轮的均值和标准差。
终端和文件中的 loss 是所有 rank 的均值，transition/replay/reset 等计数是全局和；只有 rank 0
打印、记录和保存 checkpoint。完整逐轮记录写入 `metrics.jsonl`，并生成：

- `run_config.json`：训练、tracker、motion、固定 DR 和在线 replay 契约；
- `normalization.json`：截至 checkpoint 的在线 running statistics；
- `update_XXXXXX.pt`、`last.pt`：INTACT、优化器、scheduler 与在线进度；
- `history.json`：所有 update 的结构化记录；
- `train.log`：正式脚本捕获的完整终端输出。

## 五步 residual policy 训练

新的 residual 入口保留 frozen SPV5-2 tracker 作为 skill prior，Residual Policy 根据
`16-token world context + tracker deploy-time policy feature` 输出动作修正。连续五步真实
observation 重新计算当前 policy action，Causal Forward Predictor 预测相对当前状态的五个
非链式 pose delta，并重建未来绝对 pose。Forward 与 tracking 均只计算 root position、root
orientation 和 joint position，不预测或优化未来 velocity；tracking loss 只通过 action
Jacobian 更新 Residual Policy，Forward/Backward loss 则更新 Context Encoder 和对应 predictor。

```bash
./scripts/run_residual_training.sh \
  /path/to/checkpoint.pt \
  /path/to/motion_directory \
  /path/to/runs/residual_v1 \
  --num-envs 4096 \
  --batch-size 512 \
  --wandb-project intact-residual-tracking
```

W&B 默认开启，并记录训练 loss、predictor 分组误差、residual action、梯度、replay 状态以及
SPTracking 同名的八项真实 rollout tracking error。Warmup 的零 residual rollout 作为 frozen
tracker baseline，日志同时给出当前误差相对 baseline 的 ratio 和 improvement。无网络机器可
使用 `--wandb-mode offline`，完全禁用则传 `--no-wandb`。完整张量契约、梯度路由和指标说明见
[Residual training flow](docs/residual_training_flow.md)。

在线 rollout 默认随机打散每个 vector slot 的初始 episode timeout phase，但保留所有 reset 后
的真实 transition：teleport boundary 本身不进入五步窗口，reset 后状态可以作为窗口起点，
Context 不随 episode reset 清空。

## 可选离线导出

离线 collector 仍保留用于数据审计、复现实验或 probe，不是正式训练的前置步骤。它根据
checkpoint 中保存的 `cfg` 重建环境和 SPV5-2 actor，并严格加载 actor 权重：

```bash
uv run intact-tracking-collect \
  --checkpoint-file /path/to/checkpoint.pt \
  --motion-path /path/to/motion_directory \
  --output-dir /path/to/rollouts/run_000 \
  --num-envs 16 \
  --transitions 1000000 \
  --shard-size 100000 \
  --world-session-steps 3000 \
  --device cuda:0
```

单个 motion 文件可改用 `--motion-file /path/to/motion.npz`。当前内置推理运行时要求
checkpoint 的 actor 为 `SPV52HeightContactEstimatorActor`，机器人 asset 为
`tracking_bfm_g1` 或 `tracking_bfm_spv1_g1`；不满足时会显式拒绝，而不会静默加载成
其他结构。

默认采集协议：

- 保留 checkpoint 中的 startup domain randomization；同一 world session 内物理参数不变。
- 移除 `step/interval` 扰动；如需 disturbance 数据，显式添加 `--include-disturbances`。
- `auto_reset=False`，先保存真实 terminal observation，再手动 reset。
- 任一 vector slot 结束时同步 reset 全部 slot，避免未结束 slot 的 observation history 被额外推进。
- 每 3000 个控制步重新采样 startup DR，并分配新的 `world_id`。
- manifest 保存 checkpoint SHA-256、task id、MJLab 版本、采集配置和重置协议。

分片是 mmap-readable 的 NumPy column store。若确实需要旧的离线训练，可运行
`uv run intact-tracking-train-offline --manifest ...`；该兼容入口按 physics world 拆分数据，
但不再是 `intact-tracking-train` 或 `scripts/run_training.sh` 的默认行为。

## 端到端 smoke test

最小 smoke 在同一进程内启动真实 MJLab、冻结 tracker、在线采样，然后以
`batch-size=1` 执行一次完整 INTACT optimizer step。它使用正式模型宽度、Forward、
SIGReg、共享四槽 actor 和完整 16-token context：

```bash
./scripts/run_smoke_test.sh \
  /path/to/checkpoint.pt \
  /path/to/motion_directory
```

输出写入唯一的 `runs/smoke.XXXXXX/`。设备和输出根目录可覆盖：

```bash
DEVICE=cuda:1 OUTPUT_ROOT=/data/smoke \
./scripts/run_smoke_test.sh /path/to/checkpoint.pt /path/to/motions
```

双卡 smoke 使用同一入口：

```bash
GPUS=0,1 OUTPUT_ROOT=/data/smoke \
./scripts/run_smoke_test.sh /path/to/checkpoint.pt /path/to/motions
```

脚本会校验 tracker 冻结、startup DR 在 rollout 期间不再采样的运行契约、随机 motion 已参与 rollout、
至少 105 个因果环境步、固定 16-token context、有限 loss、精确一个 optimizer step、在线
normalization 及非空 `last.pt`。

## 开发验证

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv build
```

当前测试覆盖共享四槽 actor、goal endpoint stop-gradient、physical successor attached
gradient、固定 16-token contract、单步 Direct action、跨 shard episode 索引、离线与
在线的因果 same-world context、跨 episode context、raw-zero previous action、replay 容量、
逐步滑动采样、全局 normalization 合并、DDP 梯度同步和 world-disjoint split。真实 checkpoint 的端到端
验证由在线 smoke 脚本完成。

## 当前边界

当前完成“冻结 tracker 在线 rollout → 内存 causal replay → 即时 INTACT 联合更新”的闭环。
当前架构版本为 `single_step_effect_v1`；旧版 145 维 action-block actor checkpoint 与新的
29 维单步 actor 不兼容，Stage-I 需要重新训练。
在线训练没有独立 validation split；需要泛化评估时，应另启固定 DR seeds/worlds 的只读评估。
当前 checkpoint 不保存 replay 内容，因此尚不支持 bit-exact 中断续训。Stage II RL action
head 尚未加入；在正确 context 相对 no/wrong/shuffled context 的收益通过验证前，不进入
Stage II。当前正式 launcher 支持单机多卡；多节点 torchrun 尚未作为受支持配置验证。
