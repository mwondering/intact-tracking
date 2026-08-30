# INTACT Tracking

本仓库提供一个完整闭环：使用仓库内的 MJLab 环境与冻结的 SPV5-2 tracker
执行 rollout，生成可移植训练数据，再训练 context-conditioned INTACT。环境构造、MDP
terms、策略结构、G1 MJCF 与 mesh 均随本包安装；运行时不需要另一个代码仓库或额外的
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

控制频率为 50 Hz。环境动作记为 `u_s ∈ R^29`，一个模型 action block 包含
`B=5` 个环境动作：

```text
a_t = [u_Bt, ..., u_Bt+B-1] ∈ R^145
```

默认训练窗口包含 `H=5` 个 action block。核心变量为：

```text
z_t       = E(o_t, w_t)
z_t+1     = E(o_t+1, w_t)
z_g       = E(o_g, w_t)

m_local   = z_t+1 - z_t
m_goal    = sg(z_g) - z_t
```

两种 intent 进入同一个 actor，四槽 grammar 不变：

```text
[z_t, m_t, z_t ⊙ m_t, A(a_t-1)] → p(a_t | z_t, m_t, a_t-1)
```

`w_t` 由固定 16 个 interaction token 编码，并通过共享 FiLM 调制 `z`，不会成为
第五个 actor slot：

```text
κ_i = [p_i, a_i, p_i+1]
C_t = [κ_t^1, ..., κ_t^16]
w_t = ContextEncoder(C_t)
```

每个 token 覆盖 5 个控制步，即 100 ms；16 个有效 token 对应 1.6 s 的历史交互证据。
正式训练不允许填充缺失 context。

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

## 采集 rollout

采集器根据 checkpoint 中保存的 `cfg` 重建环境和 SPV5-2 actor，并严格加载 actor
权重。只需要显式给出 checkpoint、motion 数据和输出目录：

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

分片是 mmap-readable 的 NumPy column store。训练至少需要三个不同 `world_id`，数据集
按 world 而不是 clip 随机拆分。

## 正式训练

```bash
./scripts/run_training.sh \
  /path/to/rollouts/run_000/manifest.json \
  /path/to/runs/intact_tracking_e5 \
  --epochs 5 \
  --batch-size 256 \
  --workers 4
```

脚本默认使用仓库 `.venv` 和 `cuda:0`；可通过 `PYTHON_BIN=/path/to/python` 与
`DEVICE=cuda:1` 覆盖。其余参数直接转发给训练 CLI。输出目录必须为空，脚本完整遍历
每个 epoch，并拒绝 smoke batch 上限与 padded context。默认保持 `B=5`、`H=5`、
固定 16-token context、INTACT 模型和联合损失配置。

训练统计量只使用 train worlds。缺失的 previous action 先在原始动作空间补零，再做
z-score。输出目录包含：

- `run_config.json`：训练架构、超参数和 world split；
- `normalization.json`：仅由 train worlds 估计的统计量；
- `epoch_XXX.pt`、`last.pt`：模型与优化器 checkpoint；
- `history.json`：Forward、physical、goal 与 SIGReg 指标。

## 端到端 smoke test

最小 smoke 使用三个独立静态 world 采集 360 条 transition，随后以 `batch-size=1`
执行一次完整 INTACT optimizer step 和一次 validation batch。它使用正式模型宽度、
Forward、SIGReg、共享四槽 actor和完整 16-token context：

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

脚本会校验 transition/world 数量、固定 16-token contract、有限 loss、精确的一个训练
batch、一个验证 batch 及非空 `last.pt`。

## 开发验证

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv build
```

当前测试覆盖共享四槽 actor、goal endpoint stop-gradient、physical successor attached
gradient、固定 16-token contract、Direct recurrent plan、跨 shard episode 索引、因果
same-world context、raw-zero previous action 和 world-disjoint split。真实 checkpoint 的
端到端验证由 smoke 脚本完成。

## 当前边界

当前完成“仓库内 rollout → causal window → INTACT 联合训练”的闭环。大规模数据采集与
正式训练需在目标算力上执行；Stage II RL action head 尚未加入。在正确 context 相对
no/wrong/shuffled context 的收益通过验证前，不进入 Stage II。
