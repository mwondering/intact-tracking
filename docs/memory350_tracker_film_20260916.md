# 冻结 tracker 内部 FiLM：压缩观测与环境 latent 对照

2026-09-16。实现当前讨论确定的结构；不把 FiLM 放在新增 residual action 网络中。主实验比较整套 actor/critic 是否使用环境 latent，不将结果单独归因于 actor。

后续设置已更新：本页的 4×1024、500 步预热任务已在 baseline u145 / latent u159 保存停止。当前任务改为每卡 8192 个环境、零预热，两组从相同初始化重新开始；见 [4×8192 冷启动记录](memory350_tracker_film_4x8192_cold_20260916.md)。

## 两组的最终定义

| 部分 | latent 组 | baseline（constant） |
|---|---|---|
| Actor FiLM 的观测分支 | 可训练 1645→512→256→128 | 完全相同 |
| Actor FiLM 的环境输入 | 冻结 context encoder 的 64 维 latent | 固定 64 维零向量 |
| FiLM 条件网络 | 拼接得到 192→128，再生成各层 gain/bias | 完全相同 |
| Critic 的观测分支 | 可训练 6330→1024→512→256→128 | 完全相同 |
| Critic 融合 | 128 维压缩观测＋64 维 latent | 128 维压缩观测＋64 维零向量 |
| Critic 输出网络 | 192→256→128→1，普通 MLP | 完全相同 |

Baseline 的 FiLM 仍然读取当前观测，因此能学习随 motion、phase 和状态变化的调制。它不再是只从固定输入生成一组全局常数。两组参数维度、初始权重与训练协议一致；baseline 的 latent 输入列接收零值。Actor 和 critic 的观测编码器、其余参数均完全独立。

两组均运行冻结的 context 推理与相同 Memory350 历史维护，以保持采样流程及计算开销一致。Baseline 的 actor 和 critic 均不读取真实 latent 数值；保留推理仅用于相同执行条件与诊断。这里“无 latent”指控制和价值网络没有获得该输入；原 critic 的 privileged observations 继续保留。

## Actor 与梯度路径

实际 checkpoint 的 tracker 控制 MLP：1645→2048→2048→1024→1024→512→256→128→29。

在 512、256、128 三个隐藏层的激活后施加：

`h' = (1 + delta_gamma(compressed_obs, z)) * h + beta(compressed_obs, z)`。

原 tracker 的控制权重、感知/参考编码器及全部归一化统计冻结。FiLM 新增的观测压缩、条件网络、gain/bias 输出头与 Gaussian 探索标准差可训练。FiLM 输出头初始化为零，动作均值精确等于原 tracker。Gain/bias 采用线性输出，没有额外 tanh 限幅；最终动作为调制后的 tracker 均值加 Gaussian 探索，没有外接 residual 网络。

第一次 FiLM 之前的 tracker 计算不记录梯度；第一次 FiLM 之后保留计算图，冻结的 Linear 负责向前面的 FiLM 传递梯度。新观测压缩分支单独保持梯度。不能把整个 tracker 前向包进 no_grad，也不能 detach 调制后的动作。

Context 输出先 detach，再 L2 归一化；actor/critic 均使用同一处理。FiLM baseline 在进入网络前直接生成零槽，不通过真实 latent 乘零来屏蔽。Critic baseline 只读取自己的观测，压缩完成后由 head 的固定槽补零。Critic 从头训练，全量更新，不使用 FiLM。

## 运行协议

沿用当前手部各 0–2.5 kg、小腿各 0–4 kg 加原 tracker DR；uniform motion，原奖励与末端高度 termination，wrist pitch/yaw ±10 Nm，原 action scale/PD 与执行器力矩限制。此首轮实验的冻结 tracker 预热为 500 步，后续入口默认已改为 0；长期 Memory350 跨 episode 保留现有规则。默认训练无总轮数上限，`--iterations` 可显式限制累计完成 update 数。

训练入口：`intact_tracking.cli.memory350_tracker_film_train`。

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m intact_tracking.cli.memory350_tracker_film_train \
  --conditioning latent --context-checkpoint /path/to/context.pt \
  --output-dir runs/tracker_film_latent --training-ranks 4 --num-envs 1024
```

Baseline 使用 `--conditioning constant` 与独立输出目录，其余参数一致。通过 CUDA_VISIBLE_DEVICES 指定各组四卡。`--resume` 恢复模型、优化器与统计，禁止跨组续训。`--wandb-mode offline` 用于本地验证，在线模式使用所配置的 project/entity。

评测入口：`intact_tracking.cli.memory350_tracker_film_eval`，支持 warm/cold、完整随机 DR 或现有八组 DR bank、真实/零/配对交换 latent，以及全局 body/root 指标。不传 `--checkpoint` 时评测原冻结 tracker，作为额外参考。历史 residual saturation 指标在这里标为不适用。

## 实现与验证

- 模型：[memory350_tracker_film.py](../src/intact_tracking/memory350_tracker_film.py)
- 训练：[memory350_tracker_film_train.py](../src/intact_tracking/cli/memory350_tracker_film_train.py)
- 评测：[memory350_tracker_film_eval.py](../src/intact_tracking/cli/memory350_tracker_film_eval.py)
- 单元测试：[test_memory350_tracker_film.py](../tests/test_memory350_tracker_film.py)
- 实测产物：`runs/limb_context_20260916_tracker_film_implementation_smoke/`

14 项新增单元测试通过；与压缩 policy、物理协议和精度配置相关的合计 33 项测试通过。覆盖恒等初始化、冻结主干中的梯度传递、压缩观测参与 FiLM 学习、baseline 不读取 latent、真实 latent 对动作的影响、PPO Gaussian 概率与 minibatch 重排、critic 的压缩后拼接，以及 checkpoint 严格恢复。

真实仿真预检已完成：两组各进行 2 次 PPO 更新；latent 组从 u2 恢复到 u3，模型、Adam 状态及归一化统计恢复摘要一致。两个 rank 使用最终选定的 context u35857 完成 2 次 PPO 更新，actor/critic 参数和 critic normalizer 的跨 rank 摘要一致。两组共 53 个原 tracker 权重/统计张量保持逐元素不变；新增观测压缩和 FiLM 参数均有实际更新。Actor 可训练参数 1,262,877，critic 可训练参数 7,254,401。

两组初始可训练权重完全相同。各自仿真产生的 critic 经验归一化统计不要求逐位相同；共同 tracker 预热采用同一协议与 seed，GPU 轨迹不保证逐位一致。Baseline 更新后 critic 的 latent 输入列仍为零。真实、baseline、原 tracker 三种评测入口均完成 32 步 cold 预检并输出全局 body/root 指标。短程预检不构成控制性能结论。证据：`constant_audit.json`、`latent_audit.json`、`resume_audit.json`、`ddp_latent/completion.json`、`verification.json`。

第一次实测在模型构建、350 步预热与初始化审计后，因旧日志包装器强制 online、使用了默认凭据而遇到 W&B 权限错误，中断日志保留在 `latent.log`。新增入口提供并尊重 `--wandb-mode`，后续预检显式 offline。正式 launcher 沿用既有实验凭据文件，通过环境变量传递且不写入实验记录。

## 正式四卡对四卡实验

用户授权停止 predictor 后，八 rank 在 u35857 完成更新并保存，正常退出。固定 `runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/update_035857.pt`，SHA256 为 `3c489cf5457889bf88441175ec4f10f4448104f44f7914eaaf762a9c3659d8f7`；此后不随文件 `last.pt` 改变。

运行目录：`runs/limb_context_20260916_tracker_film_obs_latent_4x1024/`。

| 组别 | GPU | 每卡环境 | Actor / critic latent | W&B |
|---|---|---:|---|---|
| baseline | 0,1,2,3 | 1024 | 均为 64 维零槽 | [tracker-film-2043327d59a3](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/tracker-film-2043327d59a3) |
| latent | 4,5,6,7 | 1024 | 均为真实 64 维 latent | [tracker-film-c809ef7d8e71](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/tracker-film-c809ef7d8e71) |

两组各 4096 个环境，完整 `/data_zcy/wxy/motion_data_correct/motion_data_full` 数据集按 rank 分片，uniform motion。相同 seed 121、rollout 24、5 epochs、4 mini-batches、actor LR 1e-4、critic LR 5e-4、FP32、500 步原 tracker 预热；每轮各 98,304 个 transition。均从恒等 FiLM 与全新 critic 开始，训练无轮数上限，每 250 轮保存。

启动协议、源码快照、进程身份以及 predictor 停止记录保存在运行目录。`launch.py` 使用分开的 torchrun 会话启动两组，`baseline/metrics.jsonl` 与 `latent/metrics.jsonl` 为逐轮本地记录。正式运行检查另存 `startup_audit.json`。

启动检查已通过：两组均实际加载 129,827 条 motion、48,085,337 帧，并进入 PPO。审计时 baseline 完成 6 轮、latent 完成 7 轮，最近 5 轮采样＋更新耗时分别约 3.19 s / 2.99 s；此处仅为早期速度。两组 W&B 服务端状态均为 running，均已收到第 3 轮及 294,912 个 transition 的指标。随后本地检查已分别到 13 / 15 轮；baseline 的 latent 置零/打乱动作差为 0，latent 组动作已有非零敏感性，这仅验证输入链路，不代表性能改善。
