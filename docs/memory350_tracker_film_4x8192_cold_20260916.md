# Tracker FiLM：每卡 8192 环境、零预热

2026-09-16，按用户要求将 baseline 和 latent 两组从每卡 1024 增至 8192 个环境，并取消进入 PPO 前的 500 步冻结 tracker 预热。两组使用相同 seed 和全新 FiLM/critic/优化器从头训练，保留原 tracker 和 context u35857 的冻结权重。此前短程训练分别在 baseline u145 / latent u159 完成保存并正常停止，原 checkpoint、日志和 W&B 记录保留。

## 当前配置

| 项目 | Baseline | Latent |
|---|---|---|
| GPU | 0–3 | 4–7 |
| 每卡环境 | 8192 | 8192 |
| 每组总环境 | 32768 | 32768 |
| 训练前预热 | 0 步 | 0 步 |
| Actor FiLM 条件 | 压缩 obs＋64 维零槽 | 压缩 obs＋64 维真实 latent |
| Critic 输入 | 压缩 obs＋64 维零槽 | 压缩 obs＋64 维真实 latent |
| W&B | [tracker-film-278d667f9885](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/tracker-film-278d667f9885) | [tracker-film-11604356cc97](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/tracker-film-11604356cc97) |

网络沿用[前一版实现](memory350_tracker_film_20260916.md)：actor 的观测压缩为 1645→512→256→128，拼接 64 维 latent 后产生 FiLM，调制冻结 tracker 的 512/256/128 隐藏层；critic 独立使用 6330→1024→512→256→128，拼接后接 192→256→128→1。Actor 和 critic 无参数共享。

Context 固定为 `runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/update_035857.pt`，SHA256：`3c489cf5457889bf88441175ec4f10f4448104f44f7914eaaf762a9c3659d8f7`。

完整 motion 数据集、uniform 采样、seed 121、rollout 24、5 epochs、4 mini-batches、actor LR 1e-4、critic LR 5e-4、FP32 均保持一致。两组每轮各采集 786,432 个 transition，训练无轮数上限，每 250 轮保存。手部各 0–2.5 kg、小腿各 0–4 kg、wrist pitch/yaw ±10 Nm，原奖励和全部 termination 保留。

## 零预热的实现和验证

入口 `--tracker-warmup-steps` 默认改为 0；runner 在该值为 0 时跳过整个 tracker 预热函数，不执行预热环境步。Memory350 从短期/长期有效历史都为 0 的状态开始，在正式 PPO 交互中逐步积累。空历史仍可以经过 encoder 得到默认 latent，历史是否为空由有效步/块数判断。

14 项相关单元测试通过。单卡真实仿真使用 8192 环境、零预热及与正式任务相同的 rollout/epoch/mini-batch 配置，完成 4 次 PPO 更新。初始短期有效步和长期块数均为 0，随后历史正常增长；全部模型参数有限，原 tracker 的 53 个权重/统计张量逐元素不变，新增观测压缩分支有实际更新。此预检不构成控制性能结论。

运行目录：`runs/limb_context_20260916_tracker_film_obs_latent_4x8192_cold/`。`protocol.json`、`source_snapshot/`、`preflight.json`、`previous_stopped.json` 保留配置、源码、预检和旧任务停止证据。`launch.py` 启动两个独立 torchrun；`audit_startup.py` 检查正式规模、冷启动历史、冻结 checkpoint、两组初始权重、PPO 指标和 W&B 服务端状态，并记录 `startup_audit.json`。

正式启动审计已通过：两组实际加载完整 129,827 条 motion / 48,085,337 帧，各 4×8192 环境，初始短期与长期有效历史均为 0，actor/critic 初始参数摘要相同。W&B 服务端均为 running，已收到训练指标。2026-09-16 04:00:20 UTC 检查时 baseline / latent 分别到 u11 / u12；早期最近 5 轮约 7.45 s / 7.07 s，显存分别约 38.8–38.9 GiB / 40.2–40.3 GiB 每卡。原任务已停止，新任务无轮数上限，未修改学习率、epochs 或 mini-batch 数。
