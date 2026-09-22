# Residual PPO checkpoint 的冻结模型依赖

> **最新定稿 Pipeline 的 checkpoint 格式。** 最终 `checkpoint_final.pt` 已保存，实际完成 6008 次更新，内嵌依赖校验通过；权重与运行记录见 [定稿文档](final_pipeline_20260920.md)。

当前 proprio122/history5/tracker-action residual checkpoint 已可在标准推理入口中独立加载全部模型权重。冻结 context encoder、归一化统计、输入协议和 tracker 构建配置一同保存；tracker、residual 和 92 维 DR 解码头的权重仍位于原 actor state 中。

## 文件内容

| 字段 | 内容 |
| --- | --- |
| `actor_state_dict` / `policy` | 原冻结 tracker、residual、动作分布、92 维 DR 解码头 |
| `critic_state_dict` / `optimizer_state_dict` / `rsl_rl` | 原有 PPO 训练状态 |
| `frozen_context.payload.model` | 仅 `context_encoder.*` 权重 |
| `frozen_context.payload.model_config` | context encoder 架构配置 |
| `frozen_context.payload.normalization` | 原始归一化统计，包含 proprio122 与 29 维动作的均值/标准差 |
| `frozen_context.payload.context_input_contract` | noisy proprio122、control action、历史与 reset 输入协议 |
| `frozen_tracker.cfg` | 原 tracker 的环境和 actor 构建配置；权重复用 `actor_state_dict['tracker.*']` |
| `inference_bundle_version` | 当前为 `1` |

原始文件 SHA256 保留为来源标识，另存内嵌内容校验值。加载时验证内容、模型来源匹配及输入协议；损坏的完整包会报错，不会静默改用外部文件。旧格式 checkpoint 仍按原始路径加载。

只保存阶段一的编码器推理依赖，不保存阶段一 forward predictor 或其优化器。本次实际文件从 170,718,571 字节增加到 179,180,893 字节，增加约 **8.1 MiB**。

## 当前训练与后续训练

本次训练进程当时已经加载旧保存函数，因此使用 `scripts/embed_memory350_checkpoints.py` 在 CPU 上处理新保存的文件。自 `checkpoint_5200.pt` 起，保存完成后自动补齐内嵌依赖，最终 `checkpoint_final.pt` 和 `checkpoint_interrupted.pt` 也已完成。该补齐进程现已退出；运行时扫描间隔为 5 秒，加上读写校验时间，刚保存的文件可能短暂仍是旧格式。

处理过程保留所有原有字段，对原 actor、critic、优化器、更新计数、配置及自适应采样状态做整体校验，写入临时文件后验证并原子替换。未重启正式训练，也未修改模型训练方式、超参数或更新上限。

状态与最新已补齐文件记录在当前 run 的 `monitor/checkpoint_embedding/status.json`，每次校验记录在 `events.jsonl`。该进程绑定正式训练的 PID 和启动时刻，正式训练结束后处理剩余 checkpoint 并退出。

后续启动的 `memory350_policy_train` 及基于它的 proprio native 训练会直接保存完整格式，无需这个补齐进程。冻结依赖在初始化时缓存于 CPU，context encoder 不加入 PPO 优化器，仍保持冻结。

## 推理加载

标准入口 `intact_tracking.cli.memory350_proprio_native_policy_eval` 优先使用内嵌的 context encoder、tracker 权重及 tracker 配置，不再需要原来的阶段一权重文件和 tracker 权重文件。

```bash
.venv/bin/python -m intact_tracking.cli.memory350_proprio_native_policy_eval \
  --checkpoint /path/to/checkpoint_5300.pt \
  --motion-manifest /path/to/motions.txt \
  --output runs/portable_eval.json \
  --memory-start warm --warmup-steps 1000 --steps 500
```

代码中可使用 `load_policy_context(state, device=...)` 和 `embedded_tracker(state)` 加载内嵌依赖。后者返回 tracker 配置和权重，可通过 `tracker_state_dict` 传给 residual actor。

这里的自包含指模型权重与配置；运行仍需要兼容的仓库代码、仿真依赖、机器人资源和所选动作数据。当前训练启动/续训入口以及一些专用诊断脚本仍使用原来的来源路径参数，本次更新的独立加载入口是上述标准推理/评测入口。交互历史从运行时重新建立，checkpoint 不包含仿真现场或历史缓存。

## 验证

- 61 项相关测试通过，覆盖旧格式兼容、原始文件删除后加载、输出一致、错误包拒绝、原子写失败保护、并发替换保护及 runner 保存。
- 对真实 `checkpoint_5200.pt`：内嵌编码器与原编码器在相同 short/long 输入下的 latent 最大绝对差为 **0**，归一化统计完全一致，tracker 的 **53 个 state 项完全一致**，DR 解码头保持 `[92,128]`。
- 短仿真测试禁止读取原始 context/tracker 模型文件，并传入不存在的 tracker 路径：16 个环境、400 步冻结 tracker 历史、32 步 residual 推理成功，外部模型文件读取尝试为 0。该测试只验证可加载与可推理，不用于性能比较。

实际校验记录位于当前 run 的 `diagnostics/checkpoint_bundle_validation_20260920/verification.json`。
