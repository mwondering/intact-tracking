# Residual PPO 的 ONNX 部署

> **最新定稿 Pipeline 的部署格式（2026-09-20）。** 最终模型的 ONNX、JSON 与 runtime 已导出并通过数值校验；固定版本位置与配置见 [定稿文档](final_pipeline_20260920.md)。

导出入口为 `intact_tracking.cli.memory350_policy_export`。它将冻结 tracker、完整 context encoder（含归一化）和 residual 动作网络导出到一个 `policy.onnx`，同时生成 SP_Tracking 风格的 `policy.json` 与 `deploy_metadata.json`。DR 辅助预测头只用于训练，不参与动作推理。

```bash
uv pip install --python .venv/bin/python -e '.[deploy]'
.venv/bin/python -m intact_tracking.cli.memory350_policy_export \
  --checkpoint /path/to/checkpoint_5700.pt \
  --output-dir /path/to/new_export_directory
```

输出目录须为新目录。导出无需启动仿真或读取动作数据；完整格式 checkpoint 不依赖原始 tracker/context 权重文件。旧 checkpoint 可使用其记录的原始文件路径。导出器检查 ONNX 结构并用 ONNX Runtime 验证不同历史状态，全部通过后才发布目录。

## 文件与自动导出

每个版本的部署目录包含：

- `policy.onnx`：全部动作推理权重，ONNX opset 18，固定 batch=1，FP32。
- `policy.json`：SP 基础字段、模型哈希、输入布局、历史协议、关节参数、精度说明与数值校验结果。
- `deploy_metadata.json`：同一份部署元数据，保留原 SP 文件命名习惯。
- `policy_runtime.py`：独立运行时，只依赖 NumPy 和 ONNX Runtime。

本次训练期间由 CPU 后台进程 `scripts/watch_memory350_onnx.py` 自动导出最新完整 checkpoint。周期版本位于 `ppo_8gpu8192/deploy/checkpoint_<iteration>/`；最终版本为 `deploy/checkpoint_final/`，对应 6008 次更新。`deploy/latest` 原子切换到最近通过验证的版本，目前指向最终版本；导出进程已完成并退出。训练目录中的 `policy.onnx`、`policy.json`、`deploy_metadata.json`、`policy_runtime.py` 是指向最新版本的入口。

部署复制时建议复制整个版本目录，保证 ONNX 与 JSON 属于同一次导出。运行时校验 ONNX 哈希，并先解析固定版本路径，避免加载时碰上 latest 更新。后台状态和事件记录位于当前 run 的 `monitor/onnx_export/`。

`scripts/run_144000_residual.py --phase train ...` 后续启动正式训练时默认启动该导出进程，可用 `--no-export-onnx` 关闭；smoke 不启动。直接使用底层训练 CLI 时，可独立运行导出命令或 watcher。导出进程不占用 GPU、不改变 PPO，也不会因导出失败停止训练。

## 部署调用

将版本目录整体复制到部署端，安装 `numpy`、`onnxruntime`，然后：

```python
from policy_runtime import Memory350Policy

policy = Memory350Policy('/path/to/export', threads=1)

# 每个控制周期调用一次。tracker_obs 保留原 SPV5-2 的 8199 维原始观测布局。
action = policy.step(
    tracker_obs,
    reset_boundary=episode_or_motion_changed,
    parameters_changed=False,
    command_applied=previous_command_actually_sent,  # 首次调用可不传
)

# nominal PD target；延迟、平滑等由部署控制器负责。
target_joint_pos = default_joint_pos + action_scale * action
```

`action` 为 29 维确定性 policy command，即 tracker mean + residual mean，不是关节角或力矩。当前 unbounded residual 不施加 tanh 或限幅。若部署控制器修改了上一步命令，`command_applied` 应传实际发送的 policy command，处于 PD scale/offset/delay 之前的单位；不传时使用上一步 ONNX 输出。

部署方沿用原 tracker 的传感器、参考轨迹和 FK 观测构造。不要再次归一化输入：模型已包含 tracker 和 context 的归一化。proprio 从 50 帧 term-major 历史中逐项提取最新帧，不能简单截取历史最后 122 个值。

## JSON 与图接口

保留原 SP 的 `format=motion_tracking_sim2real_policy`、`run_name`、`iteration`、`checkpoint`、`in_keys`、`out_keys`、`in_shapes`、`num_actions` 等字段。额外的 context 历史使接口扩展为：

| 接口 | 形状 | 说明 |
| --- | --- | --- |
| `memory350_residual_observation` | `[1,104085]` | 单个 float32 扁平输入，字段偏移见 JSON 的 `input_layout` |
| `action` | `[1,29]` | 最终确定性动作 |
| `context_latent` | `[1,64]` | 本周期 latent，运行时用于更新历史 |

扁平输入依次包含原 tracker 观测 8199、短历史 `[50,273]`、短历史 mask `[50]`、长期 chunk `[30,10,273]`、长期 mask `[30]`、前四帧 latent `[4,64]`。一个 interaction 是 `(proprio122_before, command29, proprio122_after)`。只提供已经完成的交互；当前动作对应的 next-state 不能进入输入。

原 tracker 的四块观测依次为 `robot_root_quat(4)`、`estimator_history(6100)`、`reference_encoder_input(1900)`、`robot_key_body(195)`。新增历史输入和 `context_latent` 输出需要部署端支持；配套 runtime 已实现，原有仅接受 8199 维 tracker 输入的客户端需要接入它。

JSON 还包含 `joint_names`、`joint_stiffness`、`joint_damping`、`default_joint_pos`、`action_scale`、`anchor_body_name`、`body_names`、`control_dt`。参数来自 checkpoint 对应的编译后 nominal 机器人模型，未施加训练 DR。

## 历史与重置

短历史保留最近 50 个已完成交互，溢出的旧交互每 10 个组成一个 chunk，保留最后 30 个完整 chunk。不足 10 个的 pending 尾部暂不送入编码器；历史按时间从旧到新、左侧补零，mask 标记有效部分。

episode/motion 边界时不记录跨边界交互；将旧历史中的完整 chunk 归档，丢弃不足一个 chunk 的尾部，清空短历史及 latent 历史，保留完整长期 chunk。`parameters_changed=True` 表示开始新的物理 session，会清空所有历史。首次调用以空历史编码，使用模型计算出的 latent，而不是强行令 latent 为零。

## 精度与验证

ONNX 为 FP32，使用标准展开的 attention。训练期间 CUDA 的 BF16 context inference 以及 CUDA fused Transformer 快速路径存在数值差异，不保证与 FP32 ONNX 逐位一致。验证采用 FP32 actor/context，并关闭参考 PyTorch 的 fused MHA fastpath；不会改变正式训练的计算设置。

每次导出检查空历史、部分历史、完整短/长期历史以及重置后的历史组合，比较原始 PyTorch actor/context、导出模块和 ONNX Runtime，容差 `atol=rtol=2e-4`，实际误差逐项记录在 `policy.json` 中。

持续推理验证脚本为 `scripts/verify_memory350_onnx.py`：在真实仿真中让 ONNX 输出驱动一个 world，逐步与 PyTorch 比较，检查重置、长期历史、PD 元数据，并在禁止导入 PyTorch/仿真库的独立进程中重放记录。历史维护另有 1600 步逐元素测试，对照训练端 `InteractionMemory`，覆盖边界、缓存覆盖和物理 session 改变。

本次真实 `checkpoint_5500.pt` 验证通过 420 步、3 次重置，长期历史填满 30 个 chunk；动作最大绝对误差 `6.68e-6`，latent 最大绝对误差 `6.62e-6`，PD/动作元数据与环境一致，独立进程重放通过。记录位于当前 run 的 `diagnostics/onnx_deployment_validation_20260920/verification.json`。

首轮静态校验在本机单线程 CPU 上约 12–14 ms/步。该值不含传感器/参考观测准备和控制器开销，目标部署硬件的端到端时延应单独测量。
