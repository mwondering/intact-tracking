# Heavy residual：从 2000 checkpoint 切换为 uniform 续训

日期：2026-09-21。当前实验版本：`resume2000_uniform_mass5x`。

根据用户要求，两组原 adaptive 训练已正常保存并停止：latent 完成 4528 次更新，baseline 完成 5296 次更新。新的两组训练均重新加载各自原始 `ppo_4gpu8192_aux108/checkpoint_2000.pt`，仅将 motion 采样切换为 uniform，并继续沿用已经确认的 mass-only 辅助监督设置。原 checkpoint 文件采用从 0 开始的编号，其内部 `completed_updates=2001`，恢复后的第一次更新为 2002。

<!-- LIVE_STATUS_BEGIN -->
最新状态（2026-09-22）：baseline/vanilla 已按用户要求在 **12136** 次更新后正常保存并停止，四 rank 权重一致、最终 checkpoint 及 ONNX/JSON 验证通过，GPU 4–7 已释放，拟用于 [RMA student 蒸馏](heavy_rma_student_plan_20260922.md)。Latent 继续运行。2026-09-21 12:12:21 UTC 的原启动验证见 [正式训练验证记录](../runs/144000-exp-heavy-residual-comparison/resume2000_uniform_mass5x/train_verification.json)，当时 latent/baseline 分别完成 2010/2017 次更新。
<!-- LIVE_STATUS_END -->

| 配置 | Latent | Baseline |
|---|---|---|
| GPU | 0–3 | 4–7 |
| 每卡环境数 / 每组总环境数 | 8192 / 32768 | 8192 / 32768 |
| Actor latent 输入 | 冻结 heavy encoder 的输出 | 全零，网络结构相同 |
| Motion sampling | uniform | uniform |
| Failure rewind | 关闭 | 关闭 |
| 旧 adaptive 统计 | 丢弃，不加载 | 丢弃，不加载 |
| DR 辅助损失总系数 | 0.5 | 0 |
| 有效监督维数 | 8 | 0 |
| 更新次数上限 | 无，等待用户停止 | 无，等待用户停止 |
| W&B name | `144000-exp-heavy-residual-latent-uniform` | `144000-exp-heavy-residual-baseline-uniform` |

W&B project 为 `intact-preview-v2`，group 为 `144000-exp-heavy`。使用独立 run ID，保留旧 adaptive 分支的训练记录：

- [Latent uniform](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/heavy-residual-latent-uniform-10466328a8)
- [Baseline uniform](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/heavy-residual-baseline-uniform-0c9e488aae)

## 采样与恢复语义

Uniform 在每个 rank 的 motion 分片内均匀选 motion，再均匀选择允许的起始帧。关闭 adaptive 分支和失败回放；旧访问次数、失败计数、EMA 与 rewind 历史不继承。恢复不依赖旧 checkpoint 旁的 sampling 文件，新 checkpoint 中 `motion_sampling_state=None`。保留在配置里的 adaptive 超参数不参与 uniform 采样。

模型、预测头、优化器、观测归一化和学习到的 action std 均从 2000 checkpoint 完整恢复；命令行中的 `initial-action-std=1.0` 不会覆盖恢复后的 std。冻结 tracker 仍为原 144000 checkpoint，context encoder 仍为 heavy 训练的 `update_008816.pt`。

数据集保持 `/data_zcy/wxy/motion_data_correct` 下完整数据，扫描 224651 条、排除不可执行的 4171 条，最终 220480 条，每 rank 55120 条。Manifest SHA-256：`59b8e336c152e4773133bcdcd86cf86fd6545c3c0e6cc00ce67912d107c740ac`。

物理环境不变：10% nominal、90% HDR，四肢负载仍采用 256 档组合分配；双手各 0–2.5 kg、双小腿各 0–4 kg。负载 COM 在局部坐标系内 xyz 独立采样于 ±5 cm，并更新合成质量、质心和惯量。关闭负载 COM 监督不影响它的物理随机化。完整 DR 范围见 [NDR/HDR 表格](dr_ranges_ndr_hdr.md)。

PPO 保持 rollout 24、epochs 5、mini-batches 4、actor LR 1e-4、critic LR 5e-4、entropy 0.005、FP32、seed 121；residual 输出不限幅、scale=1。预测头仍输出 108 维，只监督 torso COM xyz、摩擦系数、四肢附加质量共 8 维；baseline 辅助损失为 0。

## 预测头实测

检查的是 residual actor 共享隐藏层上的现有 DR 预测头，直接读取 checkpoint 权重，不重新拟合。测试创建 512 个新 HDR 世界，256 个负载质量组合各 2 个世界，seed 20260922；使用已有对比评测的 256 条 motion，uniform motion 采样，各 checkpoint 使用自身确定性策略生成历史。

先 warmup 500 步，再运行 256 步、每 4 步采样一次。仅统计短期与长期历史完整的样本，各世界等权：4528 checkpoint 共 22065 个有效预测，2000 checkpoint 共 21949 个有效预测，两者均覆盖全部 512 个世界。没有裁剪预测输出，评测前后 actor 状态完全一致。

下表对比共同的 8 个监督量。MAE 使用物理单位；R² 为 1 减去各世界等权的逐步预测 MSE 与目标跨世界方差之比，**不是**对世界平均预测计算的相关系数。

| 预测量 | 2000 MAE | 停止前 4528 MAE | 2000 R² | 停止前 4528 R² |
|---|---:|---:|---:|---:|
| Torso COM x | 0.928 cm | 0.681 cm | 0.913 | 0.957 |
| Torso COM y | 1.034 cm | 0.801 cm | 0.899 | 0.941 |
| Torso COM z | 1.820 cm | 1.673 cm | 0.704 | 0.752 |
| 摩擦系数 | 0.219 | 0.165 | 0.691 | 0.828 |
| 左手附加质量 | 0.235 kg | 0.199 kg | 0.820 | 0.875 |
| 右手附加质量 | 0.233 kg | 0.231 kg | 0.823 | 0.839 |
| 左小腿附加质量 | 0.568 kg | 0.487 kg | 0.605 | 0.702 |
| 右小腿附加质量 | 0.560 kg | 0.487 kg | 0.636 | 0.711 |

停止前预测头已经能区分负载大小，但小腿质量比手部质量误差更大。在 3–4 kg 档，左小腿平均真实质量 3.518 kg、预测 3.185 kg，低估 0.333 kg；右小腿平均真实质量 3.500 kg、预测 3.199 kg，低估 0.301 kg。逐步绝对误差的 95 分位分别约 1.27、1.28 kg。摩擦预测存在约 +0.123 的平均偏差。

新 uniform 训练的预测头也恢复至 **2000 checkpoint**，并未沿用 4528 权重。因此上表右侧描述的是停止前旧分支，左侧描述新的恢复起点。两次评测采用相同物理采样与协议，但各策略产生的历史不同；这是一组新物理世界上的初测，不是留出 motion 或多随机种子的泛化结论。预测质量改善也不能单独解释策略跟踪效果，uniform 对 HDR 跟踪的影响需要后续 checkpoint 的配对评测。

原始结果与图：

- [4528 完整指标](../runs/144000-exp-heavy-residual-comparison/resume2000_uniform_mass5x/prediction_head_stopped_latent_u4528/summary.json)
- [2000 恢复起点完整指标](../runs/144000-exp-heavy-residual-comparison/resume2000_uniform_mass5x/prediction_head_resume_parent_u2001/summary.json)
- [4528 散点图 PNG](../runs/144000-exp-heavy-residual-comparison/resume2000_uniform_mass5x/prediction_head_stopped_latent_u4528/prediction_scatter.png) / [PDF](../runs/144000-exp-heavy-residual-comparison/resume2000_uniform_mass5x/prediction_head_stopped_latent_u4528/prediction_scatter.pdf)
- [4528 逐步预测数据](../runs/144000-exp-heavy-residual-comparison/resume2000_uniform_mass5x/prediction_head_stopped_latent_u4528/predictions.npz)
- [负载质量分档校准](../runs/144000-exp-heavy-residual-comparison/resume2000_uniform_mass5x/prediction_head_stopped_latent_u4528/payload_mass_calibration.json)

散点图每个点为一个世界的平均预测；图中 R² 与表格一致，采用逐步误差，不是仅对散点计算。

## 验证与复现

相关 39 项单元测试通过。两组分别完成真实 4-rank × 8192 环境的恢复 smoke test，各运行 3 次更新；验证恢复后的模型、优化器、归一化状态与父 checkpoint 一致，四 rank 参数一致，日志实际为 uniform、无 rewind，且不存在 adaptive 状态文件。恢复保存的 checkpoint 包含冻结 tracker 与 context encoder。

- [单元测试记录](../runs/144000-exp-heavy-residual-comparison/resume2000_uniform_mass5x/unit_tests.json)
- [Smoke 验证](../runs/144000-exp-heavy-residual-comparison/resume2000_uniform_mass5x/smoke_verification.json)
- [旧分支停止记录](../runs/144000-exp-heavy-residual-comparison/resume2000_uniform_mass5x/stopped_previous_training.json)

启动入口为 [resume_144000_heavy_uniform.py](../scripts/resume_144000_heavy_uniform.py)，核心标志为 `--motion-sampling uniform --resume-uniform-sampling --allow-dr-aux-change`；使用显式 uniform 迁移，不使用 `--reset-adaptive-sampling`。

```bash
.venv/bin/python scripts/resume_144000_heavy_uniform.py --phase smoke --arm both
.venv/bin/python scripts/verify_heavy_uniform_resume.py --phase smoke
.venv/bin/python scripts/resume_144000_heavy_uniform.py --phase train --arm both
.venv/bin/python scripts/verify_heavy_uniform_resume.py --phase train
```

这些输出目录已经使用，启动器会拒绝覆盖。实际训练输出分别位于 `runs/144000-exp-heavy-residual-{latent,baseline}/ppo_4gpu8192_resume2000_uniform_mass5x`；同级 `latest_training_process.json` 指向新进程，监控与 ONNX 自动导出均已随新训练启动。
