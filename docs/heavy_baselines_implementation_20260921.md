# 144000-exp-heavy：RMA teacher / Any2Track 实施记录

2026-09-21。已实现两种 baseline 的训练、续训、纯 nominal / 纯 HDR 评测、完整 checkpoint 与 ONNX/JSON 导出。设计依据和共同训练参数见[原方案](heavy_rma_any2track_baseline_plan_20260921.md)。用户最新确认：两组均从零使用 **uniform 连续训练**，关闭 failure rewind，不再设置 2000 轮重启，不设更新上限。运行记录见 [uniform 训练记录](heavy_baselines_uniform_20260921.md)。

正式规模保持每组 **4 × 8192**。最初的功能验证使用单卡及四卡 × 64 环境；启动器先执行 4 × 8192 预检，通过才进入正式训练。当前实际运行状态以最新 uniform 训练记录为准。

## 实际网络与梯度

| 项目 | RMA teacher | Any2Track / AnyAdapter |
|---|---|---|
| 条件信息 | 当前真实 108D DR 参数，按范围映射到 [-1,1] | 79 帧历史，每帧含过去状态64与总 raw action29 |
| Encoder | actor、critic 各自独立 108→256→128→64 ELU | Conv1D 93→64(k9/s5)→64(k6/s3)，SiLU，展平256→128 |
| Encoder 优化 | actor PPO / critic value loss | 20 步自回归世界模型损失；PPO 阶段冻结 |
| 策略 | 1645 tracker 特征 + 64 embedding + 29 tracker action，外接 residual [512,256,128] | 冻结 tracker 的各层添加全秩线性 adapter，首层接 embedding，后续接融合隐藏层 |
| 最终动作 | tracker + 无界 residual | adapted tracker 直接输出，不再叠加 tracker |
| Actor 可训练参数，含 std | **1,127,418**，含 actor DR encoder | **8,301,114**，不含预测训练的 encoder |
| Critic 可训练参数 | **7,439,297**，含独立 DR encoder | **7,435,777** |
| History encoder / world model | 无 | **111,168 / 676,897** |
| 现有 Memory350 / DR 辅助头 | 不加载 / 不启用 | 不加载 / 不启用 |

两者保留原 SPV5-2A 归一化、reference encoder、estimator 和所有原 actor 权重。新增 residual 输出层 / adapter 全零初始化，因此初始确定性动作与 tracker 完全一致；高斯 std 统一为 1.0，entropy coefficient=0.005。RMA 的编码器在每个 PPO minibatch 内计算，物理输入可以 detach，编码器输出保留梯度。

Any2Track 的世界模型输入为 state65 + embedding128 + action29，隐藏层 `[512,512,256,256,256,128]`，输出33维增量。gyro / qdot 缩放0.05；通过 dt=0.02 积分 joint position 和 projected gravity。root height 只用于预测训练。loss 按坐标求和、按全部 transition 求平均，gyro/gravity/q/qdot/height 权重为 500/500/1/0.5/500。

每次采集24步后，各环境独立选择一个20步窗口，先执行世界模型5 epochs × 4 minibatches、Adam lr=1e-4，然后重算整份 rollout 的 embedding，再执行共同的 PPO 更新。行为 old log-prob / value 保持采集时的值。世界模型更新不增加环境交互步数。窗口 RNG 独立并保存到 checkpoint；四卡同步世界模型与 encoder 的梯度。

实现记录三处明确的移植处理：动作沿用 SP raw command；自回归历史按完整93维 `(s_t,a_t)` 更新；gravity 按实际向量范数归一化，并把完整方向误差乘有效 transition mask，排除参考代码在 reset 处引入的常数误差。reset / motion boundary 处重置历史，预测重新锚定真实状态，不跨边界积分。历史 bank 为每环境 `79+24` 帧，约0.292 GiB/rank，不逐 transition 复制79帧。

RMA 的108维包含 torso COM、torso相对质量扰动、friction、Kp/Kd/armature、四肢负载质量与 COM。输入来自 nominal 恢复和负载惯性合成后的物理状态，不包含额外外力、encoder bias、delay 或 alpha，因此是这108维因素的 oracle 对照。

nominal 也使用真实物理坐标：例如零负载质量在 [-1,1] 归一化后为 -1，并非全零输入。

## 入口与阶段控制

核心代码：

- [RMA actor/critic](../src/intact_tracking/heavy_rma_teacher.py)、[AnyAdapter actor/critic](../src/intact_tracking/heavy_anyadapter.py)。
- [世界模型与历史缓存](../src/intact_tracking/anyadapter_world_model.py)、[在线交替优化](../src/intact_tracking/anyadapter_training.py)。
- [环境条件提供器](../src/intact_tracking/heavy_baseline_env.py)、[训练 CLI](../src/intact_tracking/cli/heavy_baseline_train.py)。
- [自动阶段控制器](../scripts/run_144000_heavy_baselines.py)。

查看正式命令，不启动进程：

```bash
.venv/bin/python scripts/run_144000_heavy_baselines.py --method rma_teacher
.venv/bin/python scripts/run_144000_heavy_baselines.py --method any2track
```

运行两个独立的控制器（各自保持运行，可放在两个终端或进程管理器中）：

```bash
.venv/bin/python scripts/run_144000_heavy_baselines.py --method rma_teacher --gpus 0,1,2,3 --run
.venv/bin/python scripts/run_144000_heavy_baselines.py --method any2track --gpus 4,5,6,7 --run
```

默认产物为 `runs/144000-exp-heavy/baselines/{rma_teacher,any2track}_uniform/continuous_uniform/`。启动器检查所分配 GPU 没有其他进程，并执行 **4 × 8192**、单条短 motion 的2次更新功能预检；完整 motion 库的额外内存仍在正式加载时核验。正式训练使用完整过滤后的220480条 motion，逐 rank 分片，校验共同清单 SHA256；每卡820 nominal、7372 HDR。只允许正式共同规模和 PPO 参数，测试必须显式设置 `--bounded-smoke`。

默认正式训练阶段为 `continuous_uniform`，从头训练直到用户停止。没有 2000 轮边界，没有计划内重启。W&B group=`144000-exp-heavy`，实验名分别为 `144000-exp-heavy-rma-teacher-uniform` 和 `144000-exp-heavy-any2track-uniform`。Uniform 不积累、保存或恢复 adaptive 统计。

后续中断时重新执行同一个控制器命令，从同一目录的最新 checkpoint 恢复模型、优化器、normalizer 和更新次数，继续 uniform 训练；恢复状态做完整摘要检查。旧 adaptive 两阶段流程仅通过显式 `--motion-sampling adaptive` 保留用于历史复现，不参与当前训练。

新增 W&B 组为 `Teacher/*`、`WorldModel/*`、`Adapter/*`，保留 tracker 分组与 `Residual/*` 动作幅度指标。AnyAdapter 的 correction 定义为 adapted mean 减原 tracker mean，仅用于记录。记录预测分项、optimizer steps、embedding drift、预测更新后 PPO 前 KL 及动作漂移；动作 KL/漂移的诊断样本为各 rank rollout 第一步的前1024个环境。

`Policy/optimizer_steps` 与 `WorldModel/optimizer_steps` 分别记录累计更新步数。`Perf/torch_peak_{allocated,reserved}_gib` 记录每次 rollout+update 的 PyTorch 分配器峰值，不能视为包含 Warp 分配器的整卡峰值；另保留更新后整卡可用显存与耗时指标。

## 评测与部署

```bash
.venv/bin/python -m intact_tracking.cli.heavy_baseline_eval \
  --checkpoint <checkpoint.pt> --motion-manifest <motions.txt> \
  --physics nominal --repeats 2 --paired-starts --steps 500 --output <result.json>

.venv/bin/python -m intact_tracking.cli.heavy_baseline_eval \
  --checkpoint <checkpoint.pt> --motion-manifest <motions.txt> \
  --physics hdr --repeats 2 --paired-starts --steps 500 --output <result_hdr.json>

.venv/bin/python -m intact_tracking.cli.heavy_baseline_export \
  --checkpoint <checkpoint.pt> --output-dir <new_export_directory>
```

nominal/HDR 分别在清单的**全部 motion**上运行全 nominal / 全 HDR 环境，不拿混合环境中的10%子集代表 nominal。`--physics mixed` 保留训练分布的诊断选项。沿用配对初始状态、逐步 force 摘要和失败后幸存环境保护；`--frozen-tracker-only` 在相同构建和条件缓存下返回原 tracker。默认 cold history，reset 清空 AnyAdapter 的79步历史。单次评测最多4096个环境；整份长清单可以用以下批处理入口，不截断清单，并校验协议后跳过已完成批次：

```bash
.venv/bin/python scripts/evaluate_heavy_baselines.py \
  --checkpoint <checkpoint.pt> --motion-manifest <full_manifest.txt> \
  --physics both --motions-per-batch 256 --output-dir <new_evaluation_directory>
```

每个方法使用同一清单、分批大小和 seed；保留逐批配对物理/力摘要和 per-step traces，供共同有效前缀比较。汇总表按 episode 加权，同时记录失败率与覆盖率。

checkpoint 内含冻结 tracker、actor/critic，以及 RMA 两套 DR encoder 或 AnyAdapter history encoder + world model；续训还保存所有优化器和窗口 RNG。推理不需要 context u8816 或原 tracker 文件。

导出生成 `policy.onnx`、`policy.json`、`deploy_metadata.json`、`policy_runtime.py`。JSON 保留 SP 的关节顺序、PD 参数、action scale、输入输出形状，并列明新增输入。导出先检查 ONNX 图，再用四组输入比较原 actor、导出 PyTorch 包装和 ONNX Runtime，容差2e-4。RMA ONNX 需要真实108维物理坐标，内部归一化；runtime 拒绝缺失参数。AnyAdapter ONNX 包含 encoder 和 adapter，runtime 维护严格因果的79步历史；world model 不参与部署。

## 验证范围

功能测试与仿真产物保存在 `.runtime/heavy_baselines_validation/`，这些是小规模检查，不是新实验的正式训练结果。

- 单元测试验证 teacher 梯度、adapter 梯度、encoder/PPO 隔离、真实 PPO 更新、初始动作和最终 Gaussian log-prob、完整 optimizer 恢复、因果历史、边界重锚、物理积分/损失单位、一次采样刷新与部署 runtime。
- 两组真实单卡 ×64环境、24步 rollout、完整5×4 PPO 训练；AnyAdapter 同时执行20步、5×4世界模型训练。
- 四卡 ×64环境检查参数同步，以及短测试边界2→3的采样刷新续训；生产计数规则另由阶段测试固定为2001→2002。
- 两个真实 checkpoint 的 ONNX/JSON 检查和四组数值对齐均通过；最大 action 误差分别为2.82e-5、2.92e-5，低于2e-4验收容差。
- 纯 nominal / 纯 HDR 评测入口的真实仿真检查，及原 residual、HDR、采样和导出相关回归测试。

以上为早期实现验证；**4 × 8192** 实机预检和正式训练状态另记于最新 uniform 训练记录，避免将小规模检查当作正式规模验证。

相关76项测试及额外2项 W&B 日志回归测试通过。可用 `.venv/bin/python scripts/verify_heavy_baselines.py` 重核现有 smoke 产物；[机器可读验证记录](../.runtime/heavy_baselines_validation/verification.json)包含精确恢复摘要、四卡一致性、后续采样继承、评测配对和导出误差。
