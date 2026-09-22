# 144000-exp-heavy：RMA student 实施记录

日期：2026-09-22。依据：[已确认方案](heavy_rma_student_plan_20260922.md)。Vanilla 已保存并停止于 12136 次更新；本机 GPU 4–7 用于 student，latent、RMA teacher 和 Any2Track 保持原训练。

<!-- LIVE_STATUS_BEGIN -->
正式四卡训练已从第 112 轮成功恢复。2026-09-22 01:26:50 UTC 核对时已完成 **134 次蒸馏更新**，仍在连续训练，无更新次数上限；近期实际墙钟约 **2.9 秒/轮**，每卡总显存约 **22 GiB**。模型、优化器和归一化状态精确恢复，四 rank 参数一致；`gradient_accumulation_steps=1`，tracking 与 termination 指标已补齐，自动监控状态 healthy。[启动与恢复验证](../runs/144000-exp-heavy/baselines/rma_student_uniform/startup_verified.json)。W&B 在线记录：[144000-exp-heavy-rma-student-uniform](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/heavy-rma-student-0a4413f276)。
<!-- LIVE_STATUS_END -->

**模型与监督**

固定 teacher 为 `checkpoint_9000.pt`（内部完成 9001 次 PPO 更新），已复制到 [student 实验目录](../runs/144000-exp-heavy/baselines/rma_student_uniform/teacher_checkpoint_9000.pt)。SHA-256 为 `fbd26367b3cc014f2d7c2fa181bb7801912fefe017f4dee68286d5e4f53ec9e2`。Student 不追踪正在训练的 teacher 后续版本。

Teacher 的 tracker、residual MLP、观测归一化和 actor DR encoder 全部冻结。只训练新的 **44608 参数** adaptation module：逐帧 `122→128→32` MLP、三层时间卷积、`96→64` 输出层。输入为 50 帧本体历史，约 1 秒；输出对齐 teacher **actor** 的 64 维 embedding。Critic 不参与蒸馏，也不保存 critic 权重。

唯一训练目标为 `mean((student_embedding - teacher_actor_dr_encoder(theta108))²)`。Teacher DR 参数仅用于生成监督标签。Student 自己执行确定性动作并生成在线训练历史；动作 MSE 仅作诊断，权重为 0。RMA student 从零初始化 adaptation，不加载 u8816 context encoder。

每帧包含关节角、关节速度、投影重力、角速度、上一 raw command、带噪关节力矩。当前观测已经包含上一动作，生成本次动作前不会读取本次动作或未来数据。真实 reset/motion teleport 清空对应世界历史并加入新一帧；不足 50 帧的样本照常监督。归一化统计由四 rank 真实帧共同计算，padding 在归一化和逐帧编码后屏蔽。

完整 tracker 继续读取 reference 和原部署观测。其 `robot_key_body` 由带噪关节/陀螺仪信号计算 FK，未为 student 引入 simulator 真值 body 状态。新增 adaptation 的输入不含 reference、真实 DR 或 critic 特权字段。

**实际训练配置**

| 项目 | 实现 |
|---|---|
| GPU / 环境 | 本机 4、5、6、7；每卡 8192，共 32768 |
| 数据 | 完整过滤后的 220480 条；每 rank 55120 条 |
| Manifest SHA-256 | `59b8e336c152e4773133bcdcd86cf86fd6545c3c0e6cc00ce67912d107c740ac` |
| 采样 | 原 teacher uniform；adaptive、failure rewind 均关闭 |
| DR | 每卡 820 nominal + 7372 HDR，原 256 个四肢质量档组合和 COM/惯量合成 |
| Rollout / epoch | 24 步 / 1 epoch |
| Batch / 梯度累积 | 每卡每 batch 49152；每轮 4 次 optimizer step；**不累积梯度** |
| Optimizer | Adam，LR `5e-4`，gradient norm clip `1.0`，FP32 |
| 保存 / 定期评测 | 每 100 次蒸馏更新保存；每 500 次配对评测 |
| 时长 | 连续训练，无更新次数上限，无 2000 轮重启 |
| W&B | 项目 `intact-preview-v2`，group `144000-exp-heavy`，name `144000-exp-heavy-rma-student-uniform` |

原方案正式配置中的 microbatch 累积已按本次沟通移除。完整 49152 batch 在短流程和完整数据集训练中均成功运行：单条 motion 预检的 PyTorch 学习阶段峰值约 8.3 GiB，完整 motion 库下约 16.25 GiB；包含 Warp 等分配后，每卡总显存约 22 GiB。通用优化器保留可选 microbatch 的验证路径，正式入口固定整批更新。

训练保留原 500 步 episode、奖励、termination、传感器噪声、SP delay/smoothing 和 HDR 推力。Residual 保持无 tanh、无限幅、scale=1。Teacher PPO 更新数与 student 蒸馏更新数分别记录，不能直接作为同训练预算比较。

**验证与日志**

已通过 30 项相关测试（RMA student 5 项、现有 heavy baseline 15 项、uniform 协议 10 项），包括移除/污染特权字段不影响 student、注入 teacher embedding 后动作一致、冻结控制网络无梯度、历史重构与边界、microbatch 数学等价、两卡 DDP 对单卡合并 batch 的更新等价、模型/优化器/归一化/RNG 恢复后下一步一致，以及环境 Python scalar/Tensor 混合日志无遗漏。

[四卡预检](../runs/144000-exp-heavy/baselines/rma_student_uniform/preflight.json) 运行 3 次蒸馏更新，然后加载 checkpoint 再完成第 4 次更新；四 rank 的 adaptation 和冻结控制权重 hash 一致。开发阶段发现并修复了不同 rank 首次结束 episode 时日志字段集合不同导致的 collective 不一致：现在先形成跨 rank 的字段并集，再同步数值和有效计数。该问题仅发生在有限步数测试，未改动其他训练进程。

[第 112 轮正式验证](../runs/144000-exp-heavy/baselines/rma_student_uniform/first_training_verification_112.json) 确认 adaptation 已更新、冻结控制权重与 teacher 标签编码器逐元素保持不变、四 rank 参数一致，以及每轮 4 次独立 optimizer step。初始至第 112 轮的 latent MSE 为 `13.680 → 1.899`，同状态动作 RMSE 为 `0.3513 → 0.0494`；这是早期蒸馏拟合情况，不替代 tracking 评测。

原环境的 tracking/termination 日志使用 Python float/int，初段蒸馏循环只接收 Tensor，导致前 112 轮遗漏这些原环境 scalar 指标。修复后重新通过四卡训练和恢复预检，从第 112 轮继续；此前模型、优化器、归一化状态保留。后续保留原 `Metrics/motion/*` 10 项 tracking 误差及 `Episode_Termination/*` 四项统计。恢复事件记录见 [resume_logging_fix_112.json](../runs/144000-exp-heavy/baselines/rma_student_uniform/resume_logging_fix_112.json)。

[评测接口验证](../runs/144000-exp-heavy/baselines/rma_student_uniform/eval_interface_smoke/pairing_validation.json) 使用 16 条 motion × 2、64 步查询，在 nominal/mixed 两种环境验证 student 与固定 teacher 的初始状态、物理世界、motion 起点、奖励与力脉冲前缀相同，并验证 student 从一帧冷启动累积至完整历史。这是功能验证，不代表训练完成后的性能。

部署导出比较实际 actor、PyTorch 导出包装和 ONNX Runtime，绝对/相对容差为 `2e-4`；四个输入样例覆盖 0/1/23/50 帧。恢复后 checkpoint 的动作最大绝对差约 `2.61e-5`。Standalone runtime 另验证 54 次连续调用，包含冷启动、部分历史、完整历史和 reset。

W&B 记录 `Distill/*`、每维 `DistillLatent/*`、按 nominal/HDR、冷启动/完整历史及四肢质量档统计的 `DistillGroup/*`，以及原 `Metrics/*`、`Episode_Reward/*`、`Episode_Termination/*`、`Train/*`、`Residual/*`、`Perf/*`。尚无结束 episode 的 rollout 不伪造零 episode length。`Distill/gradient_accumulation_steps=1` 显式记录整批更新。

**Checkpoint、评测与部署**

Student checkpoint 内嵌 adaptation、输入归一化、冻结 tracker、冻结 residual controller；恢复部分另含 teacher actor DR encoder、Adam、各 rank RNG、更新数及 source hash。加载恢复后重建仿真世界并清空交互历史，恢复学习状态，不承诺逐仿真步重现旧轨迹。初始化保存 `checkpoint_0.pt`；数字文件名就是已经完成的蒸馏更新数；正常停止保存 `checkpoint_final.pt`。

CPU watcher 自动导出 checkpoint 并发布 `deploy/latest`。ONNX 输入为原 8199 维 tracker observation、`[1,50,122]` 历史和 int64 有效帧数；输出为 29 维总动作和 64 维 embedding。JSON 含关节顺序、PD、action scale、历史时序和 reset 规则。`policy_runtime.py:RMAStudentPolicy` 只依赖 NumPy/ONNX Runtime，不需要真实 DR、外部 teacher 或 context 文件。

每 500 次更新使用固定 256-motion 协议，对比当前 student、固定 teacher9000、停止后的 vanilla12136、取快照时最新 latent。每个场景 512 环境，nominal/mixed 分开；比较共同有效轨迹上的五项误差及各自完整轨迹成功率。Student 在查询起点清空历史；latent 保留原长期记忆协议，报告中明确该方法差异。导出在 CPU 执行；小规模评测仅在 GPU7 空闲显存足够时与训练共享设备，不停止训练。

**运行入口**

```bash
# 同规模短流程和恢复验证
.venv/bin/python scripts/run_144000_rma_student.py --smoke

# 启动；已有正式 checkpoint 时恢复；默认使用 GPU4-7
.venv/bin/python scripts/run_144000_rma_student.py --run

# 自动导出、健康检查和每500次更新配对评测
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/watch_rma_student.py

# 手工导出
.venv/bin/python -m intact_tracking.cli.heavy_rma_student_export \
  --checkpoint /path/to/student.pt --output /path/to/deploy
```

模型与数据接口见 [heavy_rma_student.py](../src/intact_tracking/heavy_rma_student.py)、[rma_student_env.py](../src/intact_tracking/rma_student_env.py)；优化与保存见 [rma_student_distillation.py](../src/intact_tracking/rma_student_distillation.py)，训练入口见 [heavy_rma_student_train.py](../src/intact_tracking/cli/heavy_rma_student_train.py)。评测复用 `heavy_baseline_eval`，导出实现位于 `rma_student_export.py`。
