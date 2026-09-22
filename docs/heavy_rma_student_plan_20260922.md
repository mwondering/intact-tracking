# 144000-exp-heavy：RMA student 蒸馏方案

日期：2026-09-22。状态：**方案已获用户确认并实现；实际配置和启动验证见 [实施记录](heavy_rma_student_implementation_20260922.md)。** 实施时取消原先的梯度累积，采用完整 minibatch。

Vanilla 在完成 **12136** 次更新后正常保存退出，本机 GPU **4、5、6、7** 已释放。最终 [checkpoint](../runs/144000-exp-heavy-residual-baseline/ppo_4gpu8192_resume2000_uniform_mass5x/checkpoint_final.pt) 包含优化器、冻结 tracker 和 context；四 rank 参数一致、actor/critic 权重有限、嵌入依赖校验通过。最终 [ONNX/JSON 部署目录](../runs/144000-exp-heavy-residual-baseline/ppo_4gpu8192_resume2000_uniform_mass5x/deploy/checkpoint_final/) 已通过导出数值检查。停止记录见 [stop_for_rma_student_20260922.json](../runs/144000-exp-heavy-residual-baseline/stop_for_rma_student_20260922.json)。Latent、RMA teacher、Any2Track 继续训练。

**1. 蒸馏目标与观测边界**

采用 RMA 第二阶段的核心机制：冻结已训练策略与物理参数编码器，用本体感知及动作历史估计 teacher 的 embedding；由 student 自己控制环境，在它访问的状态上进行在线监督。原论文采用 embedding MSE 和时间卷积历史编码器，历史为 50 步；本项目沿用其方法，机器人、输入维度、控制频率和 residual 外壳按现有 SP 平台适配。[RMA §III-B、§IV-B](https://arxiv.org/html/2107.04034v1#S3.SS2)

“只观测本体感知历史”在此落实为：**student 的 adaptation module 只读取本体感知和已经发出的控制指令；完整 tracking actor 继续读取目标 motion/reference。** Reference 是跟踪任务指令，不能从整个策略中删除，否则相同机器人状态无法区分不同目标动作。Student 的动作路径不接收真实 DR、critic 特权观测或 simulator 真值 body 状态。

当前 teacher 的 actor 计算：

\[
z^T=E_T(\theta_{108})\in\mathbb R^{64},\qquad
a^T_t=a^{\mathrm{tracker}}_t+
R_T(f^{\mathrm{tracker}}_t,z^T,a^{\mathrm{tracker}}_t).
\]

Student 将其中的条件替换为：

\[
\hat z_t=A_\phi(h_t)\in\mathbb R^{64},\qquad
a^S_t=a^{\mathrm{tracker}}_t+
R_T(f^{\mathrm{tracker}}_t,\hat z_t,a^{\mathrm{tracker}}_t).
\]

仅优化新建的 \(A_\phi\)。冻结 tracker、teacher residual MLP、teacher actor DR encoder 及其归一化状态。Teacher 的 critic 拥有另一套 DR encoder；它的 embedding **不是本次监督目标**。本阶段使用监督学习，不训练 critic 或 PPO。Student 编码器从零训练，不加载 heavy context u8816。

核对时最新完整 teacher 为 `runs/144000-exp-heavy/baselines/rma_teacher_uniform/continuous_uniform/checkpoint_9000.pt`，内部 `completed_updates=9001`，SHA-256 为 `fbd26367b3cc014f2d7c2fa181bb7801912fefe017f4dee68286d5e4f53ec9e2`。建议以此作为首版 teacher，在启动时复制到 student 实验目录并固定 hash；正在继续训练的 teacher 后续权重不自动替换蒸馏目标。

**2. 历史输入与网络**

每帧采用现有部署可得的 **122 维**信号：关节角 29、关节速度 29、投影重力 3、机身角速度 3、上一步控制指令 29、带噪关节力矩 29。复用 `current_proprio()` 的观测顺序、噪声样本及传感器语义；动作段核验为已经下发的合成 raw command，位于 SP delay/smoothing 之前。Teacher 与 student 的历史均记录实际用于该次 rollout 的指令。

默认窗口 **50 帧**；在当前 50 Hz 控制频率下覆盖约 **1 秒**。以当前传感器帧和上一步动作构造 \(p_t\)，令 \(h_t=[p_{t-49},\ldots,p_t]\)；生成 \(a_t\) 时无法读取它自身或未来观测。这一对齐方式会在训练与 runtime 中统一声明。

建议网络：逐帧 MLP `122 → 128 → 32`，随后三层 Conv1d `(32,32,k=8,s=4)`、`(32,32,k=5,s=1)`、`(32,32,k=5,s=1)`，最后 `flatten(3×32) → 64`。隐藏层使用 ELU，末层线性输出；含偏置共 **44608** 个可训练参数。监督的是 64 维策略 embedding，不是直接回归 108 维物理参数。

新增 student 自己的逐特征观测归一化，训练统计跨 rank 同步，评测和部署冻结。缺失历史在归一化及逐帧编码后按有效 mask 清零，mask 只表示真实采样帧。Reset、motion teleport 和显式环境重建清空对应世界的历史；禁止跨边界拼接假连续轨迹。训练包含不足 50 帧的冷启动样本，避免“先得到完整历史才能学会存活”的循环依赖。部署使用相同的 padding、mask、时间索引和清空规则。

**3. 损失与采集方式**

首版只使用：

\[
\mathcal L_{\mathrm{adapt}}
=\frac1{64}\left\|A_\phi(h_t)-\operatorname{stopgrad}
\big(E_T(\theta_{108})\big)\right\|_2^2.
\]

真实 DR 只在监督标签生成路径使用。每次 student rollout 都计算对应世界的 teacher actor embedding 标签；student 从自己的预测 embedding 生成动作，因此数据包含估计不准时的轨迹。采用确定性动作均值执行，环境本身保留原传感器噪声、DR、delay/smoothing 和推力随机化。

动作蒸馏误差先作为诊断：在**同一状态、同一 reference、同一 tracker feature/action**下，分别使用 \(z^T\) 与 \(\hat z\) 计算动作均值差。跟踪环境里的两条独立轨迹不能直接逐时刻作动作监督。首版 action loss 权重为 0，保留清楚的 RMA 方法对照；若 embedding loss 收敛而动作误差和 tracking 仍差，再另开带 action loss 的消融，而不是中途修改本组定义。

**4. 训练配置**

| 项目 | 方案 |
|---|---|
| 实验名 / W&B group | `144000-exp-heavy-rma-student-uniform` / `144000-exp-heavy` |
| 设备 / 并行 | 本机 GPU 4–7，4 rank，每卡 8192 环境，共 32768 |
| 数据集 | `/data_zcy/wxy/motion_data_correct` 完整过滤后的 220480 条 motion，与现有 teacher 使用同一 manifest 和分片规则 |
| Motion / 起始帧采样 | 沿用 teacher 的 uniform；关闭 adaptive 和 failure rewind |
| 物理环境 | 每卡 820 nominal + 7372 HDR；沿用原 NDR、四肢负载、COM 和惯量合成，以及 256 个质量档组合分配 |
| 物理生命周期 | 延续当前每 world 固定物理参数、原 reset controller 随机化和力脉冲规则 |
| 控制 / episode | 50 Hz / 最长 500 步；保留原 termination 和 tracking 指标 |
| Rollout | 每次收集 24 步，四卡共 786432 个样本 |
| Student 优化器 | Adam，初始 LR `5e-4`，梯度范数裁剪 `1.0`，FP32 |
| 监督更新 | 每批新 rollout 训练 1 epoch，4 个 minibatch；每卡每 batch 49152，整批反向传播，不累积梯度 |
| 保存 / 评测 | 每 100 次蒸馏更新保存；每 500 次以固定快照进行 nominal / mixed DR 评测 |
| 训练时长 | 连续训练，不设 2000 轮重启或训练上限；支持断点恢复 |

以上 optimizer 和监督 batch 是 student 蒸馏的新配置，不能将“蒸馏更新次数”与 teacher PPO 更新次数等同。2026-09-22 实施时用户指出梯度累积问题，改为完整 minibatch，每轮 4 次反向传播/同步/optimizer step。四卡 8192 环境短流程中完整 batch 已运行，PyTorch 学习阶段峰值约 8.3 GiB；完整数据集总显存另行验证。历史存储复用连续帧与窗口索引，避免为每个 rollout step 复制整个 50 帧窗口。

**5. 验证、日志与部署**

W&B 沿用当前项目和分组。新增 `Distill/latent_mse`、各 embedding 维 NMSE/R²、`Distill/action_rmse`、梯度范数、history 有效比例、rollout/优化耗时、samples/s；保留现有 `Metrics/*` tracking 误差、成功率、episode length 和 residual/base action 幅度。分别报告 nominal/HDR、冷启动/满窗口，以及四肢负载档位下的结果。

首版验收包含：从 student 输入删去或随机打乱全部特权字段，输出保持一致；人工给 student 注入 teacher 的 \(z^T\) 后，两者动作数值对齐；只存在 adaptation 参数梯度；冻结网络与归一化 hash 稳定；边界不会污染相邻环境历史；单卡与四卡梯度累积一致；resume 后优化器、计数、归一化和 RNG 恢复一致。恢复时重建仿真世界并清空交互历史，明确记录这属于恢复学习状态，不声称逐仿真步完全复现。

评测与固定 teacher、停止后的 vanilla、同期 latent 使用相同 motion、物理世界、初始状态及推力序列，报告成功率和共同有效轨迹上的五项 tracking 误差。Teacher 对比始终采用蒸馏源快照。冷启动评测从空 student 历史开始；若补充 warm-history 评测，历史由 student 自己生成并单列结果。现有 256-motion 协议用于前后可比，另用新 DR seed 检查泛化，不将这些 motion 称为训练集之外的留出集。

Checkpoint 应内嵌 student 历史编码器、归一化、冻结 tracker、冻结 residual head、teacher 来源/hash；训练恢复包另保存生成标签所需的 teacher actor DR encoder、optimizer、RNG、计数和运行配置。导出 ONNX + SP 风格 JSON + runtime，推理输入只保留 tracker 所需的观测/reference 和本体历史，不要求真实 DR 或 teacher 源文件；包含冷启动、部分历史、满历史、reset 四类数值一致性检查。

**6. 代码落点与实施顺序**

新增 `heavy_rma_student.py`、`rma_student_distillation.py`、配套 distill/eval/export CLI 与四卡启动器。复用 heavy 环境、过滤 manifest、uniform sampler、噪声观测、部署元数据和 paired evaluation。先完成观测/冻结/teacher 动作对齐检查，再执行 4 × 8192 的短流程检查，最后启动完整数据集连续蒸馏。

需要单独增加接受 **64 维 embedding** 的共享 residual 前向入口。当前 `RMATeacherActor._residual_input(..., latent_override=...)` 的 override 实际仍是 **108 维物理参数**，之后还会经过 DR encoder，不能把 student 的 64 维输出直接塞进去。Student 对象也不能继承一个 forward 中强制读取 `teacher_physics` 的路径，否则部署仍依赖真值。
