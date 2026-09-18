# Memory350 从头训练：四卡、nominal10、逐参数 nominal50

用户要求从头启动四卡训练，每卡 8192 个 A 环境，约 10% 为完整 nominal，其余环境的每个固定参数独立执行 50% nominal / 50% 原 DR 分布采样，不设训练上限。

运行目录：`runs/limb_context_20260917_memory350_nominal10_independent50_scratch_4x8192`。

## 环境与初始化

- GPU 0–3，每卡 819 个完整 nominal、7373 个 DR，共 32768 个 A 环境。B 为对应的 nominal 反事实模拟槽。
- 每卡最后 128 个 A world 留作验证，其中 13 个 nominal、115 个 DR；训练分区为 806 个 nominal、7258 个 DR。验证 world 不参与训练 normalization 或 nominal 方向校准。
- DR 内独立混合四肢负载、COM 各轴、躯干质量、每关节 armature 和 encoder bias；足底摩擦沿用共享系数的一次独立决定。参数跨 reset 固定。
- 连续负载范围：左／右手各 U(0,2.5) kg，左／右小腿各 U(0,4) kg。其他连续参数范围、观测噪声、motion/reset 采样及 DR 力脉冲沿用新采样方案。完整 nominal 不带负载、encoder bias 或力脉冲。
- encoder、predictor、AdamW 和 scheduler 全部重新初始化；无 resume checkpoint。预训练 tracker 仅用于采集轨迹。
- normalization 由本次训练 replay 预热后重新统计。nominal 单位方向由随机初始化 encoder 的完整训练历史校准一次，此后冻结并保存。

## 模型、目标与训练控制

Memory350 short50＋long30×10；encoder attention 深度 2/4/4、宽度128、latent64；五步 predictor 宽度512、深度6。保留 encoder LayerNorm，并在表征项使用单位 latent。

| loss | 系数 |
|---|---:|
| 五步 teacher-forced prediction | 1 |
| 五步 recursive prediction | 0.5 |
| 同 world/episode/motion、±5 步局部正样本 | 0.01 |
| DR 单位 latent 到固定 nominal 方向的距离，拟合十步 A−B 响应 RMS 映射 2D/(D+0.3) | 0.04 |
| 同 DR world/session、跨 motion、完整且不重叠历史正样本 | 0.008 |
| nominal latent 固定方向，1−cos | 0.01 |

弱负样本计算移除。使用原全量 motion 数据集及 motion-balanced replay。四卡各 batch1024、microbatch256，全局 batch4096，每 update 四次 optimizer step。初始 LR=3e-4，cosine 衰减尺度为 8000 update，之后保持 1e-5 下限；该尺度不是训练上限。`until_user_stop=True`，无 stop-after-updates，无自动 plateau 停训。

## 启动与核验

```sh
.venv/bin/python -B scripts/run_memory350_nominal_direction.py \
  --mode train --from-scratch --num-gpus 4 \
  --nominal-fraction 0.1 --dr-nominal-probability 0.5 \
  --limb-max-masses-kg 2.5 2.5 4 4 \
  --run-root runs/limb_context_20260917_memory350_nominal10_independent50_scratch_4x8192 \
  --launch
```

先以相同采样规则进行四卡短训练，核验真实物理采样、固定方向和参数同步、六项 loss 加权和。正式首轮保存点再核验四卡每卡 8192、全量 motion、随机初始化、optimizer 步数从零开始，以及 DR 分区每个参数的实际 nominal 掩码比例。

状态以运行目录中的 `smoke_verification.json`、`train_verification.json`、`train_process.json` 和 `monitor/status.json` 为准。监控仅观察，不自动停止或重启训练。

上一轮从 u15000 续训的任务已在 u15732 正常停止并保留全部记录；暂停前准备的八卡从头训练未启动。本轮使用独立目录。

## 已验证启动

29 项相关测试通过；四卡短训练完成两个真实更新，模型及固定方向在四卡一致，跨 motion 正样本平均每 microbatch 5.75 对。

正式 torchrun PID 11509，worker PID 11565–11568，使用 GPU 0–3。首个 checkpoint 为 update1 / optimizer step4，验证随机初始化、全局 batch4096、六项损失加权和、四卡参数及锚点一致。固定方向由333个完整 nominal 训练历史样本校准，source_update=0，未使用验证数据。

正式四个 rank 每卡均为 819 nominal／7373 DR。DR 分区67个标量参数的 nominal 掩码比例在48.216%–51.960%之间。全量129827条 motion 分片加载完成。

CPU 监控进程 PID 16720 每30秒检查训练状态和 checkpoint；已观察 update16→28→38 持续推进，随后训练超过 update52。W&B 服务端确认 running 并返回本轮训练指标：[训练曲线](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/m350nomdir-08bc5be5e4a1)。

正式首轮预热为500步，尚无完整且不重叠的跨 motion 历史配对，该项首轮按有效性掩码贡献零；短训练用1000步预热已确认该项参与优化。固定验证集同样没有这类配对，其零损失不能解释为跨 motion 一致性改善。
