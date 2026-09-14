# Memory350：恢复一半 nominal A 的阶段一训练

2026-09-11，用户要求恢复 nominal 占一半的训练配置，并启动修正后的原版 Memory350。
本次从头初始化 encoder 和 predictor；冻结 tracker 使用原 checkpoint。

- 正式入口：`python -m intact_tracking.cli.forward_memory_nominal_train`。
- 启动器：`scripts/run_memory350_nominal_stage1.py`。
- 新目录：`runs/limb_context_20260911_memory350_nominal50`。
- GPU 0–3；首先每卡 8192 个 A 环境，只有实际 OOM 才降至每卡 4096。
- 每卡训练 8064 个 world：4032 nominal + 4032 DR；固定验证 128 个 world：64 + 64。
  偶数 local world ID 为 nominal，奇数为 DR，训练和验证的 world 不重叠。
- Nominal：恢复所有已扩展的随机物理字段到 compiled defaults，清零 encoder bias、
  persistent joint offset 和四肢负载；关闭随机推扰及其 reset 后的触发计时器。
- DR：原 tracker 的 startup DR、随机推扰及双手/双小腿各自独立 U(0,4 kg)；
  实际 DR 物理字段与原采样逐元素一致。观测噪声、动作机制和初始状态扰动保留原设置。
- B：全部为干净 nominal，恢复对应 A 的初始状态并重放相同五步物理 PD targets。
- Memory：原版 short50 + 30×10 long，跨 reset 保留长期，历史不重叠；latent 64 维。
  encoder 为 128 宽，chunk/long/final 深度 1/2/2；不是 encoder2x。
- Predictor：512 宽、6 层、8 heads；10 步历史、预测 5 步。
- 损失保持 `L_pred + 0.01 L_positive + 0.02 L_relation`，response scale=0.75；
  正样本仍为同 world/episode/motion 的精确 ±5 步。
- 完整 motion_data_full，129827 条；global batch 4096，microbatch 256/rank，
  每 update 4 次优化，seed 717+rank，预热 500 步，AdamW 3e-4。
- 8000 updates 是 cosine LR 时间尺度，之后保持 1e-5，无自动停止；仅训练阶段一。
- 每 100 updates 验证；恢复 nominal NMSE 和 nominal A−B 响应误差的单独记录；
  best checkpoint 继续按固定验证 DR NMSE 选择。记录训练 batch 的实际 nominal 比例。
- 此次沿用 motion-balanced replay；50% 指物理 world 配置，每个随机 batch 的比例会波动。

运行与验证状态以新目录的 `state.json`、`stage1/progress.json`、`startup_verification.json`
为准。W&B 使用原 `intact-forward-predictor` 项目，独立 nominal50 训练分组。

新实现位于 `memory350_nominal_rollout.py` 和独立训练入口，历史 all-DR 入口用于复现旧实验。
训练配置和 checkpoint 显式记录 `nominal_a_fraction=0.5`，拒绝将旧 all-DR checkpoint
及其归一化作为此次运行的 resume 输入。

验证包括真实力脉冲强制触发、部分环境 reset 后 nominal 仍无外力、DR 仍正常受推扰，
compiled nominal 物理逐字段恢复、DR 字段保持不变、全局训练/验证分组，以及四卡优化与
checkpoint 参数一致性。物理核验保存在 `.runtime/memory350_nominal50/physics_verification.json`。

启动核验已通过：18 项相关测试通过；128-world 实际模拟器运行 400 步、经历 60 次
环境重置，nominal 实际外力峰值为 0，DR 为 9.973 N（按力分量绝对值）；nominal A/B
五步关节位置 RMS 差为 7.06e-5 rad。四卡每卡 8192 环境的短测完成 2 updates / 8
optimizer steps，三个 encoder 阶段梯度均有限且非零，无 OOM。

正式运行启动于 2026-09-11，完整 129827 motions / 48085337 frames 已加载。启动核验时
已完成 11 updates / 44 optimizer steps；首个 checkpoint 为 update 1，四 rank 参数
SHA256 完全一致。归一化只使用 32256 个训练 world，独立验证 world 不参与统计。
恢复的 nominal 验证指标、checkpoint 的 nominal_a_fraction=0.5 和 W&B 配置均已核对。
此记录只确认训练启动与实现正确性，不代表已经得到收敛后的 latent 质量结论。

[正式 W&B](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/m350n50-c90f7b5cd5be)
；[启动核验](../runs/limb_context_20260911_memory350_nominal50/startup_verification.json)
；[当前进度](../runs/limb_context_20260911_memory350_nominal50/stage1/progress.json)。

2026-09-11 完成冻结 update 5000 的表征测试：共同基准 nominal 跨 motion 单位化距离为 0.067，
DR 簇内 / 簇间距离比为 0.625（同一批输入上的 u1800 为 0.855）；负载/推扰组为 0.831
（u1800 为 1.082）。环境信息随训练增强，但同 DR 跨 motion 的单簇结构仍不充分，
同轮次 all-DR 对照的环境识别指标也并非全部落后。
[完整结果及交互图](../runs/latent_cluster_probe_memory350_nominal50_u5000_20260911/README.md)。
