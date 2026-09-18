# 各固定 DR 参数独立混入 50% nominal

已实现并在 mjwarp GPU 环境中核验。每个标量参数独立执行：50% 取 nominal，50% 保留原分布的随机值。COM 的三个轴、四个肢体负载、各关节的 armature 和 encoder bias 分别作独立决定。足底摩擦沿用原配置：同一环境的所有足底几何体共享一个系数和一次决定。

| 参数 | nominal 分支 | 原随机分支 |
|---|---|---|
| 左手／右手额外负载，各自独立 | 0 kg | U(0, 2.5) kg |
| 左小腿／右小腿额外负载，各自独立 | 0 kg | U(0, 4) kg |
| 躯干 COM 的 x/y/z 偏移，各自独立 | 0 m | U(-0.075, 0.075) m |
| 躯干质量变化 | 0 kg | U(-1, 1) kg |
| 足底摩擦系数 | 编译模型值，约 0.6 | U(0.3, 2.0) |
| 每关节 armature 倍率 | 1 | U(0.8, 1.2) |
| 每关节 encoder bias | 0 rad | U(-0.01, 0.01) rad |

负载上限仍由配置控制；表中是本轮使用的双手 2.5 kg、双小腿 4 kg 配置。nominal 指编译模型参数，例如 COM 是零**偏移**，不是把模型本来的质心位置置零。负载被置零后会同时重建对应的质量、质心和惯量。

时变力脉冲、逐步观测噪声、motion 与 reset 状态采样保持原设置。本轮修改的是固定环境参数的分布。

## 16,384 个环境的 GPU 核验

使用原 16,384 环境实验的八组配置、种子与实际 mjwarp 初始化流程。对每个环境保存原随机参数、nominal 参数、独立掩码和实际混合后参数；所有比较均排除额外的整环境 nominal 对照。没有执行 policy 交互或训练。

| 指标 | 原均匀采样 | 各参数 50% nominal |
|---|---:|---:|
| 四肢总负载均值 | 6.51 kg | 3.26 kg |
| COM 偏移长度均值 | 7.20 cm | 4.47 cm |
| COM 偏移小于 2 cm | 0.96% | 24.49% |
| COM 三轴偏移全为零 | 0% | 12.25% |
| 摩擦系数恰为 nominal | 0% | 49.84% |
| 躯干质量恰为 nominal | 0% | 50.04% |
| 躯干质量绝对变化均值 | 0.500 kg | 0.250 kg |

38 个物理参数的 nominal 掩码比例为 49.43%–50.59%；29 个 encoder-bias 参数为 49.08%–51.04%。所有未置为 nominal 的随机参数与原始采样值逐项完全一致。各环境连续三次完整 reset 后，物理参数、encoder bias 和 nominal 掩码均未改变。

随机分支仍覆盖原来的远端范围。例如实际采到的摩擦范围为 0.30027–1.99994，左／右小腿最大负载分别为 3.99895／4.00000 kg。单个参数的远端区间概率变为原先的一半；多个参数同时处于远端的组合概率会更小。因此，这种采样增加 nominal 附近的比例，但不保证 latent 距离八档等频。本轮尚未重跑每环境 50 秒的 latent 评估。

## 使用与兼容性

Memory350 encoder 训练入口（包括继承其参数的 DR-center 入口）增加：

```text
--dr-nominal-probability 0.5
```

底层配置为 `FixedDRRolloutConfig.dr_nominal_probability=0.5`，适用于随机 `tracker_dr_plus_limb_payload`。不允许与显式固定负载的专家配置混用。老入口的默认值仍是 0，保持旧实验复现；本轮采样核验脚本默认使用 0.5。恢复训练时会检查该概率，防止静默改变 DR 分布。

原本启用整环境 nominal50 对照的训练器仍保留其对照组；新的按参数混合规则作用于其中的 DR 组。这是两个独立设置。

实现：[独立混合事件](../src/intact_tracking/independent_nominal_dr.py)、[DR 配置](../src/intact_tracking/limb_context_dr.py)、[GPU 核验脚本](../scripts/sample_independent_nominal_dr.py)。相关采样、配置、旧默认值兼容和恢复训练检查已通过测试。

数据与审计：

- [汇总统计](../runs/limb_context_20260917_all_fixed_dr_half_nominal/summary.json)
- [全部环境参数与掩码](../runs/limb_context_20260917_all_fixed_dr_half_nominal/parameter_bank.npz)
- 各 shard 的 `metadata.json` 记录实际配置及三次 reset 的验证结果，`parameters.npz` 保存分片参数。
