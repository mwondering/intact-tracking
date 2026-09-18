# 同预算共享 baseline 与八个固定 DR 专家

用户要求停止新训的八个独立 policy，并以八卡、每卡 1024 环境快速训练一个 1000 轮共享 baseline。baseline 使用连续均匀随机 DR，随后在专家各自对应的完整固定 DR 上比较 1000 轮 checkpoint。

运行目录：`runs/limb_context_20260914_uniform_shared8x1024_vs_specialists_u1000`。

八个专家已保存退出，退出轮数为 1105、1044、1109、1087、1079、1038、1112、1107；比较预先固定各组 `checkpoint_update_001000.pt`，其 SHA256、原始来源与评测结果记录在本次目录的 `protocols/selection.json`。本次目录另保留 checkpoint 硬链接以及评测 JSON/trace 副本。

## 训练条件

| 条件 | 共享 baseline | 单个专家 |
|---|---|---|
| 环境数 | 8 卡 × 1024 | 1 卡 × 8192 |
| 物理参数 | 每个 world 独立均匀随机采样，startup 后保持固定 | 同组所有 world 复用一个完整固定 DR |
| 每轮 PPO 样本 | 8192 × 24 = 196608 | 8192 × 24 = 196608 |
| 对比轮数 | 1000 | 1000 |
| 累计 PPO 样本 | 196608000 | 196608000 |

其余设置对齐：同一冻结 tracker 与原 action scale/PD gains；residual 直接 MLP 输出，无 tanh、幅度缩放或输出 clamp；左右 wrist pitch/yaw 扭矩 ±10 N·m；手部负载范围 [0,2.5] kg、小腿 [0,4] kg；原奖励与末端高度 termination；全程 uniform motion 采样；FP32；seed121；actor LR 1e-4、critic LR 5e-4、entropy 0.0002；rollout24、5 epochs、4 minibatches；相同 500 步冻结 tracker 启动预热。

网络与专家相同且没有 latent/router：actor 1645→512→256→128，拼接当前 tracker raw action29，再经 157→256→128→29；critic 独立使用 6330→1024→512→256→128，拼接同一 action29，再经 157→256→128→1。

共享 baseline 使用已有 DDP：每次 optimizer step 平均八个 rank 的梯度，advantage 使用全局 rollout 统计，critic normalization 每步汇总全局矩。完整 129827-motion / 48085337-frame 数据集按已有 `[rank::world_size]` 方式分片，八卡联合覆盖完整数据；不缩减 motion 数据集。每个专家此前各自加载完整数据集。

## 评测与验证

双方复用 512 条 motion、相同起点、seed20001、最多1000步，以及相同 DR bank 的完整物理参数；八张卡各评测一种固定 DR。主要指标是共同存活窗口内 body/joint pos 误差，并报告 root pos/rot、失败率、覆盖率和 motion 配对 bootstrap 区间。

每轮样本量及总样本量按单个 policy 对齐；八个专家合计使用八倍训练样本。训练分布和分布式布局仍不同，因此该试验验证固定环境专门训练相对通用训练的收益，不是仅改变参数共享的严格消融；一个训练 seed 的结果也不等于稳定的跨 seed 结论。

八卡、每卡1024环境的两轮实测已通过，最终八个 rank 的 actor、critic、critic normalizer 哈希一致。模型、PPO、reward、termination、动作协议和初始化逐项与全部八个专家核对，八个 rank 的随机 DR 采样互不相同。20 项相关测试通过。正式任务由 `scripts/run_residual_uniform_shared_comparison.py` 执行，1000 轮后自动停止训练、评测八组固定 DR 并生成 `comparison.json`、`comparison.csv` 和 `README.md`。

正式启动：2026-09-14T06:39:37.296527+00:00。训练和八组评测现已完成，最终恰好 1000 轮、196608000 个 PPO transition；PPO 更新平均 2.713 秒，累计计算约 45.2 分钟。最终配置审计通过，八个 rank 的模型参数及 critic normalizer 哈希一致。W&B 服务端状态为 `finished`，已收到第 1000 轮及完整样本计数，见 `wandb_final_verification.json`。曲线：[uniform-shared-e8b9efef2bd2](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/uniform-shared-e8b9efef2bd2)。八个专家及本次 baseline/评测进程均已退出，八张卡已释放。

## 1000 轮结果

下表均为共享 baseline / 对应专家。body/joint 使用同一 motion 的共同存活窗口；root 使用各自 episode 窗口。正的 body 改善率表示专家误差更低。

| 完整固定 DR 的负载标签 | body pos (cm) | 专家 body 改善 | joint pos (rad) | root pos (cm) | 失败率 |
|---|---:|---:|---:|---:|---:|
| B00 四肢 0 kg | 3.480 / 2.457 | +29.40% | 0.4636 / 0.4005 | 14.690 / 11.798 | 0.00% / 0.20% |
| B01 四肢各 1 kg | 3.229 / 2.891 | +10.48% | 0.4813 / 0.4533 | 15.628 / 16.074 | 0.39% / 0.00% |
| B02 四肢各 2 kg | 3.487 / 3.686 | −5.71% | 0.5157 / 0.5603 | 22.670 / 19.055 | 0.00% / 0.00% |
| B03 双手各 2.5 kg、双小腿各 4 kg | 3.872 / 4.039 | −4.32% | 0.5582 / 0.6105 | 28.382 / 22.019 | 1.37% / 0.39% |
| B04 仅双手各 2.5 kg | 3.646 / 3.608 | +1.03% | 0.5385 / 0.6032 | 18.965 / 17.176 | 0.98% / 0.59% |
| B05 仅双小腿各 4 kg | 3.373 / 2.540 | +24.71% | 0.4647 / 0.3954 | 20.234 / 15.467 | 0.39% / 0.20% |
| B06 仅左手 2.5 kg、左小腿 4 kg | 3.475 / 3.145 | +9.51% | 0.5098 / 0.5081 | 21.093 / 16.516 | 0.59% / 0.39% |
| B07 仅右手 2.5 kg、右小腿 4 kg | 3.476 / 3.110 | +10.53% | 0.5032 / 0.5153 | 19.998 / 16.049 | 0.20% / 0.39% |

标签只描述负载；各组仍有 bank 中完整静态 DR。B00 零负载不等于 nominal。

八组等权汇总：body pos 从 3.505 cm 降到 3.184 cm，降低 9.14%；motion 配对 bootstrap 的 95% 区间为 [8.31%, 9.94%]。joint pos 从 0.50438 rad 变为 0.50583 rad，误差增加 0.29%；对应改善率区间 [−0.71%, +0.16%] 跨零，整体没有改善证据。整体区间先对每条 motion 的八个 DR 结果等权平均，再对 512 条 motion 做 2000 次配对重采样，保留同一 motion 跨 DR 的相关性。

Root pos 从 20.207 cm 降到 16.769 cm（降低 17.01%），root rot 从 0.05897 rad 降到 0.05440 rad（降低 7.76%）。失败从 20/4096（0.488%）降到 11/4096（0.269%），平均覆盖率从 99.733% 升到 99.851%。Root 和失败率这里是描述性汇总，不提供跨训练 seed 的显著性结论。

专家在 6/8 组 body 指标上更好，其中 B00、B01、B05、B06、B07 的配对 95% 改善区间完全大于零；B04 为 [−0.27%, +2.29%]，优势不明确。B02、B03 的 body 和 joint 指标均更差，但 root 更好，B03 的失败率也更低。因此结果支持部分环境中专门训练的收益，尚不支持所有 DR 上专家全面优于通用 policy，也不能直接推出 latent 路由 MoE 必然有效。

## 结果文件

本次运行目录内：

- `baseline/checkpoint_update_001000.pt`：最终对照 checkpoint，SHA256 `8dc1fca1bb51ca8fe29d4bf0b71b9d9402bf05f1c0bca38ad8a8620d64a883fd`。
- `comparison.json` / `comparison.csv` / `README.md`：八组配对结果、单组区间及配置审计。
- `overall_comparison.json`：八组等权 body/joint 汇总及配对区间。
- `comparison.png` / `comparison.svg`：body、joint、root 与 body 改善区间对比图。
- `evaluation/` / `reference_specialists/`：两侧评测 JSON、逐步 trace 及专家 checkpoint 引用。
- `training_final_audit.json` / `baseline/completion.json`：最终配置、样本量及八 rank 一致性检查。
- `source_snapshot/`：正式启动时的源码快照；`protocols/` 固定数据、物理参数和专家 checkpoint。

## 全局坐标系指标补测

为回答用户对全局指标的追问，使用双方原第 1000 轮 checkpoint 重新评测。旧 body 位置/姿态会按机器人水平位置及 yaw 对齐参考；新增指标直接比较世界坐标中的参考与实际状态，保留整体漂移。body 覆盖原配置的 22 个 link，root/anchor 是 pelvis。

评测仍为原八组完整固定 DR、同一 512 条 motion、起点、seed20001、FP32 和最多 1000 步。所有 20 项原始/新增指标都保存逐步 trace，并统一按每条 motion 双方共同存活窗口统计。八组等权后对 512 条 motion 做 2000 次配对 bootstrap。

| 指标 | baseline | 专家 | 专家改善 |
|---|---:|---:|---:|
| 全局 body 位置 (cm) | 20.7632 | 17.1627 | +17.34% |
| 全局 pelvis 位置 (cm) | 20.1597 | 16.6640 | +17.34% |
| pelvis 水平位置 (cm) | 19.9425 | 16.4415 | +17.56% |
| pelvis 高度 (cm) | 1.1650 | 1.1318 | +2.85% |
| 全局 body 姿态 (rad) | 0.1528 | 0.1466 | +4.02% |
| 全局 pelvis 姿态 (rad) | 0.0589 | 0.0544 | +7.77% |
| 全局 body 线速度 (m/s) | 0.2186 | 0.2050 | +6.26% |
| 全局 body 角速度 (rad/s) | 0.9184 | 0.8803 | +4.15% |
| 全局 pelvis 线速度 (m/s) | 0.1640 | 0.1495 | +8.83% |
| 全局 pelvis 角速度 (rad/s) | 0.3765 | 0.3549 | +5.72% |

全局 body 位置改善 17.34%，配对 95% 区间 [14.75%, 19.94%]；global root 位置改善 17.34%，区间 [14.71%, 20.00%]。本次重测中，原对齐后 body 位置改善 9.11%，joint 位置改善 −0.30%（区间跨零）。

| DR | 全局 body 位置 baseline / 专家 (cm) | 全局位置改善 | 全局 body 姿态改善 |
|---|---:|---:|---:|
| B00 all_0kg | 15.202 / 12.008 | +21.01% | +28.04% |
| B01 all_1kg | 16.112 / 16.242 | -0.81% | +4.56% |
| B02 all_2kg | 23.239 / 19.775 | +14.90% | -7.99% |
| B03 hands_2p5_shins_4kg | 29.085 / 22.803 | +21.60% | -12.59% |
| B04 hands_2p5kg | 19.546 / 17.501 | +10.46% | -9.07% |
| B05 shins_4kg | 20.865 / 15.587 | +25.29% | +27.25% |
| B06 left_hand_2p5_shin_4kg | 21.572 / 16.874 | +21.78% | +4.37% |
| B07 right_hand_2p5_shin_4kg | 20.485 / 16.510 | +19.40% | +4.92% |

7/8 组全局 body 位置改善，且这七组的配对 95% 改善区间都大于零。B01 为 −0.81%，区间 [−4.82%, +3.12%]，没有明确差异。B02/B03 虽然对齐后的 body 和 joint 误差变大，但全局位置分别改善 14.90% / 21.60%；这两组全局 body 姿态误差仍增加 7.99% / 12.59%。结果进一步支持专门训练可改善多数测试 DR 的全局轨迹跟随，同时保留局部姿态与全局位置表现不同的事实。

本次失败数为 baseline 22/4096、专家 12/4096；上次是 20/4096、11/4096。重测的 checkpoint、物理指纹和初始状态一致；GPU 接触仿真存在非逐位确定性，因此如实保留两次结果。全局表使用本次共同存活窗口，不能与上次按各自存活窗口汇总的 root 值逐项混用。

12 项相关测试通过，全部 16 次评测及配对核验通过；完整 trace 可重建每条 episode 的全部指标，全局 anchor 的位置/旋转/速度与原指标定义数值一致。仅扩展评测，不更新 policy 参数或训练设置。单训练 seed 和八专家总预算八倍的适用边界仍然存在。

文件均在本次运行目录 `global_evaluation/`：`README.md`、`comparison.json`、`comparison.csv`、`comparison.png`、`comparison.svg`，另有原始评测及 `validation.json`。原评测数据保留不覆盖。
