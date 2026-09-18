# FiLM latent 早期增益检查：4×8192，零预热

2026-09-16。正式训练继续运行，未改训练参数。本检查读取当前四卡对四卡任务日志，并对两组各自的 `checkpoint_update_000250.pt` 做确定性配对评测。冻结 context 均为 u35857；没有使用旧 4×1024 checkpoint。

## 对齐训练曲线

固定比较第 301–400 轮：latent reward 低 0.40%，对齐 body position 误差低 2.04%，root position 高 5.06%，joint position 高 4.58%。两组最近 critic value loss 约 0.193，长期记忆完整率均为 100%。Latent 组同一观测下打乱 latent 的动作 RMS 差约 0.0267；全部 FiLM 动作相对原 tracker 的 RMS 修正约 0.1346。前者不是控制增益或可加分解的贡献比例。

曲线：`training_comparison.png/svg`。训练日志来自不同轨迹及探索动作，不能当作配对评测。

## 共同 u250 checkpoint 配对评测

从既有 512-motion 测试列表用 seed 20260916 固定抽取 128 条 motion；每条两个不同的连续随机 DR，共 256 个世界。四组使用同一起点、同一物理指纹、噪声 seed 20001、FP32、cold memory，最长 1000 步或到 motion 结束。共同存活窗口比较，并先平均同一 motion 的两个 DR，再按 motion 做 2000 次配对 bootstrap。所有配对审计通过，四组均无失败，覆盖率均为 100%。

| 指标 | baseline | latent | latent 误差降低 | motion 配对 95% 区间 |
|---|---:|---:|---:|---:|
| 对齐 body pos (cm) | 3.1652 | 3.0423 | +3.88% | [+2.45%, +5.27%] |
| 全局 body pos (cm) | 17.6078 | 18.7886 | -6.71% | [-10.89%, -2.23%] |
| Root pos (cm) | 17.0744 | 18.3637 | -7.55% | [-11.94%, -2.95%] |
| Joint pos (rad) | 0.4583 | 0.4774 | -4.18% | [-5.30%, -3.16%] |

误差降低为负表示 latent 更差。平均 episode return：baseline 97.5527，真实 latent 97.2840，latent 置零 97.3219，latent 错配 97.1636。

## 同一 latent policy 的输入干预

错配在同 motion、同 phase、不同 DR 的两个世界间互换在线 latent，只在双方均存活且短历史完整时启用，占实际评测步数 85.57%。真实 latent 相对错配在局部 body 上有小收益：错配使局部 body 误差增加 1.72%，但全局 body/root 的变化区间跨零。置零 latent 时局部 body/root 差异区间也跨零，joint 误差反而下降约 1.08%。干预只改变 actor 输入，评测不运行 critic。

只看第 350 步之后、两份 DR 都仍有评测样本的 51 条 motion：latent 相对 baseline 的局部 body 误差降低约 4.56%，全局 body/root/joint 误差分别增加 10.25% / 10.67% / 6.85%。该子集与全体 motion 构成不同。按归一化四肢负载总量分轻/重两半，latent 的全局 body/root 在两半都未优于 baseline；这些分层仅为描述性点估计。

## 判断与边界

当前证据支持：额外 latent 没有提供整体控制优势；正确 latent 对局部姿态有可测的小幅作用。动作对 latent 敏感不等于总体指标改善。Baseline 同样具有 obs 条件化 FiLM，因此这组比较衡量额外环境信息的增益。通用反馈补偿占主导、额外信息仅有弱作用，是当前数据支持的候选解释；无法单独定位到 encoder、actor、critic 或 reward。

训练日志在 150–250 轮期间显示 latent 组 critic loss 一度更高，到 301–400 已接近，不能据此认定 critic 是原因。这里为单训练 seed、较早 u250 checkpoint 的 128-motion cold 评测；置信区间只描述 motion 采样，不包含训练随机性。Cold 评测会重新清空记忆，区别于训练后期跨 episode 保留长期记忆。350 步后子集提供补充，不能替代完整 warm-history 评测，也不能判定最终收敛结果。

原始证据：`training_windows.json`、`reward_components.json`、`evaluation_protocol.json`、四组 JSON/逐步 traces、`paired_results.json`。


证据目录：`runs/limb_context_20260916_tracker_film_obs_latent_4x8192_cold/diagnostic_early_latent_benefit`。

[训练曲线](../runs/limb_context_20260916_tracker_film_obs_latent_4x8192_cold/diagnostic_early_latent_benefit/training_comparison.png) · [完整配对结果](../runs/limb_context_20260916_tracker_film_obs_latent_4x8192_cold/diagnostic_early_latent_benefit/paired_results.json)
