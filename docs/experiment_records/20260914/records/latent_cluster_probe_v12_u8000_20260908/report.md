测量结果：这个 checkpoint 的 nominal latent 已经形成明显的紧簇；固定 DR world 也有很强的跨 motion 一致性。nominal 存在少量与 motion/phase 有关的尾部波动，尚非严格常量。当前结果没有提供必须把 nominal 硬压到原点的依据。

Checkpoint：`/data_zcy/wxy/intact-tracking/runs/forward_predictor_transformer_v12_8192_run3/update_008000.pt`；SHA256：`d22246545f9530b2b1154e6e68e8632c41add1eb82fb9830830a1d6b52fcb0bf`。

2026-09-08，在 GPU 0 上用冻结的原 tracker 采集 1,024 个 world × 3,000 steps（3,072,000 transitions）。512 个 compiled nominal world，512 个固定 startup-DR world；无额外 payload，采样 seed=81208。encoder、tracker、归一化均来自原 checkpoint，没有更新权重。原训练为 4 卡，配置中的 11 个 motion 是每卡分片；本次单卡读取同一目录的全部 42 个 motion 文件、15 个动作家族。它们属于训练 motion 目录，本次是新的物理环境/轨迹采样，不是未见 motion 的泛化测试。

只保留完整的连续 100 帧 causal history。每 100 steps 采一个主窗口，同一 world 的主窗口不重叠；reset 与 motion resample 会清空历史。另采 +5 帧窗口，只用来对照局部相似性。共 24,345 个主样本：nominal 12,159，DR 12,186；两类完整历史覆盖率分别为 79.16% 和 79.34%。每个 DR world 覆盖 4–8 个 motion，中位数 6。归一化 phase 是当前 reference frame / (motion length - 1)。

表征训练使用单位化 latent 的距离，因此主结果在单位化后的原始 64 维空间计算。下面的距离都是 RMS 欧氏距离；余弦是配对样本的平均值。远 phase 的 nominal 配对来自不同 world；相近 phase 的跨环境配对并非完全相同状态或反事实轨迹。

| 配对条件 | 配对数 | 距离 RMS | 平均余弦 | 距离 P95 | RMS 的描述性 95% 区间 |
| --- | ---: | ---: | ---: | ---: | --- |
| nominal：不同 motion | 12,159 | 0.07597 | 0.997115 | 0.18371 | [0.07366, 0.07799] |
| nominal：同 motion，phase 相差至少 0.25 | 12,158 | 0.07164 | 0.997434 | 0.17452 | [0.06888, 0.07413] |
| nominal / DR：同 motion，phase 相差不超过 0.02 | 12,110 | 0.58936 | 0.826327 | 0.81720 | [0.58590, 0.59293] |
| 同一 DR world：不同 motion | 12,186 | 0.23481 | 0.972432 | 0.52191 | [0.22725, 0.24262] |
| 不同 DR world：同 motion，phase 相差不超过 0.02 | 12,130 | 0.74414 | 0.723127 | 1.14174 | [0.73501, 0.75383] |
| nominal：相邻 +5 帧（95% 历史重叠） | 11,662 | 0.01484 | 0.999890 | 0.02089 | [0.01368, 0.01617] |
| DR：相邻 +5 帧（95% 历史重叠） | 11,621 | 0.03593 | 0.999354 | 0.06413 | [0.03435, 0.03757] |

nominal 跨 motion 的距离为 0.0760，同 motion 远 phase 为 0.0716；nominal / DR 同 motion 近 phase 为 0.5894，约是 nominal 跨 motion 距离的 7.76 倍。同一 DR world 跨 motion 距离为 0.2348，不同 DR world 同 motion 近 phase 为 0.7441，前者仅为后者的 31.6%。

nominal 相对其中心的单位 latent RMS 半径为 0.05365，中位半径 0.01768，P95 半径 0.12817。该 RMS 半径是 nominal / DR 配对距离的 9.1%。原始 nominal latent 的中心范数为 7.10511，原始 RMS 半径 0.38165：它聚在一个非零点附近。PCA 图中的零点是数据中心化后的坐标，并非 encoder 原始原点。

为避免“所有 latent 都有相似公共偏置”带来的误判，另做了只训练诊断读出的分析：

- nominal 的 motion 均值在独立 world 上解释其剩余中心化方差的 13.8%；motion × 10 个 phase 分桶均值解释 29.0%。说明紧簇内部仍保留一些运动相关结构；这两个比例指小范围的 nominal 簇内方差。
- 同时留出物理 world 和动作家族后，nominal / DR 固定正则的线性读出 balanced accuracy 为 95.21%，ROC-AUC 为 0.99442。这说明可分性，不单独代表紧致性。
- 对 495 个有两侧样本的固定 DR world，用一半动作家族估计各 world 中心，在另一半动作家族测试，中心解释的方差为 88.2%；最近中心的环境识别 top-1 为 56.65%，top-5 为 83.86%，均匀 top-1 机会水平为 0.20%。不同 DR 参数连续分布，相近环境之间仍可重叠。

物理审计确认 nominal 的展开模型字段和 encoder bias 恢复误差均为 0；采样结束后 fixed-DR 检查通过。BF16 与 FP32 对第一批完整历史的平均余弦为 0.99999607。另外检查了主窗口间隔、phase 范围和 latent 有限性；统计函数通过常量零半径与合成 motion 信号恢复检查。

这些结果只支持当前 checkpoint、该 motion 目录和完整历史下的表征结论。短历史/失败后的不确定性、未见动作、把 nominal 替换成常量后的预测精度、闭环控制收益没有在本次测试中测量。配对区间按 query world 做 400 次 bootstrap，伙伴样本有重复，区间仅作描述；没有把所有 pair 当成完全独立样本。

文件：

- [整体分布、nominal motion/phase 与距离图](latent_clusters.png)，[PDF](latent_clusters.pdf)。
- [前 8 个 DR world 的跨 motion/phase 分布](fixed_dr_world_clusters.png)：固定选取前 8 个 world，未按可分程度选择。
- [nominal 各 motion 中心距离](nominal_motion_centroids.png)。
- [完整指标](cluster_metrics.json)、[采样配置与覆盖率](metadata.json)、[检查结果](verification.json)。
- `latents.npz` 保存所有 latent、标签、当前状态与采样步；`physics.npz` 保存诊断用物理标签；`nominal_reference.npz` 保存此次测得的 nominal 参考均值，仅作诊断，不写入模型。

复现（输出目录需为空）：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python scripts/probe_forward_context_clusters.py --checkpoint runs/forward_predictor_transformer_v12_8192_run3/update_008000.pt --output runs/latent_cluster_probe_v12_u8000_repeat --num-envs 1024 --steps 3000
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/analyze_forward_context_clusters.py runs/latent_cluster_probe_v12_u8000_repeat
```
