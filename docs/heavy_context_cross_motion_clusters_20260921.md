# Heavy encoder：同一环境、不同 motion 的 latent 聚集

本图的主问题是：固定物理环境后，换不同 motion 产生的 latent 是否仍然聚在一起。颜色表示**环境身份**，点形表示 **motion 身份**，冻结 tracker 和 residual latent policy 分面展示。

## 采集与作图口径

- 固定 `update_008816.pt` context encoder、冻结 tracker 和 uniform residual 的 `checkpoint_2800.pt`；没有训练或更新 encoder/policy。
- 新建 256 个 HDR 物理环境，seed=20260924，每个环境的 108 维持续物理参数固定。所有环境都执行同一组 8 个 motion，形成 environment × motion × policy 的完整组合。
- 从此前 256 条训练目录 motion 中，按原文件帧数至少 700 筛出 86 条，再以 seed=924 随机选择 8 条。没有根据 latent 聚集效果或 rollout 成败选择 motion；这也不是留出的 motion family 测试集。
- 每个 policy/motion 使用独立仿真和空的短、长记忆，运行 650 个控制步。每个点只取一个完整的 50+300 步历史窗口；同一环境/motion 的 tracker、residual 取首个共同有效时刻。没有平均多个窗口，也没有把相邻帧当成不同 motion 来增加簇内点数。
- 同一环境的物理与初始控制器抽样在全部 motion/policy 中一致。delay、smoothing 和 joint offset 在本实验中固定；同一 motion 的两种 policy 初态与完整外力序列配对。控制协议沿用[跨 policy 实验](heavy_context_cross_policy_representation_20260921.md)。
- 采集前以 seed=921 固定选定 16 个展示环境：E006、E007、E030、E036、E047、E058、E065、E067、E087、E102、E143、E151、E181、E195、E205、E226。不会根据聚集效果替换展示环境。缺少完整历史的环境单独记录。
- 所有拥有完整 8-motion 配对数据的环境一起拟合一次无监督 t-SNE，输入为 64 维单位 latent，不输入环境或 motion 标签，不按环境先平均，不对不同 policy 做单独对齐。简洁主图只显示预选环境；原始三面板版本还以淡灰色显示其他环境。所有面板使用同一次拟合的坐标。
- 主 t-SNE 固定 perplexity=30、seed=921；另外保存 perplexity=50、seed=922 及 PCA 供敏感性检查。距离与检索均在原始 64 维计算。

| 点形标签 | Motion 文件对应动作 |
|---|---|
| M1 | 100style / Quail_BR |
| M2 | 剪刀工具使用 |
| M3 | 洗碗与厨房清理 |
| M4 | 避障移动 |
| M5 | 挥手 |
| M6 | 人群避障 |
| M7 | 排球垫球 |
| M8 | 跪姿循环 |

完整文件名与预先固定的选样规则见[采集协议](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/protocol.json)。

## 更新后的主图

256 个环境全部取得了两种 policy × 8 个 motion 的完整历史，共 **4096 个独立历史窗口**；没有排除环境。预先选定的 16 个展示环境全部保留。每个 policy 面板中，每种颜色对应 8 个不同形状的点。

![t-SNE：相同环境跨八个 motion 的 latent 聚集](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_tsne_clean.png)

[高清 PNG](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_tsne_clean.png) · [可放大的 PDF](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_tsne_clean.pdf) · [包含其他环境背景的三面板版本](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_clusters.png)

图中同色的不同形状总体聚在一起，tracker 和 residual 两个面板都保留这一结构。颜色直接表示完整物理环境的身份，点形 M1–M8 对应八条不同 motion。环境和 motion 标签仅用于着色、标记和评估，没有参与降维。

简洁版增加明确的 t-SNE 坐标轴并去掉其他环境的灰色背景；每个面板仍为 16 个预选环境 × 8 个 motion，共 128 个点。坐标逐点等于原始保存值，没有重新拟合、抖动或移动点，也没有删去预选环境中的离群点。[出图记录](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_tsne_clean.json) 和[展示坐标](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_tsne_clean.npz)保留了核对依据。

### 原始 64 维中的支持证据

| 指标 | Frozen tracker | Residual latent policy |
|---|---:|---:|
| 同环境、不同 motion 距离 RMS | 0.4212 | 0.4385 |
| 不同环境、相同 motion 距离 RMS | 1.0049 | 1.0003 |
| 环境内 / 环境间 RMS | 0.4191 | 0.4383 |
| 同环境跨 motion 平均 cosine | 0.9113 | 0.9039 |
| 按环境 ID 的轮廓系数 | 0.3925 | 0.3646 |
| 按 motion ID 的轮廓系数 | -0.0217 | -0.0183 |

同环境不同 motion 的距离约为跨环境距离的 **42%–44%**；环境分组的轮廓系数为正，motion 分组为负。这些原空间结果支持图中“表征主要按环境组织、跨 motion 保持一致性”的解释，不只依赖二维投影。

![原始 64 维的环境内外距离](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_distances_64d.png)

### 排除同 motion 的环境检索

每个 query 在其他 7 个 motion 的 latent 中找同一环境；每个候选环境有 7 个 gallery 示例。随机 Top-1 为 1/256 = **0.3906%**。

| Query → Gallery | Top-1 | Top-5 | Top-1 环境 bootstrap 95% 区间 |
|---|---:|---:|---:|
| tracker → tracker | 98.29% | 99.80% | [97.75%, 98.78%] |
| tracker → residual | 98.19% | 99.66% | [97.66%, 98.73%] |
| residual → tracker | 98.39% | 99.71% | [97.85%, 98.93%] |
| residual → residual | 98.24% | 99.76% | [97.71%, 98.78%] |

并非所有 motion 都同样稳定：M8 跪姿循环作为 query 时，同 policy 检索率分别为 **87.50% / 87.89%**，其他 motion 均至少为 **98.44%**。图中保留了偏离同色簇的星形点，没有删除难例。因此可以表述为“同环境跨多种 motion 总体形成紧凑聚集”，不应表述为“所有 motion 的 latent 完全重合”。

### 敏感性检查与原始结果

- [另一 perplexity / seed 的 t-SNE](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_clusters_sensitivity.png) · [PDF](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_clusters_sensitivity.pdf)。
- [相同样本的 PCA 图](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_clusters_pca.png) · [PDF](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/environment_motion_clusters_pca.pdf)。
- [完整汇总](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/summary.json) 与 [原始 latent、窗口选择、距离、检索索引、二维坐标](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/analysis/analysis_arrays.npz)。
- [独立复核](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/verification.json)：用 Euclidean 距离重算四个检索方向，逐 query 验证排除同 motion；重算两种 policy 的距离分布；确认 4096 个完整窗口无完全重复、16 个展示环境没有替换。

该八 motion 实验采用新的 256 个环境和固定的八条长 motion，统计口径不同于此前的 511 环境、两 motion 评测；不把两组数字的差异解释成 encoder 性能变化。


## 定量检查如何对应这张图

1. **同环境跨 motion 距离**：每个环境 8 个点之间的 28 个距离，所有环境等权。比较对象为相同 motion 下不同环境的距离，避免把换环境与换 motion 混在一起。
2. **排除同 motion 的环境检索**：对每个 query，检索库去掉所有环境中与它相同 motion 的点。每个候选环境有其余 7 个 motion 的示例，检验最近邻是否仍来自同一环境。这个口径不能直接与之前每环境只有一个 gallery 的 Top-1 数字比较。
3. **原空间轮廓系数**：分别以环境 ID、motion ID 为标签，检查分组与 64 维几何的相容性。这里的环境 ID 对应实验中具体的物理样本，不是在证明连续 DR 自然分成少数离散类别。

各距离对和来自同一环境的 8 个 query 都不是独立观测。检索区间按环境整体 bootstrap；只描述本次固定 checkpoint、motion 集和物理种子的结果。

## 脚本与复现

- [采集脚本](../scripts/probe_heavy_cross_policy_latents.py) 新增 `--fixed-motion-index`，让一个视图中的所有环境执行同一 motion；原有两视图入口保持默认行为。
- [分析与出图脚本](../scripts/plot_heavy_environment_motion_clusters.py)。
- [简洁 t-SNE 出图脚本](../scripts/render_heavy_motion_tsne.py)：读取现有 `analysis_arrays.npz`，运行 `.venv/bin/python scripts/render_heavy_motion_tsne.py runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921` 即可重新导出 PNG、PDF 和展示坐标，不重新采集或拟合。
- [完整采集命令与退出状态](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/collection_complete.json)。
- [指标自检](../runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/metric_self_check.json)：有环境信号时跨 motion 检索正确，错配环境标签后失败，仅有 motion 信号时降至随机水平。
- 采集和分析源码快照、逐步数据与日志均保存在 `runs/144000-exp-heavy/environment_motion_clusters_u8816_20260921/`。

重新采集需使用一个新的输出根目录。已有数据的分析也拒绝覆盖 `analysis/`；在包含全部 `tracker_m0`–`tracker_m7`、`residual_m0`–`residual_m7` 和 `protocol.json` 的新目录中运行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=.runtime/latent_auto_cluster_deps \
  .venv/bin/python scripts/plot_heavy_environment_motion_clusters.py <fresh_root>
```
