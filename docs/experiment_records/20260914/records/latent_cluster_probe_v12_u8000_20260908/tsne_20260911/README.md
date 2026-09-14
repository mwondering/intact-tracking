本目录是 update_008000.pt 的 latent t-SNE，可先打开 [交互图](tsne_interactive.html)，或看 [三联细节图](tsne_detail.png)。

使用前面已采集的 latent，未重新训练。全样本是 24,345 个不重叠的完整 100 步历史窗口：12,159 个 nominal，12,186 个 DR，覆盖 512 个固定 DR world 和 42 个 motion 文件。phase 是 motion 文件内的归一化进度，不是对不同动作做语义对齐后的步态相位。

细节图先用固定随机种子 20260911 均匀选定 16 个 DR world，再保留其全部主窗口；nominal 每个 motion 最多选 3 个窗口。总计 503 点，其中 DR 379 点、nominal 124 点。M16 只有 1 个 nominal 主窗口。所选每个 DR world 覆盖 5–7 个 motion，没有覆盖所有环境与 motion 的完整组合。选样在 t-SNE 拟合之前完成，环境和 motion 标签只用于着色。

三联图从左到右按环境、具体 motion 文件、phase 着色，使用完全相同的点和坐标。黑色是 nominal；DR 颜色表示固定物理参数的 world ID。

| 图 | 用途 |
| --- | --- |
| [交互图](tsne_interactive.html) | 切换着色、悬停查看 motion 文件和 phase、缩放；点击两个点计算原始 64 维 L2 距离、单位化 L2 距离和余弦相似度。HTML 自带 Plotly，可在浏览器中离线打开。 |
| [细节图 PNG](tsne_detail.png) / [PDF](tsne_detail.pdf) | 比较预先随机选出的 16 个 DR 环境和 nominal。 |
| [全样本总览 PNG](tsne_overview.png) / [PDF](tsne_overview.pdf) | 使用全部 24,345 点；环境面板以灰色显示其余 DR，突出相同的 16 个 DR world。 |
| [真实距离热图 PNG](raw_latent_distances.png) / [PDF](raw_latent_distances.pdf) | 同一批 503 点在原始 64 维空间的两两欧氏距离，按环境、motion、phase 排序；深色代表距离小。 |
| [Nominal 细节](tsne_nominal.png) | 从全样本嵌入取出 nominal，按 motion 和 phase 着色；没有单独重新拟合。 |
| [参数对照图](tsne_settings.png) | 相同 503 点，比较 raw latent 的 perplexity 10/30/50，以及单位化 latent 的 perplexity 30。 |

从细节图看，所选 DR 环境多数各自聚集；相同位置改为 motion 或 phase 着色后，多数簇内包含多种颜色，说明环境信号跨 motion 保留。仍可看到少量偏离主簇的点、局部混合和 motion 相关结构。这些图不能证明全部 512 个 DR 环境都能完全区分，也不能单凭可视化证明下游 policy 会利用这些信号。

距离热图可看到较暗的环境内对角块及更亮的部分环境间块。nominal 在 t-SNE 上占据较大面积，不能解释成它在原始 latent 空间中更分散：t-SNE 使用局部相似性概率，簇面积和全局间距没有统一的原始距离尺度。要判断两个环境的距离是否够大，应结合热图或交互图中的 64 维距离。算法作者也明确指出 t-SNE 保留的是相似性概率，不能直接比较高低维距离。[t-SNE 作者 FAQ](https://lvdmaaten.github.io/tsne/)

主图输入为原始 64 维 encoder 输出，Euclidean metric；没有按维标准化，也没有先降到较低维度。PCA 仅用于初始化。细节图 perplexity=30、1,500 iterations；总览 perplexity=50、1,000 iterations；random_state=20260911、learning_rate=auto、early_exaggeration=12、Barnes–Hut angle=0.5。单位化 latent 只在参数对照图中使用。细节与总览分别拟合，不能对照它们的绝对坐标。实现参数说明见 [scikit-learn TSNE 文档](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)。

checkpoint：`/data_zcy/wxy/intact-tracking/runs/forward_predictor_transformer_v12_8192_run3/update_008000.pt`  
SHA256：`d22246545f9530b2b1154e6e68e8632c41add1eb82fb9830830a1d6b52fcb0bf`

选中的 DR world：527, 584, 589, 666, 682, 724, 753, 791, 794, 842, 876, 880, 897, 904, 919, 984。每个图的坐标、输入数据哈希、拟合设置、选样行号保存在 [tsne_manifest.json](tsne_manifest.json) 和 `embedding_*.npz`。行号相对于过滤掉 `neighbor` 后的主窗口数组。

复现脚本为 [plot_forward_context_tsne.py](/data_zcy/wxy/intact-tracking/scripts/plot_forward_context_tsne.py)。新增绘图依赖安装在临时目录 `/tmp/intact-tsne-deps`，没有改动训练虚拟环境。

```bash
# 在 /data_zcy/wxy/intact-tracking 下运行；临时依赖目录不存在时先安装
uv pip install --python .venv/bin/python --target /tmp/intact-tsne-deps --no-deps scikit-learn==1.9.1 joblib==1.6.0 threadpoolctl==3.6.0 plotly==7.0.0 narwhals==2.26.0
env PYTHONPATH=/tmp/intact-tsne-deps OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python scripts/plot_forward_context_tsne.py \
  --input runs/latent_cluster_probe_v12_u8000_20260908 \
  --output runs/latent_cluster_probe_v12_u8000_20260908/tsne_20260911
```

已检查绘图输出及嵌入数值。交互距离计算用两个真实样本在模拟 DOM 中校验，结果与独立计算一致；未运行实际浏览器自动化测试。见 [verification.json](verification.json)。

具体 motion ID 对照：

| ID | motion 文件（相对于 lafan_qingtong） |
| --- | --- |
| M00 | dance1_subject1.motion.npz |
| M01 | dance1_subject2.motion.npz |
| M02 | dance1_subject3.motion.npz |
| M03 | dance2_subject1.motion.npz |
| M04 | dance2_subject2.motion.npz |
| M05 | dance2_subject3.motion.npz |
| M06 | dance2_subject4.motion.npz |
| M07 | dance2_subject5.motion.npz |
| M08 | fallAndGetUp1_subject1.motion.npz |
| M09 | fallAndGetUp1_subject4.motion.npz |
| M10 | fallAndGetUp1_subject5.motion.npz |
| M11 | fallAndGetUp2_subject2.motion.npz |
| M12 | fallAndGetUp2_subject3.motion.npz |
| M13 | fallAndGetUp3_subject1.motion.npz |
| M14 | fight1_subject2.motion.npz |
| M15 | fight1_subject3.motion.npz |
| M16 | fight1_subject3_612_680.motion.npz |
| M17 | fight1_subject5.motion.npz |
| M18 | fightAndSports1_subject1.motion.npz |
| M19 | fightAndSports1_subject4.motion.npz |
| M20 | jumps1_subject1.motion.npz |
| M21 | jumps1_subject2.motion.npz |
| M22 | jumps1_subject5.motion.npz |
| M23 | run1_subject2.motion.npz |
| M24 | run1_subject5.motion.npz |
| M25 | run2_subject1.motion.npz |
| M26 | run2_subject4.motion.npz |
| M27 | singlejump/jumps1_subject1.motion.npz |
| M28 | sprint1_subject2.motion.npz |
| M29 | sprint1_subject4.motion.npz |
| M30 | walk1_subject1.motion.npz |
| M31 | walk1_subject2.motion.npz |
| M32 | walk1_subject5.motion.npz |
| M33 | walk2_subject1.motion.npz |
| M34 | walk2_subject3.motion.npz |
| M35 | walk2_subject4.motion.npz |
| M36 | walk3_subject1.motion.npz |
| M37 | walk3_subject2.motion.npz |
| M38 | walk3_subject3.motion.npz |
| M39 | walk3_subject4.motion.npz |
| M40 | walk3_subject5.motion.npz |
| M41 | walk4_subject1.motion.npz |

