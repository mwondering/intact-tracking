# 144000-exp-heavy：8500 轮 latent 聚类与物理结构

2026-09-21，CPU 离线评估。结论：相同物理环境在不同 motion 下的 latent 聚合明确，较 2000 轮进一步改善；连续物理参数的邻域结构更清晰，但当前结果没有支持“已经形成少数边界清楚的天然离散类别”。新增四肢负载质量有明显结构；负载质心中，双手已有一定改善，双小腿的整体邻域区分仍很弱。

评估对象固定为 `runs/144000-exp-heavy/stage1_proprio122_16384/update_008500.pt`，与 `update_002000.pt` 使用完全相同的验证历史比较。评估期间八卡训练继续运行，未修改训练参数、损失或采样。

## 数据与口径

- 八卡各自的 128 个验证 world 均排除在训练和归一化统计之外。筛选出 868 个有有效跨 motion 配对的 DR world，共 2892 对历史；包含 256 个四肢负载质量档位中的 252 个。
- 每对具有相同 world/session、不同 motion ID，短 50 步和长 300 步历史完整；archive age 至少 355 步。用于检索和聚类的每个 world 取固定的一对历史，共 868 对；逐对核对原始交互行，没有完全重复的历史行。
- 聚类输入为每个 world 的单窗口 64 维单位 latent；不先平均 world、不白化、不输入 DR 参数、不在 PCA 空间聚类。2000/8500 使用相同 world 和相同配对行。
- 几何汇总另外计算 world 内均值和 world 间距离；这是描述性统计，不是聚类输入。nominal 的完整历史统计含 123 个窗口、75 个 world；图中每个 nominal world 只取一个窗口。
- 这是训练中持续使用的固定验证缓存，不是刚采集的新 rollout，也不是未用于任何模型选择的最终测试集。跨 motion ID 不等于跨 motion family。以下结果不能代替闭环控制收益评估。

## 同一环境的聚合与连续物理结构

| 指标 | 2000 轮 | 8500 轮 |
|---|---:|---:|
| 跨 motion 同环境检索 Top-1 ↑ | 95.28% | **98.96%** |
| 跨 motion 同环境检索 Top-5 ↑ | 98.39% | **99.88%** |
| 环境内 RMS / 环境中心间 RMS ↓ | 18.18% | **16.63%** |
| DR 参数距离与 latent 中心距离 Spearman ↑ | 0.641 | **0.749** |
| 同环境跨 motion cosine，2892 对 ↑ | 0.9445 | **0.9510** |
| 完整历史 nominal 到固定锚点 RMS ↓ | 0.01939 | **0.01138** |
| DR 中心协方差有效秩 | 13.73 | **17.74** |

检索为 868 个 world 各一个当前 query、一个另一 motion 的 archived gallery，按单位 latent 的距离找同一 world；随机 Top-1 为 0.115%。8500 轮为 859/868 个 query 找回正确 world。这里 gallery 包含目标环境的示例，不是直接从 latent 预测一个未提供示例的新环境类别。

RMS 比值的分子为每个 world 内所有有效当前/历史视图到该 world 均值的均方距离，再对 world 等权平均后开方；分母为不同 world 均值之间的 RMS 距离。距离相关性使用训练 schema 的 108 维、16 个因子等权 DR 距离；pairwise distance 并非独立统计观测，不据此报告显著性 p 值。

## 能否稳定分成有限个类别

K-means 预先固定 K=8/16/32，`n_init=20, random_state=921`。868 个 world 分成两折，每次只用一折的当前视图拟合中心，在另一折上分别给当前/archived 视图分配类别。下表为两折平均，所有聚类计算都在原始 64 维单位 latent 上完成。

| K | 2000 跨 motion 类别一致率 | 8500 跨 motion 类别一致率 | 8500 ARI | 8500 当前视图轮廓系数 |
|---:|---:|---:|---:|---:|
| 8 | 74.77% | 76.38% | 0.543 | 0.0377 |
| 16 | 66.36% | 69.47% | 0.466 | 0.0301 |
| 32 | 59.10% | 62.67% | 0.395 | 0.0107 |

一致率高于按边际类别频率独立分配的约 12.83% / 6.57% / 3.50%，但轮廓系数仍低，说明在这些分组粒度下类别分离有限。并非所有指标都随训练增加而改善：K=8/16 的轮廓系数较 2000 轮略低，尽管跨 motion 标签一致性有所提高。

HDBSCAN 也分别对两种视图独立拟合，固定 `min_samples=8, allow_single_cluster=True`，EOM。8500 轮结果：

| min_cluster_size | 当前视图非噪声簇 | 另一 motion 视图非噪声簇 | 未归类比例 |
|---:|---:|---:|---:|
| 16 | 1 簇，16 点 | 1 簇，17 点 | 98.04%–98.16% |
| 32 | 1 簇，32 点 | 1 簇，32 点 | 96.31% |

这些设置没有找出覆盖大多数 DR 环境的自然密度分组。“噪声”是算法的未归类标记，不代表模拟环境非法或训练样本错误。结果与连续 DR 形成连续结构的解释相容，但不能证明所有算法和所有尺度都无法聚类。256 个负载档位是连续采样的分层，档位内其他参数和质量数值仍不同，不应直接作为 256 个天然类别。

## 哪些物理因素体现在 latent 邻域中

对每个当前 query，排除同一 world，在其他 world 的 archived latent 中找最近 10 个邻居。比较它们的真实参数差异与所有其他 world 的平均参数差异。下表为**参数差异 MSE 相对随机邻居的下降比例**，越高表示这个因素在 latent 邻域中越一致；多轴/多关节组先对组内坐标等权平均。参数按配置范围归一化。

这不是参数预测准确率、不是 decoder 的 R²，也不是单因素干预实验；其他同时变化的参数也会影响 latent 距离。

| 参数因子 | 2000 轮 | 8500 轮 |
|---|---:|---:|
| torso COM x | 72.62% | 70.84% |
| torso COM y | 72.39% | 70.06% |
| torso COM z | 46.01% | 58.85% |
| 摩擦 | 62.74% | 64.32% |
| torso 质量 | 7.13% | 6.61% |
| 左手负载质量 | 60.28% | **65.33%** |
| 右手负载质量 | 60.54% | **66.03%** |
| 左小腿负载质量 | 34.06% | **54.41%** |
| 右小腿负载质量 | 37.14% | **52.97%** |
| 左手负载 COM xyz | 0.38% | **10.03%** |
| 右手负载 COM xyz | 0.67% | **13.58%** |
| 左小腿负载 COM xyz | -0.95% | **0.39%** |
| 右小腿负载 COM xyz | 0.31% | **1.41%** |
| Kp，29 维平均 | 12.84% | 13.79% |
| Kd，29 维平均 | 0.14% | 0.24% |
| armature，29 维平均 | 0.12% | 0.02% |

负载质量，特别是双小腿负载，比 2000 轮更能影响局部邻域。双手负载 COM 开始体现出结构，但双小腿 COM 在此指标上接近随机，尚不能说新增质心因素已经充分学好。由于这是原 latent 的几何诊断，仍不能据此断言小腿 COM 在 latent 中完全没有可解码信息。

## 图与复现

![8500 latent 几何与物理邻域](../runs/144000-exp-heavy/latent_geometry_u8500_20260921/latent_clustering_8500.png)

图中 PCA 仅用于显示：nominal+DR 前两维解释 29.2% 方差，仅 DR 的前两维解释 17.5%；不能仅凭二维投影判断原 64 维距离。所有定量指标使用原空间计算。

- [导出 PDF](../runs/144000-exp-heavy/latent_geometry_u8500_20260921/latent_clustering_8500.pdf)
- [几何结果、checkpoint 和验证缓存 SHA-256](../runs/144000-exp-heavy/latent_geometry_u8500_20260921/summary.json)
- [完整聚类与参数邻域结果](../runs/144000-exp-heavy/latent_geometry_u8500_20260921/clustering_summary.json)
- [保存的 world、分折和所有聚类标签](../runs/144000-exp-heavy/latent_geometry_u8500_20260921/clustering_labels.npz)
- [本次补充分析脚本](../runs/144000-exp-heavy/latent_geometry_u8500_20260921/analyze_clusters.py)
- [缓存重编码与几何脚本](../scripts/evaluate_native_encoder_geometry.py)

已从保存标签独立重算全部 K-means 一致率和 HDBSCAN 簇/噪声数量；检索排序也以 Euclidean 距离独立复算并与原 cosine 排序逐项一致。离线依赖沿用 `.runtime/latent_auto_cluster_deps` 中的 scikit-learn 1.8.0，训练虚拟环境未改动。

```sh
# 重新编码时使用一个尚不存在的新目录。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python scripts/evaluate_native_encoder_geometry.py \
  --run runs/144000-exp-heavy/stage1_proprio122_16384 \
  --updates 2000 8500 --output <new_output_directory>

# 复算现有缓存上的聚类和图。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=/data_zcy/wxy/intact-tracking/.runtime/latent_auto_cluster_deps \
  .venv/bin/python runs/144000-exp-heavy/latent_geometry_u8500_20260921/analyze_clusters.py
```
