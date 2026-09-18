**随机 DR 中直接抽 latent，自动确定聚类数**

2026-09-16。按用户最新澄清，从随机 DR 的真实轨迹中随机抽取单个 latent，直接聚类；不先计算环境均值或环境原型，不在 64 维空间随机生成虚构向量。DR 参数只保留作审计与后续环境配置关联，不输入聚类算法。

本次是 CPU 离线诊断，当前策略、encoder、router 和训练环境均未修改。诊断依赖安装在 `.runtime/latent_auto_cluster_deps`，通过单独进程的 PYTHONPATH 使用，没有改动训练虚拟环境的依赖。

**数据与采样**

- 使用当前冻结 u35857 权重重新编码过的原始 LaFAN tracker 历史。encoder SHA-256：`3c489cf5457889bf88441175ec4f10f4448104f44f7914eaaf762a9c3659d8f7`。
- 512 个独立连续随机 DR world，原 tracker 背景 DR 加手部 0–2.5 kg、小腿 0–4 kg；不是 256 档位网格，也不是仅负载变化的 PPO 轨迹。
- 共有 14006 个短50＋长30chunks完整的真实 latent 窗口。每个 world 均匀随机选一个窗口，得到 512 个点；避免同一 world 的大量重复帧主导密度。仅将每个点归一化到单位长度，不做均值、白化、PCA 或物理投影。
- 随机种子 731、1731、2731 重复三次抽窗口。每次使用相同的 512 个物理 world，窗口可能重复或历史重叠；这些不是三个独立环境测试集或三次 policy 训练。
- 已回读保存的 source_row，逐项确认聚类输入就是原缓存中的单个单位 latent；全部 18 组结果的类别数和噪声数已从保存标签独立重算。

**初测结果**

| 算法及设置 | 三次抽样的实际非噪声类别数 | 噪声点数量，共 512 点 |
|---|---|---|
| HDBSCAN，min_cluster_size=16 | 1、1、1 | 404、377、373 |
| HDBSCAN，min_cluster_size=32 | 1、1、1 | 480、429、373 |
| HDBSCAN，min_cluster_size=64 | 1、1、1 | 447、429、448 |
| Affinity Propagation，默认 median preference | 40、48、42 | 0、0、0 |
| Bayesian GMM，上限16 | 1、2、1 | 0、0、0 |
| Bayesian GMM，上限32 | 1、2、1 | 0、0、0 |

HDBSCAN 固定 `min_samples=8`、EOM、`allow_single_cluster=True`；允许输出单簇，不强制至少两类。其噪声占比为 72.85%–93.75%。被标为噪声的点仍来自真实合法 DR，不应直接从控制评测或训练覆盖范围中删除。

Affinity Propagation 使用 damping=0.9、max_iter=1000、convergence_iter=50、random_state=731；各次采用各自相似度中位数作为默认 preference。三次分组之间的 ARI 为 0.405、0.354、0.314，表明抽取不同历史后分组有明显变化。ARI 衡量两个分区在忽略类别编号置换后的相似性，不是运行时 router 的准确率。

Bayesian GMM 使用 Dirichlet-process 权重先验、concentration=0.1、对角协方差、reg_covar=1e-6、n_init=3、max_iter=1000，固定初始化种子731。这里类别数按硬分配实际占用的 component 统计，完整 mixture weights 也已保存。两类时均为 509 个点加 3 个点；它没有形成适合直接训练有限个均衡专家的分区。上述结论仅适用于这些设置和抽样，不意味着其他协方差、先验或更大样本量也会相同。

三类方法本轮均无收敛警告。所有配置在查看结果前固定，没有依据本轮输出调参或把某次抽样选择为最终路由。

**如何理解“自动确定数量”**

[HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)不要求预先给出 K，但需要最小簇规模和密度相关参数，并可留下未归类点。[BayesianGaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)用先验和数据推断有效分量，实现中仍有截断上限。[Affinity Propagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html)自动选代表点，数量受 preference 影响。省去显式 K 不等于省去分组粒度的选择。

这批数据尚未提供一个可信的自动专家数量。结果与“DR 连续变化、不一定存在少量明显密度簇”的解释相容，但不能仅凭这轮抽样证明 latent 没有结构或不能分组。已确认负载等参数可从 latent 线性读出，和存在若干天然离散类别，是两个不同问题。

若目标是用若干专家覆盖连续环境，可进一步比较 [DP-means](https://arxiv.org/abs/1111.0352)：它在 k-means 的类内距离目标上加入簇数量惩罚，距离现有中心足够远时可产生新簇，不预先指定 K。需要选择距离/数量惩罚尺度；该方法尚未在本次数据上运行，不能宣称优于上表算法。

正式流程仍可保持“随机 DR → 随机单窗口 latent → 自动聚类 → 保存原参数与类别关联”。但单窗口标签首先属于样本，是否能够作为 DR 环境标签，需要用同一 DR 的其他 motion/history 样本检查。可以检查标签投票与一致性，不必构造环境平均原型。还应记录每类包含多少不同 DR，而非仅有多少 latent 帧；完整 startup 参数或可验证的物理状态需随正式采集保存，才能重建专家训练环境。

**可复核产物**

- [完整配置、结果与来源](../runs/limb_context_20260916_random_latent_auto_clusters/summary.json)
- [实际抽样行、latent、参数及类别](../runs/limb_context_20260916_random_latent_auto_clusters/samples_and_labels.npz)
- [诊断脚本](../scripts/probe_random_latent_auto_clusters.py)

```sh
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=/data_zcy/wxy/intact-tracking/.runtime/latent_auto_cluster_deps .venv/bin/python -B scripts/probe_random_latent_auto_clusters.py --readout runs/limb_context_20260916_dr_readout_u35857 --output runs/limb_context_20260916_random_latent_auto_clusters
```

输出目录需尚不存在；使用 scikit-learn 1.8.0、joblib 1.5.3、threadpoolctl 3.6.0，以及项目现有 NumPy/SciPy。旧的环境原型聚类不是本页结果的数据来源或预处理步骤。
