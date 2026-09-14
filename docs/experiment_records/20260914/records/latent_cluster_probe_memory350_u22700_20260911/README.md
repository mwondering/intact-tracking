Memory350 原版 update 22700 与 v12 update 8000 的 latent 实测。

两个冻结 encoder 读取同一次 rollout 的交互，各自使用训练时的归一化和历史结构。每个 profile 运行 3,200 控制步，保持物理参数跨 reset 不变。共同 motion 集为此前使用的 42 个 LAFAN/Qingtong 文件，是 memory350 全训练目录的一个子集。

主要发现：共同基准中的同 DR 跨 motion 单位化距离，v12 为 0.239，Memory350 为 0.448。带负载组中 Memory350 的簇内/簇间距离比为 0.883，同环境跨 motion 波动较大。排除重复使用历史交互后，这个趋势仍在。

Memory350 仍有可读出的环境信息：带负载组的跨 motion family 环境中心识别 Top1 为 25.8%，v12 为 21.5%。不同读出指标并不都给出相同排序，不能只根据簇内/簇间距离比判断表征或下游控制整体优劣。

下面距离均为 RMS 欧氏距离，即先对样本对的距离平方求均值，再开根号。单位化表示逐个 latent 除以自身 L2 范数。主要比较只选 v12 完整 100 步、memory350 完整短期 50 步和长期 30×10 步的同一批 query。

“同 motion、相近 phase”指 motion 文件相同、归一化进度差不超过 0.02。“同一帧”则要求 motion 文件和整数 motion_step 完全相同。这些条件没有强制实际状态、动作或交互历史相同。

**共同基准：512 nominal + 512 原始 DR，无额外负载**

完整共同 query：23,550；nominal 11,777，DR 11,773。

| 比较条件 | 配对数 | v12 单位化 | Memory350 单位化 | v12 原始 | Memory350 原始 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal，不同 motion | 11,777 | 0.0699 | 0.4794 | 0.4972 | 3.4502 |
| 同一 DR，不同 motion | 11,773 | 0.2386 | 0.4477 | 1.7064 | 3.2360 |
| 同一 DR，不同 motion，历史不重叠 | 11,773 | 0.2368 | 0.4520 | 1.6930 | 3.2673 |
| 不同 DR，同 motion、phase 差≤0.02 | 11,687 | 0.7471 | 0.6688 | 5.3546 | 4.8411 |
| 不同 DR，同 motion、同一帧 | 139 | 0.7259 | 0.6319 | 5.2053 | 4.5762 |
| Nominal–DR，同 motion、phase 差≤0.02 | 11,741 | 0.5907 | 0.5953 | 4.2206 | 4.2970 |
| Nominal–DR，同 motion、同一帧 | 274 | 0.6003 | 0.5565 | 4.2881 | 4.0155 |
| Nominal–nominal，同 motion、同一帧 | 167 | 0.0252 | 0.1848 | 0.1798 | 1.3284 |

同环境跨 motion 距离 / 不同环境匹配 motion、phase 距离：v12 **0.319**，Memory350 **0.669**。只使用历史不重叠的同环境样本对时，该比例分别为 **0.317** 和 **0.676**。较低比例表示相对环境间间隔，同环境波动更小。

| 固定同一 checkpoint 的推理方式 | 同 DR 跨 motion，单位化 RMS | 不同 DR 匹配 motion/phase，单位化 RMS | 二者比值 |
| --- | ---: | ---: | ---: |
| 正确长期记忆 | 0.4477 | 0.6688 | 0.669 |
| 屏蔽长期记忆 | 0.4732 | 0.3402 | 1.391 |

| 跨 motion family 的环境中心识别 | 可评估 DR world | 测试样本 | Top1 | Top5 | 中心解释方差 R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| v12 | 497 | 6,487 | 54.97% | 81.98% | 0.8485 |
| Memory350 | 497 | 6,487 | 23.03% | 51.49% | 0.4850 |

环境中心由一半 motion family 的样本估计，再识别另一半 family 的样本。新旧模型使用同样的训练、测试划分；这是一项表征读出检查。

[Memory350 t-SNE](common/plots/tsne_memory350.png) · [新旧 t-SNE 对照](common/plots/tsne_comparison.png) · [交互图](common/plots/tsne_interactive.html) · [真实单位化距离热图](common/plots/unit_distance_heatmaps.png) · [距离分布](common/plots/distance_distributions.png) · [t-SNE 参数对照](common/plots/tsne_settings.png)

**新版训练物理配置：512 原始 DR + 独立四肢 U(0,4 kg) 负载**

完整共同 query：13,170；nominal 0，DR 13,170。

| 比较条件 | 配对数 | v12 单位化 | Memory350 单位化 | v12 原始 | Memory350 原始 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 同一 DR，不同 motion | 13,170 | 0.3380 | 0.9343 | 2.4161 | 6.6168 |
| 同一 DR，不同 motion，历史不重叠 | 13,170 | 0.3393 | 0.9300 | 2.4258 | 6.5871 |
| 不同 DR，同 motion、phase 差≤0.02 | 12,988 | 0.6985 | 1.0580 | 5.0027 | 7.4965 |
| 不同 DR，同 motion、同一帧 | 203 | 0.6486 | 0.9271 | 4.6411 | 6.5749 |

同环境跨 motion 距离 / 不同环境匹配 motion、phase 距离：v12 **0.484**，Memory350 **0.883**。只使用历史不重叠的同环境样本对时，该比例分别为 **0.486** 和 **0.879**。较低比例表示相对环境间间隔，同环境波动更小。

| 固定同一 checkpoint 的推理方式 | 同 DR 跨 motion，单位化 RMS | 不同 DR 匹配 motion/phase，单位化 RMS | 二者比值 |
| --- | ---: | ---: | ---: |
| 正确长期记忆 | 0.9343 | 1.0580 | 0.883 |
| 屏蔽长期记忆 | 0.5885 | 0.4455 | 1.321 |

| 跨 motion family 的环境中心识别 | 可评估 DR world | 测试样本 | Top1 | Top5 | 中心解释方差 R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| v12 | 424 | 6,203 | 21.51% | 49.43% | 0.6674 |
| Memory350 | 424 | 6,203 | 25.84% | 42.74% | 0.1939 |

环境中心由一半 motion family 的样本估计，再识别另一半 family 的样本。新旧模型使用同样的训练、测试划分；这是一项表征读出检查。

[Memory350 t-SNE](memory_training/plots/tsne_memory350.png) · [新旧 t-SNE 对照](memory_training/plots/tsne_comparison.png) · [交互图](memory_training/plots/tsne_interactive.html) · [真实单位化距离热图](memory_training/plots/unit_distance_heatmaps.png) · [距离分布](memory_training/plots/distance_distributions.png) · [t-SNE 参数对照](memory_training/plots/tsne_settings.png)

**采样与数值核对**

每 100 步保存一次 query。完整长期记忆在约 400 步形成，之后持续覆盖多个 motion。由于相邻长期窗口会重叠，另加保守的不重叠检查：较晚样本最旧 chunk 的序号必须越过较早样本已有 chunk，以及当时短期和 pending 交互未来可能进入的所有 chunk。该条件已通过跨 reset 的真实交互时间戳校验。样本并非相互独立，置信区间见 JSON 中的描述性 world bootstrap。

报告的 common_full 主表用于同 query 比较。另有 memory_full（短期 50 + 长期 300 完整）和 long_full（长期 300 完整，短期可能为 0–50 步）的指标，以覆盖 reset 后的在线输入情况；它们都保存在完整结果中。

四个 rank 共 2,048 个原始固定验证窗口复现五步 NMSE：0.0250383094，训练日志为 0.0250383085，相对差异 3.72e-08。把验证输入还原成原始交互再写入在线 memory bank，得到的 latent 与训练验证路径一致到很小的数值误差。

[推理核对](inference_verification.json) · [历史不重叠条件核对](disjoint_control_verification.json) · [完整汇总](summary.json)

这些结果比较两个已经训练好的 checkpoint。它们的训练数据、DR、损失配置和训练轮数也不同，不能把差异全部归因于 memory 架构。屏蔽长期记忆是在推理时干预，不能代替单独训练的无 memory 对照。同环境跨 motion 波动同时包含运动状态、历史内容和其他轨迹差异，不能单凭距离把所有变化都归因于 motion 标签。聚类紧凑程度也不能替代 predictor 或下游 policy 的性能测量。

t-SNE 标签只用于着色，16 个 DR world 在拟合前随机选定。新旧模型分别拟合 t-SNE，只能比较颜色混合和局部聚集，不能比较绝对坐标、簇面积或全局空隙；热图及交互数值给出真实 64 维距离。[算法作者说明](https://lvdmaaten.github.io/tsne/)

**复现记录**

Memory350 checkpoint：`/data_zcy/wxy/intact-tracking/runs/limb_context_20260909_memory350/stage1_8192/update_022700.pt`。

SHA256：`b26a54e77490821c33ffe2b7e5f70b9e38ca7022c94a0574066610c5a567601b`。

各 profile 的 metadata.json 记录完整采样参数和实际物理配置；latents.npz 保存每个 query 的三种 latent、motion、phase、world 和记忆计数；analysis/*_pairs.npz 保存实际配对及原始 query 行号；plots/tsne_manifest.json 保存随机选样和拟合设置。

脚本：`scripts/probe_memory350_latent_clusters.py`、`scripts/analyze_memory350_latent_clusters.py`、`scripts/plot_memory350_latent_clusters.py`、`scripts/verify_memory350_latent_inference.py`、`scripts/report_memory350_latent_clusters.py`。
