5000 轮复核：本次弱正样本、弱负样本和 response scale 调整的组合，明显改善了 DR 表征的跨 motion 聚类，尤其是带四肢负载和随机扰动的训练分布。固定验证集的 DR 第 5 步递归预测 NMSE 同时上升 26.9%；误差增量主要来自完全没有长记忆的样本。

两版均为 nominal50-memory350 扩容模型的 `update_005000.pt`，均完成 5000 次训练更新、20000 次 optimizer step。本次训练正常到达上限，自动比较成功完成。比较使用相同模型规模、训练种子、motion 数据、物理采样配置、学习率曲线、normalization、固定验证窗口，以及完全相同的诊断原始轨迹与样本配对。

| 比较对象 | checkpoint |
|---|---|
| 旧扩容版 | [旧 u5000](../../limb_context_20260911_memory350_encoder2x_nominal50/stage1_8192/update_005000.pt) |
| 新弱样本版 | [新 u5000](../stage1_8192/update_005000.pt) |

新版的修改为：弱正和弱负损失总系数各 0.002；弱负样本 margin 为 1.0，不匹配 motion/phase；response scale 从 0.75 改为 0.5；预测窗口仍为 5 步。新增弱损失只作用于完整 short50＋long30×10 历史。

以下距离均为 **64 维 latent 经 L2 归一化后的两点 L2 距离，再对样本对取 RMS**，不是训练监督中的响应 RMS。表征主结果要求完整 350 步历史。

| DR＋四肢负载／扰动 | 旧扩容 u5000 | 新弱样本 u5000 |
|---|---:|---:|
| 同一 DR、不同 motion，越小越紧凑 | 0.7532 | 0.5193 |
| 同一 DR、不同 motion、原始历史不重叠 | 0.7586 | 0.5253 |
| 不同 DR、同 motion、phase 差 ≤ 0.02 | 0.9673 | 1.2044 |
| 不同 DR、不同 motion | 1.1211 | 1.2452 |
| 同 DR 跨 motion／不同 DR 匹配 motion-phase 距离比，越小越好 | 0.7787 | 0.4312 |
| 跨 motion family 环境识别 Top-1，原始历史不重叠 | 28.18% | 58.53% |
| 同上 Top-5 | 47.59% | 80.91% |

这个诊断集包含 512 个 DR 环境、13997 个完整历史查询。严格的环境识别对照在 292 个已知物理环境、2761 个测试查询上进行：用一部分 motion family 的 latent 拟合环境中心，在另一部分 motion family 上测试，并排除全部训练／测试原始历史重叠。Top-1 提升 30.35 个百分点，按查询 world 配对 bootstrap 的 95% 区间为 [+26.68, +33.87] 个百分点。该区间衡量此诊断集上的查询不确定性，不代表跨训练种子的波动。

| 普通 DR，没有额外四肢负载／扰动 | 旧扩容 u5000 | 新弱样本 u5000 |
|---|---:|---:|
| nominal、不同 motion | 0.0474 | 0.0477 |
| 同一 DR、不同 motion | 0.4234 | 0.4028 |
| 同一 DR、不同 motion、原始历史不重叠 | 0.4266 | 0.4114 |
| 不同 DR、同 motion、phase 差 ≤ 0.02 | 0.6927 | 0.8557 |
| 同 DR／不同 DR 匹配 motion-phase 距离比 | 0.6112 | 0.4708 |
| 跨 motion family 环境识别 Top-1，原始历史不重叠 | 11.93% | 21.86% |

普通 DR 的改善主要体现在环境间分离，同环境进一步收紧的幅度较小；nominal 的紧凑性基本保持。普通 DR 的严格识别对照包含 409 类、3152 个测试查询，Top-1 增益 95% 区间为 [+7.58, +12.30] 个百分点。

图中能看到新版更多环境形成紧凑的同色团块，团块内仍包含不同 motion 和 phase；仍有拉长、分裂或彼此混合的环境。因此目前有充分证据支持“更容易从 latent 识别环境”，尚不能说所有 DR 环境都已形成独立、稳定的单簇。

- [DR＋负载 t-SNE 对比图](memory_training/plots/tsne_comparison.png)：上排旧版，下排新版；三列分别按环境、motion、phase 着色。
- [DR＋负载真实 64 维距离热图](memory_training/plots/unit_distance_heatmaps.png)：左旧右新，共用色标；新版同环境对角块更紧凑。
- [DR＋负载交互图](memory_training/plots/tsne_interactive.html)：可点击查询原始和归一化 latent 的真实距离。
- [普通 DR t-SNE 对比图](common/plots/tsne_comparison.png)及[交互图](common/plots/tsne_interactive.html)。

t-SNE 显示的是预先固定选择的 16 个 DR 环境，不使用环境标签拟合。两版分别拟合，图上的全局间隔不能跨模型作数值比较；上表和热图直接使用 64 维 latent。

固定预测验证的主要问题发生在记忆不足时。下表是 **递归预测第 5 步** 的 pooled NMSE，分母为相同样本上维持初始状态不变的预测误差。

| 验证条件 | 窗口数 | 旧扩容 u5000 | 新弱样本 u5000 | 误差变化 |
|---|---:|---:|---:|---:|
| 全部 DR | 1030 | 0.03587 | 0.04551 | +26.9% |
| DR：完整 short50＋long30 | 157 | 0.02808 | 0.02427 | −13.6% |
| DR：long30 完整，short 不足 50 | 226 | 0.02788 | 0.02794 | +0.2% |
| DR：有 1～29 个 long chunk | 494 | 0.02566 | 0.02723 | +6.1% |
| DR：完全没有 long chunk | 153 | 0.08168 | 0.13830 | +69.3% |
| 全部 nominal | 1018 | 0.01116 | 0.00933 | −16.4% |

153 个无长记忆窗口占 DR 验证窗口的 14.9%，贡献了 DR 总平方误差净增量的 **96.8%**。按 world 配对 bootstrap，全部 DR 误差变化 95% 区间为 [+14.8%, +42.2%]，无长记忆子集为 [+43.8%, +104.9%]，均显示退化。完整历史子集的区间为 [−27.8%, +6.2%]，仍包含无变化，所以 −13.6% 目前只作为点估计的改善趋势。

这表明本次组合修改改善了完整历史下的 DR 聚类，并暴露了无长记忆区间的预测退化。新增弱损失只覆盖完整历史与这一现象相符，但尚不能据此确定因果来源。弱正、弱负和映射 scale 同时改变，本实验不能分离三者贡献；也尚未验证下游 policy 收益。

复核记录：[review_verification.json](review_verification.json)。两版 checkpoint SHA256 校验通过；两个诊断集全部轨迹标签与原缓存一致；独立重算了 60 个样本对类别／模型组合的 300 个距离统计量，均与自动报告一致；motion、phase 和历史不重叠约束检查通过。原始缓存重编码与保存的在线 encoder 输出逐位一致，新旧模型的 BF16／FP32 编码一致性检查也通过。

预测拆分：[prediction_context_audit.json](prediction_context_audit.json)，包含各历史条件下第 1～5 步误差、配对置信区间及误差增量贡献。可在仓库根目录运行 `.venv/bin/python runs/limb_context_20260912_memory350_encoder2x_nominal50_weakpairs/comparison_005000/review_prediction_context.py` 复现，只读取冻结预测数组和固定验证文件，使用 CPU。

[自动比较报告](README.md) · [完整几何与识别结果](summary.json) · [固定验证预测结果](prediction.json)
