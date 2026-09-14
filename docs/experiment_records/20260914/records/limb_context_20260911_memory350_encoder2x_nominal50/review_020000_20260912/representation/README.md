# Memory350 encoder 扩容：20000 轮表征提取比较

两版同为 nominal50、完整数据集训练，encoder 1.04M → 2.03M，predictor 和 64 维输出不变。
冻结两版 update 20000，重新编码同一批保存的 float32 原始轨迹。严格核对 checkpoint、归一化、物理与历史 masks；重编码旧 u5000 与当时在线输出逐位相同。

主诊断为 tracker DR＋四肢 U(0,4 kg) 负载和随机推扰：512 个固定 DR world、42 条 LAFAN/Qingtong motion，13997 个 short50＋long30×10 完整样本。
这些 motion 是训练大数据集的受控子集。这里的跨 motion 指环境中心读出时按 motion family 分开，不能解释为未见训练 motion 或未见环境泛化。

| DR＋负载场景的表征指标 | 原版 | 扩容版 |
|---|---:|---:|
| 同 DR、跨 motion 的单位化距离 RMS，越低越好 | 0.78830 | 0.77692 |
| 同 DR、跨 motion、历史不重叠的距离 RMS | 0.79375 | 0.78278 |
| 不同 DR、匹配 motion/phase 的距离 RMS | 1.00490 | 1.04307 |
| 同环境 / 环境间距离比，越低越好 | 0.78446 | 0.74484 |
| DR latent raw 范数均值 | 7.25503 | 7.25022 |
| DR 单位化 latent 的协方差有效秩 | 6.68097 | 6.99110 |

距离来自真实 64 维向量，先逐个 L2 单位化；不比较两模型未对齐的坐标。扩大环境间距离本身不充分，结合跨 motion 环境识别判断。

| 环境中心识别协议 | 候选环境 / 测试样本 | 原版 Top-1 | 扩容版 Top-1 | 提高百分点 [95% CI] |
|---|---:|---:|---:|---|
| common / motion_split | 501 / 7282 | 19.80% | 21.79% | +1.99 [+0.30, +3.71] |
| common / motion_split_disjoint_history | 409 / 3152 | 17.16% | 19.19% | +2.03 [-0.22, +4.20] |
| memory_training / motion_split | 423 / 6502 | 25.47% | 31.45% | +5.98 [+4.19, +7.81] |
| memory_training / motion_split_disjoint_history | 292 / 2761 | 24.66% | 31.47% | +6.81 [+4.17, +9.50] |

普通协议用一半 motion family 拟合每个已知 world 的 latent 中心，在另一半 family 查询最近中心。
补充不重叠协议只用 ≤1600 步的前半段参考样本，并要求每个后续测试样本的最旧长期 chunk 已超过该 world 全部参考样本及待进入长期的短期/临时历史，且相隔至少 50 步；测试 motion family 仍独立。
区间为固定中心条件下对查询 world 的配对 bootstrap，只描述此诊断集，不是跨训练种子置信区间。

| 相同权重推理时去掉长期 memory：DR＋负载 | 原版 Top-1 | 扩容版 Top-1 |
|---|---:|---:|
| 保留 memory | 25.47% | 31.45% |
| 屏蔽 memory | 0.88% | 0.94% |

这是推理时的输入消融；不能等同于从头训练 Short50。扩容的环境读出优势出现在保留长期 memory 时。

| 固定验证日志的表征指标（四 rank 均值） | 原版 | 扩容版 |
|---|---:|---:|
| 同 world/episode/motion 的 ±5 步正样本 cosine | 0.98690 | 0.98941 |
| latent 距离与 A−nominalB 响应距离的相关系数 | 0.67415 | 0.67606 |
| 表征关系损失 | 0.10537 | 0.10604 |
| 打乱 latent 后 DR 五步误差倍数 | 4.57058 | 4.21867 |

局部 cosine 提高、关系相关性变化很小。两版 predictor 都依赖 latent；打乱造成的损失倍数不能用作表征质量的单一排名。

当前解释：扩容使该受控 DR 诊断集上的环境信息更易跨 motion 读出，主要是环境间距离扩大，簇内距离仅小幅缩小。这比单看五步预测误差提供了额外证据，但不证明已获得与 motion 无关的纯环境表示，也不证明 PPO tracking 会提高。

[完整几何指标](summary.json) · [历史不重叠识别控制](recognition_controls.json) · [训练表征日志](../representation_training_metrics.json)
