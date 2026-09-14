从完成检查的 u10000 续训至 u12000；每新增 1000 轮检查跨 motion 的 DR 环境识别。

弱正样本权重 0.004 → 0.008；弱负样本权重 0.004 → 0.008；负样本 margin 1.0 → 1.1；RMS 映射参数保持 0.3。
恢复 encoder、predictor、AdamW 和原 scheduler，学习率保持原下限 1e-5；nominal50、扩容 memory350、数据和采样保持。
模拟器、replay 和弱样本档案重新预热 500 控制步。新 run 保留父 checkpoint 的 SHA 和独立 W&B 记录。

主指标：用一组 motion family 建立已知 DR 环境中心，在另一组 family 上测试 Top-1，排除原始历史重叠。
本阶段目标：DR＋负载跨 motion Top-1 达到 80%；当前 u10000 基线为 73.45%。这是待验证的训练目标。
普通 DR 和 DR＋四肢负载／扰动分别报告；固定 42 条诊断 motion，不代表全部训练 motion 或下游 PPO 性能。

| 总轮数 | 新增轮数 | 普通 DR 跨 motion Top-1 | DR＋负载跨 motion Top-1 |
|---|---:|---:|---:|
| 10000 | 0 | 26.43% | 73.45% |
| 11000 | 1000 | 28.74% | 79.14% |
| 12000 | 2000 | 26.65% | 79.03% |

| 总轮数 | 普通 DR 簇内 | DR＋负载簇内 | DR 完整350步 NMSE | DR 无长期记忆 NMSE |
|---|---:|---:|---:|---:|
| 10000 | 0.4496 | 0.4562 | 0.01984 | 0.15537 |
| 11000 | 0.4437 | 0.4248 | 0.02025 | 0.15821 |
| 12000 | 0.4441 | 0.4207 | 0.01945 | 0.15992 |

这轮测量三项修改与继续训练的联合效果，不能拆分各项参数的独立贡献。

[父模型 u10000 检查](/data_zcy/wxy/intact-tracking/runs/limb_context_20260912_memory350_encoder2x_nominal50_weakpairs_tune004_s03/comparison_010000/review.md) · [趋势数据](trend.json) · [启动配置](launch_contract.json)
- [u11000 完整检查、置信区间和 t-SNE](comparison_011000/README.md)
- [u12000 完整检查、置信区间和 t-SNE](comparison_012000/README.md)
