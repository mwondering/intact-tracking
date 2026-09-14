从完成检查的原配置 u8000 续训至 u10000；每新增 1000 轮检查跨 motion 的 DR 环境识别。

弱正样本权重 0.002 → 0.004；弱负样本权重 0.002 → 0.004；RMS 映射参数 0.5 → 0.3。
恢复 encoder、predictor、AdamW 和原 scheduler，学习率保持原下限 1e-5；nominal50、扩容 memory350、数据和采样保持。
模拟器、replay 和弱样本档案重新预热 500 控制步。新 run 保留父 checkpoint 的 SHA 和独立 W&B 记录。

主指标：用一组 motion family 建立已知 DR 环境中心，在另一组 family 上测试 Top-1，排除原始历史重叠。
普通 DR 和 DR＋四肢负载／扰动分别报告；固定 42 条诊断 motion，不代表全部训练 motion 或下游 PPO 性能。

| 总轮数 | 新增轮数 | 普通 DR 跨 motion Top-1 | DR＋负载跨 motion Top-1 |
|---|---:|---:|---:|
| 8000 | 0 | 25.76% | 65.19% |
| 9000 | 1000 | 25.06% | 72.15% |
| 10000 | 2000 | 26.43% | 73.45% |

| 总轮数 | 普通 DR 簇内 | DR＋负载簇内 | DR 完整350步 NMSE | DR 无长期记忆 NMSE |
|---|---:|---:|---:|---:|
| 8000 | 0.3971 | 0.5099 | 0.02037 | 0.15104 |
| 9000 | 0.4471 | 0.4661 | 0.01996 | 0.15183 |
| 10000 | 0.4496 | 0.4562 | 0.01984 | 0.15537 |

这轮测量三项修改与继续训练的联合效果，不能拆分各项参数的独立贡献。

[父模型 u8000 检查](/data_zcy/wxy/intact-tracking/runs/limb_context_20260912_memory350_encoder2x_nominal50_weakpairs/continuation_005000_010000/comparison_008000/README.md) · [趋势数据](trend.json) · [启动配置](launch_contract.json)
- [u9000 完整检查、置信区间和 t-SNE](comparison_009000/README.md)
- [u10000 完整检查、置信区间和 t-SNE](comparison_010000/README.md)
