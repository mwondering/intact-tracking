新的 nominal50-memory350 扩容实验已配置为从头训练 5000 次更新，主要验证 DR 环境的表征是否改善。

实验目录：[limb_context_20260912_memory350_encoder2x_nominal50_weakpairs](../runs/limb_context_20260912_memory350_encoder2x_nominal50_weakpairs)。
旧扩容基线：[update_005000.pt](../runs/limb_context_20260911_memory350_encoder2x_nominal50/stage1_8192/update_005000.pt)，SHA256 `885b0b5964153f050f0bae2f535a3838410c146f3829bf8c5045fa184b234bf2`。

| 配置 | 本次实验 |
|---|---|
| GPU | 物理卡 4、5、6、7 |
| 环境数 | 每 rank 8192；8064 训练，128 验证；两部分各 50% nominal |
| 模型 | memory350；short50＋long30×10；encoder 深度 2/4/4，宽度128；latent64 |
| 参数量 | encoder 2,026,688；predictor 19,059,798 |
| 初始化 | 从头训练，seed717；继承旧扩容版本的初始化方法 |
| 数据 | 相同完整 129827 motion 文件目录；相同 tracker 和 DR＋四肢负载分布 |
| 更新量 | 5000 次训练更新＝20000 次 optimizer step；完成后停止训练并自动评估 |
| 学习率 | 保持旧版 8000 更新的 cosine 曲线；5000 是独立停止上限 |
| 预测窗口 | 5 个控制 step |
| 原有损失 | teacher＋0.5 recursive＋0.01 local positive＋0.02 response relation |
| RMS 映射 | `2*D/(D+scale)`，scale 0.75→0.5 |
| 新弱正样本 | 同一固定物理环境、同 physics session、不同 motion；总系数 0.002 |
| 新弱负样本 | microbatch 内不同 DR world 的全部无序配对；不筛 motion 或 phase；总系数 0.002 |

弱正损失为 `mean(1-cosine(unit_anchor, unit_positive))`。弱负损失为 `mean(relu(1.0-unit_distance)^2)`；距离达到 1.0 后该项不再推远。两项权重直接乘到总损失，不再乘外层 representation_weight。

新增弱损失只使用完整 short50＋long30×10 历史；原有局部正样本和响应关系损失的资格规则保持原样。所有 nominal world 属于同一种动力学，不参与新增的 DR 负样本分类。

每个 DR world 额外保留 4 个 float32 原始历史快照，采集间隔至少 200 个控制 step。每次训练用当前 encoder 重新编码正样本；没有保存旧 latent 作为监督。正样本要求 physics session 相同、当前 motion 不同、两段历史不共享任何原始交互。对长期 memory 的 chunk 序号和仍可能写入长期的 short/pending 尾部一并检查。物理参数重采样后，旧 session 的快照失效。弱样本选择使用独立 RNG，保持原有 anchor／局部正样本的随机抽样序列。

代码与启动验证：30 项相关测试通过；包括跨 motion 原始快照、历史不重叠、旧 replay 被覆盖后的留存、物理参数变化失效、正负样本梯度和 microbatch 切分。公共 objective 重构前后的默认输出及全部参数梯度逐位一致。四卡、每卡 8192 环境的两轮启动测试通过，正式训练配置通过旧扩容实验的物理、模型和训练条件对照检查。

监督进程：[run_memory350_weak_pairs.py](../scripts/run_memory350_weak_pairs.py)。完成 5000 轮后自动运行 [evaluate_memory350_weak_pairs.py](../scripts/evaluate_memory350_weak_pairs.py)，结果写入实验目录的 `comparison_005000/README.md` 和 `summary.json`。

评估复用已保存的 64 组原始查询快照：普通 DR 和带四肢负载／随机扰动的训练分布各一组。每个 profile 中，两版 encoder 处理完全相同的轨迹、motion、phase、历史窗口和物理参数。主指标要求完整 350 步历史：

- 同一 DR、不同 motion 的距离，以及严格排除历史重叠后的距离。
- 不同 DR 在相同 motion／相近 phase 下的距离；另报不匹配 motion 的环境间距离。
- 簇内／簇间距离比。
- 跨 motion family 的环境中心 Top-1、Top-5；另做时间和历史不重叠的识别对照。
- 相同固定验证 A/B 窗口上的 DR／nominal 五步 NMSE。
- t-SNE、真实 64 维距离热图、距离分布和可点击查询真实距离的离线 HTML。

评估预检已复现旧扩容 u5000 的预测误差，原始历史重编码与保存的在线输出逐位一致；同 checkpoint 比较的差异为零，打乱 latent 后环境识别率降至接近随机水平。

本实验同时改变弱正样本、弱负样本和 response scale，因此只能判断这组三项修改的联合效果。环境中心识别测试已知物理环境在不同 motion family 下的可读性，不等同于 PPO 收益。t-SNE 各模型独立拟合，图上全局距离不作为数值比较依据。
