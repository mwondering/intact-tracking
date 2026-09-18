# 共享 MoE 的簇内 DR 与逐步路由诊断

用户要求验证两个问题：16个expert簇内的latent是否紧凑、物理DR是否仍然差异很大；同一固定DR是否会随motion、episode或交互过程切换expert。

本次使用共享encoder MoE的`checkpoint_update_001000.pt`，SHA256为`5239793a01de2c95045a0bb39365a2d539360fd5e0fa5a6df115b2acc835c0d1`。运行目录为`runs/limb_context_20260914_moe_routing_diagnostic`。这不是无共享版MoE的诊断。

## 已完成的主要发现

**完整64维latent能读取四肢负载；当前16类硬路由大量丢失负载信息，尤其是小腿。** 不能把“expert簇内负载混杂”解释成“encoder没训练过负载”或“latent里没有负载信息”。

Context来源直接核验checkpoint内的`training_physics`及实际负载审计：nominal50、Memory350扩容、response10/predictor5、update15000；50% nominal，50%原DR加四肢各自独立U(0,4) kg。四肢实际最大负载均接近4 kg，nominal负载为0。共享和无共享MoE的训练配置都保存同一context SHA256 `db5d7bd7df026db163b9eedf588d37dcc81b55551d28afe3744f67a8ab2642ac`，与原始训练产物逐字节相同。手部2.5 kg上限是后续PPO设置。

两组完整交互seed、每组1024world、mean和sample动作均已完成。对sample动作的两个DR seed交换拟合/测试，在训练侧768world拟合、256world选超参数，另一批1024world测试，每world取32个充分记忆样本：

| 用于预测负载的信息 | 左手R² | 右手R² | 左小腿R² | 右小腿R² |
|---|---:|---:|---:|---:|
| 实际expert编号，按训练簇均值预测 | 0.113 | 0.354 | -0.027 | -0.005 |
| 64维单位latent，线性回归 | 0.931 | 0.933 | 0.799 | 0.786 |
| 64维单位latent，MLP | 0.940 | 0.951 | 0.799 | 0.779 |

R²不是分类识别率；1为完全预测，0相当于均值预测。线性回归重量MAE为0.142/0.143/0.387/0.401 kg。mean动作与sample结论一致，原始latent与单位latent解码结果也几乎一致。

按原latent距离重新拟合16中心、增加为64中心，两条小腿负载的分组R²仍不高于0。原中心的测试inertia为0.330，重新拟合16中心为0.337，因此没有证据表明只需修好在线中心收敛就能解决负载分组。白化后16中心对小腿的R²改善至0.096/0.161，但仍弱。按真实四肢重量分16类能得到0.71–0.75的R²，仅作为使用真实参数的参照，不能当作已验证的控制方案。

当前expert划分的方差解释率主要集中在COM x/y及足部摩擦，约74%/74%/69%；两条小腿约1%/2%。其head仅输入压缩obs及tracker action，连续latent仅用于选head，没有再进入head。

实际簇内/簇间单位latent L2：mean为0.789/1.351，sample为0.782/1.344。充分记忆且同motion时，每步切换率为0.840%/1.050%，在50 Hz下总体平均0.420/0.525次每秒；不同motion的主expert一致率82.73%/82.20%；23.97%/27.33%的切换下一步立即返回原expert。中心完全冻结，因此这些切换与在线中心更新造成的重分配不同。

[完整结论及图表](../runs/limb_context_20260914_moe_routing_diagnostic/FINDINGS.md) · [逐步及簇内统计](../runs/limb_context_20260914_moe_routing_diagnostic/README.md) · [负载解码协议](../runs/limb_context_20260914_moe_routing_diagnostic/payload_decoder/README.md) · [聚类度量对照](../runs/limb_context_20260914_moe_routing_diagnostic/cluster_metric_probe/README.md)

## 交互协议

- 两组独立静态DR采样seed：30401、30402，每组1024个world。
- 每个seed分别运行确定性均值动作与checkpoint的Gaussian采样动作。两种动作模式的完整静态DR逐项核对。
- 完整129827条motion目录，uniform motion，保留原初始状态噪声、观测噪声、推力、末端高度termination及1000步episode上限。
- 与训练相同的500步冻结tracker预热，接着运行3000步MoE交互；参数和KMeans中心均冻结，没有PPO更新。
- 使用原Memory350缓存BF16 context推理，actor FP32。完整长期chunk跨trial保留，短历史和不完整尾部按原协议清理。
- 每步保存actor真实路由输出、motion及frame、trial编号、短/长记忆计数、原始latent范数、最近两个中心的距离差，以及重置/失败标记；每10步保存单位归一化64维latent。
- 交互前后核验全部物理观测及提取的67维静态DR完全一致，核验router全部buffer完全一致。

## 统计口径

主分析窗口是MoE第500至2999步，充分记忆定义为short50及long30×10都有效。相邻step切换率分别报告同一motion、充分记忆、短记忆补充期、reset/motion边界。不同motion间一致率先为每个trial选主expert，再在同world的不同motion间比较；每个trial至少有50个充分记忆样本。另报告每world主expert占比及1/5/10/50/100/350步延迟后的expert差异率。

簇内latent距离使用路由实际采用的单位向量L2距离，不是每坐标RMS。每world至多抽32个非相邻时间候选中的样本，主簇内/簇间配对明确排除同world，另外报告同world跨motion的latent距离。

67维静态DR包含4个负载、1个躯干附加质量、3个躯干COM偏移、1个足部滑动摩擦、29个armature比例和29个encoder bias。额外负载导致的惯量/COM变化由负载及固定附着几何确定，不重复计数。足部14个碰撞几何共享摩擦值，运行时核验。

每个DR坐标除以完整训练范围；分别报告六个参数族的簇内/随机距离、方差解释率，每个expert内每个参数的均值、标准差及10/50/90%分位数。综合DR距离对六族等权，避免29维参数块支配结果。将完整DR世界身份随机置换100次作为方差解释率的随机参照，保留原路由轨迹与占用比例。

DR组成的主结果按每个world实际进入该expert的时间加权，同时保存每world只按主expert归组的结果。逐帧数量不等于独立样本数，不用数百万帧虚报统计精度。

## 实现与验证

新增诊断入口`intact_tracking.cli.moe_routing_diagnostic`、统计模块`moe_routing_diagnostics.py`及汇总脚本`scripts/report_moe_routing_diagnostic.py`。不修改训练、推理或路由实现。五项测试验证实际切换、reset排除、DR范围归一化/权重、不同world的距离配对和等距中心核验。64world、100交互步的真实GPU预检通过。

首次正式诊断在seed30401的两个动作模式中触发额外交叉核验：该检查对obs的连续latent副本重算距离，并要求`topk`首编号与actor的`argmin`严格相同，没有考虑归一化的stride舍入和等距最小值的排序差异。修正为读取actor实际消耗的同一个strided输入，并直接验证已选中心的距离等于最小值（平方距离容差2e-6），同时记录`topk`/`argmin`编号差异和实际距离偏差。增加了等距中心可接受、真正选错中心必须失败的测试。两个中断运行均归档并重新采样，没有使用其未完成数据；另一个seed的原严格检查运行继续完成。

正式诊断使用0、2、4、6卡，每个进程只负责一组采样条件。结果和图表完成后记录于运行目录的`README.md`与`summary.json`。

重跑完成后，对actor真实输入核验的最小距离偏差为0，`argmin`和`topk`编号不一致计数也为0；未检测到实际分配错误。旧的两次额外交叉核验中断不能作为policy路由错误的证据。每个worker实际导入的collector版本记录在`execution_manifest.json`；collector结果中的`source_sha256`是退出时磁盘文件的哈希，不能替代该执行版本记录。

增加两个纯离线探针：`scripts/probe_moe_latent_payload.py`按完整world与DR seed划分拟合/测试，比较expert编号、线性latent解码和MLP；`scripts/probe_moe_cluster_metric.py`比较保存中心、重拟合16/64中心、白化及真实负载分组。所有新中心只用于离线统计，没有替换进入旧expert执行。`scripts/report_moe_routing_findings.py`汇总结果和预测图。

## 解释范围

这是u1000固定权重下、重新采样DR与交互轨迹的闭环诊断。采样动作复现checkpoint的探索分布，但不恢复训练当时的完整动态优化轨迹。两个seed为DR/交互seed，不是两个训练seed。簇内物理参数接近仍不等于最优控制修正必然接近；反过来，参数跨度较大也不能单独证明该expert无法控制它们。

现在能定位到连续latent转为16类编号时的负载信息丢失，以及实际轨迹中的路由变化；还不能声称它们解释了MoE全部控制劣势。下一步控制实验应保留expert的连续latent输入，并独立验证路由度量/稳定性；不能仅依据这些离线结果宣布控制提升。五项诊断测试通过，原训练及policy代码没有因本次诊断改动。
