**按 latent 分类，但用明确参数设置环境**

2026-09-16。分类和环境构造之间可以通过参数库连接：先从物理参数生成环境，记录其稳定 latent 和类别；训练某类专家时，直接抽取该类已经保存的参数行。无需先构造虚拟 latent 再反演完整 DR。

每条记录保留 `参数 theta → 环境原型 mean_latent → class_id`。训练专家 k 时，从 `class_id == k` 的参数集合中采样，用原物理接口初始化 world，并保持该 world 的 DR 跨 episode 不变。在线选专家时才使用当前历史的 latent 推断类别。专家的 PPO 数据分配依据固定参数表，避免随瞬时 latent 改变训练归属。

**已经生成的参数表**

复用 `payload256_top5_moe/calibration`：当前 u35857 encoder、仅四肢负载变化、背景 nominal。256 种明确负载组合，每种有多个独立 world 的历史；256 个原型只由原标定的 fitting worlds 得到。这些标定轨迹由冻结 tracker 产生。

按原型的 64 维 latent 直接做欧氏 KMeans，K=4/8，固定 seed=731，各十个初始解，仅按拟合 inertia 选择，不使用物理参数或 validation/test 查询调整聚类。

| 类别数 | 每类的明确参数行数 | 独立 world 查询与固定类别的一致率 | 最低单类召回 |
|---|---|---:|---:|
| 4 | 80、97、15、64 | 98.56% | 84.55% |
| 8 | 32、32、64、33、16、32、15、32 | 98.33% | 80.60% |

一致率使用原标定留出的 1024 个完整 world、14473 条完整历史查询。测试 world 使用同一套 256 种负载组合，只是 world/history 独立；motion 文件也未强制互斥。因此不是未见物理参数或严格跨 motion family 泛化结果，不能与此前全背景 DR、world 参数留出的 87%–89% 聚类结果直接比较。此结果也不是专家控制收益。

- [4 类参数库](../runs/limb_context_20260916_latent_class_parameter_bank/bank_k04.json)
- [8 类参数库](../runs/limb_context_20260916_latent_class_parameter_bank/bank_k08.json)
- [分组与留出查询结果](../runs/limb_context_20260916_latent_class_parameter_bank/summary.json)
- [参数行回读、类别及分组结构核验](../runs/limb_context_20260916_latent_class_parameter_bank/bank_verification.json)
- [导出脚本](../scripts/export_latent_dr_partitions.py)

例如 4 类版本的 class 0 包含 `[0.83333337, 0, 0, 0]`、`[0.83333337, 0, 0, 1.33333337]` 等明确负载行，顺序为左手、右手、左小腿、右小腿，单位 kg。初始化时从这一类的 rows 中取行即可；每个专家覆盖多个物理设置，不只训练在一个均值参数上。

文件同时保存 encoder/tracker 身份、来源哈希、物理协议、聚类中心及每行的原型 ID。全部 256 行在每份表中恰好出现一次，保存质量与原型质量逐项一致；原拟合中心、原 Top-1 指标和新分组评分均已复核。

**分组本身仍需改进和验证**

此次原始 latent 的 KMeans 主要按双手负载分组。在 16 种双手负载组合中，有 15 种的全部 16 种小腿设置始终分到同一类，K=4 和 K=8 都如此。唯一例外是双手均为 0 kg 的组合。高一致率因此不能被解读为已经细致区分了四肢 DR：当前分组几乎没有细分小腿负载。

参数表解决“类别如何落实到可执行环境”的问题。是否直接采用这些原始 latent 类别，仍应由专家交叉闭环评测决定；若希望专家覆盖不同腿部动力学，需要比较经标定的 latent 度量或其他分组方式。调整度量后重新标注参数库即可，参数库的构造和采样流程不变。

**训练接口与连续分布**

底层 `UniformLimbPayload` 已使用 `self.mass[num_envs,4]`，并沿用经过核验的质量、质心与惯量更新。因此接入训练时，只需让初始化采样器从指定类别抽取参数行，生成这张 per-world 质量表；还应保存抽样行 ID 和参数库哈希用于恢复及物理核验。

现有 `residual_uniform_train --dr-bank` 只接受旧版八个完整固定 DR profile，本次导出的表尚未接入该命令，不能直接作为其替换文件。本次没有启动专家训练或改变运行中的物理设置与 router。

类别可能是参数空间中的不规则区域。文件中的 min/max 只作描述，不能直接作为四个独立 uniform 区间：这样采出的新参数未必属于该类。先使用明确参数库；需要连续覆盖时，离线增加连续随机参数、采集完整历史并归类，再加入库。之后可拟合 `物理参数 → latent 类别` 的小分类器加速新参数筛选，但必须在独立新参数环境上验证，不能默认插值正确。

这一轮参数库是离散网格。通用策略对照应采用相同的网格及权重，专家总训练预算另行报告；连续随机负载应作为额外的未见参数评测。每次改换 encoder、背景 DR 或重新聚类，都需维护对应版本的参数库和专家身份。

复算命令（输出目录须尚不存在）：

```sh
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python -B scripts/export_latent_dr_partitions.py --calibration runs/limb_context_20260916_payload256_top5_moe/calibration --output runs/limb_context_20260916_latent_class_parameter_bank --classes 4 8
```
