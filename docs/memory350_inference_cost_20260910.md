# Memory350 context encoder 推理成本

2026-09-10。当前实现单环境推理为毫秒级；每卡 8192 环境、每控制步更新一次
latent 时，推理和记忆维护开销显著，适合在接入大规模 PPO 前优化冻结后的缓存。

## 实测范围

- 硬件：物理 GPU 0，NVIDIA H100 80GB HBM3；BF16 autocast，权重与原始历史 FP32。
- 主训练始终继续运行，GPU 与 benchmark 共享。结果含训练竞争，不能当作独占
  GPU 的性能上限，也不能直接外推到机器人端设备。
- 使用新旧各自固定 update 7500 的 encoder；参数冻结、eval、无梯度。predictor
  未加载到 GPU；不包含 tracker、actor、critic 或 simulator 的运行时间。
- 所有短期和长期历史均填满。测量在线读取、排序、归一化和 encoder 的总调用，
  CUDA synchronize 后记录 host wall time。
- 新版按当前默认每批 512，8192 环境循环 16 批；旧版在线入口一次处理全部环境。
  下表比较实际入口。脚本另测同为 512 分批的纯网络耗时，原始数据均保留。
- 为减少训练阶段波动的偏差，新旧入口随机交替测量，各 40 次，预热后取中位数。

| 同时处理环境数 | 旧 context200 | 新 Memory350 | 新版 P90 |
|---|---:|---:|---:|
| 1 | 1.879 ms | 4.570 ms | 5.119 ms |
| 8192 | 69.478 ms | 125.011 ms | 201.396 ms |

8192 环境下，当前新版在线 encode 约为旧版的 1.80 倍。独立测得新版 append
记忆维护约 17.136 ms，P90 26.074 ms；该测试使用合成输入，每步约 1% 环境
reset，包含维护 episode/step 元数据。它与 encode 分开测量，二者中位数相加
只能用作粗略预算。

以 125 ms/步估算，每次 24 步 PPO rollout，仅 context encode 就是约 3 秒/卡。
这不是完整 PPO iteration 耗时；模拟器、tracker、策略网络和优化阶段需要另测。
单环境 encoder 的 4.57 ms 低于 50 Hz 的 20 ms 周期，但也不构成完整控制链路
或机器人端实时性的验证。

## 参数和显存

| 项目 | 旧 context200 | 新 Memory350 |
|---|---:|---:|
| encoder 参数量 | 453,312 | 1,035,328 |
| 8192 环境持久历史存储 | 626.56 MiB | 1926.31 MiB（1.88 GiB） |
| 当前入口单次推理临时显存峰值增量 | 10014.44 MiB | 约 967.50 MiB |

临时显存的差异受分批方式影响：旧版一次处理 8192，新版每批 512。不能解释为
结构本身必然有上述显存比例。新版 encoder FP32 权重约 3.95 MiB；整套阶段一
模型约 2009 万参数，其中大 predictor 不属于后续冻结 context 推理的开销。

只粗计线性层和 attention 的乘加，不计归一化、激活、softmax、数据搬运与同步，
旧版约 1.041 亿 MAC/环境，新版约 1.080 亿，增加约 3.7%。原始时间窗口从
200 扩到 350，并不对应一次 350-token 的全局 attention；最终 encoder 仅接收
CLS、1 个 memory token 和 50 个短期 token。

## 可优化的具体开销

1. 现在每个控制步都重新编码全部 30 个长期 chunk。encoder 冻结、归一化固定后，
   已完成 chunk 的表示可以精确缓存，新增完整 chunk 时只编码新增内容。
2. 长期 attention 的输出只在 chunk 集合更新时重算；短期 50 步与最终 context
   encoder 继续逐步更新。reset 可能一次提交多段，此时更新对应缓存；物理参数
   session 改变时按原协议清空。
3. 当前在线实现逐批读取 raw memory，并包含 `bool(tensor.any())`、计数器转
   Python 整数等同步检查。分批、数据整理和同步也值得单独优化。

冻结后若仅保留短期/待提交原始交互、30 个 128 维 chunk 表示及一个 memory
表示，8192 环境的数据存储粗估约 0.44 GiB，另加少量元数据。

benchmark 用同一输入比较分阶段预计算与原始 forward，输出逐元素完全一致，
验证了缓存计算的分解方式。**完整在线缓存尚未实现或端到端测时，因此不声称
已有某个加速倍数。** 阶段一训练期间权重不断更新，训练 replay 仍应使用原始
交互按当前参数编码，不能直接沿用冻结推理的缓存策略。

## 记录

- 脚本：`scripts/benchmark_memory350_inference.py`。
- 全部时延样本、显存、分阶段结果：
  `runs/limb_context_20260909_memory350/inference_benchmark_20260910/benchmark.json`。
- 评估进程已退出，检查时正式训练为 11751 updates；GPU 0–3 仅有原四个训练
  worker，未暂停、重启或修改训练。
