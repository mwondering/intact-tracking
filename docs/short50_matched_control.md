# Short50 与 Memory350 的匹配训练对照

2026-09-10 用户要求新增仅使用短期 50 步的版本，从头训练，验证预测效果是否下降。

## 对照设计

- 新目录：`runs/limb_context_20260910_short50`；GPU 0–3，每卡 8192 A 环境
  （8064 训练、128 验证），与原 Memory350 共用 GPU；原训练保持运行。
- 数据：完整 `/data_zcy/wxy/motion_data_correct/motion_data_full`，四 rank 相同分片。
- DR：冻结 tracker checkpoint 原始 DR，加双手和小腿中部独立 U(0,4 kg) 负载；
  episode 上限 1000 步、原 stage1 termination、观测噪声、推扰等保持相同。
- predictor：512 宽、6 层、8 heads，10 步历史、5 步预测；latent 64 维。
- context：**CLS + 当前 episode 最近 50 个完整交互，共 51 个 token**；128 宽、
  2 层、4 heads。没有长期 encoder，也没有 memory token。
- 损失仍为 `L_prediction + 0.01 L_positive + 0.02 L_relation`，
  `response_distance_scale=0.75`。正样本为相同 world/episode/motion 的精确 ±5 步。
- AdamW、LR、权重衰减、BF16、replay、全局 batch 4096、microbatch 256/rank、
  每 update 4 次优化、预热 500 步及随机种子 717+rank，均与 Memory350 匹配。
- 训练不设轮数上限，8000 仍仅为学习率 cosine horizon。预先确定重点比较
  100、500、1000、3000、5000、**7500 updates**；不依据早期方向选择结论点。

新 encoder 和 predictor **均未加载训练过的权重**。构造时使用与 Memory350
相同的随机初始化过程，再删除长期路径；公共层权重、短期位置编码和构造后的 RNG
状态逐元素一致。全尺寸验证记录：
`.runtime/short50_v1/full_initialization_verification.json`。

| 参数量 | Short50 | Memory350 |
|---|---:|---:|
| encoder | 434,112 | 1,035,328 |
| encoder + predictor | 19,493,910 | 20,095,126 |

这是移除长期模块的结构对照；保留的 predictor/短期公共层容量一致，整体参数量
随长期模块移除而减少，不能额外声称这是完全等参数量的结构比较。

## 采样与验证保持一致

为了只改变模型的信息来源，训练沿用原来的 Memory350 采集器和 replay，包括
相同的正样本筛选与可用性判定。采集器仍维护长期字段用于保证抽样协议一致，但
**Short50 模型不读取它们**。它不依赖长期字段内容或 mask，也不跨 reset 输入历史。

正式训练启动时严格核对 DR 配置、四 rank 实际负载哈希、环境数、motion 分片、
episode 上限、损失和优化配置，任一不匹配就报错，不静默退到较小环境数。

复用 Memory350 的训练环境归一化统计和原始固定验证文件；验证 world 仍不参与
训练和归一化估计。八份验证文件逐字节复制并记录 SHA256，同一 update 的两个
模型用完全相同的状态、动作、目标、归一化和窗口做预测。

GPU 共用可能改变耗时，但比较使用相同优化步数与配置，不能将两个运行的 wall time
直接当作模型效率对照。模拟器接触计算存在已记录的数值波动，因此训练轨迹不保证
逐位一致；相同验证数据则通过文件哈希严格保证。

## 监控与评估

- 自动更新的曲线、同轮次结果和配对分组见 [预测结果](short50_prediction_results.md)。
- 用户追问后的 [4500 轮副作用监测](memory350_short50_monitor_20260910.md)：整体持平，
  长期可用时有收益，长期为空时明显退化；这两组现已单独持续监测。
- 训练入口：`intact_tracking.cli.forward_short50_train`。
- 监督器：`scripts/run_short50_stage1.py`。
- W&B 账户：`2486344338@qq.com`，项目 `intact-forward-predictor`。
- 正式训练分组：`limb_context_20260910_short50-stage1`；短测单独为 `...-stage1-smoke`。
- 每 100 updates 验证，直接记录 `matched_comparison/short50_to_memory350_nmse_ratio`。
  大于 1 表示仅短期版本在相同验证集、相同 update 下预测误差更高。
- 关键 checkpoint 由 `scripts/evaluate_short50_comparison.py` 补做逐窗口配对评估、
  reset 后短期不足 50/10 步分组，以及按 world bootstrap 的 95% 误差比区间。
  结果放在 `comparison/update_XXXXXX/`，W&B 分组为 `...-paired-evaluation`。
- 这些对照不启动 PPO。是否提高策略性能仍需后续阶段二实验。

## 实现验证与状态

17 项相关 CPU 测试通过，包括：仅 51 个 token；长期内容/mask 任意变化不影响
输出；padding 不泄漏无效数据；所有保留参数可训练；公共随机初始权重及 RNG
状态一致；共享验证 world/归一化检查。

四卡、每卡 8192 环境的短测完成 2 updates / 8 optimizer steps，无 OOM；
原 Memory350 训练继续推进。正式训练已经加载全部 129,827 条 motion，完成启动
核验：四 rank 的实际 DR/负载、共享归一化和八份验证文件哈希均一致；首次
checkpoint 的四 rank 参数哈希一致，参数有限且不含长期模块。
记录：`runs/limb_context_20260910_short50/startup_verification.json`。

正式训练 W&B：
[Short50](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/s50-ad9b02965fba)，
参考 [Memory350](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/m350-07a735338e82)。

### 早期结果，尚未收敛

100 updates（400 optimizer steps），同一批 2048 个窗口、489 个验证 world：

| 指标 | Memory350 | Short50 |
|---|---:|---:|
| 五步 NMSE，按 rank 平均，与训练日志口径一致 | 0.269845 | 0.267713 |
| 五步 NMSE，合并全部窗口 | 0.270024 | 0.267904 |
| 短期不足 50 步、已有长期历史 | 0.283045 | 0.283761 |

整体误差比 Short50/Memory350 为 0.99215，按验证 world 配对 bootstrap 的
95% 区间为 [0.98673, 0.99717]；短期不足 50 步分组的误差比为 1.00253，
区间 [0.99646, 1.00812]。这只表明当前早期整体误差接近，不能据此判断收敛后
memory 是否有收益。bootstrap 区间描述当前 checkpoint 在验证 world 上的差异，
不包含训练种子之间的不确定性。

逐窗口结果：`runs/limb_context_20260910_short50/comparison/update_000100/result.json`；
[配对评估 W&B](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/s50cmp-13b3856b2771)。
继续按预先确定的相同训练轮次比较，重点结论仍待 7500 updates。

两个独立作业的 nominal 数值复现检查在绝大部分样本上相近，但尾部有差异：
pose 误差 p99 分别约 0.000337 / 0.000330；Short50 启动检查中最大关节角
重复误差为 0.18184 rad（Memory350 为 0.02099 rad）。这属于同一模拟配置的
接触数值复现诊断，不是两个模型的预测误差；因此不声称训练轨迹逐位一致。
共享固定验证样本避免了测试目标随各作业数值波动变化。

原先 baseline、Memory350 模型及训练代码保持不变；保护哈希记录在
`.runtime/short50_v1/protected_sources.json`。所有新增代码、缓存和结果均在当前项目内。
