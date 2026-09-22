# Residual 与 tracker 144000 的 adaptive sampling 核对

2026-09-20，按要求核对采样配置、实际执行代码、日志和续训状态。此前 residual 的概率参数沿用了 tracker，但通用训练入口强制关闭了 failure rewind；这一行为差异现已修复。

## 参考身份与范围

- Tracker：`checkpoint_144000.pt`，SHA256 `717fa6e627f368f880bf71ecbc4021da02d8af9e069d5bf907fe45fdcfb79e49`。
- 原训练保存的 Git commit：`2fd6aa20efc346f0b9dcf552a0a27b3c9594d592`。保存的差异文件没有修改采样实现；该 commit 的采样文件与当前原仓库文件逐字节一致，SHA256 `447a11cab33329867a2ef72cde042e3c56f79e8eb5059e3e85655f72e9815a35`。
- 对应 [W&B 原训练运行](https://wandb.ai/2486344338-zhejiang-university/obs_ablation_bfm/runs/ptb9g6sm)，名称以 `2026-09-10_16-19-06_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward` 开头。
- 核对对象是平地 motion/bin 的选择、帧起点、失败回退、统计累计和 EMA。Residual 的模型、损失、DR/nominal 分布及 8 卡规模保持原定稿设置。

## 参数与实际行为

| 项目 | Tracker 144000 | 修复前 residual | 修复后 residual |
|---|---|---|---|
| Sampling mode / strategy | adaptive / branch | 相同 | 相同 |
| 均匀分支概率 | 0.5 | 相同 | 相同 |
| Temperature | 0.25 | 相同 | 相同 |
| Bin 宽度 | 1 秒，50 控制步 | 相同 | 相同 |
| 完成访问 / 失败访问先验 | 1 / 0 | 相同 | 相同 |
| EMA 尺度 / 每轮衰减 | 1000 / 0.999 | 相同 | 相同 |
| Bin、motion 概率上限 | auto，均值倍数 200 | 相同 | 相同 |
| Sequence-length-agnostic | false | 相同 | 相同 |
| 失败前随机偏移窗口 | 100 步，仅 adaptive 分支 | 相同 | 相同 |
| Failure rewind | 开启；概率 1/3 | **关闭** | **恢复原配置** |
| Rewind min/max steps | 0 / 0，采用当前 bin 起点 | 分支未执行 | 与 tracker 相同 |
| 完整 bin 统计诊断文件 | 每更新输出 | 关闭 | 仍关闭，仅影响文件输出 |

`adaptive_uniform_ratio=0.1` 是旧配置的后备值；两者实际由 `adaptive_sampling.random_probability=0.5` 覆盖。`adaptive_failure_rate_window_iterations` 的缺省/None 不影响显式配置的 1000 轮 EMA。

新实现从 tracker 配置保留完整 `rewind` 对象，包括开关、概率和偏移。失败触发 rewind 时保留当前 motion，返回失败所在 bin 的起点；其余 reset 才重新抽取 motion/bin。实际环境的参数在每个 rank 初始化后再次核对并写入 checkpoint。

## 为什么 final top1 约为 100

原代码和 residual 都先对纯 adaptive 分布施加概率上限，然后以 50% 均匀分支混合。若有效 bin 数为 N：

```text
p_final = 0.5 × p_adaptive + 0.5 / N
final_top1_over_uniform = 0.5 × adaptive_top1_over_uniform + 0.5
```

因此，adaptive 分支峰值约 200 时，最终混合分布峰值约 **100.5**。200 是 adaptive 分支配置的均值倍数上限，不是要求 `sampling_final_top1_over_uniform` 始终达到的目标。稀疏程度、motion 上限和原实现的浮点容差也会影响实际峰值。

本次读取原运行的 TensorBoard 和 W&B，两者在 step 144000 一致：

| 原日志字段 | 实际记录 |
|---|---:|
| sampling_final_top1_over_uniform | 80.38584 |
| sampling_adaptive_top1_over_uniform | 159.98831 |
| sampling_adaptive_probability_cap_over_mean | 156.67290 |
| sampling_uniform_branch_probability | 0.391682 |

这些是训练日志聚合值。固定配置项 200 和 0.5 在旧日志中也没有显示为其常数值，因此不能把该面板数值直接当作 sampler 配置。该 checkpoint 所对应运行的 final 指标并非 200；不能为使曲线变成 200 而擅自把概率上限翻倍。原始读取结果保存在[采样核对记录](../runs/144000_exp/sampling_parity_20260920/tracker144000_wandb_sampling.json)。

当前修复前的一个采样记录为 adaptive top1=200.47520、final top1=100.73760、uniform=0.5，符合上述公式。此次没有改变概率上限或指标公式。

## 其他差异的核对

- 完整 command 配置比较只发现 rewind 开关和诊断快照间隔的有效差异。
- 分别调用原仓库和当前仓库的配置构建器构建实际 command 后，修复后的共同字段仅诊断快照间隔不同；初始状态噪声、起点边界和采样默认值也一致。构建结果见[实际配置核对](../runs/144000_exp/sampling_parity_20260920/constructed_command_parity.json)。
- 原代码中的地形分支不参与本次平地训练；residual 的同步分组扩展在 group size=1 时不改变抽样或 RNG 消耗。
- Bin 构造、有效区间、长度权重、失败率、temperature、概率上限、均匀/adaptive 分支、bin 内偏移、失败前偏移、访问/失败计数、EMA 及采样触发逻辑均已核对。
- 原训练为 4 卡，每 rank 55120 条 motion；当前为 8 卡，每 rank 27560 条。两者均按 rank 分片、独立累计统计，所以绝对 bin 概率、有效 bin 数和已学出的采样分布不会逐点相等。相同分片、统计和随机种子下的采样行为才是代码对齐的检验条件。
- 按用户要求在第 6008 次更新重置过历史统计。本次修复不再次重置，继续保留其后累计的访问、失败、pending 计数和 EMA 时间基准。

## 验证及续训

- 59 项测试通过。其中差分测试读取上述 SHA256 固定的原采样代码，以相同统计和 RNG，在初始、稀疏失败和广泛失败场景下逐项比较抽样 motion、帧起点、rewind、指标、计数、EMA 和最终 RNG 状态，结果完全一致。
- 8 GPU × 256 environments 的续训验证完成 25 → 27 次更新。模型、优化器、normalizer、8 个 rank 的采样计数及 EMA 与源 checkpoint 完整恢复；所有 rank 均启用原 rewind，更新后参数一致且有限。
- 正式训练在 **7283 次更新**完成后正常保存，从该 checkpoint 启动修正版。新运行目录为 `runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_sampling_parity_20260920/`，不设新的更新上限。
- 正式规模的[恢复核对](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_sampling_parity_20260920/launch_verification.json)已通过：模型、优化器、normalizer、冻结依赖和 8 个 rank 的统计/EMA 精确恢复，原采样概率参数与 DR 物理实例保持一致；首次新更新为 7284。
- 一次性迁移开关为 `--align-sampling-to-tracker`。它仅允许旧 native 运行恢复 tracker rewind，其他概率参数、模型、优化器配置、分片和训练规模仍严格校验。通常 resume 保持严格一致性检查。

正式运行的恢复核对及后续状态见[运行记录](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_sampling_parity_20260920/README.md)。[预检结果](../runs/144000_exp/stage2_proprio122_history5_tracker_action_auxdr_com_friction_sampling_parity_20260920/preflight_verification.json)包含测试报告、源码哈希和小规模运行证据。
