# Memory350 context encoder 扩容实验

2026-09-11。用户要求将当前 context encoder 参数量扩大约一倍，检验 predictor 误差能否下降。

| 模块 | 原版 | 扩容版 |
|---|---:|---:|
| 10 步 chunk Transformer 深度 | 1 | 2 |
| 30 段长期 memory Transformer 深度 | 2 | 4 |
| 短期＋memory 融合 Transformer 深度 | 2 | 4 |
| context 隐藏宽度 / heads | 128 / 4 | 128 / 4 |
| context encoder 参数 | 1,035,328 | 2,026,688 |
| predictor 参数 | 19,059,798 | 19,059,798 |
| 阶段一总参数 | 20,095,126 | 21,086,486 |
| 输出 latent | 64 | 64 |

Encoder 为原来的 1.9575 倍；本实验采用加深网络这一种扩容方式，不把结果外推为所有扩容方式的结论。
Predictor 结构保持 512 宽、6 层、8 heads，仍与 encoder 共同从头训练，不冻结其训练权重。

## 固定对照

- 原版参考：`runs/limb_context_20260909_memory350/stage1_8192`。
- 新目录：`runs/limb_context_20260911_memory350_encoder2x`，原训练代码和 checkpoint 保留。
- 新旧公共 predictor/encoder 层初始权重及模型构造后的 RNG 状态一致；新增层随机初始化。
  没有加载原版训练过的权重。
- 全 `motion_data_full`：129827 motions，四 rank 相同分片。每卡 8192 A 环境，
  其中 8064 训练、128 验证；nominal B 使用配对的模拟槽。
- 冻结 tracker 原始 DR、观测噪声和独立四肢 U(0,4 kg) 负载不变，小腿负载位于中部。
  Episode 上限 1000，stage1 保留原 termination。
- 短期 50 步与长期 30×10 步不重叠；跨 reset 的记忆规则和 replay/正样本筛选保持一致。
- 原版归一化和八份固定验证文件原样复用，检查文件 SHA256、验证 world 隔离及实际物理参数。
- 全局 batch 4096、microbatch 256/rank、每 update 4 次优化、BF16、AdamW、LR、损失权重不变。
  `representation_weight=.01`、`representation_relation_weight=2`、`response_distance_scale=.75`。
- 训练不设轮数上限，8000 仅为 LR cosine horizon，随后 LR 最低为 1e-5；本轮不启动 PPO。

## 评价和监控

预先固定主要比较 **22700 updates**，与当前用于 PPO 的原版 encoder 所在轮次一致。
在 100/500/1000/3000/5000/7500/10000/15000/20000/22700 做同轮次 checkpoint 配对评估。
保留每 100 轮固定验证曲线，展示相同预算内的原版/扩容版误差和训练损失。

主要指标为固定验证集的五步 NMSE；附带 1–5 步误差和分组：长期可用/为空、短期满 50/
不足 50/不足 10 步、满 30 段长期记忆。按物理 world 配对 bootstrap 计算误差比区间。
原版和扩容版在每次配对评估使用同一设备、精度和 batch 大小。相同 update 不代表相同计算量。
一个训练 seed 的结果不代表跨训练种子稳定性；固定验证窗口覆盖训练目录，不代表未见 motion 泛化。

所有训练使用用户 `2486344338@qq.com` 的 W&B：项目 `intact-forward-predictor`，
正式组 `limb_context_20260911_memory350_encoder2x-stage1`，短测与配对评估单独分组。
每次配对评估把原版/扩容版 NMSE、变化和区间打印到启动日志，并更新报告。

## 执行状态

已实现扩容模型、匹配训练入口、独立评估和自动报告。CPU 检查通过，包括实际完整模型参数量、
公共随机初始化及 RNG 一致性、新增层有限非零梯度、旧格式严格加载和训练配置隔离。
四 GPU、8192 环境/rank 短测已完成 2 updates / 8 optimizer steps；实际 checkpoint 的四 rank
参数一致，全部模型参数和优化器状态有限，W&B 短测 run 正常结束。

2026-09-11，按用户明确授权停止 GPU 0–3 的四个原占用进程；GPU 4–7 原进程保持运行。
正式训练已启动，完整加载 129827 motions / 48085337 frames。首次正式启动在参数更新前
因 JSON 列表与运行时元组的类型差异被配置核对拒绝；修复等值比较、补充回归测试并归档
失败记录后从头启动，没有改变物理参数、训练预算或模型结构。

正式启动核验通过：四卡各 8192 A 环境，原版实际 DR 和数据分片一致；归一化及八份固定
验证文件 SHA256 均与原版相同。第 1 轮 checkpoint 的四 rank 参数一致且优化器状态有限；
启动核验时已到 34 updates，后续观察推进至 51 updates。W&B 服务端已读回正式训练记录。
这些是启动验证，不是扩容降低预测误差的结论。所有新增代码、缓存和输出均在本项目内。

[正式 W&B](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/m350scale-12b5f4c4b590)；
[启动核验](../runs/limb_context_20260911_memory350_encoder2x/startup_verification.json)；
[四卡短测核验](../runs/limb_context_20260911_memory350_encoder2x/smoke_verification.json)；
[启动修复记录](../runs/limb_context_20260911_memory350_encoder2x/startup_recovery.json)。

- 启动器：`scripts/run_memory350_scale_stage1.py`。
- 训练入口：`intact_tracking.cli.forward_memory_scale_train`。
- [自动报告](../runs/limb_context_20260911_memory350_encoder2x/report.md)。
- [启动状态](../runs/limb_context_20260911_memory350_encoder2x/state.json)。
# 后续运行

原 all-DR 2x 已按用户要求在 update 4968 停止。当前对照改为两版均 50% nominal，
原版 GPU 0–3、2x GPU 4–7，见 [nominal50 对照说明](memory350_encoder2x_nominal50.md)。
