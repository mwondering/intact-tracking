# Memory350 固定 nominal 方向与 response10（2026-09-17）

用户授权 AB 项最终权重 0.04，删除弱负样本项，八卡、每卡 8192 A 环境，继续 Memory350。用户随后明确确认 nominal 锚定权重为 0.01，并要求持续监控训练。

运行根目录：runs/limb_context_20260917_memory350_nominal_direction_ab004_anchor001。
来源：runs/limb_context_20260912_memory350_response_window_ablation/response10/stage1_8192/update_015000.pt。

## 六项损失

| 项目 | 定义 | 最终系数 |
|---|---|---:|
| teacher prediction | 真实历史下五步逐步预测 | 1 |
| recursive prediction | 五步递推预测 | 0.5 |
| local positive | 同 world/episode/motion、精确 ±5 步，1−cos | 0.01 |
| DR–nominal response | SmoothL1(||normalize(z)−c||, 2D/(D+0.3), beta=0.25)，D=RMS(10×70 标准化 A−B 响应) | 0.04 |
| cross-motion positive | 同非 nominal world/session、不同 motion、完整且不重叠历史，1−cos | 0.008 |
| nominal anchor | 每个有可用历史的 nominal 样本，1−dot(normalize(z),c) | 0.01 |

响应项只平均有效 DR 样本；nominal 项只平均有效 nominal 样本。响应目标逐片段计算，不比较不同 world 的响应，也不跨片段平均。十步窗口内 reset、motion/phase 不连续或物理参数变化只屏蔽响应项。

弱负样本计算已移除。继承 schema 中的 weak_negative_margin=1.1 仅为源 checkpoint 兼容字段，权重严格为 0。

## 固定方向与恢复

恢复源模型、完成 replay 预热之后，第一次 optimizer 更新之前，只从训练分区抽取具有完整 short50 和 long30 的 nominal 历史。编码后先单位归一化，再八卡求和并归一化，得到固定方向 c。各卡检查方向哈希相同。校准不使用验证数据。

c 是不参与优化的 buffer。完整方向、来源 update、有效样本数和 SHA256 写入 nominal_anchor.json 及每个 checkpoint；后续恢复直接加载，不重新计算。encoder 的 LayerNorm、64 维 latent 和 predictor 输入路径保留。

## 运行设置

- GPU 0–7，每卡 8192 A 环境，含 128 个独立验证 world；另有对应 nominal B 槽。
- A 为 50% nominal、50% 原 tracker DR 加四肢独立 U(0,4kg) 负载。
- 全量 motion 数据集，沿用原 motion-balanced replay。
- Memory350 short50 + 30×10 长历史，attention 深度 2/4/4，latent64，五步 predictor。
- 每卡 batch512、microbatch256，全局 batch4096，每 update 四次 optimizer step。
- 恢复 u15000 的模型、AdamW、scheduler、normalization；学习率沿用 1e-5 下限，T_max 保持 32000 optimizer steps。
- 新八卡阶段重新预热并生成八组独立验证数据，验证响应标签同样是十步。
- 无总 update 上限，直到用户停止；不会自动衔接 PPO。

入口：src/intact_tracking/cli/forward_memory_nominal_direction_train.py。
启动器：scripts/run_memory350_nominal_direction.py。
核验：scripts/verify_memory350_nominal_direction.py。
结果以根目录 smoke_verification.json、train_verification.json、进程记录和 train.log 为准。

测试覆盖新距离定义、归一化、掩码、六项权重、无负样本计算、后五步标签影响响应而不影响预测、各 encoder 层梯度、训练分区校准及冻结锚点恢复。

第一次八卡 smoke 因测试 motion 符号链接未被扫描器识别而退出，日志保存在 failed_smoke_symlink_scan；已改为真实文件副本。

## 正式启动核验

27 项相关测试通过，八卡 smoke 完成两个真实更新，模型和锚点哈希均在八卡一致，跨 motion 正样本实际参与优化。正式任务 torchrun PID 47530，从 u15000 / optimizer step60000 接续。u15001 保存点验证通过：模型结构和 normalization 与源 checkpoint 相同，AdamW 步数接续到60004，encoder 与 predictor 均已更新，六项 loss 的实际加权和正确。固定方向由3139个完整 nominal 训练样本校准，八卡一致且写入 checkpoint。

W&B 服务端已确认 running 并返回 u15001 的新目标训练指标：[训练曲线](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/m350nomdir-211562e09daf)。

正式 replay 重新预热，首轮跨 motion 弱正样本尚无完整的不重叠配对，按既有有效性掩码贡献零；smoke 已用更长预热验证该项产生有效梯度，后续正式配对随历史积累出现。

正式 u15100 已观察到跨 motion 正样本：平均每个 microbatch 69.062 对，loss=0.0408708。再次核对六项总损失加权和正确。

## 持续监控

`scripts/monitor_memory350_nominal_direction.py` 每 30 秒检查训练进程的 PID 和启动时间、八个 worker、更新进度、各项 loss 的有限性及加权和、checkpoint 和 GPU 使用情况。每个新 checkpoint 在 CPU 上核验八卡参数一致、固定方向逐项不变、优化器步数和损失配置。监控不改变训练参数，也不根据一次陈旧观测停止或重启训练。

最新状态为运行根目录下的 `monitor/status.json`，逐次记录为 `monitor/history.jsonl`；监控自身的进程记录为 `monitor/process.json`。`latest_global_train` 是八卡汇总指标，`latest_rank0_train` 是更新更频繁的 rank 0 指标，二者不能直接当作同一批次比较。

固定验证数据在预热后生成，当时尚无完整且不重叠的跨 motion 配对。因此固定验证的 `weak_positive_pairs=0`，对应 loss 为零不能证明跨 motion 一致性已经改善。训练配对已正常出现；监控显式标记这项验证缺失。

早期固定验证 u15001→u15400：DR–nominal 响应距离 loss 从 0.186263 降到 0.161238；DR 五步 NMSE 从 0.043829 到 0.044183；nominal anchor cosine 从 0.987053 到 0.985331。训练目标在下降，但固定验证 nominal 锚定尚未改善，暂不能据此断言环境结构或泛化提升。

## u15500 独立缓存检查

`scripts/evaluate_memory350_nominal_direction_cached.py` 在 CPU float32 上重新编码原版 u15000 和新方案 u15500，使用已有缓存的完整 350 步历史，沿用原 motion-family 划分及历史不重叠要求。评估不占训练 GPU。

| 缓存 | 环境中心数 | 查询数 | 原版 Top1 | 新方案 Top1 | 差值与 world bootstrap 95% CI（百分点） |
|---|---:|---:|---:|---:|---|
| DR＋四肢负载 | 292 | 2761 | 81.75% | 80.26% | −1.485 [−2.872, −0.144] |
| 普通 DR | 409 | 3152 | 26.59% | 26.46% | −0.127 [−1.473, +1.195] |

完整结果在运行根目录的 `evaluation_u15500/README.md` 和 `summary.json`。原版 checkpoint 哈希与此前评估相同，两组此次均使用 CPU float32，原始缓存重建 cosine >0.9999。原历史报告使用 GPU bf16，原版识别率的小幅数值差异不能与本次模型变化混为一谈。

这是已知环境的跨 motion 识别，不是未见环境泛化，也不是匹配训练步数的单因素消融。早期 DR＋负载结果小幅下降，尚无环境结构改善的证据；继续既定训练并在后续固定 checkpoint 使用同一缓存比较。
