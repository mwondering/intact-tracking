本次测试值得优先采用的是长期 chunk 复用和显式因果 mask。两者合用，离线训练计算耗时降低约 17.5%；按当前整轮耗时估算，可节省约 12%。单纯延后记忆统计读取没有可确认的收益。compile 更快，但存在需要单独验证的数值差异。

测试于 2026-09-12，H100 80GB，PyTorch 2.13.0+cu130，当前 nominal50-memory350 扩容模型 u5000 权重，BF16，microbatch 256。完整训练计算测试包含 4 次优化器更新，每次累积 4 个 microbatch，保留原损失及全部三个视图。

| 方案 | 训练计算耗时 / 轮 | 计算耗时减少 | 整轮耗时估算 | 整轮耗时减少估算 |
|---|---:|---:|---:|---:|
| 原实现 | 1.699 s | 0.0% | 2.53 s | 0.0% |
| 只复用长期 chunk | 1.577 s | 7.2% | 2.41 s | 4.8% |
| 只显式声明因果 mask | 1.468 s | 13.6% | 2.30 s | 9.1% |
| chunk 复用 + 显式 mask | 1.402 s | 17.5% | 2.24 s | 11.7% |
| 显式 mask + compile（保留舍入） | 1.049 s | 38.2% | 1.88 s | 25.6% |
| chunk 复用 + 显式 mask + compile（保留舍入） | 0.946 s | 44.3% | 1.78 s | 29.7% |

整轮基准为正式训练 u5400→u5500 的 2.532163 秒/轮，取自 metrics.jsonl 的时间戳，早于任何 GPU 基准测试。估算只减去离线测得的计算节省量；模拟器采样、replay、四卡 DDP 通信、验证及保存仍按原耗时保留，因此不能把以上估算当作已完成的四卡集成测试。

每个方案的完整训练计算测了 2 轮；另做了两次独占 GPU 的随机交错 microbatch 测试，分别每方案 24 次和 12 次，排除编译与预热。原版与 chunk 复用的单 microbatch 耗时分别为 102.57→95.81 ms、104.58→98.04 ms；复用收益在两个窗口均约 6%–7%。完整累积更新的减少比例略大。

长期 chunk 复用的实现

保存的 256 对真实 anchor / 局部正样本中，长期窗口位移 -1 / 0 / +1 的数量为 30 / 202 / 24，全部逐元素核验可复用。同一物理 world/session 的 ±5 步局部正样本只可能新增一个 10 步 chunk。原型始终编码 anchor 的 30 个、weak 的 30 个和 positive 的 1 个边界 chunk，然后 gather 共享摘要：每次从 23040 次 chunk 编码降为 15616 次，减少 32.2%。

memory attention、最终 context attention 及全部 768 个视图都保持原计算；没有采用“跳过无效 weak 视图”这个额外优化。共享摘要在同次前向中保留计算图，两个视图的梯度共同回传；参数更新后重新编码。生产集成需要 replay 额外传出 anchor/positive 的 memory_total 差值，不需要在 GPU 上比较原始浮点 chunk。

CPU–GPU 同步

- 仅将记忆统计保留为 GPU int64：8064 个训练 world、75 个存储 chunk、每步约 0.67% reset，5 步 finish_step 从 7.739→7.693 ms（各 40 次）。配对均值节省量 0.065 ms，bootstrap 95% 区间 [-0.080, 0.197] ms；差异很小，不足以作为整轮提速依据。
- profiler 确认统计方案去掉了每 5 步 15 次 aten::item / _local_scalar_dense，但双方仍有 70 次 aten::nonzero；动态布尔索引仍需同步。原始历史、reset / physics invalidation、有效性 mask、统计计数严格一致。
- 更有效的同步优化在 predictor：TransformerEncoder 原来每次通过 GPU tensor → Python bool 检测 mask 是否因果。已核验它固定为严格上三角 mask，可直接传 is_causal=True。每 microbatch 有 1 次 teacher-forced + 5 次 recursive 调用，因而每轮涉及 96 次判断。这个改动还消除了 fullgraph 编译的阻断点。

数值检查及编译结果

| 方案 | loss 相对差 | 全模型梯度相对 L2 差 | 梯度余弦 |
|---|---:|---:|---:|
| 只复用长期 chunk | 0.000000% | 0.030347% | 0.999999954 |
| 只显式声明因果 mask | 0.000000% | 0.000001% | 1.000000000 |
| chunk 复用 + 显式 mask | 0.000000% | 0.030347% | 0.999999954 |
| 显式 mask + compile（保留舍入） | 0.019665% | 3.597220% | 0.999402424 |
| chunk 复用 + 显式 mask + compile（保留舍入） | 0.018839% | 3.602101% | 0.999403219 |

- chunk 复用：真实 batch loss 完全一致；BF16 全梯度相对差约 0.0303%，来自共享后的归约次序变化。额外 FP64 测试覆盖窗口 -1/0/+1、空/部分/完整 history、两视图共同回传及更新后重算，每个 encoder 参数梯度在 1e-10 容差内一致。
- 显式因果 mask：loss 完全一致，梯度相对差约 6.5e-9。
- 默认 compile：单 microbatch 102.57→63.53 ms（与 chunk 复用合用），但全梯度相对差约 5.9%。不能只报告这项速度。
- 开启 emulate_precision_casts 保留 BF16 中间舍入后，梯度差约 3.6%，loss 差约 0.02%。所有梯度有限，但这仍是单 batch 数值变化，尚未测它是否改变收敛或 DR 聚类。暂未加入正在进行的可比性续训。
- 编译设置为 fullgraph=True、dynamic=False、triton.cudagraphs=False；编译固定形状的 encoder 与 predictor forward，保留递归状态推进和损失代码。未使用静默 eager fallback。保留舍入版本首次编译/验证预热约 75 s，合用共享 chunk 的新图约 23 s；这些时间不计入稳态吞吐。

数据与范围限制

- anchor、局部正样本及 prediction target 来自未改动的 validation_rank_0.pt。该文件早于 weak archive，因此用保存的另一组真实完整历史填充 70 个 weak 视图，其标签仅作为形状一致的性能夹具；这不是新表征评估、有效的弱样本训练实验或新的 latent 结果。
- 当前训练后期长期 history 已满；保存的 validation 含部分 history。两者在当前 dense 实现中计算张量形状相同，但真实数据流上的收益仍需集成测时。
- 测时保持权重固定，AdamW 使用 lr=0 运行同一更新内核；不保存测试权重。完整更新重复使用同一 batch，不包含真实 replay materialization 或 DDP。
- 正式计时前先编译预热；在 4–7 上的同一个续训短暂停顿后计时，确认 GPU 利用率为 0 再开始。三个窗口分别约 15.3、4.3、25.3 秒，总暂停 44.84 秒；每次有独立 45 秒自动恢复 watchdog，均正常恢复，没有重启模拟器/replay。
- 所有正式训练数学源文件 SHA256 保持一致。测试后正式续训已到 u5795，原学习率、损失权重和每 1000 轮检查安排保持原配置。11 项测试通过。

建议先把 chunk 复用和显式因果 mask 用于下一次训练，预计整轮约 2.53→2.24 秒（吞吐约 +13%）。compile 作为后续单独短程收敛验证项，不能因为本地吞吐更高就视为已证明等价。

文件：

- [汇总 JSON](summary.json)
- [完整训练计算与保留舍入编译结果](precision_preserving/network.json)
- [第一次隔离测时与默认 compile 数值差](isolated_gpu/network.json)
- [记忆同步独立测试](isolated_gpu/memory.json)
- [源文件未改动检查](protected_sources_verification.json)
- [暂停审计 1](isolated_gpu/network_pause_audit.json)、[暂停审计 2](isolated_gpu/memory_pause_audit.json)、[暂停审计 3](precision_preserving/network_pause_audit.json)
- 基准脚本：scripts/benchmark_memory350_training_optimizations.py；独立原型：scripts/memory350_training_benchmark_components.py；一致性测试：tests/test_memory350_training_optimizations.py。

相关实现参考：[PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。数值舍入选项以本机 torch/_inductor/config.py 的 emulate_precision_casts 实现为准。
