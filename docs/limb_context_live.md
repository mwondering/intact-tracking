# 四肢负载 + nominal 监督 context 实验状态

**2026-09-14 当前控制实验：** 用户确认八张卡分别训练八个完整固定 DR 的独立 residual MLP 专家，不输入 latent；共同参照为已固定的通用 A update 4000。原 MLP/MoE 在 4020/3300 轮保存退出。配置及监测协议见 [八专家实验](fixed_dr_specialists8_20260914.md)，当前任务指针为 `.runtime/limb_context/current_ppo.json`。下方为此前实验记录。

2026-09-07，用户授权实施并推进至最终结果。仅在本项目内写文件。


## 当前有效约定（2026-09-10，覆盖下方历史配置）

- **Short50 / Memory350 4500 轮补充监测**：整体五步 NMSE 0.032730 / 0.032704，
  基本持平。有长期历史、短期不足 50 步时 Memory350 误差低 22.52%；长期为空时
  反而高 73.29%。冻结 Memory350 权重屏蔽已有长期信息，误差上升 77.55%。
  当前风险指向缺少长期信息时的退化，训练日志提示这类样本覆盖不足；原因仍需
  受控训练确认。报告器已纳入空长期和完整短期分组，原训练与 7500 主要比较继续。
  [副作用诊断及证据](memory350_short50_monitor_20260910.md)。

- **2026-09-10 Short50 从头训练对照已启动**：只移除长期 encoder 和 memory token，
  保留当前 episode 的 50 步短期交互；DR、全数据集、每卡 8192 环境、1000 步
  episode、损失和优化设置均与 Memory350 一致。两个作业共用 GPU 0–3，4–7
  保持原用途。共用固定验证文件和归一化统计，四 rank 的实际 DR/负载哈希已核对。
  17 项相关测试及四卡短测通过，正式训练在独立
  `limb_context_20260910_short50-stage1` W&B 分组持续运行。
  预先确定在 100/500/1000/3000/5000/7500 同轮次 checkpoint 配对评估；7500 为
  重点结论点，当前早期差异不用于宣称收敛后优劣。
  [设计和监控](short50_matched_control.md)；[自动更新的预测结果](short50_prediction_results.md)。

- **2026-09-10 Memory350 配对消融已完成**：固定 update 7500，在 2048 个普通验证
  窗口 / 489 个 held-out world 上，屏蔽长期 memory 后五步误差 +79.1%，错配其他
  world 的 memory 后 +113.7%；短期不足 50 步时，屏蔽 memory 后误差为 2.50 倍。
  原始预测数值复现训练日志，原本无 memory 的负对照逐元素一致。新旧版本还存在
  DR、环境数、episode 上限和损失配置差异，不能据此声称新版训练整体优于旧版。
  [分析、长度消融及图表](memory350_effect_analysis_20260910.md)。
  按用户当次指令，GPU 1–3 的其他三个进程已退出；该次清理后只保留 Memory350，
  当时观察到 7868 updates，训练继续且无上限。随后新增 Short50 对照，见上方记录。

- **新 Memory350 阶段一已独立实现并开训**：GPU 0–3，各 8192 A 环境（8064 训练＋128
  独立验证），完整 129827 motions / 48085337 frames，episode 上限 1000 步，轮数无上限。
  四卡短测完成 2 updates / 8 optimizer steps，没有 OOM；正式运行已通过 checkpoint
  四 rank 参数一致性、数据分片、验证隔离及 W&B 服务端读回检查。
  原版源文件哈希保持不变。新目录 `runs/limb_context_20260909_memory350`，
  [正式 W&B](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/m350-07a735338e82)，
  分组 `limb_context_20260909_memory350-stage1`。
  [实现、测试和当前状态](memory350_training.md)。

- 新实验 DR 以**冻结 tracker 的 checkpoint 保存配置**为准，再加两手、两侧小腿中部
  各自独立 U(0,4) kg 负载；阶段一和两个 PPO 版本统一。保留原 DR 事件、观测噪声、
  动作设置和初始状态扰动。0 kg 仅指额外负载为零，原 tracker DR 仍启用。
  新模式已接入训练、续训和评估，GPU 0 的 128-env / 500-step 模拟器检查通过。
- 旧 context200 阶段一已按用户要求在 **24988 updates** 保存后正常退出，GPU 0–3
  已释放。旧 `baseline_123` 继续挂起；`film_123` 为未完成的失败状态，不计作第三个 seed。
  新方案为不重叠的短期 50 步 + 长期 30×10 步，长期经两级编码得到一个 memory latent；
  episode 上限计划改为 1000 步。关系项乘数 2、`response_distance_scale=0.75`
  已接入新配置；分层记忆模型现已单独实现并开训，详见上方最新记录。
  [设计、实现状态和旧任务停止记录](limb_context_memory350.md)。

- 2026-09-09 最新补测：seed 122 的 baseline / FiLM 均已完成 5000 updates。
  0/2/4 kg 的 body error（cm）分别为 baseline 4.184 / 4.401 / 6.979，
  FiLM 3.730 / 4.339 / 7.037，变化 −10.85% / −1.42% / +0.84%；失败数 B/F
  为 2/2、36/40、407/423。4 kg 未复现 seed 121 的优势，重载收益尚不稳定。
  0/4 kg 已核验定期测试，2 kg 本次补齐；配对 bootstrap、共同存活时段和原始 traces
  均保留于 `additional_eval/seed122_update_005000_all_0_2_4/`，W&B
  `limb-extra-21ce656ee371` 全部 49 项 summary 核验通过。
  [详细测试](limb_context_seed122_latest.md)。这些 PPO checkpoint 仍使用旧 100-step encoder。
- 2026-09-09 LocoFormer 调研完成：[跨 trial 记忆、TXL 缓存与扩展实验方案](locoformer_context_scaling.md)。
  方案尚未应用到训练。200-step 阶段一最近 10 次记录的普通训练 batch 完整历史平均
  约 7.53%，固定普通验证 batch 约 3.91%（rank 0）；建议优先验证 reset 清空是否限制表征学习。
- 2026-09-09 新任务已启动：从头训练 200-step context encoder/predictor，其余阶段一设置保持。
  新目录 `runs/limb_context_20260909_context200`，4 GPU、全数据集、nominal 监督、latent 64，
  无停止上限，用户决定收敛。baseline_122 完成后已自动通过四卡短测，在 GPU 0–3
  启动正式阶段一；最近 W&B 已同步至 6200 updates。baseline_123 延后至 encoder 停止。
  GPU 4–7 上 film_122 也已完成，并开始 film_123；本次补测未中断现有训练。
  W&B 仍在 `intact-forward-predictor`，独立 `limb_context_20260909_context200-stage1` 分组。
  200-step 的 CPU 全模型检查、42 项相关测试及实际四卡短测已通过，正式训练运行中。
  [配置、episode 边界说明与验证](limb_context_200.md)。
- 2026-09-09 最新结果：新批次 seed 121 的 baseline / FiLM 均完成 5000 updates。
  相同 4096 motions/starts、500 步及原完整评估终止标准下，0/2/4 kg 的 body error
  （cm）为 baseline 4.143 / 4.526 / 7.364，FiLM 3.679 / 4.316 / 7.118；FiLM 分别
  降低 11.19% / 4.63% / 3.34%，joint 降低 8.89% / 1.88% / 1.21%。失败数依次
  3/3、46/49、502/449。共同存活时段方向一致。0/4 kg 复用并验证定期结果，本次
  在 GPU 2/3 补齐 2 kg；当前训练进程 PID 和 resume 历史不变，补测已完成并退出。
- 相对保留 EE 终止的旧批次（同 seed 121 / 5000 updates），2/4 kg 的 body 相对
  优势增加 1.10 / 0.99 个百分点，0 kg 缩小 0.89 个百分点；4 kg FiLM 自身 body
  从 6.902 cm 升至 7.118 cm、失败 370→449，baseline 也变差。因此相对优势扩大
  不等于绝对重载表现改善，且训练样本量翻倍，无法将变化单独归因于 termination。
  seed 122 在最新共同 checkpoint 2100 的 4 kg 测试中，FiLM body/joint 暂高
  4.59% / 3.89%、失败 391→427；该 seed 尚未完成，不混入 5000 轮结论。
  [详细结果](limb_context_no_ee_latest.md)；原始结果、逐步 traces、配对 bootstrap、
  旧/新同策略对照和训练连续性检查位于
  `runs/limb_context_20260908_no_ee_4gpu/additional_eval/update_005000_all_0_2_4/`。
- 2026-09-08 新实验已启动：`runs/limb_context_20260908_no_ee_4gpu`。用户明确结束旧批次，
  授权使用完整 8 GPU 并停止其他占卡程序。baseline 使用 0–3，FiLM 使用 4–7；每卡
  8192 env，每组全局 32768 env。仅比较 baseline / FiLM，保留配对 seeds 121/122/123，
  每组从头初始化 residual actor / critic、训练 5000 updates；第一阶段沿用已获准的
  frozen update 7800 encoder，不重新训练、不续接旧 PPO 权重。
- 新实验训练统一关闭 `ee_body_pos`，其余 reward、观测、网络、LR、熵、残差界限和
  独立 U(0,4) 四肢负载协议保持。uniform 前 1000 updates 后从准确 checkpoint 切换
  adaptive，failure rewind 关闭；每 100 轮按原完整终止标准测同一 4096 motions 的
  0/4 kg。每组四卡在评估期间暂停 PPO，独立评估器使用组内前两张卡。
- 新两组四卡短测各完成 3 updates，所有 rank 的模型和 critic 统计哈希一致，冻结
  tracker 未变，初始公共主干完全相同。正式 seed 121 两组均加载 129827 motions，
  分片 [32457,32457,32457,32456]，全部 8 rank 确认实际活动项仅为 time_out、anchor_pos、
  anchor_ori。正式初始 critic 均值最大差异 4.66e-10，平方矩一致。跨独立运行的检查
  改为比较第一、第二矩（rtol=1e-6、atol=1e-8），避免原始 sum 的固定绝对容差随
  world 数增加而过严；同一分布式任务内仍要求哈希完全一致。
- 新批次从头训练与四卡规模的独立核验：`smoke_4gpu8192_scratch/audit.json`、
  `ppo_4gpu8192_scratch/startup_verification_seed_121.json`。W&B 使用独立分组
  `limb_context_20260908_no_ee_4gpu-stage2-4gpu8192-scratch`，复用的阶段一沿用原 run。
- 旧批次已归档：seed 121 两组均完成 5000；baseline 122 保存于 1000、FiLM 122
  保存于 379，其余正式任务取消。原始模型、训练日志、周期评估和补充结果完整保留。
  5000 轮 0/2/4 kg 的 FiLM body 改善 12.08% / 3.53% / 2.34%，joint 改善
  4.91% / 1.65% / 1.62%；baseline/FiLM 失败分别 3/3、28/36、413/370。
  仅 seed 121 完整，不是跨 seed 或全目录 IID 最终结论。详见
  [旧批次结果](limb_context_original_termination_results.md)，
  [W&B 归档](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/limb-archive-dc45d33c84b6)。
- 新旧批次每组每轮样本量分别 32768×24 和 16384×24；最终报告会比较相同 seed 121、
  5000 updates 下的 latent 优势，并明确差距变化同时包含训练规模变化。
- 2026-09-08 用户要求后续 residual 训练去掉 `ee_body_pos`。新批次的 baseline、
  FiLM、concat、constant（包括短测）默认采用 `no_ee_body_pos`；保留其余终止项。
  当前已启动的两批实验根目录固定为 `original`，其待运行配对种子也沿用同一任务。
  没有停止或重启当前 PPO，新配置尚未开始正式训练。独立评估保留原完整终止标准。
  4500 轮 4 kg 的 frozen / baseline / FiLM 失败数为 561 / 417 / 375，其中
  触发 `ee_body_pos` 的为 550 / 413 / 366；仅该项触发的为 549 / 412 / 366。
  触发统计允许其他项同时发生，不能将这些数直接换算成取消终止后的成功率。
  原始证据及来源 SHA256：
  `runs/limb_context_20260907_adaptive1000/progress_reviews/ee_body_pos_termination/failure_counts.json`。
- 2026-09-08 13:53 UTC 冻结 tracker 的 0/2/4 kg 对照已完成：原始 checkpoint_72000.pt，
  相同 4096 条 motions、起点、时长、物理指纹及奖励，逐步配对检查通过。Frozen / baseline
  4500 / FiLM 4500 的 body error（cm）分别为：0 kg 2.248 / 4.298 / 3.808；2 kg
  4.542 / 4.507 / 4.313；4 kg 8.617 / 7.079 / 6.948。失败数分别为 2/3/5、38/34/31、
  561/417/375（每组 4096）。
- 相对冻结 tracker，FiLM 在 0 kg 的 body/joint 误差上升 69.42% / 53.18%；2 kg body
  降低 5.04% 但 joint 上升 11.13%；4 kg body/joint 降低 19.37% / 11.28%，失败率从
  13.696% 降到 9.155%。共同存活时段比较方向一致。因此目前观察到的是重载补偿与
  nominal 精度之间的取舍，不能把相对 residual baseline 的收益解释为相对原 tracker
  的全面提升。该结果仍只有 residual 训练 seed 121。
  产物：`runs/limb_context_20260907_adaptive1000/additional_eval/frozen_tracker_vs_update_004500_all_0_2_4/`；
  [W&B 冻结 tracker 对照](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/limb-extra-62474cef6cbe)
  已上传九份原始结果与逐步 trace，并读回核对。训练进程保持连续，终端已报告。
- 2026-09-08 13:39 UTC 第 4500 轮 0/2/4 kg 比较已齐：每组每档仍是相同 4096 条
  motions、相同起点、最多 500 步；0/4 kg 复用并核验已完成的定期测试，2 kg 本次补测。
  baseline / FiLM body error（cm）依次为 4.298 / 3.808、4.507 / 4.313、7.079 / 6.948；
  FiLM body 降低 11.39% / 4.31% / 1.85%，joint 降低 4.36% / 1.99% / 1.38%。
  失败数依次为 3 / 5、34 / 31、417 / 375；4 kg 失败率为 10.181% / 9.155%。
  三档共同存活时段的 body/joint 误差方向一致。单 seed 的 motion bootstrap 误差比区间
  均低于 1；4 kg 失败率差区间为 [-1.685, -0.366] 个百分点，跨训练 seed 尚未完成。
- 4 kg 相比第 2000 轮：FiLM body 误差 +0.29%、baseline +6.55%；FiLM 失败数从
  426 降到 375。因此相对误差优势扩大，同时 FiLM 自身的失败率改善，不能把相对误差
  改善等同于绝对精度改善。全部原始结果、逐步 trace 和配对比较见
  `runs/limb_context_20260907_adaptive1000/additional_eval/update_004500_all_0_2_4/`；
  [W&B 4500 轮 0/2/4 kg](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/limb-extra-422f7fa69fb4)
  已上传并读回核对，终端报告已写入。baseline 121 已完成 5000，baseline 122 在 GPU
  0、1 启动；FiLM 121 在 GPU 2、3 继续。额外评估仅使用 GPU 0、1，训练进程未重启。
- 2026-09-08 10:02 UTC 补测第 2000 轮、四肢各 2 kg：与上述 0/4 kg 使用同样的
  4096 条 motions、起点及最多 500 步。baseline / FiLM body error 为 0.042065 /
  0.041888 m（FiLM 低 0.42%），joint error 为 0.643973 / 0.643693 rad（低 0.04%），
  失败 38 / 36，失败率 0.928% / 0.879%。body/joint 误差比的 motion bootstrap 95%
  区间均覆盖 1；当前 2 kg 基本持平。共同存活时间段 body/joint 变化为 -0.34% / -0.03%。
  原始结果、配对核验及逐步 traces 在
  `runs/limb_context_20260907_adaptive1000/additional_eval/update_002000_all_2/`；
  [额外评估 W&B](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/limb-extra-0d7681f6d03c)
  已上传并读回核对。仅使用 GPU 0、1，PPO 进程和续训 checkpoint 参数保持一致，训练继续。
- 2026-09-08 09:42 UTC 最新复核：两组同为第 2000 轮，每端 4096 条固定 motions。
  0 kg body error 为 baseline 3.971 cm / FiLM 3.464 cm（FiLM 低 12.75%）；4 kg
  为 6.644 / 6.928 cm（FiLM 高 4.27%），失败 402 / 426（9.814% / 10.400%）。
  对比第 1000 轮，两组 4 kg body error 分别上升 7.28% / 9.20%，当前未见切换后的
  端点性能改善；该前后对比不能将原因单独归结为 adaptive。共同存活时间段的比较方向
  一致。复核报告为 `runs/limb_context_20260907_adaptive1000/progress_reviews/update_002000/review.md`。
  训练和 W&B 均正常，继续既定的至少 5000 updates。
- 用户要求从第 1000 轮 checkpoint 切换 adaptive。当前目录为
  `runs/limb_context_20260907_adaptive1000`：baseline 121 使用 GPU 0、1，FiLM 121
  使用 GPU 2、3，每卡 8192 env；其他 seed 和对照仍由该目录的调度器依次执行。
- 原目录的 uniform 前 1000 轮指标、初始 checkpoint、1000 轮 checkpoint 及已完成的
  端点测试已核对 SHA256 并导入新目录。baseline 原进程停在 1046，但续训明确取
  第 1000 轮；原 1001–1046 的 uniform 更新不进入新曲线。FiLM 原进程准确停在 1000。
- adaptive 只用于 motion/bin 起点采样：branch 策略、uniform 分支概率 0.5、temperature
  0.25，其余参数随 `run_config.json` 显式保存。四肢各自独立 U(0,4 kg)、其他 DR/noise
  关闭、failure rewind 关闭。actor/critic、Adam、归一化及冻结 context 都从选定状态恢复；
  仿真 episode 重新开始。新增检查严格比较恢复前后的完整模型与优化器 tensor 摘要。
- 后续每组均采用 1000 uniform + 至少 4000 adaptive 的共同训练安排。每个 rank 独立
  保存其 motion shard 的 adaptive 访问/失败计数和 EMA 迭代数；恢复时检查文件 SHA256、
  motion shard、维度及参数，重建属于仿真 episode 的未结束访问记录。
- 切换验证：四种网络的两卡 × 8192 env adaptive 短测通过；baseline 额外续训通过模型、
  优化器及两 rank 采样统计恢复检查，并完成独立 0/4 kg 端点短测。23 项相关 CPU 测试通过。
  端点评估的长等待改为 CPU Gloo 集合通信，避免等待 rank 占用评估所需 GPU。
- 1000 轮固定端点：0 kg body error 为 baseline 0.036870 m / FiLM 0.031733 m；
  4 kg 为 0.061931 / 0.063440 m，失败数 363 / 370（各 4096 episodes）。这是 adaptive
  开始前的比较基准。切换记录与前缀审计分别见新目录的 `sampling_curriculum.json`、
  `uniform_prefix_import.json`；终端报告见新目录 `logs/endpoint_reports.log`。
- 2026-09-08 08:06 UTC：正式 baseline/FiLM 的首个 adaptive update 均为 1001，完整模型/
  Adam/归一化摘要与各自 1000 轮 checkpoint 严格一致；数据、负载、奖励、采样配置在两组
  之间完全匹配。检查时进度 1025 / 1022，W&B 服务端已读回 adaptive 标志与恢复审计。
  记录：`runs/limb_context_20260907_adaptive1000/adaptive_continuation_verification.json`。
  当前 W&B：[baseline](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/limb-c09fe09a2baa)、
  [FiLM](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/limb-b1d4c93971a0)，
  分组 `limb_context_20260907_adaptive1000-stage2-2gpu8192-scratch`；邮箱再次核对为
  `2486344338@qq.com`。下方旧链接保留 uniform 训练历史。
- 按用户新增要求，每完成 100 次 PPO 更新，保存 `checkpoint_update_XXXXXX.pt` 并测试四肢各 0 kg、各 4 kg 两个端点。复用提前锁定的同一组 4096 motions、相同起点，每条最多 500 步。协议为 `runs/limb_context_20260907/periodic_endpoints.json`。
- 端点测试期间对应两卡暂停 PPO 更新，分别运行独立评估进程；训练侧 RNG、环境步数和 critic 归一化统计均逐次检查保持不变。测试完成后继续原训练环境。指标写入原 W&B run 的 `endpoint_eval/all_0/*`、`endpoint_eval/all_4/*`，横轴为 `endpoint_eval/checkpoint_update`，并保存逐 episode/逐 step 结果及被测 checkpoint 的 SHA256。
- 用户补充要求周期测试不反复 resume：每 100 次仅保存模型并由独立评估进程读取；训练 worker、模型、优化器及模拟器持续保留。接入新逻辑时的一次续训与此后的周期测试分开记录。2026-09-08 06:02 UTC 已确认 baseline 第 300 次、FiLM 第 200 次自动测试完成，原 worker 分别继续到第 323 / 260 次，PID 和进程启动时间不变，resume_history 均仍只有接入时的一条；两组两端指标已从 W&B 服务端读回。实际两次测试约 408 / 433 秒，进度估计已计入周期评估耗时。过程证明见 `periodic_endpoints_continuation_verification.json`，上传证明见 `periodic_endpoints_wandb_verification.json`。
- 用户要求每次测完在终端报告：已启动 `scripts/report_limb_context_endpoints.py runs/limb_context_20260907 --follow`，每 5 秒读取已通过训练状态检查的测试记录，自动在对应训练日志和 `logs/endpoint_reports.log` 追加 `[端点测试完成]` 摘要。包含模型、PPO 轮数、两端的 body/joint 位置误差、失败率、覆盖率和回报，已有结果已补报；游标避免重复输出。监听器只读取训练结果并写日志，不使用 GPU、不重启训练。单次查看最近结果可去掉 `--follow`；持续查看汇总用 `tail -f runs/limb_context_20260907/logs/endpoint_reports.log`。
- 用户要求继续监测并汇报新情况：已启动独立 CPU 观察进程 `scripts/watch_limb_context_experiment.py runs/limb_context_20260907 --follow`。同一训练 seed、相同 PPO 轮数且评估协议相同的 baseline/FiLM 齐备后，在上述汇总日志打印 `[同轮对比]`，包含误差、失败率、覆盖率、回报及上一配对轮次的数值；结果保存在 `.experiment_monitor/paired_metrics.jsonl`。同时观察训练失败、NaN/Inf、超过 35 分钟无日志进展、调度器/W&B 心跳和上传错误，只在异常出现或解除时报告。状态为 `.experiment_monitor/health.json`；不控制或重启训练，持续等待最终 `eval/comparison.json`。已验证不会把不同 seed、轮数或协议的结果配在一起。
- 接入时 baseline 121 在 update 203、FiLM 121 在 update 164 保存；这两个 checkpoint 的两端测试均已完成。旧配置未保存 baseline 的 100/200、FiLM 的 100，不能补造这些历史模型；后续 baseline 从 300、FiLM 从 200 起按 100 次间隔评估，后续新种子从初始模型和第 100 次开始。此次续训恢复模型、优化器、归一化和计数，重建模拟器 episode。切换记录为 `periodic_endpoint_transition.json`。
- 周期评估计数、独立子进程环境、错误 checkpoint/负载拒绝、W&B 横轴、分布式同步及最终缺失评估检查均已通过相关测试。最终训练审计新增周期端点覆盖和模型文件哈希检查，监控测试不替代原定的最终全量配对评估。
- 只使用 GPU 0–3；GPU 4–7 留给用户的其他任务，本实验训练和评估的选择器均禁止使用它们。
- 已核对当前实际 motion 采样：baseline / FiLM 均为 `sampling_mode="uniform"`，在各 rank 的 motion 分片内均匀选 motion，再均匀选择有效起点；adaptive motion sampling 和 failure rewind 均关闭。该项与每环境、每部位独立 U(0,4 kg) 的负载采样是两项独立设置。第 600 轮 checkpoint 的 `cfg.task` 保留源 tracker 的 `adaptive` / rewind 配置，属于源配置记录；实际环境通过 `configure_load_only(prepared.env, ...)` 覆盖后再实例化，不能把该源配置字段当成当前采样模式。运行记录中固化的协议源码 SHA256 与当前文件一致，核对记录为 `effective_motion_sampling_verification.json`。
- 按用户要求补测第 600 轮、四肢各 2 kg：使用同一组 4096 motions/starts、500 步上限，在 GPU 0 顺序完成 baseline / FiLM 独立评估，训练进程和 resume 次数不变。身体位置误差 3.8887 → 3.7390 cm（-3.85%），关节 L2 误差 0.56896 → 0.56013 rad（-1.55%）；失败 27 → 28 条，平均回报 83.3365 → 83.3209。共同存活时间段的 body/joint 误差变化为 -3.92% / -1.60%，物理指纹、motion、起点、时长、奖励及逐步轨迹配对检查通过。产物位于 `additional_eval/update_000600_all_2/`，摘要已写入终端汇总及两组训练日志；[W&B 补充评估](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/limb-extra-97d82167498d) 已上传并读回确认，分组为 `limb_context_20260907-stage2-extra-eval`。
- 用户已明确判断阶段一基本收敛并允许进入第二阶段。阶段一在 update 8116 / optimizer step 32464 保存并正常退出；此前无上限续训的 LR 已降至 1e-5。
- `stage2_release.json` 当前 `approved=true`，记录用户原话、停止步数和模型 SHA256。按预定独立验证集选模规则使用 update 7800 的 `best.pt`，五步 NMSE 0.0179188268；只读副本为 `stage1/frozen_for_stage2_update_007800.pt`，SHA256 为 `ed48887b11a6f46e6491a4e709841858e7bb378090a0cc053299a54f734ca5f6`。
- 第二阶段为 residual policy：底层 tracker 冻结，可训练残差 actor 和 critic 从头训练；配对 seed 的公共主干一致。每组两卡、每卡 8192 env，正式 PPO 仍至少 5000 updates。
- 新 PPO 目录 `ppo_2gpu8192_scratch`，短测目录 `smoke_2gpu8192_scratch`；W&B 分组 `limb_context_20260907-stage2-2gpu8192-scratch`。
- 旧四卡 warm-critic B121 在 update 957 保存并停止，不参与最终比较。阶段一在 update 6303 / optimizer step 25212 保存，转换为手动停止模式，LR 保存值 2.5950884029729785e-5。保留模型、Adam、归一化和固定验证文件；模拟器和 replay 在续训时重建。
- 从头初始化、公共主干匹配、分布式同步、GPU 限制、无上限迭代及 LR 尾段相关 39 项测试通过。新两卡 GPU 短测四种融合均已完成 3 updates，通过初始参数、冻结 tracker、有限值和卡间同步检查；正式 baseline 121 / FiLM 121 已完成数据加载并持续执行 PPO 更新。
- 独立仿真启动的初始归一化统计存在 float32 舍入差异：短测初始均值最大差异 7.45e-9。跨实验改为比较实际统计张量（rtol/atol 均为 1e-6、样本数严格相等），仍校验各自 checkpoint 与审计 SHA256，并保留同一分布式任务内卡间完全一致的要求。两个回归测试通过；四种 GPU 短测完整审计见 `smoke_2gpu8192_scratch/audit.json`。
- 2026-09-08 05:15 UTC：W&B 服务端已读回正式 baseline 121 的 update 42、FiLM 121 的 update 34。两组各加载 129827 motions / 48085337 frames，两分片为 [64914, 64913]；公共 actor/critic 初始权重严格一致，数据/物理/奖励配置相同，初始 critic 均值最大差异 4.66e-10。正式初始 checkpoint 审计和 W&B 读回记录分别保存在 `ppo_2gpu8192_scratch/startup_verification.json`、`wandb_startup_verification.json`。约 3.7 / 4.6 s per update，当前这一对到 5000 updates 约需 5 / 6.4 小时；这些早期指标不作为性能结论。
- 正式 W&B run：[baseline 121](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/limb-f4d55e6d56b7)、[FiLM 121](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/limb-289b81d74806)。后续 122/123 两个配对 seed、concat/constant 对照和固定协议评估由同一调度器继续执行。
- 无上限续训已验证：从 update 6303 恢复，第一个新记录为 6304，LR 由 2.5950884029729785e-5 连续衰减到 2.592144731058713e-5；九份固定验证/归一化文件 SHA256 不变。W&B 原阶段一 run 已读回 update 6304 和 6400，服务端配置确认无上限、关闭自动早停。当时八组正式 PPO 及四个 PPO 短测均为 `waiting_for_user`，本次收到用户明确指令后才放行。续训审计见 `runs/limb_context_20260907/stage1_manual_continuation.json`，切换记录见 `stage1_stop_for_stage2.json` 和 `stage2_release.json`。

## 已完成

- 新负载协议：四个部位各自独立 U(0,4 kg)，启动时采样，之后保持；小腿负载中心在 knee link 向下 0.15 m。
- 除负载外关闭物理/动作 DR 和观测噪声；reference reset，无额外初始姿态扰动。
- actor/critic 同时使用冻结 64 维 context，多隐藏层 FiLM；另有 baseline、concat、constant 对照。
- 最新初始化：可训练残差 actor 与 critic 从头训练，保留冻结 tracker 的初始动作均值；action std 统一为 0.25，critic 统计由当前 DR 首批观测重新累计。
- 第一阶段的 validation world 与 training replay/归一化拟合隔离；按独立验证选 best.pt。
- 模型/负载/旧 residual、forward predictor/online rollout 及配对统计测试合计 65 项通过。
- 单 motion 测试完成 baseline 5 次 PPO 更新、stage 1 两次外层更新（8 optimizer steps）；仅为实现测试。

## 正式协议

- 全目录 `/data_zcy/wxy/motion_data_correct/motion_data_full`，加载全部 NPZ，启动时核对 catalog。
- 第一阶段四 GPU，每 rank 1024 training + 128 validation worlds；global optimizer batch 4096，4 steps/update，无步数上限，关闭自动早停；warmup 500 steps。原 8000 仅为学习率衰减时间尺度，LR 到 1e-5 后保持。
- 预测收敛使用含不完整历史的独立验证 batch，表征诊断使用另一份满历史正样本 batch；不以完整历史筛选掩盖重置后的质量。
- 每 100 updates 验证；至少 2000 updates 后，过去 1000 updates best NMSE 改善 <1% 且表征诊断范围 <0.1 时记录平台诊断；不自动停训。须用户明确确认后才开始第二阶段。
- PPO：baseline seeds 121/122/123，FiLM seeds 121/122/123，concat 121，constant 121。每组两 GPU、每 GPU 8192 env（全局 16384）、24 steps/update、5000 完整 updates。B/K 用 GPU 0、1；F/C 用 GPU 2、3。
- 残差上限 0.25，actor LR 1e-4，critic LR 5e-4，5 epochs，4 minibatches，entropy coef 0.0002；固定 schedule，不做额外 critic warmup。
- encoder/归一化冻结，PPO 不执行 predictor；actor 非 latent 输入1645维，critic6330维（运行时复核）。

## 调度与产物

正式运行目录：`runs/limb_context_20260907`。
调度入口：`scripts/run_limb_context_experiment.py --run-root runs/limb_context_20260907`。
调度状态 `state.json`；每组输出目录包含 `run_config.json`、`metrics.jsonl`、checkpoint 和 `completion.json`。
2026-09-08 00:34 UTC：用户恢复 Full access，并已明确授权停止占卡计算进程。已对八个旧训练进程发送 SIGTERM，全部正常退出，八卡均已释放；记录见 `runs/limb_context_20260907/gpu_release_20260908.json`。
原调度器仍存活，已自动启动零负载 A/B 审计。另启动 128 worlds 的配对评估短测，检查 reference 时间线、失败后局部 reset 和逐步误差记录。未修改其他目录的文件。
队列会先运行零负载 A/B 数值一致性审计、四 GPU stage 1 短测、四 GPU、每卡 8192 env 的四种融合短测，然后自动运行正式训练、独立预算/冻结参数审计、最终配对评估和本地报告。
评估清单已在任何正式分数出现前固定：全目录 IID；同一组 4096 motions/starts 的全 0/1/2/3/4kg；同一组 1024 的不对称负载；1024 motions × 2 相同起点副本的 latent 干预。
评估代码保存逐步误差，以共同存活时间段的配对结果补充失败截断均值。
2026-09-08 00:39 UTC：四种融合的 4096-env / 3-update GPU 短测全部通过，FiLM 最后一轮约 3.07 s，concat 约 2.99 s。四卡 encoder 完成 2 updates / 8 optimizer steps；独立 broad validation 五步 NMSE 1.1338 → 1.0624，仅用于检查实现。
128-world / 300-step 配对评估短测全部通过：frozen、baseline initial、FiLM correct/zero/paired-swap；跨负载替换实际执行 18,482 个有效满历史步骤。baseline initial 与 frozen 没有新失败或救回差异，但闭环数值误差使 body/joint 均值相差约 0.048% / 0.240%，不能将这个量级的短测差异当作学习收益。审计见 `smoke/evaluation_audit.json`。
零负载 A/B 审计通过：3,792 个有效五步窗口，根位置 RMS 1.16e-5 m，关节位置 RMS 3.06e-4 rad，恢复误差最大 1.43e-6；接触求解存在少数较大尾部误差，完整 max/RMS 均保存在 `smoke/nominal_audit.json`。
正式 stage 1 占用 GPU 0–3，baseline 121/122/123 与 constant 121 各占 GPU 4–7；全部完成数据加载并持续训练。上述短测数字不是最终实验结果。
进度命令：`python3 scripts/status_limb_context_experiment.py runs/limb_context_20260907`。
2026-09-08 02:04 UTC：按用户要求把第一阶段上限从 6000 提高到 8000。在 update 2504 / optimizer step 10016 完成一次受控保存，再从该 checkpoint 恢复；第一个新记录为 update 2505 / optimizer step 10020。保存时 LR 为 0.00018851931729122918，延长后首次更新为 0.00018847966673336098，余弦终点从 24000 延长至 32000 optimizer steps。模型和 Adam 状态恢复，八份固定验证文件及归一化文件的 SHA256 均保持不变，数据集和验证协议相同；模拟器与 replay 按续训入口重建。原初配置保存在 `stage1/run_config.initial.json`，审计记录为 `budget_extension_8000.json`。W&B 服务端配置及 summary 已确认上限 8000，并读回 update 2505。
四个 PPO 原进程 PID 45240/45241/45242/45243 持续运行，未重启；新调度器 PID 22242 接管它们，并等待第一阶段真正完成后启动 FiLM/concat。预算延长、LR 连续性/恢复、原 cap 不提前释放依赖、进程接管、W&B 及评估队列相关九项测试通过。
已排队提前评估：三个 baseline 和 constant 全部达到 5000 完整 updates 后，在这些任务释放的 GPU 上评估 frozen/四个对照，FiLM/concat 可继续训练；最终全量评估复用通过 checkpoint/seed/干预模式验证的结果文件。子集评估不生成最终结论，共用文件锁避免重复评估，同步与子集评估新增三个测试通过。

## W&B 监控补充（2026-09-08）

用户要求全部训练在 W&B 可监控。已实现独立 JSONL 实时同步器 `scripts/sync_limb_context_wandb.py`，覆盖阶段一、八组 PPO 和五个训练短测，并能补传历史；后续调度器启动时自动维护同步进程。两个增量补传/中断恢复测试通过。
当前训练调度器启动早于 W&B 补充功能，另已启动独立的 `scripts/supervise_limb_context_wandb.py`，每 20 秒检查同步进程并在退出后重启；全部训练完成且上传结束后自行退出，无须中断现有训练来更新调度器。
2026-09-08 02:00 UTC，调整第一阶段预算时已替换训练调度器，并通过 `--adopt-running` 接管四个原 PPO 进程；W&B 同步监督已转交新调度器，原独立监督进程退出。
最初默认账号被原项目拒绝写入，已停止该同步进程。2026-09-08 01:29 UTC，使用用户提供的项目内凭据，通过 W&B viewer 接口确认邮箱为 `2486344338@qq.com`，用户名 `2486344338`，entity 为 `2486344338-zhejiang-university`；未改动全局登录配置。凭据仅存在项目内受保护且已被 Git 忽略的文件中，运行配置只保存其路径。

2026-09-08 01:42 UTC，按用户明确的项目约定，将 stage1 和 smoke_stage1 的原 run 从 `intact-preview-v2` 迁入 `intact-forward-predictor`，服务端确认 run ID 和历史记录保留。两阶段分别使用 `limb_context_20260907-stage1` / `limb_context_20260907-stage2` 分组，短测附加 `-smoke`；PPO 保留在 `intact-preview-v2`。已修正同步器的按阶段路由，后续 FiLM/concat 自动接入第二阶段分组。

- 第一阶段 run：https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/limb-7fdd4c20fbf2
- 第一阶段分组：https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/groups/limb_context_20260907-stage1
- 第二阶段当前分组：https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/groups/limb_context_20260907-stage2-2gpu8192-scratch

同步器新增各 worker 独立控制台输出及原始日志文件上传，显式提交每条指标以免最新验证点等待下次记录才显示。增量恢复、控制台隔离/部分行重试、阶段路由及配对评估队列共五项测试通过。

## 后续工作

- 正式全目录训练的持续吞吐与稳定性检查。
- 第一阶段完整训练和冻结 best checkpoint；八个 PPO 任务全部满足 5000 完整 updates。
- 固定 motion/start/load 的配对评估、端点和不对称负载、latent 替换诊断。
- 整理跨 seed 结果、失败/救回统计、误差与覆盖指标，记录是否支持 latent 性能增益。

## 第二阶段四卡修订（2026-09-08）

按用户要求，停止并保存旧单卡 PPO（B121=1788、B122=1780、B123=1852、K121=1713 updates），它们不进入正式结论。新 B/F/C/K 均从同一源 tracker 开始，4 GPU × 8192 env，5000 updates。阶段一四卡 PID 22245 被新调度器接管，训练不中断，上限仍为 8000。

只读对照了 `SP_Tracking/scripts/train_tracking_bfm_multigpu.sh`、训练入口、runner、PPO 及 motion loader：沿用 torchrun/LOCAL_RANK、按 rank 分片、梯度平均，并采用全局 advantage 统计；额外同步 critic 的 VecNorm 增量以保持各卡值函数一致。基础算法保留本实验的 plain residual PPO / GAE 语义。两个进程的 Gloo 测试检验全局统计、参数同步、仅主 rank 写日志及集体停止；相关 36 项测试通过，随后启动真实四卡短测。

新 PPO 输出：`runs/limb_context_20260907/ppo_4gpu8192`；短测：`smoke_4gpu8192`；日志：`logs/ppo_4gpu8192`。新 W&B 分组为 `limb_context_20260907-stage2-4gpu8192`，旧单卡 run 标记 superseded/excluded。阶段一的项目、run ID 与历史均保留。

旧的提前 control 评估等待器已停止；四卡调度器在全部正式训练通过独立审计后统一评估，避免两个队列同时争用释放的 GPU。

2026-09-08 02:45 UTC：四种 4×8192 / 3-update GPU 短测全部通过；各卡 actor、critic 和 VecNorm 状态哈希一致，四种融合的初始物理/reward/数据配置完全一致，rank 负载样本各不相同。正式 B121 在 GPU 4–7 完成 31 updates，约 3.62 s/update，每卡显存约 37 GB；确认四分片 [32457, 32457, 32457, 32456] 共 129827 motions / 48085337 frames。阶段一到 update 3800，继续上限 8000。详细验证保存在 `smoke_4gpu8192/audit.json` 与正式 B121 的 `remote_startup_verification.json`。
