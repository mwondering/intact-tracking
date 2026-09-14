# Memory350 实验目标完成核验

实验范围保持原定要求。验证可以得到否定整体增益的结果；这不改变实验完成标准。

| 要求 | 核验结果 | 证据 |
|---|---|---|
| 指定源码、日志、checkpoint、报告及缓存位于本项目，外部 tracker/数据只读取 | 检查实际输出路径、报告链接及进程缓存配置；未操作其他目录任务 | [final_inputs.json](../../artifacts/protocol_audits/final_inputs.json) |
| baseline GPU 0/1，latent GPU 2/3，每卡 8192 环境 | 两 rank/组，16384 环境/组，实际启动与完成记录 | [verified.json](../../artifacts/baseline_completion_audit/verified.json)、[verified.json](../../artifacts/film_completion_audit/verified.json)、[final_inputs.json](../../artifacts/protocol_audits/final_inputs.json) |
| 完整 motion_data_full 数据集 | 重新枚举 129827 个文件的路径/大小/mtime 清单；运行时两 rank 共加载 48085337 帧 | [final_inputs.json](../../artifacts/protocol_audits/final_inputs.json) |
| 冻结 tracker 全部原始 DR 和观测噪声，再加四肢负载 | 两组六项 DR、噪声和实际逐 rank 质量 SHA 一致 | [final_inputs.json](../../artifacts/protocol_audits/final_inputs.json) |
| 双手及小腿中部各独立 U(0,4 kg)，无 nominal 子集 | 源码采样规则、运行质量范围/相关性及固定局部位置已核对 | [final_inputs.json](../../artifacts/protocol_audits/final_inputs.json) |
| residual actor/critic 从头初始化，公共主干一致 | 第 0 轮实际权重逐位匹配约定 seed 新建主干；optimizer state 为空 | [initial_checkpoints.json](../../artifacts/final_output_audits/initial_checkpoints.json) |
| baseline 仅使用原 tracker observation，不读 latent | 实际参数无 FiLM、context 为空，原始输入维度 1645/6330 | [baseline_latent_architecture_audit.json](../../artifacts/baseline_latent_architecture_audit.json)、[baseline_005000_film_005000.json](../../artifacts/checkpoint_state_audits/baseline_005000_film_005000.json) |
| 冻结 Memory350 latent 同时用于 actor 和 critic | short50/chunk10×30、64 维 latent、两条独立 FiLM，PPO 不执行 predictor | [context_selection.json](../../context_selection.json)、[baseline_005000_film_005000.json](../../artifacts/checkpoint_state_audits/baseline_005000_film_005000.json)、[baseline_latent_architecture_audit.json](../../artifacts/baseline_latent_architecture_audit.json) |
| tracker、encoder 和 context normalization 冻结 | 实际 tracker 53 个权重/缓冲逐位一致；encoder eval/无梯度且固定 SHA | [baseline_005000_film_005000.json](../../artifacts/checkpoint_state_audits/baseline_005000_film_005000.json) |
| 每组至少 5000 PPO updates，其他训练参数一致 | 两组恰好 5000，所有优化参数均为 100000 optimizer steps，两 rank 最终一致 | [verified.json](../../artifacts/film_completion_audit/verified.json)、[verified.json](../../artifacts/baseline_completion_audit/verified.json)、[baseline_005000_film_005000.json](../../artifacts/checkpoint_state_audits/baseline_005000_film_005000.json) |
| 1000 轮后由 uniform 精确续训至 adaptive | 双方均仅一次第 1000 轮模型/优化器/归一化恢复，failure rewind 关闭 | [final_inputs.json](../../artifacts/protocol_audits/final_inputs.json) |
| episode=1000，训练去 EE、测试保留原失败条件 | 训练合同、测试失败标记和触发项相互核对 | [final_inputs.json](../../artifacts/protocol_audits/final_inputs.json)、[final_005000.json](../../artifacts/failure_breakdown/final_005000.json) |
| 每 100 轮保存测试并报告，测试不改变训练状态 | 双方 100–5000 共 50 次周期测试均通过源码的 checkpoint/protocol/状态保护审计 | [final_results.json](../../final_results.json) |
| 同一第 5000 轮测试固定 0/2/4 kg 和独立 uniform 的 cold/warm，另测冻结 tracker | 24 项×4096 motion，实际 checkpoint SHA、物理/query/motion/start/warm-up 协议已配对 | [final_reconciliation.json](../../artifacts/final_output_audits/final_reconciliation.json) |
| cold 历史为空，warm 跨 reset 保留长期 | 实际 cold 无历史，四个 warm 起始短期空、长期各 30 个完整片段 | [final_reconciliation.json](../../artifacts/final_output_audits/final_reconciliation.json) |
| 比较 tracking、失败率及覆盖率并保留证据 | 全部十项误差、共同有效时段 body/joint、配对 bootstrap 及失败触发项已检查 | [final_005000.csv](../../artifacts/tracking_details/final_005000.csv)、[final_reconciliation.json](../../artifacts/final_output_audits/final_reconciliation.json)、[final_005000.json](../../artifacts/failure_breakdown/final_005000.json) |
| 使用用户账号监控全部训练 | 2486344338@qq.com；两组各 5000 条完整更新，最终 1344 数值远端一致 | [wandb_account_audit.json](../../wandb_account_audit.json)、[wandb_training_completion_audit.json](../../artifacts/wandb_training_completion_audit.json)、[wandb_final_summary_audit.json](../../artifacts/wandb_final_summary_audit.json) |
| 给出 latent 是否提升 tracking 的明确结论 | 整体优势未证实；明确报告 0 kg 和辅助指标收益、2/4 kg 主误差代价与冻结 tracker 对照；单训练 seed 限制 | [final_results.json](../../final_results.json)、[reviewed_report.json](../../artifacts/final_output_audits/reviewed_report.json) |
| 保存报告、曲线、CSV、checkpoint 与原始轨迹 | 链接和 336 行 CSV 已检查，原自动报告保留，最终主表使用共同有效时段均值 | [reviewed_report.json](../../artifacts/final_output_audits/reviewed_report.json)、[tracking_progress_update_005000.json](../../artifacts/plots/tracking_progress_update_005000.json) |
| 实验训练、评估和报告任务正常完成 | 全部 job complete、已记录任务进程退出；监督器恢复事件未重启 PPO，源码修订可追溯 | [supervisor_state.json](../../supervisor_state.json)、[verified.json](../../artifacts/supervisor_recovery_20260910/verified.json) |
