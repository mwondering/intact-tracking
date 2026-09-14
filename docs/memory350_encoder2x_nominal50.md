# Memory350 nominal50：原版与 encoder2x 对照

2026-09-11，用户要求先检查原版 nominal50 latent，再停止原 all-DR 2x，
将 2x 也改成一半 nominal，并移到 GPU 4–7 对照。

初步 latent 测试使用冻结的原版 nominal50 `update_001800.pt`，其余模型读取相同轨迹。
结果与完整采集图表保存在
[latent 报告](../runs/latent_cluster_probe_memory350_nominal50_u1800_20260911/README.md)。

旧 all-DR 2x 已响应 STOP，在 update 4968 / 19872 optimizer steps 保存后退出。
只停止该运行，旧权重与日志保留；
[停止核验](../runs/limb_context_20260911_memory350_encoder2x/stop_verification.json)。

- 原版：`runs/limb_context_20260911_memory350_nominal50/stage1_8192`，GPU 0–3，继续训练。
- 新 2x：`runs/limb_context_20260911_memory350_encoder2x_nominal50/stage1_8192`，GPU 4–7，从头训练。
- 两版每卡 8192 worlds，其中 8064 train = 4032 nominal + 4032 DR，
  128 validation = 64 nominal + 64 DR；偶数 local world ID 为 nominal。
- Nominal 物理恢复 compiled defaults，四肢负载/encoder bias/额外随机推扰清零；
  DR 保留 tracker DR、推扰及四肢独立 U(0,4 kg) 负载。
- 相同 tracker、129827 motions、seed 717、short50 + disjoint long30×10、64-D latent。
- 只将 chunk/long/final attention depth 从 1/2/2 改到 2/4/4；
  encoder 参数 1,035,328 → 2,026,688，predictor 保持 19,059,798。
- Predictor 和公共 encoder 层的初始权重相同；新增层独立初始化且恢复后续 RNG stream。
- 新 2x 复用原版 nominal50 的 normalization 和八份固定验证文件，逐文件 SHA256 校验。
  严格核对训练配置、实际物理采样、nominal 恢复和运动数据分片；拒绝 all-DR reference。
- 相同 optimizer/LR/损失、global batch 4096、microbatch 256/rank、4 optimizer steps/update。
  8000 updates 是 cosine 时间尺度，之后保持 LR 1e-5，训练直到用户停止。
- 按相同 update 对照，单独记录 nominal 和 DR NMSE；定期逐窗口配对评估并生成报告。
  一次训练种子，较深 encoder 每步计算量更多。Latent 聚类和 policy 收益仍须独立评估。

独立入口 `intact_tracking.cli.forward_memory_scale_nominal_train`；
启动器 `scripts/run_memory350_scale_nominal_stage1.py`；
核验脚本 `scripts/verify_memory350_scale_nominal_startup.py`。
GPU 4–7 经用户明确指定，可在显存充足时与已有任务共享，不停止无关进程。

[运行状态](../runs/limb_context_20260911_memory350_encoder2x_nominal50/state.json)
；[启动核验](../runs/limb_context_20260911_memory350_encoder2x_nominal50/startup_verification.json)
；[持续对照报告](../runs/limb_context_20260911_memory350_encoder2x_nominal50/report.md)。

正式启动核验通过：完整 129827 motions / 48085337 frames；四 rank 参数完全一致，
归一化和八份验证文件与原版 nominal50 一致，训练/验证 nominal 比例及实际物理已核对。
核验时完成 24 updates / 96 optimizer steps，首个 checkpoint 为 update 1；13 项相关测试通过。

[正式 W&B](https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/m350scale-n50-022050a4d5bb)。
用户随后明确授权结束 GPU 4–7 的占卡程序；四个占卡进程及其 uv 启动器均已退出，
记录见运行目录 `gpu_occupier_stop.json`。GPU 0–3 为原版 nominal50，4–7 为 encoder2x nominal50。

首个 update 100 同轮逐窗口预测评估已完成，分别统计 nominal 和 DR。
DR 误差相对原版 +1.55%，nominal +2.90%；仅是早期观察，不作为收敛结论。
