# Memory350 response-window 对照

两条分支均训练到 **u15000**；predictor 都只监督 **5 步**。

|分支|GPU|恢复点|A−B 标签|
|---|---|---:|---:|
|[response10](response10/)|0–3|11000|10 步|
|[response5](response5/)|4–7|12000|5 步|

每 1000 轮自动检查相同缓存上的跨 motion 表征；最后直接对比两个 u15000 checkpoint。
按 DR＋负载严格跨 motion Top1 选型，普通 DR 和五步预测一起报告。两条分支恢复时点不同，已明确记录。

两组完成后，自动选择 encoder 并启动从头训练的 residual PPO：baseline 占 0–3，latent 占 4–7；两组都没有总轮数上限。
Actor：**1645 → 512 → 256 → 128，拼接 64 维 latent → 192 → 256 → 128 → 29**。
Baseline 使用同网络、latent 槽全零；压缩器和 head 可训练，tracker/encoder 冻结。
Critic 保留原 6330 维输入，逐层压缩到 128 后使用相同的 64 维 latent 槽。

- [实时状态](state.json)
- [表征启动检查](startup_verification.json)
- [W&B 远端上传核验](wandb_upload_verification.json)
- [单元检查记录](test_results.json)
- [PPO 实机验证与自动衔接](PPO_READY.json)
- [完整方案与恢复说明](../../docs/memory350_response_window_ablation.md)

`smoke/` 是小规模程序验证，不是正式 PPO 效果。正式 PPO 等待 u15000 选型后才开始。
