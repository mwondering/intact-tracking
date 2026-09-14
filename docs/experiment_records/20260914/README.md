# 实验记录归档：2026-09-14

归档时间：2026-09-14T05:54:58.738785+00:00。收录 1824 个文件、100 个实验目录，共 57.4 MB。

包含训练配置、评测协议、汇总指标、对比报告、验证记录，以及报告直接引用且不超过 2 MB 的 PNG/SVG 图。文本和数据保留原文；[manifest.json](manifest.json) 记录每份文件的服务器来源路径和 SHA256。

这是一次静态归档。运行中的状态文件在归档期间分别读取，轮数、PID 和 W&B 状态仅代表当时快照；八卡训练在归档后继续运行。权重、原始轨迹、完整日志、W&B 缓存、离线 HTML 与大型图仍保留在实验服务器。历史报告中的绝对路径和未归档文件链接需要在原服务器访问，具体遗漏的已引用产物见 manifest。

初次归档时发现旧 supervisor 已退出，原 `state.json` 停留在较早轮数。已修复并接管原八个训练进程，未重启训练；[恢复验证](metadata/supervisor_recovery_verification.json)、[恢复后状态](metadata/training_state.after_monitor_recovery.json)、[更新后的 READY](metadata/READY.after_monitor_recovery.json) 为后补的最新记录。旧状态和 READY 仍保留用于追溯。

## 主要入口

- [最新不限幅 residual、uniform、腕部 10 N·m 配置](../../residual_uniform_unbounded_20260914.md)
- [最新八组启动核验](records/limb_context_20260914_fixed_dr_specialists8_uniform_unbounded_wrist10/startup_verification.json)、[W&B 上传核验](records/limb_context_20260914_fixed_dr_specialists8_uniform_unbounded_wrist10/wandb_startup_verification.json)、[训练状态快照](records/limb_context_20260914_fixed_dr_specialists8_uniform_unbounded_wrist10/state.json)
- [旧八组独立 policy 的 1000 轮对比](records/limb_context_20260914_fixed_dr_specialists8/analysis/compare_update_001000/report.md)
- [memory350 响应标签 5/10 步在 15000 轮的比较](records/limb_context_20260912_memory350_response_window_ablation/comparison_015000/README.md)
- [弱正负样本方案](../../memory350_weak_pairs.md)、[MoE 与 critic action 方案](../../memory350_online_kmeans16_critic_action_20260913.md)
- [v12 latent t-SNE 与真实距离说明](records/latent_cluster_probe_v12_u8000_20260908/tsne_20260911/README.md)
- [后续 PPO 默认参数快照](metadata/future_ppo_defaults.json)

旧共享 A 与新八组 B 的腕部力矩、手部负载、动作限幅和 motion 采样设置不同；旧 A 只能作为历史参考，严格共享/独立 policy 对照需要相同的新配置。

## 归档及监控修复后验证

完整 CPU 测试：576 passed、1 skipped、0 failed。为保持八卡训练运行，测试进程隐藏了 CUDA；跳过项是 GPU replay/normalization 驻留测试。详见 [validation.json](validation.json)。真实 GPU 短程 PPO、恢复、评测和八组正式启动核验见最新实验记录。

## 各实验目录

| 实验 | 文件数 |
|---|---:|
| [abc_lafan_20260907](records/abc_lafan_20260907/) | 4 |
| [abc_lafan_20260907_r2](records/abc_lafan_20260907_r2/) | 7 |
| [abc_lafan_validation](records/abc_lafan_validation/) | 11 |
| [adaptation_goal](records/adaptation_goal/) | 222 |
| [forward_nominal_v7](records/forward_nominal_v7/) | 1 |
| [forward_nominal_v7_1](records/forward_nominal_v7_1/) | 1 |
| [forward_nominal_v9_unified](records/forward_nominal_v9_unified/) | 1 |
| [forward_nominal_v9_unified_4096](records/forward_nominal_v9_unified_4096/) | 1 |
| [forward_predictor_mlp_v1_8192](records/forward_predictor_mlp_v1_8192/) | 1 |
| [forward_predictor_mlp_v2_4096](records/forward_predictor_mlp_v2_4096/) | 1 |
| [forward_predictor_mlp_v2_8192](records/forward_predictor_mlp_v2_8192/) | 1 |
| [forward_predictor_payload_v13_8192_run1](records/forward_predictor_payload_v13_8192_run1/) | 1 |
| [forward_predictor_transformer_v12_8192_run2](records/forward_predictor_transformer_v12_8192_run2/) | 1 |
| [forward_predictor_transformer_v12_8192_run3](records/forward_predictor_transformer_v12_8192_run3/) | 1 |
| [forward_predictor_transformer_v1_8192](records/forward_predictor_transformer_v1_8192/) | 1 |
| [forward_predictor_transformer_v2_8192_run1](records/forward_predictor_transformer_v2_8192_run1/) | 1 |
| [forward_predictor_transformer_v2_8192_run2](records/forward_predictor_transformer_v2_8192_run2/) | 1 |
| [forward_predictor_transformer_v2_8192_run3](records/forward_predictor_transformer_v2_8192_run3/) | 1 |
| [forward_predictor_transformer_v3_8192_run3](records/forward_predictor_transformer_v3_8192_run3/) | 1 |
| [forward_predictor_transformer_v4_8192_run3](records/forward_predictor_transformer_v4_8192_run3/) | 1 |
| [forward_predictor_transformer_v5_8192_run3](records/forward_predictor_transformer_v5_8192_run3/) | 1 |
| [forward_predictor_transformer_v7_8192_run4](records/forward_predictor_transformer_v7_8192_run4/) | 1 |
| [forward_predictor_transformer_v7_8192_run5](records/forward_predictor_transformer_v7_8192_run5/) | 1 |
| [forward_predictor_transformer_v7_8192_run6](records/forward_predictor_transformer_v7_8192_run6/) | 1 |
| [forward_predictor_transformer_v8_8192_run6](records/forward_predictor_transformer_v8_8192_run6/) | 1 |
| [forward_predictor_transformer_v9_8192_run1](records/forward_predictor_transformer_v9_8192_run1/) | 1 |
| [forward_predictor_transformer_v9_8192_run2](records/forward_predictor_transformer_v9_8192_run2/) | 1 |
| [intact_online](records/intact_online/) | 1 |
| [intact_online_largebatch](records/intact_online_largebatch/) | 1 |
| [intact_online_largebatch2](records/intact_online_largebatch2/) | 1 |
| [intact_online_largebatch3](records/intact_online_largebatch3/) | 1 |
| [intact_online_largebatch4](records/intact_online_largebatch4/) | 1 |
| [intact_online_largebatch4_sonic_filtered](records/intact_online_largebatch4_sonic_filtered/) | 1 |
| [latent_cluster_probe_memory350_nominal50_u1800_20260911](records/latent_cluster_probe_memory350_nominal50_u1800_20260911/) | 11 |
| [latent_cluster_probe_memory350_nominal50_u5000_20260911](records/latent_cluster_probe_memory350_nominal50_u5000_20260911/) | 12 |
| [latent_cluster_probe_memory350_u22700_20260911](records/latent_cluster_probe_memory350_u22700_20260911/) | 21 |
| [latent_cluster_probe_v12_u8000_20260908](records/latent_cluster_probe_v12_u8000_20260908/) | 15 |
| [limb_context_20260907](records/limb_context_20260907/) | 83 |
| [limb_context_20260907_adaptive1000](records/limb_context_20260907_adaptive1000/) | 160 |
| [limb_context_20260908_no_ee_4gpu](records/limb_context_20260908_no_ee_4gpu/) | 241 |
| [limb_context_20260909_context200](records/limb_context_20260909_context200/) | 6 |
| [limb_context_20260909_memory350](records/limb_context_20260909_memory350/) | 7 |
| [limb_context_20260910_memory350_ppo](records/limb_context_20260910_memory350_ppo/) | 244 |
| [limb_context_20260910_short50](records/limb_context_20260910_short50/) | 10 |
| [limb_context_20260911_memory350_encoder2x](records/limb_context_20260911_memory350_encoder2x/) | 11 |
| [limb_context_20260911_memory350_encoder2x_nominal50](records/limb_context_20260911_memory350_encoder2x_nominal50/) | 18 |
| [limb_context_20260911_memory350_nominal50](records/limb_context_20260911_memory350_nominal50/) | 6 |
| [limb_context_20260912_memory350_encoder2x_nominal50_weakpairs](records/limb_context_20260912_memory350_encoder2x_nominal50_weakpairs/) | 41 |
| [limb_context_20260912_memory350_encoder2x_nominal50_weakpairs_tune004_s03](records/limb_context_20260912_memory350_encoder2x_nominal50_weakpairs_tune004_s03/) | 24 |
| [limb_context_20260912_memory350_encoder2x_nominal50_weakpairs_tune008_m11_s03](records/limb_context_20260912_memory350_encoder2x_nominal50_weakpairs_tune008_m11_s03/) | 20 |
| [limb_context_20260912_memory350_response_window_ablation](records/limb_context_20260912_memory350_response_window_ablation/) | 117 |
| [limb_context_20260913_memory350_compressed_ee_adaptive_scratch](records/limb_context_20260913_memory350_compressed_ee_adaptive_scratch/) | 105 |
| [limb_context_20260913_memory350_compressed_ee_restored_u1200](records/limb_context_20260913_memory350_compressed_ee_restored_u1200/) | 3 |
| [limb_context_20260913_memory350_compressed_grid256_adaptive_scratch](records/limb_context_20260913_memory350_compressed_grid256_adaptive_scratch/) | 47 |
| [limb_context_20260913_memory350_compressed_grid256_tracker_action_adaptive_scratch](records/limb_context_20260913_memory350_compressed_grid256_tracker_action_adaptive_scratch/) | 70 |
| [limb_context_20260913_memory350_online_kmeans16_critic_action](records/limb_context_20260913_memory350_online_kmeans16_critic_action/) | 107 |
| [limb_context_20260913_memory350_online_kmeans16_uniform_dr](records/limb_context_20260913_memory350_online_kmeans16_uniform_dr/) | 23 |
| [limb_context_20260914_fixed_dr_specialists8](records/limb_context_20260914_fixed_dr_specialists8/) | 49 |
| [limb_context_20260914_fixed_dr_specialists8_uniform_unbounded_wrist10](records/limb_context_20260914_fixed_dr_specialists8_uniform_unbounded_wrist10/) | 20 |
| [memory350_speed_benchmark_20260912](records/memory350_speed_benchmark_20260912/) | 9 |
| [model_gradient_residual_v1_run1](records/model_gradient_residual_v1_run1/) | 1 |
| [preview_v2](records/preview_v2/) | 4 |
| [preview_v2_validation_IumuuB](records/preview_v2_validation_IumuuB/) | 6 |
| [preview_v2_validation_limb](records/preview_v2_validation_limb/) | 3 |
| [residual_payload_tracker_output_latent_run1](records/residual_payload_tracker_output_latent_run1/) | 1 |
| [residual_policy_latent_dr_payload_v13_run1](records/residual_policy_latent_dr_payload_v13_run1/) | 1 |
| [residual_policy_latent_dr_payload_v13_run2](records/residual_policy_latent_dr_payload_v13_run2/) | 1 |
| [residual_policy_latent_nominal_v13_run1](records/residual_policy_latent_nominal_v13_run1/) | 1 |
| [residual_policy_latent_syncfix_run1](records/residual_policy_latent_syncfix_run1/) | 1 |
| [residual_policy_latent_syncfix_run2](records/residual_policy_latent_syncfix_run2/) | 1 |
| [residual_policy_latent_syncfix_run3](records/residual_policy_latent_syncfix_run3/) | 1 |
| [residual_policy_latent_syncfix_run4](records/residual_policy_latent_syncfix_run4/) | 1 |
| [residual_policy_latent_syncfix_run5](records/residual_policy_latent_syncfix_run5/) | 1 |
| [residual_policy_latent_syncfix_run6](records/residual_policy_latent_syncfix_run6/) | 1 |
| [residual_policy_latent_v12_u7000_devicefix_run1](records/residual_policy_latent_v12_u7000_devicefix_run1/) | 1 |
| [residual_policy_latent_v12_update7000](records/residual_policy_latent_v12_update7000/) | 1 |
| [residual_policy_no_latent_dr_payload_v13_run1](records/residual_policy_no_latent_dr_payload_v13_run1/) | 1 |
| [residual_policy_no_latent_dr_payload_v13_run2](records/residual_policy_no_latent_dr_payload_v13_run2/) | 1 |
| [residual_policy_no_latent_g67_wandbfix](records/residual_policy_no_latent_g67_wandbfix/) | 1 |
| [residual_policy_no_latent_nominal_v13_run1](records/residual_policy_no_latent_nominal_v13_run1/) | 1 |
| [residual_policy_no_latent_nominal_v13_run2](records/residual_policy_no_latent_nominal_v13_run2/) | 1 |
| [residual_policy_no_latent_run1](records/residual_policy_no_latent_run1/) | 1 |
| [residual_policy_no_latent_run2](records/residual_policy_no_latent_run2/) | 1 |
| [residual_policy_payload_v13_latent_run2](records/residual_policy_payload_v13_latent_run2/) | 1 |
| [residual_policy_payload_v13_no_latent_run1](records/residual_policy_payload_v13_no_latent_run1/) | 1 |
| [residual_uniform_wrist10_hand2p5_preflight_20260914](records/residual_uniform_wrist10_hand2p5_preflight_20260914/) | 6 |
| [residual_v1_1](records/residual_v1_1/) | 1 |
| [residual_v1_2](records/residual_v1_2/) | 1 |
| [residual_v2_2](records/residual_v2_2/) | 1 |
| [residual_v2_3](records/residual_v2_3/) | 1 |
| [residual_v3_2_trunk](records/residual_v3_2_trunk/) | 1 |
| [residual_v4_2_trunk](records/residual_v4_2_trunk/) | 1 |
| [residual_v5_1_trunk](records/residual_v5_1_trunk/) | 1 |
| [residual_v5_2_trunk](records/residual_v5_2_trunk/) | 1 |
| [residual_v6_1_trunk](records/residual_v6_1_trunk/) | 1 |
| [residual_v6_2_trunk](records/residual_v6_2_trunk/) | 1 |
| [residual_v7_1_trunk](records/residual_v7_1_trunk/) | 1 |
| [residual_v7_3_trunk](records/residual_v7_3_trunk/) | 1 |
| [residual_v7_4_trunk](records/residual_v7_4_trunk/) | 1 |
| [simulator_preview_single_motion](records/simulator_preview_single_motion/) | 10 |
