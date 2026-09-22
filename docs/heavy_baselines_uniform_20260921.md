# RMA teacher / Any2Track：uniform 连续训练

2026-09-21，按用户最新要求，两组均从零使用 uniform motion sampling，关闭 failure rewind。取消原定 2000 轮重启与采样刷新，正式训练无更新上限，直到用户停止。模型与方法差异见[实施文档](heavy_baselines_implementation_20260921.md)。

两组保留相同 HDR、完整过滤后的 220480 条 motion、每组 4 GPU × 8192 环境，以及原定 PPO 超参数。均不加载预训练 Memory350，不启用 residual 的 DR 回归辅助损失。RMA 读取真实 108 维物理参数，通过可训练编码器条件化 residual actor；Any2Track 的历史编码器继续仅由世界模型预测目标训练。

两组部署在 `root@10.127.48.252:30171`、主机 `lgsl-a4-5f01-m4-7-h100gpu58`。**RMA teacher 使用 GPU 4,5,6,7，Any2Track 使用 GPU 0,1,2,3**，各自独立运行四 rank、完整数据集和 32768 个环境。

<!-- STATUS_BEGIN -->
**RMA 正式训练已启动，启动验证全部通过。** 2026-09-21 12:33:47 UTC 验证时已完成 8 次 PPO 更新，最近 5 次平均 **4.76 秒/次**。使用指定主机 GPU 4–7，4 × 8192 环境，完整过滤后的 220480 条 motion；HDR、reward、termination 和 tracker 均与现有 uniform baseline 一致。实际日志确认 uniform、无 failure rewind，且无更新上限或 2000 轮阶段转换。

- [RMA W&B](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/heavy-rma_teacher-332fcf147a)
- [完整启动验证](../runs/144000-exp-heavy/baselines/rma_teacher_uniform/startup_verification.json)

38 项相关回归测试与 RMA 4 × 8192 实机预检均通过。
<!-- STATUS_END -->

<!-- ANY2TRACK_STATUS_BEGIN -->
**Any2Track 正式训练已在 GPU 0–3 启动，完整启动验证通过。** 2026-09-21 14:01:10 UTC 验证时已完成 9 次 PPO 更新，最近 5 次平均 **7.78 秒/次**。实际完整数据集、HDR、reward、termination 与 tracker 均与现有 uniform baseline 一致；每次 PPO 更新执行 20 次 world-model optimizer step，PPO 阶段冻结 history encoder。实际日志为 uniform、无 rewind，无 2000 轮重启与更新上限。W&B 服务端确认已收到更新记录。

4 × 8192 实机预检确认 history encoder、adapter、world model 均实际更新，冻结 tracker 权重不变，四 rank 参数一致。正式训练从零初始化，未继承预检权重。

- [Any2Track W&B](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/heavy-any2track-a5302212ac)
- [Any2Track 完整启动验证](../runs/144000-exp-heavy/baselines/any2track_uniform/startup_verification.json)
- [Any2Track W&B 服务端核验](../runs/144000-exp-heavy/baselines/any2track_uniform/wandb_remote_verification.json)
- [Any2Track 正式规模预检](../runs/144000-exp-heavy/baselines/any2track_uniform/preflight_runtime_verification.json)
- [Any2Track 进程记录](../runs/144000-exp-heavy/baselines/any2track_uniform/controller_process.json)
<!-- ANY2TRACK_STATUS_END -->

启动命令：

```bash
.venv/bin/python scripts/run_144000_heavy_baselines.py \
  --method rma_teacher --motion-sampling uniform --gpus 4,5,6,7 --run

.venv/bin/python scripts/run_144000_heavy_baselines.py \
  --method any2track --motion-sampling uniform --gpus 0,1,2,3 --run
```

默认 sampling 已为 uniform。启动器先进行 4 × 8192、2 次更新的短 motion 预检，再从零加载完整数据进行正式训练；预检权重不会用于正式初始化。

产物目录为 `runs/144000-exp-heavy/baselines/{rma_teacher,any2track}_uniform/continuous_uniform/`。正式配置 `maximum_updates=null`、`until_user_stop=true`，sampling mode 为 uniform，failure rewind 为 false，`sampling_reset_generation=0`。不会因为到达 2000 次更新改变进程、采样或权重。

W&B project 为 `intact-preview-v2`，group 为 `144000-exp-heavy`；run name 为 `144000-exp-heavy-rma-teacher-uniform` 或 `144000-exp-heavy-any2track-uniform`。保留 tracker 指标分组和方法各自的 `Teacher/*`、`WorldModel/*`、`Adapter/*`。

正常中断后可重用启动命令恢复同一目录的最新 checkpoint；恢复模型、优化器、normalizer 与计数，不重置学习到的 std。Uniform 无 adaptive 统计要继承。为历史实验保留的显式 adaptive 启动选项不影响这两组默认训练。

验证覆盖：两种方法的无上限配置、超过 2000/8000 更新后的连续恢复、不允许混入旧 adaptive checkpoint、原 RMA/AnyAdapter 模型梯度与真实 PPO 更新，以及 uniform/no-rewind 采样合同。

远端环境补齐：复制当前训练机同版本 `fd 8.3.1` 至 `/usr/local/bin/fdfind`，SHA-256 为 `41bba7ba205681255e40b2cc703f668de50ddb8e5663883af417bc7c24e751b2`。首次正式启动因缺失此工具在训练更新前退出，记录保留于 [启动依赖记录](../runs/144000-exp-heavy/baselines/rma_teacher_uniform/startup_failure_missing_fdfind.json)。数据扫描与过滤逻辑保持不变。

该开发机无法通过默认 DNS 解析外部域名。已在本项目 `.runtime/limb_context/wandb_config/settings` 配置现有训练环境使用的 HTTP/HTTPS 代理；远端 W&B 已读取配置并通过认证。不修改系统 DNS；代理配置不写入 git。

- [38 项测试记录](../runs/144000-exp-heavy/baselines/rma_teacher_uniform/unit_tests.json)
- [正式规模预检](../runs/144000-exp-heavy/baselines/rma_teacher_uniform/preflight_runtime_verification.json)

正式进程的验证命令必须在训练机器执行，因为 `/proc` 中的 PID 属于远端主机：

```bash
.venv/bin/python scripts/verify_heavy_baseline_startup.py \
  --root runs/144000-exp-heavy/baselines/rma_teacher_uniform

.venv/bin/python scripts/verify_heavy_baseline_startup.py \
  --root runs/144000-exp-heavy/baselines/any2track_uniform
```

[W&B 服务端记录核验](../runs/144000-exp-heavy/baselines/rma_teacher_uniform/wandb_remote_verification.json)确认线上已收到实际更新次数。
