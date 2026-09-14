# ee_body_pos 终止项：训练配置修订

后续新实验的全部 residual 对照统一采用 `no_ee_body_pos`，保留 time_out、anchor_pos、anchor_ori。
当前批次固定 original，包含未开始的配对种子；本次没有启动或重启 PPO。新配置的性能尚未验证。

4500 轮 residual checkpoint 与冻结 tracker 使用相同 4096 motions/starts、500 步上限及原终止标准：

| 每部位负载 | 策略 | 失败总数 | ee_body_pos 触发 | 仅 ee_body_pos |
|---|---|---:|---:|---:|
| 0 kg | frozen | 2 | 2 | 2 |
| 0 kg | baseline_121 | 3 | 3 | 3 |
| 0 kg | film_121 | 5 | 4 | 4 |
| 2 kg | frozen | 38 | 37 | 37 |
| 2 kg | baseline_121 | 34 | 33 | 33 |
| 2 kg | film_121 | 31 | 28 | 27 |
| 4 kg | frozen | 561 | 550 | 549 |
| 4 kg | baseline_121 | 417 | 413 | 412 |
| 4 kg | film_121 | 375 | 366 | 366 |

每项可同时触发；不能从失败数中扣除该项来推断新成功率，因为原始轨迹在终止后没有继续采集。

机制：四个 wrist_yaw / ankle_roll 任一参考高度误差超过 0.5 m 即触发，没有连续超限缓冲。
去掉训练终止可能增加偏离后的恢复经验；adaptive sampling 的失败信号也会变化，性能收益需新批次验证。
评估继续使用原完整终止标准，避免改变成功率和误差的统计口径。

新批次参数：`--training-terminations no_ee_body_pos`；新调度队列所有 PPO 自动添加。
独立新训练默认关闭此项，既有 checkpoint 默认继承原配置。根目录 training_terminations.json 固定实验配置，普通 resume 拒绝改变。
实际活动项保存在 run_config/config/checkpoint/completion；训练日志和 W&B 对应字段为 `training_terminations/ee_body_pos_enabled`。

验证：23 项配置、续训、调度、W&B/预算回归检查，以及 1 项实际 termination manager 行为测试通过。
真实 tracker 与两组最终 checkpoint 的兼容检查记录见 configuration_verification.json。
