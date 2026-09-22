# NDR 与 HDR 的随机化范围

核对日期：2026-09-21。NDR 指原 SPV5-2A `checkpoint_144000.pt` 使用的 DR；HDR 指当前 heavy residual 两组共同使用的 DR。下表列出 DR 环境中的物理参数、控制随机化及外力扰动，不包含观测噪声。HDR 保留 NDR 的原生范围，新增四肢负载质量与负载质心偏移。

| 参数 | NDR：原 144000 | HDR：当前四肢负载环境 |
|---|---|---|
| 躯干 COM 偏移 x / y / z | 每轴 [-7.5, 7.5] cm | 相同 |
| 躯干质量增量 | [-1, 1] kg | 相同 |
| 足底切向摩擦系数 | [0.3, 2.0]，同一环境的双脚碰撞体共享 | 相同 |
| 上肢 Kp 缩放（肩、肘、腕） | [0.9, 1.1] × nominal | 相同 |
| 上肢 Kd 缩放（肩、肘、腕） | [0.9, 1.1] × nominal | 相同 |
| 腰部及下肢 Kp 缩放（腰、髋、膝、踝） | [0.8, 1.2] × nominal | 相同 |
| 腰部及下肢 Kd 缩放（腰、髋、膝、踝） | [0.8, 1.2] × nominal | 相同 |
| 各关节 armature 缩放 | [0.8, 1.2] × nominal | 相同 |
| 关节编码器固定偏置 | [-0.01, 0.01] rad | 相同 |
| 动作延迟 | {0, 1, 2} 个仿真步，即 {0, 5, 10} ms | 相同 |
| 动作平滑系数 α | [0.8, 1.0] | 相同 |
| 躯干外力脉冲 | 世界坐标系各轴 [-10, 10] N | 相同，仅作用于 DR 环境 |
| 外力持续时间 | [0.3, 0.5] s | 相同 |
| 外力触发间隔 | [3, 6] s | 相同 |
| 左手、右手附加负载质量 | 0 kg | 每只手 [0, 2.5] kg |
| 左小腿、右小腿附加负载质量 | 0 kg | 每条小腿 [0, 4] kg |
| 四肢附加负载 COM 偏移 | 无附加负载 | 每个负载的 x / y / z 各 [-5, 5] cm |
| 四肢附加载荷总质量 | 0 kg | [0, 13] kg |
| 附加负载引起的惯性变化 | 无附加负载 | 根据负载质量和 COM，同步计算连杆与负载的合成质量、COM、惯量 |

原生随机化参数在各自范围内均匀采样，动作延迟为三个离散值等概率采样。负载质量采用以下分层方式；总负载质量由四项相加，合成惯量由物理计算得到，两者不独立采样。其他随机化不共享负载档位或物理模板。

- 每个肢体分为 4 个等宽质量档位，四肢的笛卡尔组合为 4^4 = 256 组；均衡分配环境后，各环境在对应档位内独立连续均匀采样。手部档位边界为 0 / 0.625 / 1.25 / 1.875 / 2.5 kg，小腿为 0 / 1 / 2 / 3 / 4 kg。
- 负载 COM 以挂载点为中心，在对应连杆的局部坐标系内，各轴独立均匀采样。区域为边长 10 cm 的立方体，最大偏移范数约 8.66 cm。挂载点为手腕 yaw 连杆的 (0.12, 0, 0) m、小腿 knee 连杆的 (0, 0, -0.15) m。
- 当前每卡 8192 环境中，820 个为严格 nominal，7372 个使用 HDR，256 个质量组合各分配 28–29 个 DR 环境。上表 HDR 范围适用于后者；nominal 环境恢复原始物理参数、关闭负载和外力，delay=0、α=1、encoder bias=0。
- HDR 的持久物理参数在同一 world 跨 reset 固定，Kp/Kd 也固定；原 tracker 的 motor reset 会重新采样 Kp/Kd。两者范围一致，但这一时间采样语义有区别。HDR 的控制延迟和平滑仍沿用原 reset 采样。
- Kp、Kd、armature 和躯干质量的辅助监督权重为 0，不表示这些环境 DR 被关闭。最新 residual 续训仅在 latent 组监督 torso COM、摩擦和四肢负载质量；**负载 COM 监督已关闭，但负载 COM 的环境随机化仍保留**。baseline 辅助损失为 0，两组物理范围完全相同，见[最新续训配置](heavy_residual_resume2000_mass5x_20260921.md)。context encoder 的既有权重保持冻结，本次只调整 residual 辅助目标。
- 两者均使用平地；虽然 checkpoint 中有 terrain height offset 的 [-0.02, 0.02] m 配置，该项在无 terrain motion plan 的平地环境中不生效，因此未计入上表。

持久物理标签为 NDR 92 维（躯干 COM 3 + 躯干质量 1 + 摩擦 1 + Kp/Kd/armature 各 29），HDR 108 维（额外负载质量 4 + COM 12）。该维数不包含编码器偏置、动作延迟/平滑和外力脉冲。

核对来源：

- [原 144000 checkpoint](/data_zcy/wxy/SP_Tracking/logs/rsl_rl/g1_tracking/2026-09-10_16-19-06_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_4gpu_8192env_motion_data_correct/checkpoint_144000.pt) 中的 `cfg.task.events`、`cfg.task.action`、`cfg.task.sim`。
- [当前 latent 组实际运行配置](../runs/144000-exp-heavy-residual-latent/ppo_4gpu8192_resume2000_mass5x_sampling_reset/run_config.json) 的 `physics.original_events`、`physics.original_action`、`physics.heavy_payload`；baseline 使用相同物理配置。
- [负载采样及惯性合成实现](../src/intact_tracking/memory350_heavy_dr.py)、[当前原生 DR 适配](../src/intact_tracking/memory350_native_policy.py)、[原 tracker 随机化实现](/data_zcy/wxy/SP_Tracking/src/sp_tracking/tasks/tracking/mdp/randomizations.py)。
