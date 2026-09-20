# Residual PPO：探索配置、无界残差与幅度日志（2026-09-19）

当前 native PPO 新训练入口及 `scripts/run_144000_residual.py` 已使用以下设置：

| 配置 | 原设置 | 新设置 |
| --- | --- | --- |
| PPO entropy coefficient | 0.0002 | 0.005 |
| 可学习 Gaussian std 的初值 | 0.25 | 1.0 |
| residual mean | `0.25 * tanh(MLP(input))` | `MLP(input)` |
| `residual_output_mode` | 未设置，旧版 bounded 语义 | `unbounded` |
| `residual_scale` | 0.25 | 1.0；无界分支直接返回网络输出 |

新训练中的动作分布为

\[
\Delta a_t=f_\theta\bigl(h_{\mathrm{tracker}}(o_t),z_{t-4:t},a_t^{\mathrm{tracker}}\bigr),
\qquad
a_t\sim\mathcal N\bigl(a_t^{\mathrm{tracker}}+\Delta a_t,\operatorname{diag}(\sigma^2)\bigr).
\]

残差输出不经过 `tanh`、裁剪或额外缩放。当前动作链的 wrapper clip、raw-action clip、joint-target clip 均为空；启动时也会验证这一约束。残差末层仍以零权重、零 bias 初始化，因此初始策略均值等于冻结 tracker；初始 std 为 1.0，随后由 PPO 学习，并非固定为 1.0。PPO 的 entropy 项系数为 0.005。

源码位置：[`memory350_native_policy_train.py`](../src/intact_tracking/cli/memory350_native_policy_train.py)、[`memory350_policy_train.py`](../src/intact_tracking/cli/memory350_policy_train.py)、[`residual_policy.py`](../src/intact_tracking/residual_policy.py)。

旧 checkpoint 配置未包含 `residual_output_mode`，加载时继续使用原来的 bounded 计算，保证历史评估含义不变。新 checkpoint 会保存显式的 `unbounded` 配置。训练的配置一致性校验会拒绝将这次变更静默混入旧实验的恢复过程。

截至本次修改，原训练进程仍在运行，未重启或切换到新参数。新设置的训练效果和训练后输出幅度尚未测得。启动脚本仍会检查输出目录冲突和 preflight 中的源码摘要；原有 preflight 不能直接用于修改后的代码，需要为新的训练实验准备独立目录及对应验证记录。

**现有策略的实测幅度**

数据来源：`runs/144000_exp/stage2_proprio122_history5_tracker_action/ppo_8gpu8192/metrics.jsonl`，第 **3627–3726** 次更新共 100 条记录；最后一条时间为 **2026-09-19 15:04:42 UTC**。以下是仍采用旧参数的策略数据，单位为 policy command，未混入 Gaussian 探索噪声。

| 指标 | 100 次更新的平均值 |
| --- | ---: |
| residual RMS | 0.06455808 |
| frozen tracker action RMS | 1.85210281 |
| 上述两个平均 RMS 的比值 | 3.4857% |
| 旧日志 `residual_action_abs_max` | 0.24929176 |
| 打乱 latent 导致的 action delta RMS | 0.03042844 |
| 清零 latent 导致的 action delta RMS | 0.03841358 |
| Gaussian action std 日志均值 | 0.03155152 |

残差并非零输出，但相对于 tracker 的整体动作 RMS 较小。这一比值本身不能证明残差是否改善跟踪效果。打乱或清零 latent 会改变输出，也不能单独证明 latent 的使用是有益的。

旧日志中的 RMS 是各 rank RMS 的平均，`abs_max` 是各 rank 最大值的平均，表中又对更新取了平均，因此后者不是整段训练的全局最大值。旧版饱和比例还读取此前 forward 的缓存；本次已改为与当前诊断 observation batch 一致。无界残差不再记录饱和比例，评估 JSON 对该项输出 `null`。

**新训练的 W&B 指标**

| W&B key | 含义 |
| --- | --- |
| `Residual/mean_rms` | 学到的残差均值的 RMS |
| `Residual/mean_abs` | 残差均值的平均绝对值 |
| `Residual/mean_abs_max` | 所有 rank 的残差均值最大绝对值 |
| `Residual/mean_to_tracker_rms_ratio` | 残差 RMS / tracker action RMS |
| `Residual/tracker_action_rms` | 冻结 tracker 动作 RMS |
| `Residual/target_rms_rad` | `residual * joint_action_scale` 的 RMS，单位 rad |
| `Residual/target_abs_max_rad` | 上述关节目标修正的最大绝对值，单位 rad |
| `Residual/output_bounded` | 新策略为 0 |
| `training/action_std` | Gaussian 探索 std，与残差均值幅度分别记录 |

`Residual/*` 以 `completed_updates` 为横轴，每次 PPO 更新后记录一次当前 observation batch，覆盖所有 rank 的环境和动作维度，不代表整个 rollout 的时间统计。RMS 先汇总均方再开方，最大值使用跨 rank MAX；RMS 比值在全局汇总后计算。关节目标修正指标位于 SP delay、smoothing 和 joint offset 之前，不等同于机器人实际关节运动。原有 JSON、TensorBoard 和 `training/loss/*` 路径继续记录相应标量。

后续补齐了 SP/RSL 的基础 W&B 指标分类：`Loss/*`、`Policy/mean_std`、`Perf/*`、`Train/*`、`Episode/*`、`Episode_Reward/*`、`Episode_Termination/*`、`Metrics/*`。环境提供的带 `/` 名称原样保留，无分类的 episode key 才加 `Episode/`。reward、episode length 和环境指标使用所有 rank 汇总值；FPS 使用全局采样量和最慢 rank 耗时。所有这些图表使用 `completed_updates`，不混入 wall-time step。

正式训练继续使用 W&B `online` 模式，当前 proprio native smoke 使用 `disabled`。项目仍是 `intact-preview-v2`，实验 group 仍是 `144000_exp`；实验分组与上述指标分类是两个不同概念。旧进程尚未重启，因此这些新增标准分类目前只对后续启动的训练生效。日志对齐补丁通过 28 项相关回归测试，包括双进程的不等长 episode 数据汇总和指标名称/step 校验。

实现位置：[`residual_runner.py`](../src/intact_tracking/residual_runner.py)、[`memory350_policy_train.py`](../src/intact_tracking/cli/memory350_policy_train.py)。

**验证与后续事项**

100 项回归测试通过，包括：std=1.0 和 entropy=0.005；正负 10 的残差直接输出且保留梯度；旧 checkpoint 参数化兼容；bounded/unbounded 两种策略的 PPO 更新与恢复；双进程 Gloo 的全局 RMS、最大值和关节目标幅度统计；JSON、RSL logger 及模拟 W&B 接口的数值和 step 一致性。测试未启动新的 GPU 训练，也未验证新参数的学习效果。

共享 actor 隐藏层预测动力学参数的辅助头尚未接入。用户已确认排除质量辅助监督，优先 COM 三轴与摩擦；详见 [辅助头方案](residual_ppo_dr_aux_plan_20260919.md)。
