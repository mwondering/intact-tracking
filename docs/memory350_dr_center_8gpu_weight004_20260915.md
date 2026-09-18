# DR-center 八卡续训，表征权重 0.04

用户要求清理 GPU 0–3 的占卡任务，将现有 DR-center predictor 改成八卡续训，并将表征学习权重翻倍。

## 恢复来源与设置

- 原四卡任务在 update **1863** 正常响应 SIGTERM，协调保存 checkpoint 后退出；没有从较早的周期 checkpoint 回退。
- 来源：`runs/limb_context_20260915_dr_center_hand2p5_shin4/stage1_8192/update_001863.pt`。
- SHA256：`0063fce88e48e14edb59ce90d074fc15842fc51a0aa3a7e8d40b844c57d4c0da`。
- 来源 AdamW 已完成 **7452** 步，全部参数的 optimizer step 一致；scheduler `last_epoch=7452`、`T_max=32000`，学习率 **0.00026161613005045543**。
- 新目录：`runs/limb_context_20260915_dr_center_8gpu_weight004/stage1_8192`。

| 设置 | 四卡阶段 | 八卡阶段 |
|---|---:|---:|
| GPU | 4–7 | 0–7 |
| 每卡仿真环境 | 8192 | 8192 |
| 总仿真环境 | 32768 | 65536 |
| 每卡优化 batch | 1024 | 512 |
| 全局优化 batch | 4096 | 4096 |
| 每卡 microbatch | 256 | 256 |
| 每 update 的 optimizer step | 4 | 4 |
| DR 中心关系权重 | 0.02 | **0.04** |

保持每个 microbatch 的中心配对方式和全局优化 batch 不变，以八卡分担原来每轮的模型计算。四卡阶段每个 optimizer step 共 16 个 microbatch；八卡阶段仍为 16 个。

模型仍为 nominal50 Memory350 encoder2x、64 维 latent、五步 predictor。双手负载各 U(0,2.5) kg，双小腿各 U(0,4) kg；motion 和 replay 均为 uniform。DR 参数距离映射 `2d/(d+0.3)`，SmoothL1 β=0.25。总损失为：

\[
L=L_{\mathrm{teacher,5step}}+0.5L_{\mathrm{recursive,5step}}+0.04L_{\mathrm{DR-center}}.
\]

没有新增中心半径、弱正/弱负、局部正样本或 A−B 响应关系损失。八卡首次启动时沿用总 update 5000 的上限；用户随后要求取消上限，当前目标改为**持续训练，直到用户要求停止**。`--updates 8000` 仅保留原学习率衰减周期；到达后以最低学习率 `1e-5` 继续，不触发停止或自动 plateau 停止。

## 恢复语义

`--resume-new-stage` 从来源目录读取真实配置和 checkpoint，恢复模型、AdamW、scheduler 和固定归一化，在新目录记录来源，允许在全局 batch 相等时重分配 rank batch。`--retune-representation-weight` 仅允许表征系数改变，其他损失、DR 距离 schema 和 archive 配置继续严格检查。

仿真、replay 和 raw history archive 重新预热。八卡会生成自己的独立验证 world 与固定验证 batch，当前阶段的验证 history 重新开始；其在线验证曲线不能与原四卡固定 batch 当成逐样本匹配比较。之前保存的外部跨 motion 聚类诊断集仍可用于严格的 checkpoint 比较。

## 检查记录

- 57 项相关测试通过，包括新阶段的全局 batch 约束、只允许指定损失权重变化、父目录配置不可修改，以及原版/弱样本调参回归检查。
- 最初试用的单 update smoke checkpoint 已处于余弦调度终点，延长调度被现有保护正确拒绝；记录在 `first_smoke_result.json`。正式来源 u1863 尚未达到终点。随后使用实际小规模训练产生的中途 checkpoint 验证八卡恢复。
- 正式启动、实际 loss、八卡配置和恢复检查结果在 run 根目录的 JSON 审计文件中记录。
- 八卡恢复 smoke 已完成：从真实 u1 中途 checkpoint 接续到 u3，全部八卡参数哈希一致，AdamW step 接续至 3，归一化逐项相等，表征权重为 0.04，全局 batch 保持 16，八个 rank 均生成新的验证文件，新的验证 history 只包含 u2/u3。记录在 `resume_smoke_verification.json`。该单 motion 小测试检查恢复链路，跨 motion 中心有效对在正式预热后另外检查。
- 正式八卡任务于 **2026-09-15 11:13:28 UTC** 首次启动，torchrun PID **14496**。该次启动记录已归档在 `unbounded_resume/stage1_process.json` 与 `launch_contract.json`；当前活动进程以 `stage1_process.json` 为准。

W&B：<https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/drcenter8-809311ce5092>。

## 正式启动检查通过

`startup_verification.json` 验证了第一个续训 checkpoint **u1864 / 7456 optimizer step**：

- 八卡模型参数哈希完全一致；全部 AdamW state step 从 7452 接续到 7456；context encoder 和 predictor 均实际更新。
- 模型架构、DR 参数 schema 和归一化与父 checkpoint 逐项相同；余弦调度 `T_max=32000` 保持，u1864 学习率为 `0.0002615767694710516`，与原曲线对应的 step 7456 一致。
- 首轮实际优化每个 microbatch 平均有 962.97 对有效中心，中心关系损失 `0.05059585`，加权后 `0.00202383`，确认系数为 **0.04**。prediction loss `0.31635454`，总损失 `0.31837837`，与设定公式一致。
- 八个 rank 的真实负载均不超过 2.5 / 2.5 / 4 / 4 kg，各卡 4096 nominal / 4096 DR；nominal 负载和物理恢复误差均为零。全量 129827 个 motion 被完整划分到八卡。
- W&B API 已读取到服务端真实优化记录，远端状态为 `running`。新的验证 history 从 u1864 开始。

检查时已运行到 **u1933**。初测 u1864→u1900 平均 **1.6726 秒/update**，包含区间内验证与保存；原四卡稳定阶段为 2.3220 秒/update，轮吞吐约提高 **38.8%**。当前样本区间较短，速度记录在 `initial_speed.json`；对应每 1000 update 约 27.9 分钟。

## 取消训练上限

用户随后明确要求“不设置训练上限”。原进程不支持在线修改停止条件，因此于 2026-09-15 11:30 UTC 向已核验身份的 rank 0 发送 SIGTERM，八卡在 **u2315 / 9260 optimizer step** 协调保存后全部正常退出。该 checkpoint 的八卡参数哈希一致，全部 AdamW 参数的 step 均为 9260。

- 恢复来源：当前目录 `update_002315.pt`，SHA256 `8c72539c47c689659b014f792e7912769014dca09e481c749f33562ce5539f46`。
- 新进程：torchrun PID **44556**，GPU 0–7；日志 `stage1_unbounded_resume.log`。
- 移除 `--stop-after-updates 5000`，同时将 DR-center CLI 的默认上限改为 `None`，避免省略参数后仍隐式停止在 5000 轮。
- 保留 `--until-user-stop`，关闭自动 plateau 停止；保留 `--updates 8000` 的学习率周期与 `1e-5` 续训学习率下限。
- 在同一目录普通恢复，不使用 `--resume-new-stage` 或 `--retune-representation-weight`。模型、损失、0.04 表征权重、全局 batch、DR 范围、8 组固定验证数据和验证历史均保持。仿真、replay、Memory350 历史重新预热。
- AdamW 与 scheduler 从 step 9260 恢复；`T_max=32000`，恢复点学习率 `0.00024216860053809008`，不重置衰减。
- 继续使用相同 W&B run ID `drcenter8-809311ce5092`。

40 项相关测试通过，覆盖 DR-center/nominal 参数约束、无限 update 迭代以及到达余弦终点后的学习率下限。完整命令和前后参数差异在 `unbounded_resume_contract.json`；旧进程配置、停止记录和固定验证文件哈希保存在 `unbounded_resume/`。

恢复后的正式检查通过：已运行至 **u2350 / 9400 optimizer step**，`progress.json` 显示 `unbounded=true`、`stop_after_updates=null`，八个 worker 均在运行。首次恢复更新 u2316 的学习率为 `0.00024212212043849603`，与原余弦曲线 step 9264 完全一致；损失及梯度有限，实际加权中心损失仍等于原始中心损失的 0.04 倍。16 份固定验证文件与归一化文件哈希均保持相同，模型、DR schema 和优化配置一致。

W&B 服务端已确认无上限配置并收到 u2316、u2320、u2330 的新记录。最终证据保存在 `unbounded_resume_verification.json` 与 `unbounded_resume/wandb_verification.json`。
