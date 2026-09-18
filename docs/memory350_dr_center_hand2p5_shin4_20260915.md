# DR 中心距离监督：双手 2.5 kg / 双小腿 4 kg

用户授权：使用已经实现的 DR 中心关系损失和 predictor 损失训练一个新 predictor；双手负载上限各 2.5 kg，双小腿各 4 kg。

## 训练配置

- 从零初始化 context encoder 和 predictor，seed 717；冻结 tracker 为 `checkpoint_72000.pt`，没有从旧 context checkpoint 恢复权重。
- GPU 4、5、6、7，各 8192 个仿真世界，其中各 128 个为独立验证世界；每张卡严格一半 nominal，一半原 tracker DR + 四肢独立均匀负载。
- 负载范围 `[0,2.5] / [0,2.5] / [0,4] / [0,4]` kg，顺序为左手、右手、左小腿、右小腿。nominal 世界的四肢负载均为零。
- 采样上限在 startup payload 事件之前传入，因此质量、惯量和 COM 均按采样结果正确生成；标签按同一上限归一化。
- 保留原 tracker 的 COM、质量、摩擦、armature、encoder bias 和 force pulse 配置；nominal 世界仍恢复编译时物理并禁用扰动。
- 全量数据目录 `/data_zcy/wxy/motion_data_correct/motion_data_full`，motion 与 replay 都使用 uniform 采样。启动时发现 129,827 个 motion，按 rank 划分为 32,457 / 32,457 / 32,457 / 32,456 个。
- Memory350 encoder2x：short50 + long30×10、64 维 latent、chunk/memory/context 深度 2/4/4；predictor 的监督长度仍为五步。
- 每卡 batch 1024、microbatch 256，每次 update 四次 optimizer step；BF16 前向、FP32 中心距离。
- AdamW 初始学习率 0.0003，余弦跨度 8000 update，学习率下限 0.00001；本轮在 update 5000 停止。
- 每 100 update 进行固定验证，每 250 update 保存 checkpoint。

总损失为：

\[
L=L_{\mathrm{teacher,5step}}+0.5L_{\mathrm{recursive,5step}}+0.02L_{\mathrm{DR-center}}.
\]

DR 中心目标距离使用 $2d_{\mathrm{DR}}/(d_{\mathrm{DR}}+0.3)$，SmoothL1 β=0.25。局部正样本、跨 motion 弱正样本、异 world 弱负样本、A−B 响应关系和中心半径损失全部关闭。详细公式见 `docs/memory350_dr_center_variant_20260915.md`。

## 启动与审计文件

Run 根目录：`runs/limb_context_20260915_dr_center_hand2p5_shin4`。

- `launch_contract.json`：完整启动参数、GPU 分配、W&B run ID 和初始化方式。
- `stage1_process.json`：独立后台进程的 PID、启动时钟和完整命令；torchrun PID 为 51161。
- `source_sha256.json`：启动时所有研究 Python 源文件哈希。
- `released_holders.json`：清理的 GPU 4–7 自动占卡任务。仅清理 `RUN_MJLAB_SOURCE=auto`、`RUN_MJLAB_PROJECT=run` 且绑定相应 GPU 的任务，保留其他任务和全局占卡守护。
- `stage1_8192.log`：训练标准输出和错误输出。
- `stage1_8192/run_config.json`：四个 rank 的实际物理和模型配置。
- `stage1_8192/progress.json`：实际已完成的 update。
- `stage1_8192/metrics.jsonl`：固定验证和训练指标。

W&B：`https://wandb.ai/2486344338-zhejiang-university/intact-forward-predictor/runs/drcenter-b5982437b61b`。

## 启动前验证

52 项相关测试通过，覆盖两种负载上限下的组合惯量、固定种子采样保持一致、原 tracker DR 保留、nominal50 配置和中心损失。旧入口仍默认四肢上限 4 kg，只有此变种的入口默认改为 2.5 / 2.5 / 4 / 4 kg。

32 环境、一次 update 的仿真检查完成，记录在 `smoke_32` 与 `smoke_verification.json`：

| 项目 | 左手 | 右手 | 左小腿 | 右小腿 |
|---|---:|---:|---:|---:|
| 设定上限 / 标签归一化上限（kg） | 2.5 | 2.5 | 4 | 4 |
| 实际采样最大值（kg） | 2.39681 | 2.17004 | 3.99451 | 3.91353 |

nominal 负载最大绝对值为 0。这个单 motion 检查只验证物理和训练链路，不作为聚类效果结论；跨 motion 表征梯度已在上一轮实现验证中单独核实。

## 正式训练启动验证

`startup_verification.json` 已通过全部检查。检查时已完成 58 update / 232 optimizer step，四个 worker 都在运行，启动时记录的 179 个 Python 源文件哈希均未变化。

- 四卡每卡 4096 nominal / 4096 DR；训练集各 4032，验证集各 64。实际负载没有超过 2.5 / 2.5 / 4 / 4 kg；nominal 负载和物理恢复误差均为零，nominal 扰动已禁用。
- 首个 update 的优化 batch 中，平均每个 microbatch 有 40.19 个有效环境中心、803.44 对中心；跨 motion 视图有效率 15.77%。这是实际训练采样的指标，固定验证集视图有效率为 100%。
- 首轮未加权中心关系损失 0.307940，加权后 0.006159；prediction loss 10.218466，总损失 10.224624。总损失与设定公式的误差约为 `2.05e-7`。
- chunk、long、final 三段 context encoder 的梯度范数分别为 0.0980 / 0.1321 / 1.3871，均非零且有限。
- 已通过 W&B API 读取服务端实际训练 history，包含首轮中心对数、中心损失和 prediction loss；远端状态为 `running`，不是只检查本地初始化成功。
- nominal B 的重复仿真数值诊断存在 warning，完整数值保存在启动验证文件；B 在此变种中只用于诊断，不提供表征损失目标。

以上只确认训练配置和更新链路正确，尚未评估新 checkpoint 的聚类改善。

## u1500 聚类检查

2026-09-15 使用 `update_001500.pt`（SHA256 `d9a470aab284d938778a6fdde66417c9fd84e7162c1e0e8ba53ee97769e79b44`）在 GPU 2 评测，训练继续运行。四卡 checkpoint 参数哈希一致，缓存重放与在线输出的单位 latent RMS 差为零。

为双手 2.5 kg / 小腿 4 kg 的新负载范围重新采集 512 个固定 DR world、每个 3200 步、42 个诊断 motion 文件。三个模型处理完全相同的原始轨迹，使用各自保存的归一化；识别按不同 motion family 建中心和查询，并排除 350 步原始历史重叠。新负载组最后可评估 278 个环境中心、2704 个查询；普通 DR 复用原始缓存，409 个中心、3152 个查询。

| 指标 | 新版 u1000 | 新版 u1500 | 旧 response10 u15000 |
|---|---:|---:|---:|
| DR＋负载 Top-1 | 6.80% | 14.09% | 83.54% |
| DR＋负载 Top-5 | 18.42% | 33.28% | 95.08% |
| DR＋负载：同环境跨 motion 距离，历史不重叠 | 0.5119 | 0.4757 | 0.4035 |
| DR＋负载：异环境同 motion、phase 差 ≤0.02 距离 | 0.4753 | 0.6266 | 1.3140 |
| DR＋负载：簇内/簇间 | 1.0770 | 0.7592 | 0.3071 |
| DR＋负载：DR-only 中心/参数距离 Pearson | 0.2485 | 0.4219 | 0.6346 |
| 普通 DR Top-1 | 2.73% | 4.79% | 26.46% |
| 普通 DR Top-5 | 8.57% | 15.55% | 51.65% |

当前聚类仍不足，同环境和异环境的距离分布明显重叠；u1000→u1500 的两组识别率与 DR-only 距离相关性都有改善。旧参考训练了 15000 轮，且训练损失、负载范围等不同，不能由此单次早期检查断定新损失无效。新负载诊断集与历史手部 4 kg 的诊断集不同，旧参考此次的 83.54% 与历史 81.60% 不是同一评测集。

完整记录、真实 64 维距离分布、固定随机 16 个环境的 t-SNE 及交互图：`runs/limb_context_20260915_dr_center_hand2p5_shin4/cluster_check_001500/README.md`。评测代码 `scripts/evaluate_memory350_dr_center.py`；`artifact_verification.json` 验证检查点、原始历史重建、匹配条件、数值距离和旧参考普通 DR 26.46% 的复现。

## 转入八卡续训

随后按用户要求，本四卡阶段于 2026-09-15 11:03 UTC 在 **u1863 / 7452 optimizer step** 正常保存退出。后续从 `update_001863.pt` 恢复到八卡，DR 中心关系权重由 0.02 改为 0.04；每卡训练 batch 1024→512，保持全局 batch 4096，每卡仿真环境仍为 8192。八卡首次启动沿用 5000 update 的上限；用户随后要求取消上限，任务在八卡 u2315 保存后切换为持续训练，直到用户要求停止。

后续运行目录为 `runs/limb_context_20260915_dr_center_8gpu_weight004/stage1_8192`，详细恢复与验证记录见 [八卡续训记录](memory350_dr_center_8gpu_weight004_20260915.md)。
