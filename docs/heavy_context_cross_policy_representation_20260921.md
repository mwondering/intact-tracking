# Heavy context encoder：跨 policy 表征实验

本实验检验同一个冻结 encoder 在冻结 tracker 与 residual latent policy 产生的交互历史上，是否保留可迁移的环境信息。encoder 为 `144000-exp-heavy/stage1_proprio122_16384/update_008816.pt`；residual 为 uniform 续训分支 `checkpoint_2800.pt`，内部 `completed_updates=2801`。

用于展示“相同环境、不同 motion 聚在一起”的主图已更新为[八 motion 环境聚集实验](heavy_context_cross_motion_clusters_20260921.md)：环境用颜色区分，motion 用点形区分，两个 policy 分面展示。

## 实验设计

- 512 个新采样的纯 HDR 世界，seed=20260923，256 个四肢负载质量组合各 2 个世界；完整 108 维持续物理参数在 rollout 中保持不变。双手质量各 0–2.5 kg、双小腿各 0–4 kg，负载 COM 各轴 ±5 cm，保留原 tracker DR。
- 256 条已有训练目录 motion。每个世界分别使用两个不同 motion ID，第二视图把 motion 索引循环平移 128。每个视图内固定 motion ID，重置时重复该 motion；起点预留最多 350 步。不同 ID 不保证不同 motion family，也不代表未见过的 motion。
- 四个独立 rollout：tracker / motion A、tracker / motion B、residual / motion A、residual / motion B；每组 1000 个控制步。每组从空短记忆与空长记忆开始，所有历史均由该组自身 policy 产生。没有共同 tracker 预热，没有将 tracker latent 喂给 residual 作为实验历史。
- 四组逐世界的物理与初始控制器参数完全配对，同 motion 视图的初始 qpos/qvel 完全配对。为隔离 policy 改变，delay、smoothing alpha、joint offset 固定为各世界初始抽样值；episode reset 不重抽这些控制器量。这是受控表征实验，与训练中 reset 时重抽控制器量的协议有区别。
- 外力保留原脉冲幅值、持续时间和间隔范围，使用独立随机数流；失败 reset 不改变外力时钟。每步重新施加当前脉冲，避免模拟器 reset 清掉仍应作用的外力。reset 的随机数消费也与观测噪声隔离。配对检查要求同一视图的整个 1000 步外力序列 SHA-256 完全一致。
- 从第 350 步起每 50 步保存 latent、历史完整性、motion、历史统计、历史内容哈希和失败计数。主分析为从第 500 步起四组同时拥有完整 50+300 步历史的首个共同采样时刻；每个世界/策略/视图只用一个窗口，不平均多个窗口来提高检索。未满足四组完整历史条件的世界单独列出。
- 使用同一 encoder 权重和归一化。几何指标在原始 64 维单位 latent 上计算；原始未单位化 latent 也独立做 policy 分类与物理参数解码。cached/direct encoder 输出及 actor、encoder 参数不变性均检查。

## 如何解释各项检查

1. **跨 motion、跨 policy 环境检索**：query 与 gallery 的 motion ID 不同，比较 tracker→tracker、tracker→residual、residual→tracker、residual→residual。每个 gallery 世界只有一个窗口。Top-1/Top-5 衡量从已有环境示例中检索同一物理世界，不能称为“对任意全新环境类别的分类准确率”。
2. **仅 latent 的物理参数线性 probe**：五折按世界划分，训练只用 source policy 的 motion A，测试 target policy 的 motion B，测试世界的所有视图均排除在拟合之外。StandardScaler 与逐目标 RidgeCV 都仅在训练世界拟合；输出各物理因素的 R²、物理单位 MAE。没有读取 residual actor 的辅助预测头、隐藏层或当前观测来替代 latent probe。
3. **policy 泄漏检查**：类别平衡的线性 logistic probe，在训练世界的 motion A 上学习区分两个 policy，在测试世界的 motion B 上评估。可识别 policy 是反对严格统计不变性的证据；接近随机也不能证明任意 policy 下不变。原始 latent、单位 latent、历史均值/标准差对照均评估。
4. **联合可视化**：四组 latent 一起拟合 t-SNE，分别按 policy、环境及物理参数着色；没有分别拟合再旋转对齐。额外保存另一 perplexity/seed 的 t-SNE 和 PCA。t-SNE 仅作展示，环境检索、距离和参数 probe 都不用二维坐标。[t-SNE 原论文](https://www.jmlr.org/beta/papers/v9/vandermaaten08a.html)。

所有置信区间为固定 motion 集、固定 checkpoint 和物理种子下按世界 bootstrap 的描述性区间；不是多训练随机种子的置信区间。相邻时刻的重叠窗口只用于敏感性检查，不当作独立样本。

## 实测结果

**结论：环境信息能够跨 policy 迁移，但 latent 仍包含可线性解码的 policy 信息；本次结果不支持严格的 policy independence。**

512 个世界中，511 个在四组 rollout 中均取得完整历史，主分析覆盖率 99.80%。排除的 world 501 在 tracker / motion A 中没有取得符合条件的完整历史，其余三组各覆盖全部 512 个世界。排除名单和所有失败计数均保留。两种 motion 视图的完整 1000 步外力序列、物理参数和配对初始状态均通过逐项/哈希验证。

### 环境检索

下表固定 motion A 为 query、motion B 为 gallery，每格使用相同的 511 个物理世界；均在原始 64 维单位 latent 中检索。随机 Top-1 为 **0.196%**，随机 Top-5 为 0.978%。

| Query policy → Gallery policy | Top-1 | Top-5 | Top-1 世界 bootstrap 95% 区间 |
|---|---:|---:|---:|
| tracker → tracker | 94.91% | 98.63% | [92.95%, 96.67%] |
| tracker → residual | 94.32% | 99.02% | [92.17%, 96.28%] |
| residual → tracker | 94.52% | 98.63% | [92.37%, 96.29%] |
| residual → residual | 93.93% | 98.83% | [91.78%, 95.89%] |

从 tracker→tracker 的 94.91% 到 tracker→residual 的 94.32%，差异为 0.59 个百分点；不能把数值接近直接当作统计等效性证明。反向 motion B→A 的跨 policy Top-1 为 94.72% / 94.91%，结论相近。固定同 motion、只换 policy 时，各方向 Top-1 为 99.22%–100%。

历史均值/标准差这个简单对照在四个跨 motion 检索方向的 Top-1 均为 0%，Top-5 为 0.98%–1.76%；其检索主要受 motion 影响。这不是所有 raw-history 方法的上限，也没有与可训练 Transformer 基线比较。

同环境、同 motion、跨 policy 的平均 cosine 为 0.9743；同环境跨 motion 的 tracker→tracker 为 0.9268，tracker→residual 为 0.9194。这里可观察到 policy 改变带来的位移，但环境身份仍较稳定。不同采样时刻的原空间检索也保留在 summary 中；这些窗口重叠，不作为额外独立证据。

### 物理参数：只用 latent 的线性 probe

五折测试世界从未用于标准化、超参数选择或 decoder 拟合，并且测试采用另一 motion ID。表格报告原始物理坐标的 R²，多坐标因子为坐标 R² 的平均。负值表示比使用总体均值预测更差。

| 因子 | Tracker→Tracker | Tracker→Residual | Residual→Tracker | Residual→Residual | 历史统计 Tracker→Residual |
|---|---:|---:|---:|---:|---:|
| Torso COM x | 0.934 | 0.945 | 0.931 | 0.943 | 0.638 |
| Torso COM y | 0.927 | 0.921 | 0.919 | 0.915 | 0.677 |
| Torso COM z | 0.788 | 0.686 | 0.728 | 0.671 | -0.018 |
| Torso mass | 0.034 | 0.026 | 0.013 | 0.026 | -0.057 |
| Friction | 0.881 | 0.815 | 0.859 | 0.809 | 0.093 |
| Kp | 0.357 | 0.351 | 0.356 | 0.351 | -0.032 |
| Kd | -0.010 | -0.012 | -0.012 | -0.013 | -0.072 |
| Armature | -0.024 | -0.024 | -0.024 | -0.024 | -0.085 |
| Left hand mass | 0.863 | 0.844 | 0.858 | 0.851 | 0.493 |
| Right hand mass | 0.839 | 0.841 | 0.826 | 0.846 | 0.331 |
| Left shin mass | 0.685 | 0.588 | 0.686 | 0.601 | -0.160 |
| Right shin mass | 0.728 | 0.596 | 0.704 | 0.604 | 0.048 |
| Left hand COM | 0.390 | 0.356 | 0.378 | 0.353 | 0.079 |
| Right hand COM | 0.369 | 0.316 | 0.361 | 0.317 | -0.027 |
| Left shin COM | 0.019 | -0.013 | -0.011 | 0.007 | -0.057 |
| Right shin COM | 0.013 | 0.001 | -0.001 | 0.014 | -0.066 |

躯干 COM x/y、摩擦和双手负载的跨 policy 解码较好；小腿负载质量能够解码，但 tracker→residual 的 R² 从同策略的 0.685/0.728 降至 0.588/0.596。小腿负载 COM、Kd、armature 和 torso mass 的线性结果较弱。这里只能说这些线性 probe 没有充分读出，不能断言 latent 完全不包含相应信息。Torso mass 的 schema 坐标是相对质量量，不是直接以 kg 表示。

### Policy 可辨识度

训练世界 / motion A 上拟合线性分类器，测试世界 / motion B 上识别 tracker 或 residual，类别严格平衡，随机准确率为 50%。

| 输入 | 准确率 | 世界 bootstrap 95% 区间 | ROC-AUC |
|---|---:|---:|---:|
| 单位 latent（64 维） | 63.60% | [61.64%, 65.56%] | 0.704 |
| 原始 latent（64 维） | 63.60% | [61.74%, 65.56%] | 0.704 |
| 历史均值/标准差（546 维） | 90.61% | [88.85%, 92.37%] | 0.967 |

相较简单历史统计，policy 的线性可辨识度降低，但仍高于随机。因此可以写 **“在所测策略上具有良好的跨 policy 环境检索与参数解码迁移能力”**，不能写“表征严格与 policy 无关”。这组比较没有建立信息论意义上的信息量上限，两个 policy 也共享同一个 tracker。

两种 policy 确实产生不同动作：residual 的两个 motion 视图的修正量 RMS 为 0.184/0.187，总动作 RMS 为 1.610/1.620，约为总动作 RMS 的 11.4%/11.6%；冻结 tracker 的修正量为零。所有数值单位为 policy command，不是关节角 rad。

### 图

分别展示两种 policy；所有面板沿用同一次联合 t-SNE 的坐标和相同颜色范围：

![分别观察两种 policy 的 latent](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis/policies_separate.png)

[分 policy 图 PDF](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis/policies_separate.pdf) · [联合 t-SNE PNG](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis/joint_tsne.png) · [联合 t-SNE PDF](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis/joint_tsne.pdf)

![跨策略检索与 policy probe](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis/retrieval_and_policy_probe.png)

[完整物理参数迁移图](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis/physical_parameter_transfer.png) · [物理参数 PDF](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis/physical_parameter_transfer.pdf) · [另一 t-SNE 设置及 PCA](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis/projection_sensitivity.png)

主 t-SNE 的 k=10 trustworthiness 为 0.963；另一设置为 0.960。PCA 前两维总共解释约 17.9% 的方差，因此二维图不能替代 64 维定量分析，也不据此声称环境已形成天然的离散类别。

### 复核与复现

- [全部结果 JSON](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis/summary.json) 与 [选中世界、窗口、fold、decoder 预测和二维坐标](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis/analysis_arrays.npz)。
- [独立复核记录](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/verification.json)：从保存结果用 Euclidean 距离重算全部 12 个检索方向；从预测数组重算 12 组参数 R² 向量及 3 个 policy probe；复核配对外力、初态及源代码快照。
- [指标自检](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/analysis_self_check.json)：已知相同/错配 gallery、已知线性物理标签与显式 policy 标记均给出预期结果。
- [完整采集命令与设备](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/processes.json)、[checkpoint 选择](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/selection.json)、[tracker A 元数据](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/tracker_v0/metadata.json)、[residual A 元数据](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/residual_v0/metadata.json)。
- 采集脚本：[probe_heavy_cross_policy_latents.py](../scripts/probe_heavy_cross_policy_latents.py)；分析脚本：[analyze_heavy_cross_policy_latents.py](../scripts/analyze_heavy_cross_policy_latents.py)。
- 四组原始数组位于 `runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/{tracker,residual}_v{0,1}/latents.npz`；checkpoint、collector 与 analyzer 源文件均在实验目录中留有快照。
- 分 policy 图的复现脚本：[plot_separate_policies.py](../runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/plot_separate_policies.py)。

首次采集因外力随机数流没有配对而被排除，已完整保存在 `attempt_unpaired_force_rng/`，未混入本报告的任何定量结果。受控重采集通过全部验证。训练代码、已有训练配置和模型权重未因本实验而修改。

Encoder SHA-256：`decd72ae598603755d0a7df93250b2925574bd71bae3bc9a6d53eefe3d057931`。Residual snapshot SHA-256：`e1cb660e20e50ff27278701dd2ebaa4ca7aaca60ab5a9d44f5d11b5f4b5166b6`。

采集时为每组指定尚不存在的输出目录；四组必须使用相同 checkpoint、manifest、seed 和环境数，只改变 `--policy tracker|residual`、`--view 0|1`。分析程序要求该根目录内的四组名为 `tracker_v0`、`tracker_v1`、`residual_v0`、`residual_v1`，且 `analysis/` 尚不存在。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/probe_heavy_cross_policy_latents.py \
  --checkpoint runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/residual_checkpoint_2800.pt \
  --manifest runs/144000-exp-heavy/cross_policy_latents_u8816_uniform2800_20260921/motions_256.txt \
  --policy tracker --view 0 --output <fresh_root>/tracker_v0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=.runtime/latent_auto_cluster_deps \
  .venv/bin/python scripts/analyze_heavy_cross_policy_latents.py <fresh_root>
```


## 下一步优先级

- 在新的物理种子和留出的 motion family 上重复同一检索/probe 矩阵，并加入另一训练种子或不同结构的 policy。现有两个 policy 共享冻结 tracker，不能覆盖任意行为分布。
- 做固定 residual checkpoint 的 correct / zero / shuffled latent 闭环消融。交换时配对 motion、历史年龄和完整性，并一起交换 actor 实际使用的五帧 latent；报告失败率、覆盖率和共同有效前缀上的 tracking 误差。这项实验回答“policy 是否利用了环境信息”，与“latent 含有环境信息”互补。
- 使用相同 rollout 历史比较随机初始化 encoder、早期 encoder 和最终 encoder；进一步分开预测损失、cross-motion positive、DR neighborhood、nominal anchor 的作用。当前历史均值/标准差只是一个简单对照。
- 研究动态变化时的识别时间：在受控质量/摩擦切换后画 latent、参数预测和 tracking 的时间曲线；明确是否使用知道切换时刻的 oracle memory reset，避免把先验信息混入适应能力结论。

论文表述应依证据选择“跨 policy 可迁移的动力学表征”或“在所测试策略和环境分布上近似不变”。冻结 encoder、不输入 policy ID，或者二维图混在一起，本身都不等价于 policy independence。
