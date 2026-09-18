# Memory350 DR 软邻域变种，h=0.15

本变种替换 DR 参数中心距离排序项；保留正在运行的 scale=0.6 排序版本用于对照。实现后已按用户新指令，从 u9436 在物理 GPU 4–7 启动独立对照训练。

## 目标与构建

1. 使用现有 DR 物理标签距离：38 维、10 个物理因素分组归一化；推理仍只输入历史，不输入 DR 标签。
2. 仅使用非 nominal、同 world/session、不同 motion、完整且满足既有不重叠条件的 current/archive 配对。
3. 在每卡每个 microbatch 内，按 world/session 分别聚合 current 和 archive 的单位 latent；各自求均值后再单位化，得到两个视图的环境中心。两个池分别构建，同一查询不会被直接包含在自己的对侧中心计算中；沿用现有 archive 的逐配对不重叠保证，不增加跨全部行的全局时间去重。
4. 原始目标相似度：`2 ** (-(DR参数距离 / 0.15) ** 2)`。距离0、h、2h分别对应1、0.5、0.0625。每行归一化成概率。
5. 对侧候选包含同 DR 环境的跨 motion 中心，原始权重为1。不同 DR 即使标签完全相同也有相同目标权重，不人为要求其分开。
6. 预测概率：对跨视图中心的 cosine similarity 除以温度0.1，再做行 softmax。
7. 最小化目标概率到预测概率的 KL，两方向平均、每个环境等权。梯度与软标签交叉熵相同。DR 标签不反传梯度；两个视图的当前 encoder 均可得到梯度；预测器不直接受该项梯度影响。

没有固定 latent 距离或负样本 margin。但软概率与 AB 球面几何仍可能存在折中，不保证任意目标都可精确满足。

## 默认设置

| Loss | 权重 |
| --- | ---: |
| Teacher-forced | 1 |
| Recursive | 0.5 |
| 局部正样本 | 0.01 |
| DR 到 nominal 十步 A/B 距离 | 0.08 |
| nominal 固定方向 | 0.01 |
| 同 DR 跨 motion 弱正样本 | 0.008 |
| **DR 软邻域 KL（替代排序）** | **0.02** |

AB scale=0.6；h=0.15；softmax temperature=0.1。h 表示参数空间相似度半衰距离，temperature控制latent概率分布；两者不是同一个参数。保持0.02便于第一轮对照，但不代表KL项与原排序项的梯度强度相同。

## 入口与续训限制

入口：`python -m intact_tracking.cli.forward_memory_nominal_dr_soft_train`。

沿用原任务的模型、采样、batch和分布式参数，移除 `--dr-center-rank-weight`、`--dr-rank-temperature`、`--dr-rank-min-gap`、`--resume-retune-response-scale`；增加 `--dr-soft-h 0.15 --dr-soft-weight 0.02 --dr-soft-temperature 0.1`。

从当前排序版本切换时必须 `--resume-new-stage` 并使用新的输出目录；AB scale必须已为0.6，其他旧loss配置必须一致。不允许借切换变种偷偷改其他损失。同变种断点恢复要求全部loss配置一致。

配置和checkpoint记录独立 `dr_soft_version`、监督字段和物理标签schema。监控脚本支持新CLI、七项loss求和和schema检查。

## 诊断与验证

记录有效环境数、KL及加权项、目标同DR概率、目标熵、有效邻居数、模型同DR概率。有效邻居数包含同DR候选，使用每行 exp(entropy) 的均值；候选数量变化会影响软目标。这与此前422个环境、排除自身的候选统计不同，不能直接比较数值。

36项测试通过（软变种、原排序、nominal方向、监控）。覆盖已知KL最优解、h半衰定义、双视图梯度、标签无梯度、环境池化、无效/nominal排除、空批次、旧loss不变和恢复限制。另用实际scale=0.6任务配置做了CLI/metadata恢复预检，结果在 `runs/memory350_dr_soft_h015_variant_preflight_20260917/preflight.json`。

现已通过 GPU 首个训练更新核验（u9437，24项检查），尚未做该变种的聚类效果或下游PPO评估；不声明h=0.15最优。

训练记录：[u9436 分支](../runs/limb_context_20260917_memory350_dr_soft_h015_u9436_4x8192/README.md)。

## 2026-09-18：单独调整 nominal 锚点权重

从已有 soft checkpoint 创建新阶段时，可同时指定 `--resume-new-stage --resume-retune-nominal-anchor --nominal-anchor-weight 0.05`。该选项仅放行 `nominal_anchor_weight`，其余损失、soft 监督和固定锚点定义仍必须一致；普通续训仍要求全部损失配置一致。模型、优化器和原固定锚点从源 checkpoint 恢复。
