# LaFAN A/B/C：1000 轮 fine-tune 结果

本轮支持“DR 单策略在两个端点上存在相对专用策略的提升空间”。这是固定
1000 轮预算、一个训练种子下的初步结论，不是单策略能力上限的证明，也尚未
验证 preview 或 latent 能否消除差距。

## 设置

- 同一预训练 tracker，actor 和 critic 的全部初始权重、归一化统计逐 tensor
  检查与源 checkpoint 完全一致。不是 residual policy，critic 不从零训练。
- 训练原 actor 的控制 MLP 和探索 std；原估计器、reference encoder、actor
  归一化统计冻结。critic 使用原 6330 → 1024 → 512 → 512 → 1 结构并全网训练，
  DecayVecNorm 在两卡间同步。无新增观测、preview、latent 或奖励改动。
- 每组 4096 环境（2 × 2048），1000 次 PPO 更新，98,304,000 条真实 transition；
  训练 seed 121。共同 actor LR 1e-5、critic LR 5e-4，无 critic 预热。
- 40 条原始命名 LaFAN motion，每张卡均可采样全部 motion。
- A：nominal，无额外负载。B：每手、每小腿各 4 kg。
- C：25% 世界四肢均为 0 kg，25% 均为 4 kg，50% 各肢独立 U(0,4) kg；
  每世界负载固定。其他动力学随机化关闭，原观测噪声保留。
- 每策略、每测试端点 960 段：40 motion × 8 起点 × 3 评测种子
  （20001、20002、20003），每段最多 500 控制步 / 10 秒。
- 测试 motion、起始帧、每步随机种子、实际物理参数指纹配对；只评测最终
  1000 轮 checkpoint，不按结果挑选 checkpoint。

详细定义及启动命令见 [实验协议](lafan_abc_finetune.md)。本轮是共享预训练
checkpoint 后的 fine-tune 对照，不是从头进行 nominal/DR 预训练的对照。

## 完整评测

Body 为原评测定义下的 body 平均位置误差（m），Joint 为全部关节位置误差的
L2 范数（rad），不是单关节平均角误差。先在每段内平均，再等权平均各段。

| 测试环境 | 策略 | Body | Joint L2 | 失败 / 960 | 平均时长覆盖率 |
| --- | --- | ---: | ---: | ---: | ---: |
| nominal | 冻结 tracker | 0.03070 | 0.48454 | 1 | 99.95% |
| nominal | A | 0.02964 | 0.45430 | 3 | 99.84% |
| nominal | B | 0.06399 | 0.78660 | 9 | 99.72% |
| nominal | C | 0.04471 | 0.74512 | 3 | 99.85% |
| 最大负载 | 冻结 tracker | 0.11198 | 1.23396 | 263 | 84.80% |
| 最大负载 | A | 0.12512 | 1.28679 | 411 | 74.80% |
| 最大负载 | B | 0.05625 | 0.89544 | 40 | 97.56% |
| 最大负载 | C | 0.05993 | 1.00013 | 73 | 95.75% |

## 预定的两项主要比较

| 比较 | C 的 Body 增幅 [95% CI] | C 的 Joint 增幅 [95% CI] | 失败变化 |
| --- | ---: | ---: | ---: |
| nominal：C 相对 A | +50.86% [43.62%, 58.41%] | +64.01% [58.72%, 69.65%] | 3 → 3 |
| 最大负载：C 相对 B | +6.53% [3.69%, 9.54%] | +11.69% [9.34%, 14.15%] | 40 → 73 |

四项误差差异的区间均排除零。区间来自按 motion 配对 bootstrap：先平均三个
评测种子，再对 40 条 motion 重采样 5000 次。它描述本次评测的不确定性，
**不涵盖不同训练种子造成的波动**。

失败总数相同不代表失败片段相同：nominal 下 C 相对 A 新增失败 2 段、救回
2 段；最大负载下 C 相对 B 新增 45 段、救回 12 段。最大负载的失败率净增
3.44 个百分点，motion bootstrap 95% CI 为 [1.56, 5.42] 个百分点。

C 并非没有学会适应负载：相对冻结 tracker，最大负载下 Body/Joint 分别降低
46.49%/18.95%，失败 263 → 73；但 nominal 下相应误差升高 45.67%/53.78%。
因此，当前直接 fine-tune 的 DR 策略仍存在明显的端点折中，尤其损失了
nominal 精度。这个结果可以作为下一步“增加 preview/特权环境信息”的基线。

限制：相同总训练预算下，C 对每个精确端点的训练样本少于专用策略；另外，
本次为训练集合内评测。这不是“任何训练预算下单策略都不可能达到两端专家”
或“preview/latent 必然有效”的证据。失败截断轨迹也计入误差平均，所以必须
同时看误差、失败和覆盖率，不能仅凭误差宣布全面更优。

## 审计、日志和复现

- [完整数值与配对区间](../runs/abc_lafan_20260907_r2/comparison.json)
- [逐段评测数据](../runs/abc_lafan_20260907_r2/eval/)
- [独立训练核验](../runs/abc_lafan_20260907_r2/independent_training_audit.json)
- [独立评测核验](../runs/abc_lafan_20260907_r2/independent_evaluation_audit.json)
- [训练与评测入口](../scripts/run_lafan_abc.py)
- [W&B A](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/pv2-3f446d0a72e3)、
  [W&B B](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/pv2-fc409d6ddf79)、
  [W&B C](https://wandb.ai/2486344338-zhejiang-university/intact-preview-v2/runs/pv2-d990edd76495)

三组最终两卡网络状态逐字节一致；actor 冻结部分未改变；actor 和 critic 的
MLP 都实际发生了参数更新。24 份评测均核对了 checkpoint 哈希、参考时间线
及部分 reset 时幸存世界的状态/历史，确认没有额外 oracle、preview 或动作过滤。
相关 39 项回归测试通过。

早先 `runs/abc_lafan_20260907` 的中断工程运行无效，不参与上述结果；另一个
1e-4 actor LR 的 100 轮诊断出现明显退化，因此本轮三组在正式启动前统一改用
1e-5。三组正式训练过程中未修改学习设置。C 起初共享 GPU 2/3，用户要求清理
竞争进程后变为独占，C 本身没有重启或丢失进度，资源事件已记录。
