# 四专家与混合 baseline：五个独立 PPO

实现入口：[fixed_five_ppo_train.py](../src/intact_tracking/cli/fixed_five_ppo_train.py)；调度与分组：[fixed_five_ppo.py](../src/intact_tracking/fixed_five_ppo.py)。支持单 GPU 与 torchrun 多 GPU mjwarp；2026-09-17 按用户要求补充四卡训练，启动记录见[完整数据训练](fixed_five_ppo_full_training_20260917.md)。

## 行为

- 保留各固定 DR 标量独立的 50% nominal / 50% 原范围均匀采样，保留当前力脉冲和观测噪声。
- response10/u15000 encoder 与 tracker 均冻结。每个专家环境由 tracker 预热 100 秒，每 2 秒查询一次，仅使用 short50/long30 完整历史；归一化每次 latent 后取环境均值，按该中心到固定 nominal 中心的距离分四档。
- expert ID 全程固定，motion/episode reset、PPO 更新和恢复训练均不重新分类。训练阶段不再运行 encoder。
- 四个专家与 baseline 的 actor/critic 均不输入 latent 或 expert ID。每个 PPO 独立保存模型、Gaussian std、optimizer、buffer、GAE、优势归一化与 critic 观测归一化。
- baseline 在所有专家环境的物理参数副本上独立行动。它跨四类学习，但不会将专家的轨迹当作自身 PPO 数据。四专家合计与 baseline 每轮交互预算相同。
- 每个网络使用既有压缩 residual MLP，初始 residual 为零、动作 std=0.25；actor/critic 学习率默认 1e-4/5e-4，固定学习率。rollout 默认 24 步、5 epochs、4 minibatches。critic 的最终 bootstrap 使用当前 next-state 的 tracker action。
- checkpoint 恢复五套模型、optimizer、累计步数与固定标签，严格核查固定物理参数。恢复时仿真 episode 重新开始，非逐物理步精确续接。
- 预热分类后或恢复 checkpoint 时，将初始 episode 计时均匀随机化到 `[0, max_episode_length)`，打散集中超时。专家与对应 baseline 副本使用相同计时；各 rank 使用独立种子和专用 RNG。后续 reset 仍从零计时。
- 每个 PPO 分别记录正常超时 `timeouts` 和失败 `failure_terminations`；同时满足失败与超时时归入失败，两项之和等于 `episode_endings`。PPO 的超时 bootstrap 规则保持原设置。

`--num-envs N` 表示每卡、每个对照分支 N 个物理配置，总仿真环境数为 2N×卡数；不是每专家 N 个。类别不强制等频；单卡空类别跳过更新，多卡要求各 rank 均有四类样本，否则在开始 PPO 前明确报错。

多卡时仍是全局五套 PPO，每套聚合各 rank 自己的样本。每专家梯度按各 rank 环境数加权，GAE 保持本地完整轨迹并跨卡汇总优势均值/方差，critic 归一化仅在同一专家的各 rank 间同步。每次 checkpoint 以及第一轮更新后检查五套 actor、critic 和 critic 归一化的跨卡 hash 一致。每 rank 独立保存固定分类、物理参数指纹与 checkpoint；根目录保存全局日志和 checkpoint 清单。恢复可传整个训练目录或某个 rank 的 checkpoint，卡数必须保持不变。

## 已完成验证

GPU 4 上使用 128 个 DR 配置、128 个 baseline 配对副本与 32 条 motion。完成每环境 100 秒预热、两轮更新，再从 checkpoint 恢复完成第三轮；每轮 rollout 8 步、1 epoch、2 minibatches。预热后每环境取得 40–47 个完整 latent 查询。

| PPO | 环境数 | 三轮累计 transition | optimizer 步数 |
|---|---:|---:|---:|
| C0 专家 | 21 | 504 | 6 |
| C1 专家 | 15 | 360 | 6 |
| C2 专家 | 55 | 1,320 | 6 |
| C3 专家 | 37 | 888 | 6 |
| 混合 baseline | 128 | 3,072 | 6 |

五套 actor/critic 参数都发生更新，冻结 tracker 的权重及归一化 buffer 全部不变。配对物理参数完全相同；所有 checkpoint 中分类标签和物理参数指纹一致；恢复未重新预热。验证文件：[verification.json](../runs/limb_context_20260917_fixed_five_ppo_preflight/verification.json)；状态：[state.json](../runs/limb_context_20260917_fixed_five_ppo_preflight/state.json)。

单元测试覆盖独立参数/optimizer/归一化、baseline 轨迹归属、更新一个 PPO 不影响其他 PPO、五套 checkpoint 恢复、空类别和 next-state bootstrap。命令：

```sh
.venv/bin/pytest -q tests/test_fixed_five_ppo.py
```

复现预检可用以下命令，output-dir 必须为新目录：

```sh
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl \
.venv/bin/python -u -m intact_tracking.cli.fixed_five_ppo_train \
  --output-dir runs/fixed_five_ppo_new_preflight \
  --motion-path runs/limb_context_20260917_fixed_five_ppo_preflight_inputs \
  --num-envs 128 --iterations 2 --rollout-steps 8 --epochs 1 \
  --mini-batches 2 --save-interval 1
```

恢复使用相同配置，加 `--resume <output-dir>/checkpoint_final.pt` 并将 `--iterations` 改为目标累计轮数。配置、日志、分组、物理参数与五模型 checkpoint 均在 output-dir 中。

本次预检仅验证实现与恢复，不能据此认定专家优于 baseline。完整 motion 数据集已启动四卡训练；留出环境/动作上的收益评测与蒸馏尚未开展。首次预检运行中补充了冻结 tracker/零 residual 的运行时断言，其初始状态已另用 checkpoint 检查；该次源文件 hash 的采集时点说明见 verification.json。后续入口在启动时记录源文件 hash。多卡补充测试覆盖不同 rank 类别数不等时的加权梯度和更新后参数一致性。计时修复后共 11 项测试通过，新增多周期超时分散、配对计时与 RNG 独立性、超时/失败互斥计数检查；实际四卡恢复验证见[完整数据训练记录](fixed_five_ppo_full_training_20260917.md)。

默认占卡训练及其 gpu-auto-hold 启动器已停止；原有 GPU 0–3 encoder 训练保留，GPU 4–7 用于本次五 PPO 任务。
