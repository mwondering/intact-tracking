本轮检验：将冻结 tracker 的当前动作显式提供给 residual，能否改善控制效果及 latent 的有效利用。

实验目录：`runs/limb_context_20260913_memory350_compressed_grid256_tracker_action_adaptive_scratch`。

输入的是当前一步冻结 tracker 的确定性原始均值动作，即最终与 residual 相加的同一份 29 维数值。它位于 residual 合成、最终高斯采样、限幅和执行器处理之前。实现复用当前前向已经计算的输出，不额外执行 tracker，也不使用上一时刻的缓存动作。tracker 和 Memory350 encoder 都保持冻结。

两组 Actor 都先执行 `1645→512→256→128` 的观测压缩，再按 `[128维观测特征, 29维tracker动作, 64维latent]` 拼接，进入 `221→256→128→29` 的残差头。baseline 的 latent 槽置零。残差均值仍为 `0.25*tanh(head)`，最终均值仍为 tracker 动作加残差。

Critic 沿用 `6330→1024→512→256→128`，拼接 64 维 latent 后进入 `192→256→128→1`。latent 组的 critic 保留 latent 输入；本轮只改变 Actor 的 tracker 动作输入。

两组 residual Actor、Critic、优化器和归一化从零开始。新增动作输入列初始化为零；观测压缩层、残差头的原有参数保持与旧结构相同的种子初始化。新 baseline 和 latent 的初始可训练权重相同。Critic 各自从首批环境观测建立新的归一化统计，不假定两组统计逐位一致。

训练沿用 grid256：每肢 `{0,1,2,4} kg`，共 256 种组合。每卡 8192 环境，每种组合 32 个副本，每组四卡共 128 个副本。完整静态 DR 参数在副本、四个 rank 和两组之间配对；motion、phase、reset、观测噪声、推力和 Memory350 历史按 world 独立。baseline 用 GPU 0–3，latent 用 GPU 4–7。从第零轮直接 adaptive，保留末端高度终止，没有训练轮数上限。

冻结 encoder 沿用 response10 的第 15000 轮：`runs/limb_context_20260912_memory350_response_window_ablation/response10/stage1_8192/update_015000.pt`。长期记忆在同一 world 的 episode reset 之间保留，进程启动时从空记忆开始；各个 DR 副本不共享记忆。

每 100 轮执行相同的 512 motion 固定评估（四肢全 0 / 全 4 kg，cold / warm500），每 1000 轮执行正确与跨 DR 交换 latent 的闭环评估。动作敏感性诊断中，替换 latent 时保持接收方当前观测和 tracker 动作不变。评估使用独立连续 DR，避免只在训练参数表上比较。

与前一轮未输入 tracker 动作的 grid256 实验按相同更新数、相同物理参数和 motion 起点比较。主要关注 body / joint、root 跟踪、失败率，以及正确 latent 相比交换 latent 的闭环收益；动作敏感度本身不作为有效利用的证明。

入口：`scripts/run_memory350_tracker_action_ppo.py`；训练 / 评估模块分别为 `intact_tracking.cli.memory350_action_policy_train` 和 `intact_tracking.cli.memory350_action_policy_eval`。旧 compressed 入口继续默认不加入 tracker 动作，可加载原 checkpoint。
