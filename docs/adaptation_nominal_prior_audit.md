# 原始奖励 nominal 冻结先验核查

2026-09-07 02:24 UTC。仅核查既定 nominal 目标模型，不放行其他旧模型：
`runs/residual_policy_no_latent_nominal_v13_run2/checkpoint_1000.pt`，SHA256
`800da8c40016bba3263e685e9694e51e82b83a5e1c3df3e2e80543bd47ad8f1f`。

保存的 `git/intact-tracking.diff` 记录训练 commit
`692f90bf9dd9900229f8a5d201fb20e1db2c314d`，tracked worktree 无修改。
当时仅有用户的空格目录和 wandb 目录未跟踪。与当前版本比较，整个
`src/intact_tracking/environment` 和 `src/intact_tracking/rollout` 均无差异。
训练入口只调整物理开关、数据/采样配置及训练网络，不修改奖励；后续该入口
的 diff 仅涉及另一种 latent 输入模式及元数据，没有奖励修改。

模型记录 `spv52a_frozen_tracker_residual_v1`、no-latent、nominal physics、
payload disabled、tracker frozen。实际逐 tensor 比较：53 个 tracker 权重与
预处理缓冲均与原始 tracker checkpoint 完全相同。仅额外 nominal 残差头
经过训练，其尺度0.25。当前固定奖励签名仍为
`abee73632d079f08c160584b04ace15532ecf6bd9dfa8139979af518d8a1553c`。

因此可在不继承旧改奖励 teacher 的前提下，把该模型的残差头作为冻结先验，
再训练物理适配结构。新模型必须记录此来源，实际核对 SHA 和全部 tracker
张量；先验及预处理不可更新。此项核查不是对旧模型的一般豁免，训练入口
对其他未审计或改奖励初始化/蒸馏 teacher 的拒绝规则保持不变。

这不意味着新结构已达标，亦不保证 DR 可靠性；同 nominal 条件的初始函数
一致性、原始奖励签名、DR tracking 与失败率仍须分别验证。
