# Response10 latent：按距 nominal 的远近分成 16 / 64 档

本次按用户指定的一维距离进行分类：单位归一化旧 `response10/u15000` 的 encoder 输出，以独立 nominal 历史的平均 latent 为固定参考中心，计算 `r = ||unit(z) - nominal_center||₂`，在 `[0, 1.915034]` 等宽分成 16 或 64 档。编号越大越远。真实 DR 参数只用于评估，不参与分类。

Checkpoint SHA256：`db5d7bd7df026db163b9eedf588d37dcc81b55551d28afe3744f67a8ab2642ac`。

复用此前 16,384 个 DR 环境，每环境恰好 16 个充分历史窗口，在 GPU 6 上以 FP32 重新编码。另有 2,048 个 nominal 环境，随机分成 1,024 个用于估计参考中心、1,024 个用于独立检查。没有训练或重新仿真，没有修改当前任务。原始历史文件逐一核对 SHA256；物理参数、窗口和 world ID 与此前原始参数距离分档一致。

主要发现：

- 16 档中，独立 nominal 窗口的 99.44% 进入 C0。64 档 C0 更窄，覆盖 91.09%；64 档的 C0–C3 合并后与 16 档 C0 完全一致。
- 16 档中 88.52% 的随机 DR 窗口集中于 C12–C15，单独 C13 占 42.29%；当前等宽划分不提供均衡的专家样本量。
- C10–C15 的平均四肢总负载依次为 4.24 / 4.90 / 5.83 / 6.96 / 8.05 / 8.87 kg。
- 复用两组互不重复的共 4,096 个 DR 环境、分别 96 / 64 个 motion 起点的匹配状态和动作仿真，以旧 checkpoint 的 `delta_std` 重算原始 H10 标签的距离。C10–C15 的平均动态距离依次为 1.002 / 1.184 / 1.387 / 1.542 / 1.637 / 1.670。主要 DR 分布区域存在明确的平均影响梯度，但各档的动态距离分布仍重叠。
- 按环境平均 latent 距离与独立测试动态距离的 Spearman 相关，在两组分别为 0.679 / 0.664；统计只包含随机 DR，没有加入 nominal 来放大相关。
- 16 档 C0 同时收到 1,032 个 DR 窗口，占全部 DR 窗口 0.394%，来自 571 个不同环境，其中最大总负载 10.59 kg。近 nominal 的一次 latent 估计不能等同于原始参数都小，或该环境在所有 motion 上影响都小。此测试没有逐窗口对应的未来反事实响应，不能据此将这些窗口全部判定为错误。

动态测试用的是独立 probe motion 的跨 motion 平均影响，不是每个 latent 历史之后的即时未来响应。同一环境的 16 个窗口独立分类，可能进入多个档；环境数不能跨档相加。没有测试专家训练收益。

[完整报告及每档分布](../runs/limb_context_20260917_response10_radial_bins/README.md) · [16 档 CSV](../runs/limb_context_20260917_response10_radial_bins/bins_16.csv) · [64 档 CSV](../runs/limb_context_20260917_response10_radial_bins/bins_64.csv) · [独立校验](../runs/limb_context_20260917_response10_radial_bins/verification.json)

![分档占用和实测动态距离](../runs/limb_context_20260917_response10_radial_bins/radial_bins.png)

可复用的 nominal 中心、分档边界、world ID 和每个窗口的类号保存在 `runs/limb_context_20260917_response10_radial_bins/radial_router_and_assignments.npz`。给新历史计算 latent 后，采用相同单位归一化、参考中心和边界即可分档。
