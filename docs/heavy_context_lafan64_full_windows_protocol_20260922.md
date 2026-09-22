# Heavy context encoder：64 个环境遍历完整 LAFAN 的窗口级评测

状态：实验口径与本地数据清单已核对，尚未采集本协议的全量交互或拟合全量 t-SNE。此前八 motion 的图属于小规模探针，不能作为本协议的结果。

## 一个点的定义

一个点对应一个实际交互历史窗口经冻结 encoder 得到的 64 维 latent：`z[policy, environment, motion, query_step]`。不先对 motion、时间或环境求均值。

- 固定 64 组物理参数；每个环境都从头至尾遍历同一个完整 motion 清单。
- 冻结 tracker 和 residual latent policy 各执行一遍，使用相同的 64 组物理环境与 motion 清单。
- 沿用 encoder 的原生历史定义：50 条短期 interaction，加 30 × 10 条长期 interaction。采集步长为 1，每个控制步生成的 latent 都保存。
- 新 motion 从空的短期、长期历史开始，不让上一条 motion 的历史混入当前 motion 的样本。
- 每个样本记录 policy、环境 ID、motion 文件、参考帧、控制步、episode ID、短期/长期有效数和失败/重置标志。`motion identity` 只是来源标签，同一 motion 会贡献许多点。

## 数据覆盖

本地数据目录：`/data_zcy/wxy/motion_data_correct/motion_data_full/AMASS_LAFAN_Qingtong/lafan_qingtong`。

按仓库既有 `lafan_files` 规则，40 条原始命名的 motion 共 441,120 帧，全部为 50 fps；清单不依赖 latent 或交互成败。另有两份衍生文件：`fight1_subject3_612_680.motion.npz` 是短裁剪，`singlejump/jumps1_subject1.motion.npz` 的所有数组与原目录同名文件完全一致，避免将其再次加权。

在每条 motion 内仅执行相邻帧间的 interaction 时，两种 policy × 64 环境共计划 56,458,240 条 interaction。若每条 motion 的历史从空开始，且没有失败/重置/数据损坏，完整 350-interaction 历史窗口的理论上界为 54,671,360 个；实际数量必须由有效掩码和覆盖审计确定。

[逐文件清单、内容哈希和数量计算](../runs/144000-exp-heavy/lafan64_full_windows_protocol_20260922/protocol.json)；[motion manifest](../runs/144000-exp-heavy/lafan64_full_windows_protocol_20260922/motions.txt)。

## 交互与失败口径

- 环境身份指固定物理参数，不是某个并行仿真 slot。物理参数、控制器 nuisance 参数在不同 motion 和 policy 间保持一致，并记录实际值及哈希。
- 64 个环境从覆盖完整 256 组 payload 范围的物理池中预先选择；不能直接使用现有采样器的前 64 个 group，否则一个肢体的负载范围会被截窄。
- 参考时间从起点推进到终点。失败重置必须记录，不能总回到 motion 起点、遗漏后半段，却宣称遍历了整条 motion。
- 保存冷启动、完整历史和重置附近的所有 latent，分别标记有效性；主图若只展示完整历史，应同时报告未满历史样本与失败样本的比例。
- 原生 memory 在同一 motion 的 episode reset 后可以保留长期历史；这类样本需要与无 reset 的连续轨迹样本区分。不得把跨 reset 的 native history 描述为 350 个连续控制步。
- 按 environment × motion × policy 核对参考帧覆盖与窗口数量，不能只保留成功的 motion 或容易形成簇的时段。

## t-SNE 与结论

全量 latent 集合是评测数据基础。每个展示点对应集合中的一个实际 latent；颜色表示环境，两个 policy 使用同一套降维坐标后分面展示。可对同一投影另外按 motion 着色，检查运动身份是否主导分组。

步长 1 会产生约五千五百万个完整窗口，不能把八 motion 的点扩增、复制或插值来代替。如果计算限制需要分层抽样出图，必须保留全量采集数据，明确标注为“完整数据集的分层抽样 t-SNE”，公布采样规则和数量；不能称其为“每个窗口都展示的全量 t-SNE”。同样，子集拟合后的外推投影不能冒充对全量样本联合拟合的结果。

相邻窗口共享大部分交互，适合保留在可视化中，但不能作为独立统计样本。环境识别和检索应按完整 motion 分组，query/gallery 使用不同 motion，避免把相邻窗口分到两侧。统计同时给出按 environment × motion 等权的结果，防止长 motion 占据主要权重。

该实验直接检验“同一环境跨完整 LAFAN 的多种 motion 是否保持一致表征”。t-SNE 是可视化证据；还需原始 64 维中的跨 motion、跨 policy 检索或环境解码结果，才能支持相应范围内的表征迁移结论。
