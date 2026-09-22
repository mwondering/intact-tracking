# `144000-exp-heavy` Training Flowchart

本文档总结 `runs/144000-exp-heavy/` 中重载动力学上下文识别实验的训练流程。实验中冻结预训练的 SPV5-2A tracking policy，仅从零训练 Memory350 context encoder 和 forward dynamics predictor。

## Detailed training flowchart

```mermaid
flowchart TD

    Init["实验初始化<br/>冻结 SPV5-2A tracking policy<br/>加载 checkpoint_144000.pt"]

    Env["构建并行仿真环境<br/>标称环境 + 随机化负载环境 A<br/>独立标称对照环境 B"]

    DR["采样持久物理参数<br/>原始 DR + 四肢附加质量/质心<br/>质量分层：4^4 = 256 个组合"]

    Model["从零初始化可训练模型<br/>Memory350 Context Encoder<br/>Forward Dynamics Transformer"]

    Calib["训练前校准<br/>估计归一化统计量<br/>计算并冻结 nominal latent anchor"]

    Replay["初始化 replay buffer 与交互记忆<br/>短期历史：50 步<br/>长期历史：30 个 10 步片段"]

    Init --> Env
    Env --> DR
    Init --> Model
    DR --> Calib
    Model --> Calib
    Calib --> Replay

    Rollout["在环境 A 中执行冻结 tracking policy<br/>uniform motion sampling"]

    Obs["读取带噪本体感知<br/>joint position/velocity<br/>projected gravity<br/>angular velocity<br/>last action<br/>joint torque"]

    Action["读取 29 维 raw action<br/>记录实际 PD target"]

    Interaction["构造交互 token<br/>x_t = [o_t, a_t, o_{t+1}]<br/>122 + 29 + 122 = 273 维"]

    Rollout --> Obs
    Rollout --> Action
    Obs --> Interaction
    Action --> Interaction

    Memory["更新 Memory350<br/>短期历史：最近 50 个交互<br/>长期历史：30 个不重叠 10 步 chunk<br/>reset/motion switch 边界不跨越"]

    Interaction --> Memory
    Memory --> Encoder

    Encoder["Hierarchical Context Encoder<br/>10 步 chunk Transformer<br/>30 个 chunk memory Transformer<br/>融合 50 步 short history"]

    Latent["输出动力学上下文<br/>z_t ∈ R^64"]

    Encoder --> Latent

    PredictorInput["构造预测器输入<br/>privileged physical state: 71D<br/>foot features: 8D<br/>contact force: 6D<br/>contact binary: 2D<br/>actual PD target: 29D"]

    Rollout --> PredictorInput
    Latent --> Predictor

    Predictor["Context-conditioned Forward Predictor<br/>6-layer causal Transformer<br/>hidden width = 512"]

    PredictorInput --> Predictor

    TF["Teacher-forced prediction<br/>每一步使用真实状态<br/>预测未来 5 个控制步"]

    REC["Recursive prediction<br/>后续状态使用模型预测值<br/>评估多步误差累积"]

    Predictor --> TF
    Predictor --> REC

    Outputs["预测输出<br/>state delta: 70D<br/>foot features: 8D<br/>contact force: 6D<br/>contact logits: 2D"]

    TF --> Outputs
    REC --> Outputs

    Nominal["构造标称响应监督<br/>将环境 B 恢复到环境 A 的起始状态<br/>重放 A 中相同的实际 PD targets<br/>独立运行 10 个控制步"]

    Action --> Nominal
    Nominal --> Response["计算 A-B response label<br/>70D physical state delta<br/>计算 10 步 response RMS"]

    Latent --> Loss

    Loss["联合损失计算"]

    PredLoss["动力学预测损失<br/>L_TF + 0.5 L_REC<br/>state / foot / force / contact"]

    Local["局部正样本一致性<br/>同一环境相邻历史窗口"]

    Cross["跨 motion 一致性<br/>同一物理环境，不同 motion"]

    Anchor["Nominal anchor loss<br/>nominal latent 靠近固定锚点"]

    ResponseLoss["响应距离约束<br/>latent-to-anchor distance<br/>对齐 A-B response magnitude"]

    Soft["DR soft-neighborhood loss<br/>latent neighborhood 对齐<br/>108D physical parameter neighborhood"]

    Outputs --> PredLoss
    TF --> PredLoss
    REC --> PredLoss
    Response --> ResponseLoss
    Memory --> Local
    Memory --> Cross
    Latent --> Local
    Latent --> Cross
    Latent --> Anchor
    Latent --> ResponseLoss
    Latent --> Soft
    DR --> Soft

    Total["总损失<br/>L = L_pred<br/>+ 0.01 L_local<br/>+ 0.008 L_cross<br/>+ 0.08 L_anchor<br/>+ 0.1 L_response<br/>+ 0.02 L_soft"]

    PredLoss --> Total
    Local --> Total
    Cross --> Total
    Anchor --> Total
    ResponseLoss --> Total
    Soft --> Total

    Optim["反向传播与参数更新<br/>AdamW<br/>batch = 1024 / GPU<br/>microbatch = 256<br/>BF16"]

    Total --> Optim
    Optim --> Replay

    Validate["每 100 次 update 验证<br/>固定 held-out worlds<br/>incomplete/broad contexts<br/>nominal 与 DR 分开评估"]

    Replay --> Validate
    Validate --> Metric["记录指标<br/>DR five-step NMSE<br/>nominal five-step NMSE<br/>latent shuffle ratio<br/>response/geometry diagnostics"]

    Metric --> Select["选择最优 checkpoint<br/>minimum held-out DR five-step NMSE"]

    Select --> Stop{"是否停止？"}

    Stop -- "否" --> Rollout
    Stop -- "是" --> Final["保存最终模型与训练记录<br/>best.pt<br/>final update checkpoint<br/>metrics.jsonl"]
```

## Compact paper-style flowchart

```mermaid
flowchart LR

    A["Frozen SPV5-2A<br/>Tracking Policy"]
    B["Randomized Heavy-load<br/>Simulation Environment"]
    C["Noisy Proprioception<br/>+ Control History"]
    D["Memory350<br/>Context Encoder"]
    E["64-D Dynamics Latent z"]
    F["Forward Dynamics<br/>Predictor"]
    G["5-step State Prediction"]
    H["Nominal Counterfactual<br/>10-step Rollout"]
    I["A-B Response Supervision"]
    J["Representation Losses"]
    K["Joint Optimization"]
    L["Held-out Validation<br/>and Checkpoint Selection"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G

    B --> H
    H --> I
    E --> J
    I --> J
    G --> J

    J --> K
    K --> B
    K --> L
    L --> K
```

## Figure caption

> **Training pipeline of the heavy-load context identification experiment.** The pretrained SPV5-2A tracking policy is frozen during training. Interaction histories collected from randomized heavy-load environments are encoded by a hierarchical Memory350 encoder into a 64-dimensional dynamics latent. The latent conditions a forward dynamics predictor trained with teacher-forced and recursive five-step prediction losses. In parallel, an independent nominal simulator replays the same joint targets to obtain a ten-step randomized-to-nominal response signal, which supervises the geometry of the latent space. Local temporal consistency, cross-motion consistency, nominal anchoring, response-distance alignment, and soft physical-neighborhood losses are jointly optimized. Model selection is performed using held-out five-step DR prediction NMSE.

## Main implementation facts

- The tracking policy is frozen; only the context encoder and forward predictor are optimized.
- The encoder input is noisy proprioception and control history, not privileged physical parameters.
- The encoder uses 50 short-history steps and 30 non-overlapping 10-step long-history chunks.
- The context latent has dimension 64.
- The forward predictor forecasts five control steps.
- The randomized-to-nominal response target uses a separate ten-step counterfactual rollout.
- Training uses 8 GPUs, 16,384 worlds per GPU, batch size 1,024 per GPU, microbatch size 256, BF16, and AdamW.
- Validation is performed every 100 updates, and the best checkpoint is selected by held-out DR five-step NMSE.
- In the recorded run, training stopped after 8,816 updates and 35,264 optimizer steps.
