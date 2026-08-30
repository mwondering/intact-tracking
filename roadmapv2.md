# Learning the World in Context：面向跨域 Humanoid Motion Tracking 的自适应世界模型

## 1. 核心问题

我们假设已经拥有一个经过大规模 motion 数据训练的强 Motion Tracker。该 Tracker 已经掌握丰富的 humanoid motion skills，因此本文不再解决“机器人如何学会运动”的问题，而关注其部署到未知物理环境后的 Domain Adaptation：

> **一个已经掌握丰富运动技能的机器人，能否通过自身过去与当前环境的交互，在 Context 中理解这个世界，并据此调整已有的运动策略？**

传统 Domain Randomization 试图让一个固定策略对不同 dynamics 保持鲁棒；我们的目标则是让机器人主动利用 interaction history 推断当前世界：

> **Interaction → World Understanding → World-Aware Action**

整个 adaptation 过程发生在 Context 中，部署时不需要参数更新、真实 dynamics parameter 或 test-time gradient。

---

# 2. 总体架构

系统由三个模块组成：

1. **Frozen Large Motion Tracker**
2. **In-Context World Model**
3. **RL Action Head**

整体结构：

```text
                         Motion Intent
                        /             \
                       ▼               ▼
             Frozen Large Tracker   In-Context World Model
                       │               │
                       ▼               ▼
                Tracker Action    Reference Action
                       │               │
                       └───────┬───────┘
                               ▼
                         RL Action Head
                               │
                               ▼
                          Final Action
                               │
                               ▼
                             Robot
                               │
                               ▼
                    Interaction Experience
                               │
                               └────→ Context Memory
```

三个模块分别回答：

> **Tracker：通常情况下，这个 motion 应该怎么做？**

> **World Model：根据过去的交互经验，在当前这个世界里，为了实现这个 motion intent，应该怎么做？**

> **Action Head：面对这两个 action proposal，当前真正应该执行什么？**

因此可以把整个方法概括成：

> **Motion → Intent**

> **Interaction → World**

> **Intent × World → Action**

---

# 3. Frozen Large Motion Tracker

Large Tracker 是已经训练完成的 universal motion tracking policy，在整个方法训练过程中保持冻结。

输入：

> Current Robot State + Reference Motion

输出：

> **Tracker Action**

Tracker Action 表示已有 motion prior 所认为的合理动作。

因此 Tracker 负责提供 **Skill Prior**，而不是承担新环境 adaptation。

---

# 4. In-Context World Model

## 4.1 核心思想

World Model 的目标是从机器人在当前环境中积累的历史 interaction 中推断当前 physical world。

Context 包含：

> State → Action → Observed Response → Action → Observed Response → ...

这些 interaction 自然暴露当前环境的：

> actuator strength、friction、payload、mass distribution、latency、contact dynamics 等。

模型不需要显式预测这些 dynamics parameters，只需要从 interaction 中提取足够的信息，使其能够理解：

> **“在当前这个世界里，什么 Action 会实现什么 Physical Change？”**

---

## 4.2 长期 Context Memory

Context 不局限于当前 state 之前最近若干帧，而是来自机器人进入当前 environment 后已经积累的 interaction experience。

长历史首先被切分为较短的 interaction chunks，例如每个 chunk 包含 100–200 ms：

> states + executed actions + observed responses。

每个 chunk 通过 Interaction Encoder 得到一个 Context Token。

当前实现固定使用最近 16 个 Context Tokens。每个 token 覆盖 100 ms 的
state-action-response，因此模型始终接收 1.6 s、无 padding 的 environment evidence。

因此我们希望得到一个核心现象：

> **More Interaction Context → Better World Understanding**

并最终表现为：

> **More Context → Better Prediction / Better Action Inference**

Context 足够长以后自然包含不同 motion phase、contact configuration，甚至不同 motions，因此不需要人为要求 context 来自不同 trajectory。

---

# 5. JEPA Forward World Modeling

World Model 的第一项训练任务是 Action-Conditioned Future Prediction。

给定：

> Historical Interaction Context
>
> * Current State
> * Future Action Chunk

模型预测：

> **Action Chunk 执行结束后的 Future Latent**

例如在 50 Hz control frequency 下，可以使用：

> 100–200 ms Action Chunk，即约 5–10 个 actions。

完整 Action Chunk 必须显式作为输入，而不是让模型自己预测未来 policy action。

因此 JEPA 学习的是：

> **“在我从 Context 推断出的这个世界里，如果施加这一串 Actions，最终会发生什么？”**

而不是：

> “Tracker 接下来可能会怎么运动？”

这使 policy behavior 与 physical dynamics prediction 尽可能解耦。

---

# 6. JEPA 网络结构

最小网络结构如下：

```text
Historical Interaction Memory
          │
          ▼
   Interaction Encoder
          │
          ▼
    World Context Tokens
          │
          ├─────────────────────┐
          │                     │
Current State              Action Chunk
     │                          │
     ▼                          ▼
State Encoder             Action Encoder
     │                          │
     └────────────┬─────────────┘
                  ▼
           JEPA Predictor
                  │
                  ▼
        Predicted Future Latent
                  │
               compare
                  │
                  ▼
          EMA Target Encoder
                  ▲
                  │
           Real Future State
```

推荐第一版采用轻量结构：

> State Encoder：MLP
> Interaction Encoder：Small Temporal Encoder
> Context Aggregator：4-layer Transformer
> Action Encoder：MLP
> JEPA Predictor：4–6 layer Transformer
> Latent Dimension：256–384
> Target Encoder：EMA State Encoder

由于输入是结构化 proprioception 而非视觉数据，模型无需很大。

---

# 7. 从 Physical Intent 反推 Reference Action

单纯 Forward JEPA 存在一个问题：

> 必须先提供 Action，才能得到 Future Prediction。

但最终 controller真正需要的是相反的问题：

> **“我想实现这样的未来变化，在当前这个世界里应该采取什么 Action？”**

因此借鉴 INTACT 的思想，我们利用已有 rollout 自动构造 Intent-to-Action supervision。

一条真实 rollout：

```text
Current State
      │
      │ Action
      ▼
Real Future State
```

同时提供了两种监督。

### Forward

> Current State + Action → Real Future

用于训练 JEPA World Model。

### Inverse

Current State 到 Real Future State 之间的变化定义为：

> **Realized Physical Intent**

而产生这一变化的真实 Action 已经存在于 rollout 中。

因此同一条数据可以反过来训练：

> **World Context + Current State + Realized Physical Intent → Action**

不需要额外 action label 或 RL。

---

# 8. Reference Action

训练完成后，Intent-to-Action 分支可以直接用于推理。

训练时：

> Realized Physical Intent → Executed Action

部署时，将 Realized Intent 替换成：

> **Desired Motion Intent**

Desired Motion Intent 可以由当前 robot state 与 Reference Motion 所定义的 desired future state构造。

于是模型输出：

> **Reference Action**

其含义是：

> **“根据我过去在这个环境中的交互经验，为了实现当前 Motion Intent，我认为应该采取什么 Action？”**

因此：

> Tracker Action = **Skill-Aware Action Proposal**

而：

> Reference Action = **World-Aware Action Proposal**

两者具有不同的信息来源和 inductive bias。

---

# 9. World Model 的联合监督训练

World Model 第一阶段完全使用 rollout data 进行监督 / 自监督训练，不使用 RL。

同一条数据：

> Context + Current State + Action + Future State

同时训练两个方向：

```text
                     Future State
                    /            \
                   /              \
         Forward JEPA          Physical Intent
                │                   │
                │                   ▼
                │             Inverse Model
                │                   │
                ▼                   ▼
         Future Latent        Reference Action
```

即：

> **Action → Future**

和：

> **Future Intent → Action**

共享同一个 World Context representation。

Forward objective 迫使 Context 表达当前环境如何响应 action；

Inverse objective 迫使 representation 保留与控制相关的 action information。

---

# 10. RL Action Head

完成 World Model 训练后：

> Large Tracker：Frozen

> In-Context World Model：Frozen

只训练最终的：

> **Action Head**

其核心输入为：

> Current State
>
> * Motion Intent
> * Tracker Action
> * Reference Action

也可以进一步加入 World Context Token。

输出：

> **Final Action**

Action Head 使用原有 Motion Tracking Reward 在 randomized simulation environments 中通过 RL 训练。

因此它不需要重新学习完整 motion tracking，而主要学习：

> **如何融合 Skill Prior 与 World-Aware Action Proposal。**

整个信息流可以概括为：

```text
Large Tracker
     │
     ▼
Skill Prior ──────────┐
                      │
                      ▼
                  Action Head ───→ Final Action
                      ▲
                      │
World Context         │
     │                │
     ▼                │
Intent-to-Action ─────┘
     │
     ▼
World-Aware Prior
```

---

# 11. 两阶段训练

整个系统采用非常清晰的两阶段训练流程。

## Stage I：World Learning

冻结 Large Tracker，通过不同 randomized dynamics 下的大量 rollout训练：

> Interaction Encoder
> JEPA Forward Predictor
> Intent-to-Action Predictor

目标是：

> **从 Interaction Context 中理解当前 World。**

这一阶段完全是 supervised / self-supervised learning。

---

## Stage II：Control Adaptation

冻结：

> Large Tracker + World Model

只通过 RL 训练：

> **Action Head**

目标是：

> **学会利用 Tracker Action 与 World-Aware Reference Action，在不同 dynamics 中产生更好的最终动作。**

因此两个训练问题被完全解耦：

> **Stage I：Learn the World**

> **Stage II：Learn to Use the World**

---

# 12. Context 有效性验证

本工作的一个核心实验现象是：

> **机器人与当前环境交互得越多，对这个环境的理解应该越准确。**

当前阶段保持所有参数冻结，并保持 16-token 输入长度不变。通过
correct / no-information / wrong-world / shuffled context 对照，判断模型是否真正利用了
interaction evidence；可变 token 数量留作后续扩展，不进入第一版实现。

重点观察两类 Scaling：

### Prediction Scaling

> Context ↑ → Future Prediction Error ↓

### Action Scaling

> Context ↑ → Reference Action Accuracy ↑

最终进一步验证：

### Tracking Scaling

> Context ↑ → Final Motion Tracking Performance ↑

从而形成完整链条：

> **More Interaction
> → Better World Understanding
> → Better Action Inference
> → Better Motion Tracking**

---

# 13. 第一阶段关键验证

在训练最终 Action Head 之前，首先验证 World Model 是否真正成立。

重点实验包括：

**Context Scaling**

增加同一 environment 中的 interaction context，Future Prediction Error 是否下降。

**Wrong-World Context**

固定同一个 query，但提供另一个 environment 的 context，prediction 和 reference action 是否发生符合对应 dynamics 的变化。

**Long-Term Context**

长期 environment interaction memory 是否优于仅使用当前 state附近的 recent history。

**OOD Dynamics**

面对训练中没有见过的 dynamics 或 dynamics combinations，更多 context 是否仍然能够改善 prediction 和 action inference。

如果这些实验成立，就说明模型确实表现出了：

> **In-Context Physical World Learning**

而不是普通的 temporal trajectory prediction。

---

# 14. 方法核心

整个方法最终可以用三个概念概括：

### Skill

Frozen Large Tracker：

> **How do I normally perform this motion?**

### World

In-Context World Model：

> **Based on my previous interactions, how does this world work?**

### Adaptation

RL Action Head：

> **Given my skill prior and what I have learned about this world, how should I act now?**

因此，我们并不试图训练一个提前适应所有可能世界的 humanoid policy，而是希望获得一种不同的能力：

> **机器人进入一个陌生的物理世界，通过自身不断积累的 interaction experience 在 Context 中理解这个世界，并立即利用这种理解调整已有的运动技能。**

核心结构可以最终压缩为：

> **Interaction → World**

> **Motion → Intent**

> **World × Intent → Reference Action**

> **Reference Action × Skill Prior → Final Action**

## 暂定标题

**Learning the World in Context: Adaptive Humanoid Motion Tracking from Interaction**

或：

**In-Context World Models for Adaptive Humanoid Motion Tracking**
