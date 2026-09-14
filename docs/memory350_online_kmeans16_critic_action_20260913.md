# Online K-means16 with tracker action in both actor and critic

Run directory: `runs/limb_context_20260913_memory350_online_kmeans16_critic_action`.
This replaces the initial online-MoE run, which omitted the critic tracker-action
input. Its MLP and MoE checkpoints were saved at updates 100 and 65 respectively;
the corrected comparison starts both arms from scratch.

| Component | MLP baseline | Latent MoE |
|---|---|---|
| Actor observation encoder | 1645 → 512 → 256 → 128 | same |
| Actor head | one 157 → 256 → 128 → 29 | sixteen independent heads |
| Critic observation encoder | 6330 → 1024 → 512 → 256 → 128 | same |
| Critic head | one 157 → 256 → 128 → 1 | sixteen independent heads |
| Head inputs | compressed observation 128 + raw tracker mean 29 | same |
| Latent input | absent | frozen latent64, used for hard routing |
| GPUs | 0,1,2,3 | 4,5,6,7 |

Actor and critic share **no parameters**. All actor experts use one actor
observation encoder; all critic experts use a separate critic observation
encoder. The critic does not own or reference the actor's tracker module.

At each rollout step, the actor computes the deterministic frozen tracker mean.
The collector copies this current 29-dimensional mean into the observation
before evaluating the critic and recording the transition. Minibatches retain
that action with its original state. The critic therefore receives the raw
tracker action, rather than a sampled residual action, applied action, or the
previous state's action. For final-state GAE bootstrapping, the tracker mean is
computed on the final next-state observation. The extra field is excluded from
the 6330-dimensional observation normalizer and appended after compression.

Sixteen centers are initialized from online interaction histories and updated
throughout training. Centers stay fixed during each 24-step rollout and all five
PPO epochs. After PPO, globally reduced cluster means move centers at rate 0.01;
backtracking caps reassignment on that rollout at 2%. Empty centers retain their
identity. Actor and critic keep separate copies of equal routing buffers, saved
with the checkpoints; no PPO gradient updates the router.

Each GPU has 8192 independently randomized worlds, with four independent uniform
0–4 kg limb loads and original tracker DR. Both arms use the full 129827-motion
dataset, adaptive sampling from PPO update zero, original end-effector height
termination, IEEE FP32 policy forwards, and no update cap. The Memory350 context
encoder remains frozen, with its existing cached BF16 inference. Long memory
persists across episode resets under the existing history-boundary rules.

Both arms first run 500 frozen-tracker interaction steps to prepare histories;
these are not PPO updates. Initialization uses the same seed procedure and
initial compressor weights. Simulator trajectories after those steps need not
be bitwise identical across the two GPU groups.

The supervisor records health every ten seconds and history once per minute:
finite losses, critic action input, independent actor/critic parameters, and
one center update per PPO update. An invariant failure stops both arms safely.
Every 100 PPO updates, the existing 512-motion cold/warm all-0/all-4 evaluation
runs; every 1000 updates, the 128-motion continuous-DR latent-swap diagnostic
checks whether routing affects control. Evaluation uses saved, fixed centers.
Matched checkpoints are aggregated in `ppo_comparison.json`.

The requested MLP baseline has fewer parameters than the MoE; the comparison
does not by itself separate latent routing from expert capacity.
