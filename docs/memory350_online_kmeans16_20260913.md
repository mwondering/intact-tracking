# Online K-means16 MoE versus MLP

Run: `runs/limb_context_20260913_memory350_online_kmeans16_uniform_dr`.

Both policies start from zero residual and use the frozen tracker mean plus
`0.25 * tanh(residual_head)`, with one Gaussian over the final action. Actor and
critic have independent trainable observation encoders and independent heads.
All sixteen actor heads share the actor encoder; all sixteen critic heads share
the critic encoder. No trainable parameter is shared between actor and critic.

| Component | MLP baseline | Latent MoE |
|---|---|---|
| Actor encoder | 1645 → 512 → 256 → 128 | same |
| Actor head input | compressed 128 + raw tracker action 29 | same |
| Actor heads | one 157 → 256 → 128 → 29 | sixteen independent heads |
| Critic encoder | 6330 → 1024 → 512 → 256 → 128 | same |
| Critic heads | one 128 → 256 → 128 → 1 | sixteen independent heads |
| Latent | absent | 64 dimensions, used only for hard routing |
| GPUs | 0,1,2,3 | 4,5,6,7 |

Each GPU runs 8192 independently randomized worlds, with independent U(0,4 kg)
loads on the four limbs plus the original tracker DR. There is no grid256 bank.
The full 129827-motion dataset remains sharded by rank. Adaptive motion sampling
starts at PPO update zero; original end-effector height termination is enabled.
Training has no update cap. Policy forwards use IEEE FP32; the frozen context
encoder retains its existing cached BF16 inference and Memory350 reset rules.

Both arms first execute 500 deterministic frozen-tracker interaction steps,
without PPO updates. The latent arm uses the histories at steps 350, 400, 450,
and 500 to initialize sixteen centers by K-means++ / Lloyd iterations. Samples
are pooled across all four ranks. This is an online initialization from the new
training worlds, not a precomputed labeled DR classifier.

During PPO, latent vectors are L2 normalized and assigned to their nearest
Euclidean center. Only the selected actor and critic expert heads produce each
sample's output. Centers stay fixed for the entire 24-step rollout and all five
PPO epochs. After that PPO update, the saved rollout latents supply global
cluster sums/counts. Centers move 1% toward their newly assigned mean. A
backtracking step limits reassignment on this rollout to at most 2%. Empty
centers retain their index and location; online updates do not permute or
reinitialize expert identities. Centers receive no PPO gradients.

The actor and critic store independent copies of identical routing state;
updates copy all router buffers from actor to critic. Centers, cumulative
assignment counts, and update count are saved in both state dictionaries.
Evaluation freezes the saved centers. A resume restores the saved routing state
after rebuilding simulator history, preserving expert identities and optimizer
state. The source encoder remains frozen throughout.

Training logs contain per-expert rollout/current occupancy, effective expert
count, center movement, effective center rate, and reassignment fraction.
Every 100 updates, both arms receive the existing matched cold/warm all-0/all-4
tracking evaluation. Every 1000 updates, the continuous-DR matched latent-swap
diagnostic checks closed-loop sensitivity to environment routing. The MLP
baseline has fewer total parameters; this is the requested architecture versus
baseline comparison, not a parameter-count-matched ablation.

The prior grid256 tracker-action experiment was gracefully paused at baseline
1638 and latent 1400; verified immutable resume checkpoints are recorded under
its `paused_for_online_kmeans16/verification.json`.
