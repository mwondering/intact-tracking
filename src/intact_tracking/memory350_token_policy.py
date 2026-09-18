"""29-token actor/Transformer critic versus the original 1645-D MLP baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories

from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking.limb_context_policy import LimbContextCritic, configure_context_models
from intact_tracking.memory350_token_inputs import (
    BASE_ACTION, FRAME_DIMS, FRAME_NAMES, FUTURE, HISTORY, LATENT_SLICE, VALID,
    intervene_latent_history,
)
from intact_tracking.residual_policy import FrozenTrackerResidualActor

VERSION = "memory350_temporal29_transformer_vs_original_mlp_v1"
INITIALIZATION = "frozen_tracker_zero_residual_scratch_independent_value_v1"
ACTOR_CLASS = "intact_tracking.memory350_token_policy:TemporalTokenActor"
CRITIC_CLASS = "intact_tracking.memory350_token_policy:TemporalTokenCritic"
ARCHITECTURES = ("mlp", "transformer")
WIDTH, DEPTH, HEADS, FFN_DIM = 128, 2, 4, 256


def _projection(widths):
    layers = []
    for index, (a, b) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if index < len(widths) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.output = nn.Linear(WIDTH, WIDTH)
        self.ffn_norm = nn.LayerNorm(WIDTH)
        self.ffn = nn.Sequential(nn.Linear(WIDTH, FFN_DIM), nn.GELU(),
                                 nn.Linear(FFN_DIM, WIDTH))

    def forward(self, tokens, valid):
        n, length, _ = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens)).reshape(n, length, 3, HEADS, WIDTH // HEADS)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # SDPA bool masks use True for an allowed key. Every sample has valid
        # current/future targets. Dropout is zero in collection AND PPO updates.
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid[:, None, None, :], dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(n, length, WIDTH)
        tokens = tokens + self.output(attended)
        return tokens + self.ffn(self.ffn_norm(tokens))


class TemporalTokenBackbone(nn.Module):
    """Each instance owns all modality projections and Transformer parameters."""

    def __init__(self, *, privileged_dim=0):
        super().__init__()
        self.privileged_dim = int(privileged_dim)
        self.projections = nn.ModuleDict({
            "proprio": _projection((320, 256, 128)),
            "reference": _projection((269, 256, 128)),
            "error": _projection((260, 256, 128)),
            "latent": nn.Linear(64, 128),
            "tracker_action": nn.Linear(29, 128),
            "future": nn.Linear(77, 128),
        })
        self.type_embedding = nn.Embedding(7 if privileged_dim else 6, WIDTH)
        self.time_embedding = nn.Embedding(9, WIDTH)  # -4 .. +4 relative to the decision.
        nn.init.normal_(self.type_embedding.weight, std=.02)
        nn.init.normal_(self.time_embedding.weight, std=.02)
        types = [*list(range(5)) * 5, *([5] * 4)]
        times = [*[i for i in range(5) for _ in range(5)], 5, 6, 7, 8]
        if privileged_dim:
            self.privileged = _projection((privileged_dim, 1024, 512, 256, 128))
            types.append(6)
            times.append(4)
        self.register_buffer("type_ids", torch.tensor(types), persistent=False)
        self.register_buffer("time_ids", torch.tensor(times), persistent=False)
        self.blocks = nn.ModuleList([AttentionBlock() for _ in range(DEPTH)])
        self.final_norm = nn.LayerNorm(WIDTH)

    def forward(self, history, future, valid, privileged=None):
        if history.ndim != 3 or history.shape[1:] != (5, sum(FRAME_DIMS)):
            raise ValueError("Expected five synchronized token frames")
        if future.shape != (len(history), 4, 77) or valid.shape != (len(history), 5):
            raise ValueError("Expected four known future reference frames and a five-frame mask")
        # Mask inputs before their projections as well as attention keys, so
        # invalid padding can never leak NaNs or synthetic reset transitions.
        raw = history.detach().masked_fill(~valid[..., None], 0)
        pieces = raw.split(FRAME_DIMS, dim=-1)
        encoded = []
        for name, value in zip(FRAME_NAMES, pieces, strict=True):
            if name == "latent":
                value = F.normalize(value.float(), dim=-1, eps=1e-8)
            encoded.append(self.projections[name](value))
        tokens = torch.stack(encoded, dim=2).flatten(1, 2)
        tokens = torch.cat((tokens, self.projections["future"](future.detach())), 1)
        allowed = torch.cat((valid.bool().repeat_interleave(5, -1),
                             torch.ones(len(valid), 4, dtype=torch.bool, device=valid.device)), -1)
        if self.privileged_dim:
            if privileged is None or privileged.shape != (len(history), self.privileged_dim):
                raise ValueError("This value network requires its privileged observation token")
            tokens = torch.cat((tokens, self.privileged(privileged.detach())[:, None]), 1)
            allowed = torch.cat((allowed, torch.ones_like(allowed[:, :1])), -1)
        elif privileged is not None:
            raise ValueError("Actor/common-token-only backbone cannot receive privileged observations")
        tokens = tokens + self.type_embedding(self.type_ids) + self.time_embedding(self.time_ids)
        for block in self.blocks:
            tokens = block(tokens, allowed)
        # Index 20 is the CURRENT proprio token, not an average over padded history.
        return self.final_norm(tokens[:, 20])


class TemporalTokenActor(FrozenTrackerResidualActor):
    def __init__(self, *args, architecture="transformer", initialization_seed=10128,
                 initial_action_std=.25, **kwargs):
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        kwargs.pop("fusion_mode", None)
        kwargs["use_dynamics_latent"] = False
        kwargs["residual_scale"] = 1.0
        super().__init__(*args, **kwargs)
        if self.tracker.policy_input_dim != 1645 or self.distribution.output_dim != 29:
            raise ValueError("This policy requires the 1645-D / 29-action frozen tracker")
        self.architecture, self.initialization_seed = architecture, int(initialization_seed)
        self.fusion_mode = "concat" if architecture == "transformer" else "baseline"
        self.use_dynamics_latent = architecture == "transformer"
        del self.residual_mlp
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.initialization_seed)
            if self.use_dynamics_latent:
                self.backbone = TemporalTokenBackbone()
                self.head = _projection((128 + 29, 256, 128, 29))
            else:
                # User-requested original observation MLP: no latent, no extra
                # tracker action input, and no additional temporal observations.
                self.head = _projection((1645, 512, 256, 128, 29))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        cfg = dict(kwargs.get("distribution_cfg") or {})
        cfg.pop("class_name", None)
        cfg["init_std"] = float(initial_action_std)
        self.distribution = GaussianDistribution(29, **cfg)
        self.frozen_tracker_digest = tensor_digest(self.tracker.state_dict().items())

    def residual_mean(self, obs):
        if self.architecture == "mlp":
            features, base = self._base_features_and_action(obs)
            return base, self.head(features)
        h = self.backbone(obs[HISTORY], obs[FUTURE], obs[VALID])
        base = obs[BASE_ACTION].detach()
        return base, self.head(torch.cat((h, base), -1))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        base, residual = self.residual_mean(obs)
        mean = base + residual
        self.last_base_action, self.last_residual_mean = base, residual.detach()
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def update_normalization(self, obs):
        if self.architecture == "mlp":
            super().update_normalization(obs)
        # Transformer collection already populated the immutable tracker cache.

    def assert_tracker_frozen(self):
        if self.tracker.training or any(p.requires_grad or p.grad is not None for p in self.tracker.parameters()):
            raise RuntimeError("Frozen tracker acquired gradients/training state")
        if tensor_digest(self.tracker.state_dict().items()) != self.frozen_tracker_digest:
            raise RuntimeError("Frozen tracker weights or normalization changed")

    def intervene_latent(self, obs, mode, *, donors=None, eligible=None):
        if not self.use_dynamics_latent:
            raise ValueError("The MLP baseline has no latent input")
        return intervene_latent_history(obs, mode, donors=donors, eligible=eligible)

    @torch.no_grad()
    def policy_metrics(self, obs):
        base, residual = self.residual_mean(obs)
        result = {"base_action_rms": float(base.square().mean().sqrt()),
                  "residual_action_rms": float(residual.square().mean().sqrt()),
                  "residual_action_abs_max": float(residual.abs().max()),
                  "residual_output_bounded": 0.0,
                  "actor_uses_environment_latent": float(self.use_dynamics_latent)}
        if self.use_dynamics_latent:
            zero = self.intervene_latent(obs, "zero")
            donor = torch.arange(len(base), device=base.device).roll(1)
            swapped = self.intervene_latent(obs, "paired-swap", donors=donor,
                                            eligible=obs[VALID].all(-1))
            result.update(
                latent_zero_action_delta_rms=float((residual - self.residual_mean(zero)[1]).square().mean().sqrt()),
                latent_shuffle_action_delta_rms=float((residual - self.residual_mean(swapped)[1]).square().mean().sqrt()),
                token_history_full_fraction=float(obs[VALID].all(-1).float().mean()))
        return result


class TemporalTokenCritic(LimbContextCritic):
    def __init__(self, *args, architecture="transformer", critic_privileged_token=True,
                 initialization_seed=20124, **kwargs):
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        kwargs.pop("fusion_mode", None)
        super().__init__(*args, fusion_mode="baseline", initialization_seed=initialization_seed, **kwargs)
        self.architecture = architecture
        self.critic_privileged_token = bool(critic_privileged_token)
        del self.mlp
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(initialization_seed))
            if architecture == "transformer":
                self.backbone = TemporalTokenBackbone(
                    privileged_dim=self.obs_dim if critic_privileged_token else 0)
                self.head = _projection((128 + 29, 256, 128, 1))
            else:
                self.head = _projection((self.obs_dim, 1024, 512, 256, 128, 1))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        if self.architecture == "mlp":
            return self.head(self.obs_normalizer(self._flat_obs(obs)))
        privileged = (self.obs_normalizer(self._flat_obs(obs)) if self.critic_privileged_token else None)
        h = self.backbone(obs[HISTORY], obs[FUTURE], obs[VALID], privileged)
        return self.head(torch.cat((h, obs[BASE_ACTION].detach()), -1))


def configure_models(train, fusion, *, scratch_seed, architecture="transformer", critic_privileged_token=True):
    if fusion != ("concat" if architecture == "transformer" else "baseline"):
        raise ValueError("Architecture and context-loading contract disagree")
    result = configure_context_models(train, "baseline", scratch_seed=scratch_seed)
    result["actor"].update(class_name=ACTOR_CLASS, architecture=architecture, residual_scale=1.0)
    result["critic"].update(class_name=CRITIC_CLASS, architecture=architecture,
                            critic_privileged_token=bool(critic_privileged_token))
    return result


@torch.no_grad()
def audit_initial_models(actor, critic, obs, fusion):
    del fusion
    _, expected = actor._base_features_and_action(obs)
    torch.testing.assert_close(actor(obs), expected, atol=0, rtol=0)
    torch.testing.assert_close(actor(obs), actor.tracker(obs), atol=0, rtol=0)
    actor.distribution.update(expected)
    torch.testing.assert_close(actor.output_std, torch.full_like(expected, .25), atol=0, rtol=0)
    if {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}:
        raise RuntimeError("Actor and critic share parameters")
    if actor.use_dynamics_latent:
        torch.testing.assert_close(obs[HISTORY][:, -1, LATENT_SLICE], obs["dynamics_latent"], atol=0, rtol=0)
        torch.testing.assert_close(obs[HISTORY][:, -1, -29:], expected, atol=0, rtol=0)
    actor.assert_tracker_frozen()
    return {"architecture": actor.architecture, "actor_original_features": 1645,
            "critic_original_features": critic.obs_dim,
            "actor_critic_parameters_shared": False, "identical_initial_tracker_action": True,
            "actor_tokens": 29 if actor.use_dynamics_latent else 0,
            "critic_tokens": (29 + int(critic.critic_privileged_token)) if actor.use_dynamics_latent else 0,
            "baseline_actor_mlp": [1645, 512, 256, 128, 29],
            "baseline_critic_mlp": [critic.obs_dim, 1024, 512, 256, 128, 1],
            "token_width": WIDTH, "transformer_layers": DEPTH, "attention_heads": HEADS,
            "residual_output_bounded": False, "initial_action_std": .25,
            "actor_trainable_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
            "critic_trainable_parameters": sum(p.numel() for p in critic.parameters() if p.requires_grad)}
