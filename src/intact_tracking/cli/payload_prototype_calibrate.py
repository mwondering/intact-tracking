"""Calibrate fixed load prototypes from independent, full-history tracker worlds."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from mjlab.envs import ManagerBasedRlEnv

from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.cli.adaptation_eval import load_actor
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout, _load_saved_config
from intact_tracking.limb_context_protocol import TRACKER, TRACKER_SHA256, FULL_DATASET
from intact_tracking.limb_context_terminations import configure_training_terminations
from intact_tracking.memory350_inference import load_memory350_checkpoint
from intact_tracking.memory350_policy_env import Memory350PolicyWrapper
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.payload_prototype_physics import configure_physics, audit_physics, load_grid
from intact_tracking.preview_protocol import dataset_identity

VERSION = "payload256_cross_world_prototypes_v1"


def fit_prototypes(samples, ids, fit_mask, validation_mask, test_mask):
    z = F.normalize(samples.float(), dim=-1)
    centers = torch.stack([z[fit_mask & (ids == e)].mean(0) for e in range(256)])
    if not torch.isfinite(centers).all():
        raise ValueError("Every prototype needs full-memory fitting samples")
    # Arithmetic means of unit latents: preserve the historical nearest-centroid metric.
    distance = (z.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2*z@centers.T).clamp_min(0)
    nearest, order = distance.topk(5, largest=False)
    gap = (nearest[validation_mask, 1] - nearest[validation_mask, 0]).median().clamp_min(1e-6)
    candidates = gap * torch.tensor([0.1, 0.2, 0.5, 1., 2., 5., 10.])
    records = []
    for tau in candidates:
        weights = (-nearest[validation_mask] / tau).softmax(-1)
        correct_weight = (weights * (order[validation_mask] == ids[validation_mask, None])).sum(-1)
        records.append(float(-correct_weight.clamp_min(1e-8).log().mean()))
    temperature = float(candidates[torch.tensor(records).argmin()])
    metrics = {}
    for name, mask in (("validation", validation_mask), ("test", test_mask)):
        correct = order[mask] == ids[mask, None]
        weights = (-nearest[mask] / temperature).softmax(-1)
        metrics[name] = {"samples": int(mask.sum()), "top1": float(correct[:,0].float().mean()),
                         "top5": float(correct.any(-1).float().mean()),
                         "mean_max_weight": float(weights.max(-1).values.mean()),
                         "per_sample_effective_experts": float((1/weights.square().sum(-1)).mean()),
                         "load_estimate_mae_kg": ((weights[...,None]*load_grid()[order[mask]]).sum(1)-load_grid()[ids[mask]]).abs().mean(0).tolist()}
    return centers, temperature, {"metrics": metrics, "temperature_candidates": candidates.tolist(),
        "validation_nll": records, "temperature_selection": "Validation-only correct-prototype probability among nearest five"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--sample-start", type=int, default=600)
    parser.add_argument("--sample-stride", type=int, default=50)
    parser.add_argument("--motion-count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=19616)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_envs % 1024 or args.num_envs < 1024 or args.sample_start < 350:
        raise ValueError("Need at least four independent replicas per load, and full history")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    configure_policy_precision("fp32")
    _seed_everything(args.seed)
    prepared = prepare_rollout(checkpoint_file=TRACKER, num_envs=args.num_envs,
                               motion_path=FULL_DATASET, motion_file=None)
    prepared.env.seed = args.seed
    physics = configure_physics(prepared.env, args.seed, anchor_fraction=1.)
    files, dataset = dataset_identity(None,FULL_DATASET)
    # A fixed uniform subset spans the catalog without allocating all 48M frames
    # on the one calibration GPU. PPO itself still uses the complete dataset.
    order = torch.randperm(len(files),generator=torch.Generator().manual_seed(args.seed+17))
    selected = sorted(str(files[i]) for i in order[:args.motion_count].tolist())
    manifest = output/'motions.txt'
    manifest.write_text('\n'.join(selected)+'\n')
    prepared.env.commands['motion'].motion_manifest_file = str(manifest)
    configure_training_starts(prepared.env, "original")
    configure_training_terminations(prepared.env, _load_saved_config(Path(TRACKER)), "original")
    context = load_memory350_checkpoint(args.context_checkpoint, device=args.device, expected_tracker_sha256=TRACKER_SHA256)
    _seed_everything(args.seed)
    env = ManagerBasedRlEnv(cfg=prepared.env, device=args.device)
    try:
        audit = audit_physics(env, physics)
        wrapped = Memory350PolicyWrapper(env, prepared.clip_actions, context)
        obs = wrapped.get_observations()
        actor = load_actor(None, prepared, obs, wrapped).eval().requires_grad_(False)
        command = env.command_manager.get_term("motion")
        samples, valid, motions = [], [], []
        with torch.inference_mode():
            for step in range(1, args.steps+1):
                action = actor(obs)
                env.action_manager.get_term("joint_pos").record_policy_mean(action)
                obs, _, _, _ = wrapped.step(action)
                if step >= args.sample_start and (step-args.sample_start) % args.sample_stride == 0:
                    bank = wrapped.context.memory
                    full = (bank.total_chunks-bank.session_start >= 30) & (bank.short_count == 50)
                    samples.append(obs["dynamics_latent"].detach().cpu())
                    valid.append(full.cpu())
                    motions.append(command.motion_idx.cpu().clone())
                if step % 100 == 0:
                    print(json.dumps({"step":step,"samples":len(samples),"context":wrapped.latent_metrics}),flush=True)
        z, full, motion = torch.stack(samples), torch.stack(valid), torch.stack(motions)
        n = args.num_envs
        worlds = torch.arange(n).expand(z.shape[0],-1).reshape(-1)
        ids = worlds % 256
        replica = worlds // 256
        # Split whole worlds, never time points from the same history.
        fold = replica % 4
        fit, val, test = (fold < 2)&full.flatten(), (fold==2)&full.flatten(), (fold==3)&full.flatten()
        centers, tau, report = fit_prototypes(z.reshape(-1,64), ids, fit, val, test)
        report.update(version=VERSION, arguments=vars(args), context_sha256=context.sha256,
                      context_checkpoint=context.path, tracker_sha256=TRACKER_SHA256,
                      physics=physics, physics_audit=audit, full_history_samples=int(full.sum()),
                      calibration_episode_steps=int(env.max_episode_length),
                      motion_count=len(command.motion_files), sampled_unique_motions=int(motion.unique().numel()),
                      source_dataset=dataset,
                      split="Independent worlds: 50% fitting / 25% validation / 25% test; same load IDs, different histories",
                      center_formula="mean of unit latents, unnormalized means; squared Euclidean distance",
                      temperature=tau, ppo_warmup_steps=0)
        torch.save({"version":VERSION,"centers":centers,"temperature":tau,"loads_kg":load_grid(),"metadata":report}, output/'prototypes.pt')
        torch.save({"latent":z,"full":full,"motion":motion,"ids":ids,"worlds":worlds},output/'calibration_samples.pt')
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
