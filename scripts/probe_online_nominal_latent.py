"""Collect online latent and residual diagnostics from a fixed PPO checkpoint."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from omegaconf import OmegaConf

from intact_tracking import memory350_native_policy as native
from intact_tracking.cli.memory350_proprio_native_policy_train import configure_proprio_physics
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.limb_context_sampling import STATE_FIELDS
from intact_tracking.memory350_inference import load_memory350_checkpoint
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.memory350_proprio_policy import ProprioNativePolicyWrapper
from intact_tracking.memory350_tracker_action_policy import TrackerActionResidualActor


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args):
    out = Path(args.output).resolve()
    if out.with_suffix('.json').exists() or out.with_suffix('.npz').exists():
        raise FileExistsError(out)
    configure_policy_precision('fp32')
    torch.cuda.set_per_process_memory_fraction(.30)
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS['terrain_height_offset'] = native.FlatTerrainHeightOffset
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta = state['residual_policy']
    agent = OmegaConf.to_container(state['cfg'].agent, resolve=True)
    update = int(state['completed_updates'])
    header = next(x for x in state['motion_sampling_state']['ranks'] if x['rank'] == args.rank)
    if digest(header['path']) != header['sha256']:
        raise ValueError('Adaptive sampler checkpoint changed')
    sampler = torch.load(header['path'], map_location='cpu', weights_only=False)
    files = Path(args.manifest).read_text().splitlines()
    if hashlib.sha256('\n'.join(files).encode()).hexdigest() != header['motion_shard_sha256']:
        raise ValueError('Probe motion catalog must match the saved training rank')
    _seed_everything(args.seed)
    prepared = prepare_rollout(checkpoint_file=meta['tracker_checkpoint'], num_envs=args.num_envs,
                               motion_file=None, motion_path=meta['motion_path'])
    cfg = prepared.env
    physics = configure_proprio_physics(cfg, args.seed, profile=native.PROFILE)
    sampling = native.configure_sampling(cfg, 'adaptive', 0, update)
    if sampling != meta['motion_sampling'] or sampler['configuration'] != sampling:
        raise ValueError('Adaptive sampling configuration differs from training')
    cfg.commands['motion'].motion_manifest_file = str(Path(args.manifest).resolve())
    cfg.auto_reset, cfg.seed = True, args.seed
    cfg.episode_length_s = 500 * cfg.decimation * cfg.sim.mujoco.timestep
    if set(cfg.terminations) != {term['name'] for term in meta['training_terminations']['active_terms']}:
        raise ValueError('Original termination terms changed')
    context = load_memory350_checkpoint(meta['context_checkpoint'], device='cuda:0',
                                       expected_tracker_sha256=meta['tracker_sha256'])
    if context.sha256 != meta['context_sha256']:
        raise ValueError('Context checkpoint changed')
    encoder_state = torch.load(meta['context_checkpoint'], map_location='cpu', weights_only=False)
    anchor = torch.tensor(encoder_state['nominal_direction_anchor']['direction'], device='cuda:0', dtype=torch.float32)
    torch.testing.assert_close(anchor.norm(), torch.ones((), device=anchor.device))
    del encoder_state
    _seed_everything(args.seed)
    env = native.environment_factory(ManagerBasedRlEnv, cfg=cfg, device='cuda:0')
    try:
        cmd = env.command_manager.get_term('motion')
        if tuple(files) != tuple(cmd.motion_files):
            raise ValueError('Runtime loaded different motions')
        wrapped = ProprioNativePolicyWrapper(env, prepared.clip_actions, context, latent_history_frames=5)
        obs = wrapped.get_observations()
        kwargs = copy.deepcopy(agent['actor']); kwargs.pop('class_name')
        actor = TrackerActionResidualActor(obs, agent['obs_groups'], 'actor', 29, **kwargs).to(env.device)
        actor.load_state_dict(state['actor_state_dict'], strict=True)
        actor.eval().requires_grad_(False)
        env.common_step_counter = int(state['env']['common_step_counter'])
        # Reset visits are rebuilt, exactly as on simulator restart. Restore the
        # checkpoint's rank-local difficulty statistics before the first query.
        for key in STATE_FIELDS:
            target, saved = getattr(cmd, key), sampler['statistics'][key]
            if target.shape != saved.shape or target.dtype != saved.dtype:
                raise ValueError(f'Sampler shape mismatch: {key}')
            target.copy_(saved.to(target.device))
        cmd._adaptive_ema_last_iteration = sampler['ema_last_iteration']
        _seed_everything(args.seed + 1000000)
        obs, _ = wrapped.reset()
        env.episode_length_buf[:] = torch.randint_like(env.episode_length_buf, high=env.max_episode_length)
        nominal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        nominal[env.native_policy_nominal_ids] = True
        action_term = env.action_manager.get_term('joint_pos')
        scale = torch.as_tensor(action_term._scale, device=env.device).expand(env.num_envs, 29)
        fields = ('latent', 'radius', 'residual', 'base_rms', 'residual_pd_rms', 'exploration_rms',
                  'short_count', 'long_count', 'history5_count', 'motion_id', 'motion_step',
                  'episode_id', 'episode_step', 'anchor_direction_action_delta_rms')
        samples = {key: [] for key in fields}
        reset_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        failure_counts = torch.zeros_like(reset_counts)
        elapsed_start = time.time()
        std = None
        del state
        with torch.inference_mode():
            for step in range(args.warmup_steps + args.steps):
                if step % 24 == 0:
                    cmd.begin_adaptive_sampling_iteration(update + step // 24)
                # Produce both mean and sampled action so the reported residual
                # is always the learned mean correction, excluding exploration.
                actor.update_normalization(obs)
                sampled = actor(obs, stochastic_output=True)
                mean = actor.output_mean
                applied = sampled if args.mode == 'sampled' else mean
                residual = actor.last_residual_mean
                base = actor.last_base_action
                if std is None:
                    std = float(actor.output_std.mean())
                if step >= args.warmup_steps and (step - args.warmup_steps) % args.stride == 0:
                    latent = obs['dynamics_latent'][:, -64:]
                    unit = torch.nn.functional.normalize(latent, dim=-1)
                    bank = wrapped.context.memory
                    # A same-observation sensitivity check. Preserve each frame's
                    # norm and padding while replacing direction with nominal.
                    features, _ = actor._base_features_and_action(obs)
                    history = obs['dynamics_latent'].reshape(env.num_envs, 5, 64)
                    anchor_history = (history.norm(dim=-1, keepdim=True) * anchor).flatten(1)
                    anchored = actor._residual(actor._residual_input(obs, features, base,
                                                                    latent_override=anchor_history))
                    values = {
                        'latent': latent, 'radius': (unit-anchor).norm(dim=-1), 'residual': residual,
                        'base_rms': base.square().mean(-1).sqrt(),
                        'residual_pd_rms': (residual*scale).square().mean(-1).sqrt(),
                        'exploration_rms': (applied-mean).square().mean(-1).sqrt(),
                        'short_count': bank.short_count,
                        'long_count': (bank.total_chunks-bank.session_start).clamp(max=30),
                        'history5_count': wrapped.latent_history.count,
                        'motion_id': cmd.motion_idx, 'motion_step': cmd.time_steps,
                        'episode_id': wrapped.episode_ids, 'episode_step': env.episode_length_buf,
                        'anchor_direction_action_delta_rms': (anchored-residual).square().mean(-1).sqrt(),
                    }
                    for key, value in values.items():
                        samples[key].append(value.detach().cpu().numpy().copy())
                action_term.record_policy_mean(mean)
                obs, _, dones, _ = wrapped.step(applied)
                if step >= args.warmup_steps:
                    reset_counts += dones.long()
                    failure_counts += env.reset_terminated.long()
                if (step+1) % 250 == 0:
                    print(json.dumps({'step':step+1, 'phase':'warmup' if step<args.warmup_steps else 'sample',
                                      'seconds':time.time()-elapsed_start, 'memory':wrapped.latent_metrics}), flush=True)
        native.audit_native_runtime(env)
        arrays = {key: np.stack(value) for key,value in samples.items()}
        arrays.update(is_nominal=nominal.cpu().numpy(), anchor=anchor.cpu().numpy(),
                      action_scale=scale.detach().cpu().numpy(), resets=reset_counts.cpu().numpy(),
                      failures=failure_counts.cpu().numpy())
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.savez_compressed(out.with_suffix('.npz'), **arrays)
        report = {
            'version':'online_nominal_residual_probe_v1', 'mode':args.mode, 'arguments':vars(args),
            'checkpoint':str(Path(args.checkpoint).resolve()), 'checkpoint_sha256':digest(args.checkpoint),
            'completed_updates':update, 'context_checkpoint':meta['context_checkpoint'],
            'context_sha256':context.sha256, 'tracker_sha256':meta['tracker_sha256'],
            'num_envs':env.num_envs, 'nominal_worlds':int(nominal.sum()),
            'dr_worlds':int((~nominal).sum()), 'sampling_rank':args.rank,
            'motion_count':len(files), 'motion_catalog_sha256':header['motion_shard_sha256'],
            'adaptive_state':header, 'adaptive_sampling':sampling, 'physics':physics,
            'physics_audit':env.native_policy_runtime_audit, 'action_std':std,
            'recorded_control_steps':len(arrays['radius']), 'recorded_world_frames':arrays['radius'].size,
            'memory_warmup':'Same fixed residual policy as the collection arm; old memory preserved across resets',
            'production_episode_limit':env.max_episode_length,
            'production_termination_terms':list(env.termination_manager.active_terms),
            'no_optimizer_updates':True, 'runtime_seconds':time.time()-elapsed_start,
            'residual_definition':'Learned deterministic correction before exploration, clipping, action scaling and SP substep delay/smoothing',
            'scaled_residual_definition':'Correction times joint action scale, in PD angle radians, before SP filtering',
            'anchor_intervention':'Diagnostic only, not applied to environment: same observation, all filled latent frames rotated to nominal anchor, retaining each raw norm',
            'scope':'New online trajectories from a fixed current checkpoint on one restored training motion shard; not a readout of live training process buffers',
            'source_sha256':digest(__file__),
        }
        out.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:report[k] for k in ('mode','completed_updates','num_envs','nominal_worlds','recorded_world_frames','runtime_seconds')}),flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['sampled','mean'],required=True)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--num-envs',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=239019)
    parser.add_argument('--warmup-steps',type=int,default=1000)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--stride',type=int,default=5)
    run(parser.parse_args())
