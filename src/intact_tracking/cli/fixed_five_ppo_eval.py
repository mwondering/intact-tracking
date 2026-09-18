"""Paired deterministic evaluation of fixed experts against their mixed baseline.

Recreates one complete training rank's fixed DR population and assignments.
Uses new motion draws and equal initial reference states in the paired arms.
Force pulses and observation noise retain their original independent draws.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.utils import resolve_callable

from intact_tracking import global_tracking_metrics
from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.adaptation_reward_contract import capture_original_rewards
from intact_tracking.cli.adaptation_eval import METRICS
from intact_tracking.cli.fixed_five_ppo_train import physics_snapshot, write_json
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout, _load_saved_config
from intact_tracking.fixed_five_ppo import VERSION, ROLES, dispatch_indices, mirror_baseline_physics
from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking.limb_context_dr import TRACKER_DR, configure_limb_dr, audit_limb_dr
from intact_tracking.limb_context_grouped_dr import PHYSICS_FIELDS
from intact_tracking.limb_context_terminations import configure_training_terminations
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.residual_uniform_protocol import MAX_MASSES, wrist_articulation
from intact_tracking.rollout.mjlab_adapter import _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=930017)
    parser.add_argument('--motions', type=int, default=1024)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--sample-per-class', type=int, default=0,
                        help='Evaluate a fixed stratified subset; zero evaluates the complete rank')
    args = parser.parse_args()
    if not args.device.startswith('cuda') or min(args.motions, args.trials, args.steps) < 1:
        raise ValueError('Positive evaluation counts and mjwarp CUDA required')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError(out / 'result.json')
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    configure_policy_precision('fp32')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    meta = saved['metadata']
    if meta['version'] != VERSION:
        raise ValueError('Wrong five-PPO checkpoint version')
    rank = meta['rank']
    n = len(saved['partition']['labels'])
    physics_seed = meta['arguments']['seed'] + 1000003 * rank
    eval_seed = args.seed + 1000003 * rank
    tracker = Path(meta['arguments']['tracker_checkpoint']).resolve()
    if _sha256(tracker) != meta['tracker_sha256']:
        raise ValueError('Frozen tracker checkpoint changed')
    rng = np.random.default_rng(eval_seed)
    catalog = meta['motion_files']
    chosen = sorted(rng.choice(len(catalog), min(args.motions, len(catalog)), replace=False).tolist())
    files = [catalog[i] for i in chosen]
    manifest = out / 'motions.txt'
    manifest.write_text('\n'.join(files) + '\n')
    prepared = prepare_rollout(checkpoint_file=str(tracker), num_envs=2*n,
                               motion_path=meta['arguments']['motion_path'], motion_file=None)
    cfg = prepared.env
    cfg.seed, cfg.auto_reset, cfg.episode_length_s = physics_seed, False, 20.
    cfg.scene.entities['robot'].articulation = wrist_articulation(cfg.scene.entities['robot'].articulation)
    rewards = capture_original_rewards(cfg)
    if rewards != meta['reward_contract']:
        raise ValueError('Reward contract differs from training')
    physics = configure_limb_dr(cfg, physics_seed, profile=TRACKER_DR,
                                max_masses_kg=MAX_MASSES, nominal_probability=.5)
    configure_training_starts(cfg, 'original')
    termination = configure_training_terminations(cfg, _load_saved_config(tracker), 'original')
    motion_cfg = cfg.commands['motion']
    motion_cfg.motion_manifest_file = str(manifest)
    motion_cfg.resample_on_motion_end = False
    motion_cfg.resampling_time_range = (1e9, 1e9)
    motion_cfg.init_noise, motion_cfg.pose_range, motion_cfg.velocity_range = {}, {}, {}
    motion_cfg.joint_position_range = (0., 0.)
    motion_cfg.if_log_metrics = True
    _seed_everything(physics_seed)
    env = ManagerBasedRlEnv(cfg=copy.deepcopy(cfg), device=args.device)
    try:
        mirror_baseline_physics(env)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions)
        snapshot = physics_snapshot(env)
        physical_sha = tensor_digest(sorted((k, v) for k, v in snapshot.items()
                                           if k in (*PHYSICS_FIELDS, 'encoder_bias', 'privileged_physics')))
        if physical_sha != meta['physics_sha256']:
            raise ValueError(f'Evaluation DR differs from fixed training population: {physical_sha}')
        runtime_physics = audit_limb_dr(env, physics)
        labels = saved['partition']['labels'].to(args.device)
        groups = dispatch_indices(labels)
        if args.sample_per_class:
            selected = np.sort(np.concatenate([rng.choice(np.flatnonzero(labels.cpu().numpy()==i),
                min(args.sample_per_class, int((labels==i).sum())), replace=False) for i in range(4)]))
        else:
            selected = np.arange(n)
        selected_ids = torch.as_tensor(selected, device=env.device)
        both_selected = torch.cat((selected_ids, selected_ids+n))
        participants = torch.zeros(2*n, dtype=torch.bool, device=env.device)
        participants[both_selected] = True
        groups = [group[participants[group]] for group in groups]
        obs = wrapped.get_observations()
        models = []
        train = meta['training_configuration']
        for role in ROLES:
            actor_cfg = copy.deepcopy(train['actor'])
            actor_type = resolve_callable(actor_cfg.pop('class_name'))
            actor = actor_type(obs, train['obs_groups'], 'actor', wrapped.num_actions, **actor_cfg).to(args.device)
            actor.load_state_dict(saved['five_ppo']['algorithms'][role]['actor_state_dict'], strict=True)
            actual_sha = tensor_digest((name, p) for name, p in actor.named_parameters() if p.requires_grad)
            if actual_sha != saved['rank_agreement']['model_sha256'][role]['actor']:
                raise ValueError(f'Actor hash mismatch: {role}')
            actor.eval().requires_grad_(False)
            models.append(actor)
        command = env.command_manager.get_term('motion')
        if list(command.motion_files) != files:
            raise ValueError('Motion manifest ordering changed')
        metrics = (*METRICS, *global_tracking_metrics.METRICS)
        failure_names = [key for key in env.termination_manager.active_terms
                         if not env.termination_manager.get_term_cfg(key).time_out]
        result_meta = {
            'protocol': 'fixed_five_ppo_paired_training_dr_new_motion_draws_v1',
            'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': _sha256(args.checkpoint),
            'completed_training_updates': saved['five_ppo']['completed_updates'],
            'rank': rank, 'num_dr_worlds': len(selected),
            'sampled_training_world_indices': selected.tolist(),
            'class_counts': torch.bincount(labels[selected_ids], minlength=4).tolist(),
            'training_global_class_counts': meta['global_world_counts'][:4],
            'reconstruction_simulator_worlds': 2*n,
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'physics_sha256': physical_sha, 'physics_matches_training': True,
            'actor_hashes_match_checkpoint': True, 'physics_runtime_audit': runtime_physics,
            'reward_contract': rewards, 'termination': termination,
            'metric_names': list(metrics), 'global_metric_contract': global_tracking_metrics.contract(command),
            'failure_names': failure_names, 'motion_files': files,
            'policy': 'Deterministic actor mean; no sampling from policy Gaussian',
            'pairing': 'Identical fixed DR, motion, reference start, root-relative initial qpos and qvel; original independent force pulses and observation noise remain enabled',
            'scope': 'Training DR population and fixed labels, new random motion assignments from training catalog; not held-out-DR generalization',
            'aggregation': 'Per-world per-trial means with failures and coverage reported; errors stop at failure. Matched-prefix errors also stop when either arm ends.',
        }
        write_json(out / 'run_config.json', result_meta)
        del saved, snapshot
        all_trials = []
        begin = time.monotonic()
        with torch.inference_mode():
            for trial in range(args.trials):
                # Balanced motion counts across the population, independent of class IDs.
                motion_ids = np.resize(rng.permutation(len(files)), n)
                rng.shuffle(motion_ids)
                lengths = command.motion.file_lengths.cpu().numpy()[motion_ids]
                max_start = np.maximum(lengths - args.steps - 2, 0)
                starts = (rng.random(n) * (max_start + 1)).astype(np.int64)
                ids = torch.as_tensor(np.tile(motion_ids, 2), device=env.device)
                start = torch.as_tensor(np.tile(starts, 2), device=env.device)

                def sample(self, env_ids):
                    self.motion_idx[env_ids] = ids[env_ids]
                    self.motion_length[env_ids] = self.motion.file_lengths[ids[env_ids]]
                    self.time_steps[env_ids] = start[env_ids]

                command._uniform_sampling = MethodType(sample, command)
                _seed_everything(eval_seed + 10000000 + trial)
                obs, _ = wrapped.reset()
                qpos = env.sim.data.qpos.clone()
                qpos[:, :3] -= env.scene.env_origins
                torch.testing.assert_close(qpos[:n], qpos[n:], rtol=0, atol=2e-5)
                torch.testing.assert_close(env.sim.data.qvel[:n], env.sim.data.qvel[n:], rtol=0, atol=0)
                initial = command.time_steps.clone()
                torch.testing.assert_close(initial[:n], initial[n:], rtol=0, atol=0)
                horizon = (command.motion_length - initial - 1).clamp(1, args.steps)
                active = participants.clone()
                counts = torch.zeros(2*n, dtype=torch.long, device=env.device)
                returns = torch.zeros(2*n, dtype=torch.float64, device=env.device)
                totals = torch.zeros(2*n, len(metrics), dtype=torch.float64, device=env.device)
                common_totals = torch.zeros_like(totals)
                common_counts = torch.zeros(n, dtype=torch.long, device=env.device)
                failures = torch.zeros(2*n, dtype=torch.bool, device=env.device)
                failure_terms = torch.zeros(2*n, len(failure_names), dtype=torch.bool, device=env.device)
                for step in range(args.steps):
                    action = torch.zeros(2*n, wrapped.num_actions, device=env.device)
                    for group, actor in zip(groups, models, strict=True):
                        if len(group):
                            action[group] = actor(obs[group])
                    action.masked_fill_(~active[:, None], 0.)
                    env.action_manager.get_term('joint_pos').record_policy_mean(action)
                    common = active[:n] & active[n:]
                    obs, reward, dones, _ = wrapped.step(action)
                    if not torch.equal(command.time_steps[active], initial[active] + step + 1):
                        raise RuntimeError('Active reference timeline changed')
                    command._update_metrics()
                    values = torch.cat((torch.stack([command.metrics[k] for k in METRICS], -1),
                                        global_tracking_metrics.values(command)), -1)
                    if not bool(torch.isfinite(values[active]).all() & torch.isfinite(reward[active]).all()):
                        raise RuntimeError('Nonfinite active evaluation metrics')
                    totals[active] += values[active].double()
                    common_mask = common.repeat(2)
                    common_totals[common_mask] += values[common_mask].double()
                    common_counts += common.long()
                    returns[active] += reward[active].double()
                    counts += active.long()
                    failures |= active & env.reset_terminated
                    for i, name in enumerate(failure_names):
                        failure_terms[:, i] |= active & env.termination_manager.get_term(name)
                    active &= ~dones.bool() & (counts < horizon)
                    done_ids = dones.nonzero().flatten()
                    if len(done_ids):
                        # The existing evaluator uses the same private reset so
                        # surviving histories and motion references do not advance.
                        env._reset_idx(done_ids)
                        env.scene.write_data_to_sim()
                    if (step + 1) % 100 == 0:
                        progress = {'trial': trial, 'step': step + 1, 'active': int(active.sum()),
                                    'failures': int(failures.sum()), 'seconds': time.monotonic() - begin}
                        write_json(out / 'progress.json', progress)
                        print(json.dumps(progress), flush=True)
                    if not bool(active.any()):
                        break
                data = {
                    'class_id': labels[selected_ids].cpu().numpy(), 'motion_id': motion_ids[selected],
                    'training_world_id': selected,
                    'initial_step': initial[both_selected].cpu().numpy(), 'horizon': horizon[both_selected].cpu().numpy(),
                    'counts': counts[both_selected].cpu().numpy(), 'returns': returns[both_selected].cpu().numpy(),
                    'totals': totals[both_selected].cpu().numpy(), 'common_totals': common_totals[both_selected].cpu().numpy(),
                    'common_counts': common_counts[selected_ids].cpu().numpy(), 'failures': failures[both_selected].cpu().numpy(),
                    'failure_terms': failure_terms[both_selected].cpu().numpy(), 'metric_names': np.asarray(metrics),
                }
                np.savez_compressed(out / f'trial_{trial:02d}.npz', **data)
                all_trials.append({'trial': trial, 'failures': failures.reshape(2, n).sum(1).tolist(),
                                   'mean_reward': (returns/counts.clamp_min(1))[both_selected].reshape(2, -1).mean(1).tolist(),
                                   'coverage': (counts/horizon)[both_selected].reshape(2, -1).mean(1).tolist()})
                print(json.dumps(all_trials[-1]), flush=True)
        result_meta.update(trials=all_trials, wall_seconds=time.monotonic()-begin,
                           paired_initial_states_verified=True, reference_timeline_audited=True)
        write_json(out / 'result.json', result_meta)
    finally:
        env.close()


if __name__ == '__main__':
    main()
