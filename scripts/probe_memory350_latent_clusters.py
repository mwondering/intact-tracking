"""Compare frozen Memory350 and v12 on the same fresh tracker trajectories."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from intact_tracking.memory350_inference import Memory350Inference, load_memory350_checkpoint
from intact_tracking.memory350_rollout import Memory350TrackerRollout
from intact_tracking.residual_context import DynamicsContextInference, load_frozen_context_checkpoint
from intact_tracking.rollout.mjlab_adapter import _sha256
from intact_tracking.rollout.online import FixedDRRolloutConfig, FixedDRTrackerRollout


@torch.inference_mode()
def without_long_memory(context):
    c, bank = context.checkpoint, context.memory
    results = []
    for start in range(0, bank.num_worlds, context.batch_size):
        ids = bank._worlds[start:start + context.batch_size]
        raw, valid = bank.ordered_short(ids)
        state = (raw[..., :71] - c.state_mean) / c.state_std
        action = (raw[..., 71:100] - c.action_mean) / c.action_std
        next_state = (raw[..., 100:] - c.state_mean) / c.state_std
        empty = torch.zeros(len(ids), 30, 10, 171, device=raw.device)
        empty_mask = torch.zeros(len(ids), 30, dtype=torch.bool, device=raw.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            results.append(c.encoder(state, action, next_state, valid, empty, empty_mask).float())
    return torch.cat(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--reference-probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=['common', 'memory_training'], required=True)
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--seed', type=int, default=81208)
    parser.add_argument('--checkpoint-label', default='Memory350')
    parser.add_argument('--save-history', action='store_true',
                        help='Cache raw query histories for exact future checkpoint comparisons.')
    parser.add_argument('--comparison-checkpoint', action='append', default=[],
                        metavar='NAME=PATH', help='Additional Memory350 encoder on the same raw memory bank.')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty output: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    reference = json.loads((args.reference_probe / 'metadata.json').read_text())
    source = reference['rollout_config']
    model = load_memory350_checkpoint(args.checkpoint, device='cuda:0')
    old = load_frozen_context_checkpoint(args.baseline, device='cuda:0',
                                         expected_tracker_sha256=model.tracker_sha256)
    comparisons = {}
    for spec in args.comparison_checkpoint:
        name, path = spec.split('=', 1)
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name in {'v12', 'no_long', 'memory350', 'memory350_without_long'} or name in comparisons:
            raise ValueError(f'Invalid or duplicate comparison name: {name}')
        comparisons[name] = load_memory350_checkpoint(
            path, device='cuda:0', expected_tracker_sha256=model.tracker_sha256)
    config = FixedDRRolloutConfig(
        checkpoint_file=source['checkpoint_file'], motion_path=source['motion_path'],
        motion_file=source.get('motion_file'), task_id=source.get('task_id'),
        num_envs=args.num_envs, device='cuda:0', seed=args.seed,
        stochastic_policy=source['stochastic_policy'],
        randomize_initial_episode_phase=source['randomize_initial_episode_phase'],
        nominal_fraction=.5 if args.profile == 'common' else 0.,
        tracker_dr_plus_limb_payload=args.profile == 'memory_training',
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.monotonic()
    rollout_type = FixedDRTrackerRollout if args.profile == 'common' else Memory350TrackerRollout
    print(json.dumps({'event': 'initializing', 'profile': args.profile, 'config': asdict(config)}), flush=True)
    with rollout_type(config) as rollout:
        assert _sha256(rollout.checkpoint_path) == model.tracker_sha256 == old.tracker_sha256
        assert list(rollout.motion_files) == reference['motion_files']
        context = Memory350Inference(model, args.num_envs, batch_size=256, use_bfloat16=True)
        comparison_contexts = {}
        for name, checkpoint in comparisons.items():
            other = Memory350Inference(checkpoint, args.num_envs, batch_size=256, use_bfloat16=True)
            other.memory = context.memory
            comparison_contexts[name] = other
        baseline = DynamicsContextInference(old, num_envs=args.num_envs, device='cuda:0', use_bfloat16=True)
        metadata = {
            'checkpoint': model.path, 'checkpoint_sha256': model.sha256,
            'checkpoint_update': int(args.checkpoint.stem.split('_')[-1]),
            'checkpoint_label': args.checkpoint_label,
            'models': {'memory350': 'latent', **{k: 'latent_' + k for k in comparisons},
                       'v12': 'latent_v12', 'memory350_without_long': 'latent_no_long'},
            'comparison_checkpoints': {k: {'path': c.path, 'sha256': c.sha256,
                'update': int(Path(c.path).stem.split('_')[-1])} for k, c in comparisons.items()},
            'baseline_checkpoint': old.path, 'baseline_checkpoint_sha256': old.sha256,
            'tracker_sha256': model.tracker_sha256, 'profile': args.profile,
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'rollout_config': asdict(config), 'rollout_metadata': rollout.metadata,
            'motion_files': list(rollout.motion_files),
            'motion_lengths': rollout.motion_command.motion.file_lengths.cpu().tolist(),
            'context_history_steps': 50, 'memory_chunks': 30, 'memory_chunk_steps': 10,
            'contract': 'Frozen models on identical physical transitions. Memory350 retains completed long chunks across motion/episode resets; v12 clears all history. Main queries every 100 steps; long histories may overlap. Full-bank and conservative nonoverlapping subsets are analyzed separately. No latent controls the tracker.',
            'complete': False,
        }
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        np.savez_compressed(args.output / 'physics.npz', values=rollout.privileged_dynamics.cpu().numpy(),
                            names=np.asarray(rollout.privileged_dynamics_names))
        rows, coverage = [], []
        numerical_check = None
        collection_started = time.monotonic()
        with torch.inference_mode():
            for step in range(1, args.steps + 1):
                batch = rollout.step(predictor_only=True)
                context.append(batch)
                baseline.append(batch['robot_state'], batch['joint_target'], batch['reset_boundary'])
                if step % args.stride == 0:
                    bank = context.memory
                    chunks = (bank.total_chunks - bank.session_start).clamp_max(30)
                    z = context.encode()
                    old_z = baseline.encode(batch['next_robot_state'])
                    no_long = without_long_memory(context)
                    comparison_z = {name: other.encode() for name, other in comparison_contexts.items()}
                    if numerical_check is None and bool((chunks == 30).any()):
                        context.use_bfloat16 = False
                        precise = context.encode()
                        context.use_bfloat16 = True
                        mask = (chunks == 30) & (bank.short_count == 50)
                        numerical_check = {
                            'full_memory_samples': int(mask.sum()),
                            'bf16_fp32_element_rms_difference': float((z[mask] - precise[mask]).square().mean().sqrt()),
                            'bf16_fp32_mean_cosine': float(torch.nn.functional.cosine_similarity(z[mask], precise[mask]).mean()),
                        }
                        numerical_check['comparisons'] = {}
                        for name, other in comparison_contexts.items():
                            other.use_bfloat16 = False
                            comparison_precise = other.encode()
                            other.use_bfloat16 = True
                            numerical_check['comparisons'][name] = {
                                'bf16_fp32_mean_cosine': float(torch.nn.functional.cosine_similarity(
                                    comparison_z[name][mask], comparison_precise[mask]).mean())}
                    if bool((chunks == 0).any()):
                        # Masking an already-empty memory must leave outputs unchanged.
                        torch.testing.assert_close(z[chunks == 0], no_long[chunks == 0], rtol=0, atol=0)
                    motion = rollout.motion_command.motion_idx
                    motion_step = rollout.motion_command.time_steps
                    lengths = rollout.motion_command.motion.file_lengths[motion]
                    row = {
                        'latent': z.cpu().numpy(), 'latent_v12': old_z.cpu().numpy(),
                        'latent_no_long': no_long.cpu().numpy(),
                        **{'latent_' + name: values.cpu().numpy() for name, values in comparison_z.items()},
                        'world': rollout.world_ids.cpu().numpy().copy(),
                        'episode': rollout.episode_ids.cpu().numpy().copy(),
                        'nominal': rollout.is_nominal.cpu().numpy().copy(),
                        'motion': motion.cpu().numpy().copy(),
                        'motion_step': motion_step.cpu().numpy().copy(),
                        'phase': (motion_step / (lengths - 1).clamp_min(1)).cpu().numpy(),
                        'step': np.full(args.num_envs, step, dtype=np.int64),
                        'neighbor': np.zeros(args.num_envs, dtype=bool),
                        'short_steps': bank.short_count.cpu().numpy().copy(),
                        'long_chunks': chunks.cpu().numpy(),
                        'total_chunks': bank.total_chunks.cpu().numpy().copy(),
                        'pending_steps': bank.pending_count.cpu().numpy().copy(),
                        'memory_session': bank.session.cpu().numpy().copy(),
                        'v12_steps': baseline.history_valid.sum(0).cpu().numpy(),
                    }
                    if args.save_history:
                        history_dir = args.output / 'queries'
                        history_dir.mkdir(exist_ok=True)
                        raw_short, short_valid = bank.ordered_short()
                        raw_long, long_valid = bank.read_chunks(bank._worlds)
                        torch.save({
                            'format_version': 'memory350_raw_query_v1',
                            'step': step,
                            'short': raw_short.cpu(), 'short_valid': short_valid.cpu(),
                            'long': raw_long.cpu(), 'long_valid': long_valid.cpu(),
                            'world': torch.from_numpy(row['world']),
                        }, history_dir / f'query_{step:06d}.pt')
                        del raw_short, short_valid, raw_long, long_valid
                    for key in metadata['models'].values():
                        if not np.isfinite(row[key]).all():
                            raise ValueError(f'Nonfinite {key}')
                    rows.append(row)
                    status = {'event': 'collection', 'step': step, 'total_steps': args.steps,
                              'seconds': time.monotonic() - collection_started,
                              'short50_long30': int(((bank.short_count == 50) & (chunks == 30)).sum()),
                              'common_full': int(((baseline.history_valid.sum(0) == 100) & (chunks == 30)).sum()),
                              **bank.metrics()}
                    coverage.append(status)
                    print(json.dumps(status), flush=True)
                if (step % 500 == 0 or step == args.steps) and rows:
                    np.savez_compressed(args.output / 'latents.npz', **{
                        key: np.concatenate([row[key] for row in rows]) for key in rows[0]
                    })
        rollout._assert_fixed_dr()
        if context.memory.parameter_invalidations != 0:
            raise ValueError('Unexpected physics-session change')
        metadata.update(complete=True, coverage=coverage, numerical_check=numerical_check,
                        elapsed_seconds=time.monotonic() - started,
                        collection_seconds=time.monotonic() - collection_started,
                        rollout_final_metadata=rollout.metadata)
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({'event': 'complete', 'output': str(args.output),
                          'elapsed_seconds': metadata['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
