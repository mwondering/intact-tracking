"""Collect one independent mjwarp shard for the 16384-world DR encoder audit."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from intact_tracking.memory350_inference import Memory350Inference, load_memory350_checkpoint
from intact_tracking.memory350_rollout import Memory350TrackerRollout
from intact_tracking.memory350_nominal_rollout import NominalMemory350TrackerRollout, NominalMemory350RolloutConfig
from intact_tracking.rollout.online import FixedDRRolloutConfig
from intact_tracking.rollout.mjlab_adapter import _sha256


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--dr-worlds', type=int, default=2048)
    parser.add_argument('--with-nominal', action='store_true')
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--output-root', type=Path, default=Path('runs/limb_context_20260916_dr16384'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    torch.set_float32_matmul_precision('high')
    out = args.output_root/f'shard_{args.shard:02d}'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seed = 91616384 + 1009*args.shard
    torch.manual_seed(seed)
    np.random.seed(seed)
    cp = Path('runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/update_035857.pt')
    reference = json.loads(Path('runs/limb_context_20260915_dr_center_hand2p5_shin4/cluster_check_001500/capped_loads/metadata.json').read_text())
    source = reference['rollout_config']
    schema = json.loads((cp.parent/'run_config.json').read_text())['dr_center_contract']['schema']
    checkpoint = load_memory350_checkpoint(cp, device='cuda:0')
    assert checkpoint.sha256 == '3c489cf5457889bf88441175ec4f10f4448104f44f7914eaaf762a9c3659d8f7'
    count = args.dr_worlds * (2 if args.with_nominal else 1)
    config_type = NominalMemory350RolloutConfig if args.with_nominal else FixedDRRolloutConfig
    config = config_type(checkpoint_file=source['checkpoint_file'], motion_path=source['motion_path'],
                         motion_file=source.get('motion_file'), task_id=source.get('task_id'),
                         num_envs=count, device='cuda:0', seed=seed,
                         stochastic_policy=False, randomize_initial_episode_phase=True,
                         nominal_fraction=.5 if args.with_nominal else 0.,
                         tracker_dr_plus_limb_payload=True, limb_max_masses_kg=(2.5, 2.5, 4., 4.))
    rollout_type = NominalMemory350TrackerRollout if args.with_nominal else Memory350TrackerRollout
    print(json.dumps({'event': 'initializing', 'shard': args.shard, 'config': asdict(config)}), flush=True)
    with rollout_type(config) as rollout:
        assert checkpoint.tracker_sha256 == _sha256(rollout.checkpoint_path)
        assert list(rollout.motion_files) == reference['motion_files']
        assert list(rollout.privileged_dynamics_names) == schema['names']
        physics = rollout.privileged_dynamics.cpu().numpy().copy()
        nominal = rollout.is_nominal.cpu().numpy().copy()
        assert int((~nominal).sum()) == args.dr_worlds
        unit = (physics-np.asarray(schema['lower']))/(np.asarray(schema['upper'])-schema['lower'])
        assert np.isfinite(unit).all() and unit.min() > -1e-5 and unit.max() < 1+1e-5
        world_ids = np.arange(count, dtype=np.int64) + args.shard*100000
        np.savez_compressed(out/'physics.npz', values=physics, names=np.asarray(schema['names']),
                            nominal=nominal, world=world_ids)
        metadata = {'complete': False, 'shard': args.shard, 'seed': seed,
                    'checkpoint': {'path': checkpoint.path, 'sha256': checkpoint.sha256},
                    'tracker_sha256': checkpoint.tracker_sha256,
                    'motion_files': list(rollout.motion_files),
                    'rollout_config': asdict(config), 'rollout_metadata': rollout.metadata,
                    'schema': schema, 'dr_worlds': args.dr_worlds, 'nominal_worlds': int(nominal.sum()),
                    'protocol': 'Frozen original tracker, identical previous full background DR plus independent limb loads, hand maxima2.5/shin4 kg; independent seed per shard. Physical parameters fixed for whole collection, motion resets at original 1000-step cap. Actual raw Memory350 histories; no latent influences actions. Query every100 steps; windows may overlap. All raw queries saved for audit. Only full short50/long30 histories used downstream.',
                    'script_sha256': _sha256(Path(__file__)), 'queries': []}
        write_json(out/'metadata.json', metadata)
        context = Memory350Inference(checkpoint, count, batch_size=256, use_bfloat16=True)
        rows = []
        full_counts = np.zeros(count, np.int64)
        query_dir = out/'queries'
        query_dir.mkdir()
        for step in range(1, args.steps+1):
            batch = rollout.step(predictor_only=True)
            context.append(batch)
            if step % 50 == 0:
                write_json(out/'progress.json', {'step': step, 'total_steps': args.steps,
                           'seconds': time.monotonic()-started, 'dr_worlds': args.dr_worlds,
                           'full_queries_min_per_world': int(full_counts.min()), 'complete': False})
            if step % 100:
                continue
            bank = context.memory
            chunks = (bank.total_chunks-bank.session_start).clamp_max(30)
            latent = context.encode()
            assert torch.isfinite(latent).all()
            full = ((bank.short_count == 50) & (chunks == 30)).cpu().numpy()
            full_counts += full
            row = {'latent': latent.cpu().numpy(), 'world': world_ids.copy(),
                   'episode': rollout.episode_ids.cpu().numpy().copy(),
                   'motion': rollout.motion_command.motion_idx.cpu().numpy().copy(),
                   'motion_step': rollout.motion_command.time_steps.cpu().numpy().copy(),
                   'step': np.full(count, step, dtype=np.int32), 'nominal': nominal.copy(),
                   'short_steps': bank.short_count.cpu().numpy().copy(),
                   'long_chunks': chunks.cpu().numpy().copy(),
                   'total_chunks': bank.total_chunks.cpu().numpy().copy(),
                   'pending_steps': bank.pending_count.cpu().numpy().copy(),
                   'memory_session': bank.session.cpu().numpy().copy()}
            rows.append(row)
            short, valid = bank.ordered_short()
            long, long_valid = bank.read_chunks()
            path = query_dir/f'query_{step:06d}.pt'
            torch.save({'format_version': 'memory350_raw_query_v1', 'step': step,
                        'world': torch.from_numpy(world_ids), 'short': short.cpu(), 'short_valid': valid.cpu(),
                        'long': long.cpu(), 'long_valid': long_valid.cpu()}, path)
            metadata['queries'].append({'path': str(path.resolve()), 'step': step, 'sha256': _sha256(path)})
            del short, valid, long, long_valid
            if step == 500:
                context.use_bfloat16 = False
                precise = context.encode()
                context.use_bfloat16 = True
                mask = torch.from_numpy(full).to(latent.device)
                metadata['numerical_audit'] = {
                    'full_windows': int(full.sum()),
                    'bf16_fp32_unit_latent_l2_mean': float(torch.linalg.vector_norm(
                        torch.nn.functional.normalize(latent[mask], dim=-1)-torch.nn.functional.normalize(precise[mask], dim=-1), dim=-1).mean()),
                    'bf16_fp32_unit_latent_l2_max': float(torch.linalg.vector_norm(
                        torch.nn.functional.normalize(latent[mask], dim=-1)-torch.nn.functional.normalize(precise[mask], dim=-1), dim=-1).max())}
                del precise
            if step % 400 == 0 or step == args.steps:
                np.savez_compressed(out/'latents.npz', **{key: np.concatenate([r[key] for r in rows]) for key in rows[0]})
                write_json(out/'metadata.json', metadata)
            print(json.dumps({'event': 'query', 'shard': args.shard, 'step': step,
                              'full_windows': int(full.sum()), 'minimum_full_queries': int(full_counts.min()),
                              'seconds': time.monotonic()-started}), flush=True)
        rollout._assert_fixed_dr()
        assert context.memory.parameter_invalidations == 0
        np.testing.assert_array_equal(physics, rollout.privileged_dynamics.cpu().numpy())
        assert full_counts.min() >= 16, full_counts.min()
        metadata.update(complete=True, full_query_counts=full_counts.tolist(),
                        physics_unchanged=True, memory_metrics=context.memory.metrics(),
                        elapsed_seconds=time.monotonic()-started,
                        physics_sha256=_sha256(out/'physics.npz'), latents_sha256=_sha256(out/'latents.npz'))
        write_json(out/'metadata.json', metadata)
        write_json(out/'progress.json', {'step': args.steps, 'total_steps': args.steps,
                   'seconds': time.monotonic()-started, 'complete': True})
        print(json.dumps({'event': 'complete', 'shard': args.shard, 'seconds': metadata['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
