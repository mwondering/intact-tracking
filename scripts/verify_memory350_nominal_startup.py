"""Check the real four-rank nominal50 training artifacts after optimizer startup."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    project = Path(__file__).resolve().parents[1]
    root = args.run_root.resolve()
    run = (root / 'stage1').resolve()
    config = json.loads((run / 'run_config.json').read_text())
    progress = json.loads((run / 'progress.json').read_text())
    original = json.loads((project / 'runs/limb_context_20260909_memory350/stage1_8192/run_config.json').read_text())
    assert progress['completed_updates'] >= 10
    assert progress['optimizer_steps'] == 4 * progress['completed_updates']
    assert progress['unbounded']
    assert config['arguments']['nominal_fraction'] == .5
    assert config['arguments']['resume'] is None
    assert config['model'] == original['model']
    assert config['loss'] == original['loss']
    assert config['optimization'] == original['optimization']
    same_arguments = ('motion_path', 'seed', 'batch_size', 'micro_batch_size', 'warmup_steps',
                      'gradient_steps_per_update', 'replay_capacity', 'replay_sampling',
                      'context_history_steps', 'context_depth', 'context_dim',
                      'representation_weight', 'representation_relation_weight', 'response_distance_scale',
                      'updates', 'until_user_stop', 'model_learning_rate', 'weight_decay')
    for key in same_arguments:
        assert config['arguments'][key] == original['arguments'][key], key
    assert config['dataset']['loaded_global_motion_count'] == 129827
    assert config['dataset']['loaded_global_frames'] == 48085337
    ranks = config['dataset']['runtime_audits_by_rank']
    assert len(ranks) == 4
    assert {r['physical_gpu'] for r in ranks} == {'0', '1', '2', '3'}
    counts = []
    for rank in ranks:
        assert rank['nominal_training_worlds'] == rank['dr_training_worlds'] == rank['training_worlds'] // 2
        assert rank['nominal_validation_worlds'] == rank['dr_validation_worlds'] == rank['validation_worlds'] // 2
        mixture = rank['physics']['nominal_mixture']
        assert mixture['nominal_count'] == mixture['dr_count'] == rank['num_envs'] // 2
        assert mixture['nominal_restore']['model_field_max_abs_error'] == 0
        assert mixture['nominal_restore']['encoder_bias_max_abs_error'] == 0
        assert mixture['nominal_payload_max_abs_kg'] == 0
        assert mixture['dr_physics_unchanged_from_original_sampling']
        assert mixture['nominal_pulses_disabled']
        counts.append({key: rank[key] for key in ['rank', 'physical_gpu', 'num_envs',
                      'nominal_training_worlds', 'dr_training_worlds',
                      'nominal_validation_worlds', 'dr_validation_worlds']})
    checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    assert checkpoint['nominal_a_fraction'] == .5
    assert checkpoint['training_mixture_version'] == 'memory350_nominal50_v1'
    assert checkpoint['update'] >= 1
    assert checkpoint['optimizer_steps'] == checkpoint['update'] * 4
    agreement = checkpoint['distributed_parameter_agreement']
    assert agreement['passed'] and len(set(agreement['sha256_by_rank'])) == 1
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    norm_worlds = set(checkpoint['normalization']['world_ids'])
    n = config['arguments']['num_envs']
    train_n = n - config['arguments']['validation_worlds']
    assert norm_worlds == {rank * n + i for rank in range(4) for i in range(train_n)}
    validations = []
    for rank in range(4):
        path = run / f'validation_broad_rank_{rank}.pt'
        batch = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        world = batch['world_id']
        assert ((world >= rank * n + train_n) & (world < (rank + 1) * n)).all()
        assert torch.equal(batch['is_nominal'], world.remainder(2).eq(0))
        assert batch['is_nominal'].any() and (~batch['is_nominal']).any()
        assert not norm_worlds.intersection(world.tolist())
        validations.append({'rank': rank, 'samples': len(world),
                            'nominal_samples': int(batch['is_nominal'].sum()),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    logs = [json.loads(line) for line in (run / 'metrics.jsonl').open()]
    first = logs[0]
    assert first['update'] == 1 and first['optimizer_steps'] == 4
    assert all(torch.isfinite(torch.tensor(v)) for v in first['fixed_probe'].values())
    assert 'nominal_five_step_nmse' in first['fixed_probe']
    assert 'nominal_counterfactual_rms' in first['fixed_probe']
    assert all(first['memory'][f'gradient_norm_{name}'] > 0
               for name in ['chunk_encoder', 'long_encoder', 'final_encoder'])
    remote = None
    if args.wandb:
        os.environ['WANDB_API_KEY'] = (project / '.runtime/limb_context/wandb_api_key').read_text().strip()
        import wandb
        remote_run = wandb.Api(timeout=30).run('2486344338-zhejiang-university/intact-forward-predictor/' + checkpoint['wandb']['id'])
        assert remote_run.config['arguments']['nominal_fraction'] == .5
        remote = {'url': remote_run.url, 'state': remote_run.state,
                  'last_step': remote_run.summary.get('_step')}
        assert remote['last_step'] is not None and remote['last_step'] >= 1
    result = {'passed': True, 'run': str(run), 'progress_at_verification': progress,
              'checkpoint_update': checkpoint['update'], 'original_architecture_and_loss_preserved': True,
              'initialization': 'from scratch; first logged update=1 and optimizer_steps=4; no resume',
              'rank_counts': counts, 'validation': validations,
              'normalization_worlds': len(norm_worlds), 'validation_excluded_from_normalization': True,
              'ddp_parameter_agreement': agreement, 'first_probe': first['fixed_probe'],
              'wandb': remote or checkpoint['wandb']}
    (root / 'startup_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['first_probe', 'validation', 'ddp_parameter_agreement']}, indent=2))


if __name__ == '__main__':
    main()
