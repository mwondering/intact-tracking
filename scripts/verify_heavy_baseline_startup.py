"""Read-only verification of a live continuous-uniform heavy baseline on its host."""

import argparse
import json
from pathlib import Path
import socket
import time

import torch

from monitor_memory350_nominal_direction import process_identity
from verify_144000_heavy_residual import finite


def read(path):
    return json.loads(path.read_text())


def verify(root):
    active = read(root/'active_process.json')
    controller = read(root/'controller_process.json')
    output = root/'continuous_uniform'
    cfg = read(output/'run_config.json')
    rows = [json.loads(x) for x in (output/'metrics.jsonl').read_text().split('\n')[:-1] if x]
    reference = read(Path(__file__).resolve().parents[1]/
        'runs/144000-exp-heavy-residual-baseline/ppo_4gpu8192_resume2000_uniform_mass5x/run_config.json')
    wandb = read(output/'wandb_run.json')
    preflight = read(root/'preflight.json')
    if socket.gethostname() != controller['hostname']:
        raise RuntimeError('Run this verifier on the training host, not another shared-filesystem host')
    worker_pids = [int(x) for x in Path(f'/proc/{active["pid"]}/task/{active["pid"]}/children').read_text().split()]
    worker_gpu_env = {}
    for pid in worker_pids:
        values = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
        worker_gpu_env[pid] = next((x.partition(b'=')[2].decode() for x in values
                                   if x.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
    sampling = cfg['motion_sampling']
    initial = torch.load(output/'checkpoint_initial.pt', map_location='cpu', weights_only=False, mmap=True)
    checks = {
        'requested_gpus':active['gpus'] == controller['gpus'] and len(worker_pids) == 4
            and all(v == ','.join(map(str, controller['gpus'])) for v in worker_gpu_env.values()),
        'controller_trainer_and_workers_live':process_identity(controller['pid'], controller['start_ticks'])['live']
            and process_identity(active['pid'], active['start_ticks'])['live']
            and all(process_identity(pid)['live'] for pid in worker_pids),
        'continuous_uniform_no_cap':cfg['baseline_stage'] == 'continuous_uniform'
            and cfg['maximum_updates'] is None and cfg['arguments']['until_user_stop'],
        'no_2000_restart':cfg['baseline_contract']['stage1_boundary_completed_updates'] is None
            and cfg['baseline_contract']['stage2_first_completed_update'] is None and cfg['sampling_reset_generation'] == 0,
        'uniform_without_rewind':sampling['requested_mode'] == sampling['active_mode'] == 'uniform'
            and not sampling['failure_rewind_enabled'] and not sampling['rewind']['enabled'],
        'four_rank_runtime_sampling':len(cfg['sampling_runtime_audits']) == 4
            and all(x['passed'] and x['active_mode'] == 'uniform' and not x['failure_rewind_enabled']
                    for x in cfg['sampling_runtime_audits']),
        'actual_uniform_updates':all(not x['motion_sampling']['adaptive_enabled']
                                    and not x['motion_sampling']['failure_rewind_enabled'] for x in rows),
        'full_filtered_dataset':cfg['dataset'] == reference['dataset']
            and cfg['dataset']['motion_count'] == cfg['dataset']['loaded_motion_count'] == 220480,
        'matched_physics':cfg['physics'] == reference['physics'],
        'matched_rewards':cfg['reward_contract'] == reference['reward_contract'],
        'matched_terminations':cfg['training_terminations'] == reference['training_terminations'],
        'matched_tracker':cfg['tracker_sha256'] == reference['tracker_sha256'],
        '4x8192':cfg['distributed']['world_size'] == 4 and cfg['distributed']['num_envs_per_rank'] == 8192
            and cfg['distributed']['global_num_envs'] == 32768,
        'no_pretrained_context_or_aux_loss':cfg['context_checkpoint'] is None
            and not cfg['input_audit']['dr_auxiliary']['enabled'],
        'fresh_formal_initialization':initial['completed_updates'] == 0
            and cfg['input_audit']['identical_initial_action'] and cfg['input_audit']['action_std_initialization'] == 1.,
        'no_sampler_state':initial['motion_sampling_state'] is None and not (output/'sampling_state').exists(),
        'portable_frozen_tracker':'frozen_tracker' in initial,
        'contiguous_updates':len(rows) >= 5
            and [x['completed_updates'] for x in rows] == list(range(1, len(rows)+1)),
        'finite':finite(rows) and finite(initial['rsl_rl']),
        'online_wandb':bool(wandb['url']) and wandb['group'] == '144000-exp-heavy' and wandb['project'] == 'intact-preview-v2',
        'same_scale_preflight':preflight['passed'] and preflight['motion_sampling'] == 'uniform',
    }
    if cfg['method'] == 'rma_teacher':
        checks['teacher108_input'] = len(cfg['physics_input_schema']['names']) == 108 and cfg['extra_current_state_physics_privilege']
    elif cfg['method'] == 'any2track':
        wm = cfg['baseline_contract']['world_model']
        checks['history_conditioning_without_physical_inputs'] = (
            not cfg['extra_current_state_physics_privilege'] and wm['history'] == 79 and wm['embedding'] == 128)
        checks['encoder_frozen_during_ppo'] = cfg['encoder_frozen'] and cfg['encoder_freezing_scope'].startswith('PPO only')
        checks['world_model_updates_each_rollout'] = wm['horizon'] == 20 and all(
            row['loss']['WorldModel/optimizer_steps'] == row['completed_updates'] * wm['epochs'] * wm['mini_batches']
            and row['loss']['WorldModel/update_seconds'] > 0 for row in rows)
        checks['world_model_resume_state_saved'] = all(key in initial['rsl_rl'] for key in (
            'world_model_state_dict', 'world_optimizer_state_dict', 'world_optimizer_steps', 'world_window_rng'))
        checks['world_model_starts_from_zero_updates'] = initial['rsl_rl']['world_optimizer_steps'] == 0
    recent = rows[-5:]
    report = {'passed':all(checks.values()), 'checked_at':time.time(), 'hostname':socket.gethostname(),
        'checks':checks, 'completed_updates':rows[-1]['completed_updates'], 'wandb':wandb,
        'controller_pid':controller['pid'], 'trainer_pid':active['pid'], 'worker_pids':worker_pids,
        'worker_cuda_visible_devices':worker_gpu_env, 'last_metrics':rows[-1],
        'recent_update_seconds':sum(r['collect_seconds']+r['learn_seconds'] for r in recent)/len(recent)}
    (root/'startup_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed':report['passed'], 'failed':[k for k,v in checks.items() if not v],
        'completed_updates':report['completed_updates'], 'wandb':wandb,
        'seconds_per_update':report['recent_update_seconds']}, indent=2))
    if not report['passed']:
        raise RuntimeError('Heavy baseline startup verification failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify(args.root)
