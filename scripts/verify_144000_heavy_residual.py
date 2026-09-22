"""Verify the actual 8192-world payload-supervision runs and their saved artifacts."""

import argparse
import json
import math
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch

from run_144000_heavy_residual import (
    ROOT, COMPARISON, ARMS, CONTEXT, CONTEXT_SHA256, DIRECTORIES, REVISION, sha256,
)
from intact_tracking.residual_dr_aux import PAYLOAD_GROUPS

MANIFEST = '59b8e336c152e4773133bcdcd86cf86fd6545c3c0e6cc00ce67912d107c740ac'


def read(path):
    return json.loads(path.read_text())


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple,list)):
        return all(finite(v) for v in value)
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(value.isfinite().all())
    return not isinstance(value,float) or math.isfinite(value)


def write_report(name, checks, **details):
    result = dict(revision=REVISION, unix_time=time.time(), checks=checks,
                  passed=all(all(v.values()) for v in checks.values()), **details)
    COMPARISON.mkdir(parents=True, exist_ok=True)
    (COMPARISON/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'report':str(COMPARISON/name),'passed':result['passed'],
        'failures':{arm:[k for k,v in fields.items() if not v] for arm,fields in checks.items()}},indent=2),flush=True)
    assert result['passed']
    return result


def common_checks(config, state, rows, arm):
    mode = ARMS[arm][0]
    coefficient = .1 if arm == 'latent' else 0.
    agent = OmegaConf.to_container(state['cfg'].agent,resolve=True)
    aux = config['input_audit']['dr_auxiliary']
    runtime = config['physics']['runtime_audits_by_rank']
    supervised = ['com_x','com_y','com_z','friction', *PAYLOAD_GROUPS] if coefficient else []
    reference = read(ROOT/'runs'/('144000-exp-heavy-residual-'+arm)/'scale_4gpu16384/run_config.json')
    context = torch.load(CONTEXT,map_location='cpu',weights_only=False,mmap=True)
    payload = json.loads(json.dumps(context['heavy_payload_contract']))
    return {
        'finite_model_optimizer_metrics': finite(state['rsl_rl']) and finite(rows),
        'correct_mode': agent['actor']['latent_input_mode'] == agent['critic']['latent_input_mode'] == mode,
        'correct_encoder': config['context_sha256'] == CONTEXT_SHA256 == sha256(CONTEXT),
        'portable_checkpoint': all(k in state for k in ('frozen_context','frozen_tracker','inference_bundle_version')),
        'same_network_dimensions': config['input_audit']['actor_input_dimensions'] == 1994
            and config['input_audit']['critic_input_dimensions'] == 6679
            and config['input_audit']['actor_trainable_parameters'] == 1203366
            and config['input_audit']['critic_trainable_parameters'] == 7632385,
        '108_head_20_or_zero_supervised': aux['output_dim'] == 108 and aux['supervised_groups'] == supervised
            and aux['active_coordinates'] == (20 if coefficient else 0)
            and aux['coefficient'] == agent['algorithm']['dr_aux_coef'] == coefficient,
        'actual_loss_mode': all(r['loss']['AuxDR/coefficient'] == coefficient
            and r['loss']['AuxDR/enabled'] == bool(coefficient)
            and ('AuxDR/loss' in r['loss']) == bool(coefficient) for r in rows),
        'payload_metrics': all(sum(k.startswith('AuxDR/payload_') for k in r['loss']) == (32 if coefficient else 0) for r in rows),
        'excluded_targets_not_logged': all(not any(k.startswith(('AuxDR/kp_','AuxDR/kd_','AuxDR/armature_','AuxDR/mass_')) for k in r['loss']) for r in rows),
        'exploration_and_unbounded': agent['algorithm']['entropy_coef'] == .005
            and agent['actor']['initial_action_std'] == 1. and agent['actor']['residual_output_mode'] == 'unbounded'
            and agent['actor']['residual_scale'] == 1.,
        '8192_per_rank': config['distributed']['world_size'] == 4
            and config['distributed']['num_envs_per_rank'] == 8192 and config['distributed']['global_num_envs'] == 32768,
        'original_sampling_unchanged': config['motion_sampling'] == reference['motion_sampling']
            and len(config['sampling_runtime_audits']) == 4 and all(v['passed'] for v in config['sampling_runtime_audits']),
        'original_rewards_unchanged': config['reward_contract'] == reference['reward_contract'],
        'physical_stability': all(r['loss']['nominal_physics_max_error'] == 0 for r in rows),
        'balanced_256_groups': all(v['physics']['heavy_payload']['all_256_groups_present']
            and min(v['physics']['heavy_payload']['mass_group_counts']) == 28
            and max(v['physics']['heavy_payload']['mass_group_counts']) == 29 for v in runtime),
        'composite_mass_com_inertia': all(v['physics']['heavy_payload']['actual_composite_inertial_fields_verified']
            and v['physics']['heavy_payload']['nominal_inertial_fields_exact'] for v in runtime),
        'context_physics_contract': all(config['physics']['heavy_payload'][k] == payload[k]
            for k in ('max_masses_kg','com_half_width_m','nominal_fraction','mount_positions_body_m','sizes_m',
                      'mass_bins_per_limb','mass_groups','mass_sampling','com_sampling','inertia','lifetime')),
        'warp_memory_handoff': agent['release_cuda_cache_after_update'] and all(r['loss']['cuda_free_after_release_gib'] > 0 for r in rows),
        'latent_input_effect': (all(r['loss'][k] == 0 for r in rows for k in (
            'dynamics_latent_rms','latent_zero_action_delta_rms','latent_shuffle_action_delta_rms')) if mode == 'zero'
            else rows[-1]['loss']['dynamics_latent_rms'] > 0 and rows[-1]['loss']['latent_zero_action_delta_rms'] > 0),
    }


def matched_checks(configs, initial):
    left,right = (configs[a] for a in ARMS)
    return {
        'same_dataset': left['dataset'] == right['dataset'],
        'same_sampling': left['motion_sampling'] == right['motion_sampling'],
        'same_physics': left['physics'] == right['physics'],
        'same_initial_weights_and_shapes': all(torch.equal(v,initial['baseline'][key][name])
            for key in ('actor_state_dict','critic_state_dict') for name,v in initial['latent'][key].items()
            if not name.startswith('obs_normalizer.')),
        'same_critic_normalization_rule': left['critic_normalization_initialization'] == right['critic_normalization_initialization']
            and all(float(initial[a]['critic_state_dict']['obs_normalizer.count']) == 32768 for a in ARMS),
    }


def audit(phase):
    from monitor_memory350_nominal_direction import process_identity
    from monitor_memory350_residual import observe
    checks,configs,initial,details = {},{},{},{}
    for arm,(mode,gpus) in ARMS.items():
        root = ROOT/'runs'/('144000-exp-heavy-residual-'+arm)
        output = root/DIRECTORIES['smoke' if phase == 'smoke' else 'train']
        config = read(output/'run_config.json')
        state = torch.load(output/('checkpoint_final.pt' if phase == 'smoke' else 'checkpoint_0.pt'),
                           map_location='cpu',weights_only=False,mmap=True)
        first = torch.load(output/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
        rows = [json.loads(line) for line in (output/'metrics.jsonl').read_text().split('\n')[:-1] if line]
        checks[arm] = common_checks(config,state,rows,arm)
        checks[arm]['frozen_tracker'] = all(torch.equal(v,first['actor_state_dict'][k])
            for k,v in state['actor_state_dict'].items() if k.startswith('tracker.'))
        checks[arm]['shared_actor_updated'] = any(not torch.equal(v,first['actor_state_dict'][k])
            for k,v in state['actor_state_dict'].items() if k.startswith('residual_mlp.'))
        changed = (state['actor_state_dict']['dr_aux_head.weight'] != first['actor_state_dict']['dr_aux_head.weight']).any(dim=1)
        checks[arm]['head_updates_match_supervision'] = (changed.sum().item() == (20 if arm == 'latent' else 0))
        if phase == 'smoke':
            completion = read(output/'completion.json')
            export = read(output/'export_u25/policy.json')
            checks[arm].update(
                completed=completion['complete'] and completion['completed_updates'] == state['completed_updates'] == 25,
                all_ranks_agree=completion['distributed_parameter_agreement']['passed']
                    and completion['distributed_parameter_agreement']['world_size'] == 4,
                resumed=config['resume_state_audit']['passed'] and config['resume_state_audit']['checkpoint_update'] == 24,
                full_long_history=max(r['loss']['long_full_fraction'] for r in rows) > .5,
                onnx_verified=export['validation']['passed'] and export['latent_input_mode'] == mode and export['completed_updates'] == 25)
        else:
            record = read(Path(read(root/'latest_training_process.json')['process_record']))
            health = observe(root)
            wandb = read(output/'wandb_run.json')
            export = read(output/'policy.json')
            observer = read(root/'monitor/observer_process.json')
            exporter = read(root/'monitor/onnx_export/launch.json')
            previous = read(root/'scale_4gpu16384/run_config.json')
            checks[arm].update(
                fresh_unbounded=config['maximum_updates'] is None and config['arguments']['until_user_stop']
                    and not config['arguments']['bounded_smoke'] and config['arguments']['resume'] is None,
                four_workers=health['training_process']['live'] and len(health['worker_pids']) == 4,
                correct_gpus=record['physical_gpus'] == list(gpus),
                actual_updates=len(rows) >= 3 and rows[-1]['completed_updates'] >= 3,
                no_runtime_issues=not health['issues'],
                full_filtered_dataset=config['dataset'] == previous['dataset']
                    and config['dataset']['manifest_sha256'] == MANIFEST
                    and config['dataset']['loaded_motion_count'] == 220480
                    and config['dataset']['loaded_motion_counts_per_rank'] == [55120]*4,
                online_logging=bool(wandb['url']) and wandb['project'] == 'intact-preview-v2'
                    and wandb['group'] == '144000-exp-heavy' and config['arguments']['wandb_name'] == root.name,
                observer_live=process_identity(observer['pid'],observer['start_ticks'])['live'],
                exporter_live=process_identity(exporter['pid'],exporter['process_start_ticks'])['live'],
                onnx_verified=export['validation']['passed'] and export['latent_input_mode'] == mode)
            details[arm] = dict(health=health,wandb=wandb,completed_updates=rows[-1]['completed_updates'])
        configs[arm],initial[arm] = config,first
    checks['matched'] = matched_checks(configs,initial)
    if phase == 'launch':
        preflight = read(COMPARISON/'preflight_verification.json')
        checks['preflight'] = {'passed':preflight['passed'], 'source_unchanged':all(
            sha256(ROOT/name) == digest for name,digest in preflight['source_sha256'].items())}
    return write_report(f'{phase}_verification.json',checks,details=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('smoke','ready','launch'),required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase != 'ready':
        audit(args.phase)
        return
    smoke = audit('smoke')
    unit = read(COMPARISON/'unit_tests.json')
    assert unit['passed']
    paths = [*ROOT.glob('src/**/*.py'), ROOT/'scripts/run_144000_heavy_residual.py',
             ROOT/'scripts/watch_memory350_onnx.py', ROOT/'scripts/monitor_memory350_residual.py', Path(__file__).resolve()]
    result = {'passed':True,'revision':REVISION,'unix_time':time.time(),'smoke':smoke,'unit_tests':unit,
        'scope':'New 4x8192 batch/head/loss/resume/export smoke uses one motion. Full filtered catalog and runtime sampling are audited at formal launch.',
        'source_sha256':{str(path.relative_to(ROOT)):sha256(path) for path in paths}}
    (COMPARISON/'preflight_verification.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
