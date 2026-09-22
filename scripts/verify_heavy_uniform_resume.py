"""Verify actual uniform resumes before and after the full-dataset launch."""

import argparse
import json
import time

import torch

from resume_144000_heavy_uniform import ROOT, COMPARISON, ARMS, command_for
from run_144000_heavy_residual import sha256
from verify_144000_heavy_residual import finite
from intact_tracking.limb_context_sampling import state_digest


def read(path):
    return json.loads(path.read_text())


def audit(phase):
    checks, details, configurations = {}, {}, {}
    for arm, (mode, gpus) in ARMS.items():
        _, output, parent = command_for(arm, phase)
        config = read(output/'run_config.json')
        old = torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
        resumed = torch.load(output/'checkpoint_resume.pt',map_location='cpu',weights_only=False,mmap=True)
        rows = [json.loads(line) for line in (output/'metrics.jsonl').read_text().split('\n')[:-1] if line]
        source_update = old['completed_updates']
        sampling, audits = config['motion_sampling'], config['sampling_resume_audits']
        aux = config['input_audit']['dr_auxiliary']
        coef = .5 if arm == 'latent' else 0.
        checks[arm] = {
            'exact_model_optimizer_normalizer_restore':state_digest(resumed['rsl_rl']) == state_digest(old['rsl_rl']),
            'restore_audit':config['resume_state_audit']['passed'],
            'same_dataset':config['dataset'] == old['residual_policy']['dataset'],
            'same_physics':config['physics'] == json.loads(json.dumps(old['residual_policy']['physics'])),
            'same_rewards':config['reward_contract'] == old['residual_policy']['reward_contract'],
            'frozen_context_unchanged':config['context_sha256'] == old['residual_policy']['context_sha256'],
            'uniform_without_rewind':sampling['requested_mode'] == sampling['active_mode'] == 'uniform'
                and not sampling['failure_rewind_enabled'] and not sampling['rewind']['enabled'],
            'runtime_uniform_audits':len(config['sampling_runtime_audits']) == 4
                and all(a['passed'] and a['active_mode']=='uniform' and not a['failure_rewind_enabled'] for a in config['sampling_runtime_audits']),
            'no_adaptive_state_in_checkpoint':resumed['motion_sampling_state'] is None and not (output/'sampling_state').exists(),
            'old_sampler_not_loaded':len(audits)==4 and all(not a['restored'] and a['adaptive_statistics_discarded'] for a in audits),
            'eight_or_zero_supervised_coordinates':aux['active_coordinates'] == (8 if coef else 0)
                and aux['coefficient']==coef and not aux['payload_com_supervised'],
            'unchanged_training_scale':config['distributed']['num_envs_per_rank']==8192 and config['distributed']['global_num_envs']==32768,
            'portable_resume':all(k in resumed for k in ('frozen_tracker','frozen_context','inference_bundle_version')),
            'updates_continue':len(rows)>=3 and rows[0]['completed_updates']==source_update+1
                and [r['completed_updates'] for r in rows]==list(range(source_update+1,rows[-1]['completed_updates']+1)),
            'actual_uniform_updates':all(not r['motion_sampling']['adaptive_enabled'] and not r['motion_sampling']['failure_rewind_enabled'] for r in rows),
            'finite':finite(rows) and finite(resumed['rsl_rl']),
            'auxiliary_objective_preserved':all(r['loss']['AuxDR/coefficient']==coef
                and not any(k.startswith('AuxDR/payload_com_') for k in r['loss']) for r in rows),
        }
        if mode=='zero':
            checks[arm]['zero_latent']=all(r['loss'][k]==0 for r in rows for k in
                ('dynamics_latent_rms','latent_zero_action_delta_rms','latent_shuffle_action_delta_rms'))
        if phase=='smoke':
            completion=read(output/'completion.json')
            checks[arm]['completed_four_rank_smoke']=completion['complete'] and completion['completed_updates']==28
            checks[arm]['distributed_agreement']=completion['distributed_parameter_agreement']['passed']
        else:
            from monitor_memory350_residual import observe
            from monitor_memory350_nominal_direction import process_identity
            health=observe(output.parent)
            record=read(output.parent/f'train_resume2000_uniform_mass5x_process.json')
            wandb=read(output/'wandb_run.json')
            observer=read(output.parent/'monitor/observer_process.json')
            exporter=read(output.parent/'monitor/onnx_export/launch.json')
            checks[arm].update(
                original_2000_checkpoint=source_update==2001,
                no_cap=config['maximum_updates'] is None and config['arguments']['until_user_stop'],
                full_dataset=config['dataset']['motion_count']==config['dataset']['loaded_motion_count']==220480,
                four_healthy_workers=health['training_process']['live'] and len(health['worker_pids'])==4 and not health['issues'],
                correct_gpus=record['physical_gpus']==list(gpus),
                online_logging=bool(wandb['url']) and wandb['group']=='144000-exp-heavy',
                observer_live=process_identity(observer['pid'],observer['start_ticks'])['live'],
                exporter_live=process_identity(exporter['pid'],exporter['process_start_ticks'])['live'])
            details[arm]={'health':health,'wandb':wandb,'completed_updates':rows[-1]['completed_updates']}
        configurations[arm]=config
    checks['paired']={'same_physics':configurations['latent']['physics']==configurations['baseline']['physics'],
                     'same_dataset':configurations['latent']['dataset']==configurations['baseline']['dataset'],
                     'same_uniform_sampling':configurations['latent']['motion_sampling']==configurations['baseline']['motion_sampling']}
    report={'passed':all(all(fields.values()) for fields in checks.values()),'phase':phase,
            'checks':checks,'details':details,'unix_time':time.time()}
    (COMPARISON/f'{phase}_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'failures':{arm:[k for k,v in fields.items() if not v] for arm,fields in checks.items()}},indent=2))
    if not report['passed']:
        raise RuntimeError('Uniform resume verification failed')
    if phase=='smoke':
        unit=read(COMPARISON/'unit_tests.json')
        if not unit['passed']:
            raise RuntimeError('Unit tests failed')
        paths=[*ROOT.glob('src/**/*.py'),ROOT/'scripts/resume_144000_heavy_uniform.py',
               ROOT/'scripts/verify_heavy_uniform_resume.py',ROOT/'scripts/run_144000_heavy_residual.py',
               ROOT/'scripts/monitor_memory350_residual.py',ROOT/'scripts/watch_memory350_onnx.py']
        (COMPARISON/'preflight_verification.json').write_text(json.dumps({
            'passed':True,'smoke':report,'unit_tests':unit,
            'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths}},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('smoke','train'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    audit(args.phase)
