"""Audit the authorized heavy auxiliary transition, real restore, and fresh sampler."""

import argparse
import json
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf

from resume_144000_heavy_residual import ROOT, COMPARISON, OUTPUTS, ARMS, REVISION, command_for
from run_144000_heavy_residual import sha256
from verify_144000_heavy_residual import finite
from intact_tracking.cli.memory350_policy_train import state_digest


def read(path):return json.loads(path.read_text())


def json_representation(value):
    # Checkpoints retain tuples and str-valued enums; run_config.json uses
    # lists and strings. Compare their serialized values, without dropping any
    # physical field or coercing numbers to strings.
    return json.loads(json.dumps(value))


def audit(phase):
    checks,configs,details={},{},{}
    launch=phase=='launch'
    for arm,(mode,gpus) in ARMS.items():
        _,out,parent=command_for(arm,'train' if launch else 'smoke')
        c=read(out/'run_config.json')
        old=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
        resumed=torch.load(out/'checkpoint_resume.pt',map_location='cpu',weights_only=False,mmap=True)
        rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().split('\n')[:-1] if s]
        aux=c['input_audit']['dr_auxiliary']
        coef=.5 if mode=='learned' else 0.
        source_update=2001 if launch else 25
        source_config=old['residual_policy']
        agent=OmegaConf.to_container(resumed['cfg'].agent,resolve=True)
        audits=c['sampling_resume_audits']
        weights=aux['group_weights_normalized']
        checkpoint=resumed if launch else torch.load(out/'checkpoint_final.pt',map_location='cpu',weights_only=False,mmap=True)
        checks[arm]={
            'correct_parent':old['completed_updates']==source_update==resumed['completed_updates'],
            'exact_model_optimizer_std_normalization_restore':state_digest(old['rsl_rl'])==state_digest(resumed['rsl_rl'])
                ==c['resume_state_audit']['model_optimizer_state_sha256'] and c['resume_state_audit']['passed'],
            'new_objective_same_structure':aux['output_dim']==108 and aux['active_coordinates']==(8 if coef else 0)
                and agent['actor']['dr_aux_payload_com_enabled'] is False and agent['algorithm']['dr_aux_coef']==coef,
            'exact_fivefold_retained_weight':weights[:4]==[.125]*4 and weights[4:7]==[0.]*3
                and weights[7:11]==[.03125]*4 and weights[11:]==[0.]*12
                and aux['group_weight_normalization_denominator']==8.,
            'same_frozen_sources':c['context_sha256']==source_config['context_sha256']
                and c['tracker_sha256']==source_config['tracker_sha256'],
            'same_dataset':c['dataset']==source_config['dataset'],
            'same_physics':c['physics']==json_representation(source_config['physics']),
            'same_sampling_algorithm':c['motion_sampling']==source_config['motion_sampling'],
            'same_rewards':c['reward_contract']==source_config['reward_contract'],
            'all_four_samplers_reset':len(audits)==4 and all(a['reset_to_priors'] and not a['restored']
                and a['visit_prior_max_error']==a['failure_prior_max_error']==0
                and a['prior_visit_count']==1 and a['prior_failure_count']==0
                and a['pending_visit_count']==a['pending_failure_count']==0
                and a['ema_last_iteration'] is None for a in audits),
            'portable_checkpoint':all(k in resumed for k in ('frozen_context','frozen_tracker','inference_bundle_version')),
            'finite':finite(rows) and finite(checkpoint['rsl_rl']),
            'updates_continue':rows[0]['completed_updates']==source_update+1 and rows[-1]['completed_updates']>=source_update+3,
            'correct_logged_loss':all(r['loss']['AuxDR/coefficient']==coef and r['loss']['AuxDR/enabled']==bool(coef)
                and ('AuxDR/loss' in r['loss'])==bool(coef)
                and sum(k.startswith('AuxDR/payload_mass_') for k in r['loss'])==(8 if coef else 0)
                and not any(k.startswith('AuxDR/payload_com_') for k in r['loss']) for r in rows),
            'mode_unchanged':agent['actor']['latent_input_mode']==agent['critic']['latent_input_mode']==mode,
            '8192_per_rank':c['distributed']['num_envs_per_rank']==8192 and c['distributed']['global_num_envs']==32768,
        }
        if mode=='zero':
            checks[arm]['baseline_zero_latent']=all(r['loss'][k]==0 for r in rows for k in
                ('dynamics_latent_rms','latent_zero_action_delta_rms','latent_shuffle_action_delta_rms'))
        export=read(out/('policy.json' if launch else 'export_u28/policy.json'))
        checks[arm]['onnx_valid']=export['validation']['passed'] and export['latent_input_mode']==mode
        if launch:
            from monitor_memory350_residual import observe
            from monitor_memory350_nominal_direction import process_identity
            health=observe(out.parent)
            record=read(Path(read(out.parent/'latest_training_process.json')['process_record']))
            wandb=read(out/'wandb_run.json')
            observer=read(out.parent/'monitor/observer_process.json')
            exporter=read(out.parent/'monitor/onnx_export/launch.json')
            checks[arm].update(
                four_live_workers=health['training_process']['live'] and len(health['worker_pids'])==4 and not health['issues'],
                correct_gpus=record['physical_gpus']==list(gpus),
                no_cap=c['maximum_updates'] is None and c['arguments']['until_user_stop'],
                full_dataset=c['dataset']['motion_count']==c['dataset']['loaded_motion_count']==220480,
                online_logging=bool(wandb['url']) and wandb['group']=='144000-exp-heavy',
                observer_live=process_identity(observer['pid'],observer['start_ticks'])['live'],
                exporter_live=process_identity(exporter['pid'],exporter['process_start_ticks'])['live'])
            details[arm]={'health':health,'wandb':wandb,'completed_updates':rows[-1]['completed_updates']}
        else:
            completion=read(out/'completion.json')
            checks[arm].update(completed=completion['complete'] and completion['completed_updates']==28,
                four_rank_agreement=completion['distributed_parameter_agreement']['passed']
                    and completion['distributed_parameter_agreement']['world_size']==4)
        configs[arm]=c
    checks['paired']={'same_physics':configs['latent']['physics']==configs['baseline']['physics'],
                      'same_dataset':configs['latent']['dataset']==configs['baseline']['dataset']}
    if launch:
        preflight=read(COMPARISON/'preflight_verification.json')
        checks['preflight']={'passed':preflight['passed'],'source_unchanged':all(sha256(ROOT/n)==h for n,h in preflight['source_sha256'].items())}
    report={'passed':all(all(v.values()) for v in checks.values()),'revision':REVISION,
            'unix_time':time.time(),'checks':checks,'details':details}
    (COMPARISON/f'{phase}_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'failures':{a:[k for k,v in fields.items() if not v] for a,fields in checks.items()}},indent=2),flush=True)
    assert report['passed']
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('ready','launch'),required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.phase=='launch':audit('launch');return
    smoke=audit('smoke')
    unit=read(COMPARISON/'unit_tests.json');assert unit['passed']
    paths=[*ROOT.glob('src/**/*.py'),ROOT/'scripts/resume_144000_heavy_residual.py',ROOT/'scripts/run_144000_heavy_residual.py',
           ROOT/'scripts/monitor_memory350_residual.py',ROOT/'scripts/watch_memory350_onnx.py',Path(__file__).resolve()]
    result={'passed':True,'revision':REVISION,'unix_time':time.time(),'smoke':smoke,'unit_tests':unit,
            'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths}}
    (COMPARISON/'preflight_verification.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
