"""Audit the completed 16384-world smoke before the eight-GPU scratch launch."""

import json
import math
from pathlib import Path
import time

import torch

from monitor_memory350_nominal_direction import check_loss, process_identity
from run_144000_heavy_context import ROOT, RUN, command_for, sha256


def main():
    torch.set_num_threads(1)
    plan = json.loads((RUN / 'plan.json').read_text())
    run = RUN / 'smoke_16384'
    config = json.loads((run / 'run_config.json').read_text())
    baseline = json.loads(Path(plan['baseline_config']).read_text())
    progress = json.loads((run / 'progress.json').read_text())
    process = json.loads((RUN / 'smoke_process.json').read_text())
    assert not process_identity(process['pid'], process['process_start_ticks'])['live'], 'Smoke still running'
    state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    payload = config['heavy_payload_contract']['runtime_audit']
    schema = state['dr_metric_schema']
    checks = {
        'unit_tests_passed': json.loads((RUN / 'unit_test_verification.json').read_text())['passed'],
        'two_updates_eight_optimizer_steps': progress['completed_updates'] == state['update'] == 2
            and state['optimizer_steps'] == 8,
        'scratch_initialization': config['initialization'] == 'scratch' and not config['arguments']['resume'],
        'frozen_tracker_144000': state['tracker']['frozen']
            and state['tracker']['checkpoint_sha256'] == plan['tracker_checkpoint_sha256'],
        'same_model_architecture': config['model'] == baseline['model'] == state['model_config'],
        'same_loss_configuration': config['loss'] == baseline['loss'] == state['loss_config'],
        'same_effective_loss_weights': config['objective_weights'] == baseline['objective_weights'],
        'same_optimizer_batch_and_precision': (
            {k:v for k,v in config['optimization'].items() if k != 'diagnostics_interval'}
            == {k:v for k,v in baseline['optimization'].items() if k != 'diagnostics_interval'}),
        'all_model_parameters_finite': all(bool(torch.isfinite(x).all()) for x in state['model'].values()),
        'model_parameter_agreement': state['distributed_parameter_agreement']['passed'],
        'nominal1639_dr14745': payload['nominal_count'] == 1639 and payload['dr_count'] == 14745,
        '256_balanced_groups': len(payload['mass_group_counts']) == 256
            and min(payload['mass_group_counts']) == 57 and max(payload['mass_group_counts']) == 58,
        'nominal_physics_exact': payload['nominal_inertial_fields_exact'],
        'composite_mass_com_inertia_verified': payload['actual_composite_inertial_fields_verified'],
        'cube_com_not_ball': .05 < payload['com_offset_norm_max_m'] <= math.sqrt(3) * .05
            and min(payload['com_offset_min_m']) >= -.05 and max(payload['com_offset_max_m']) <= .05,
        '108_dr_coordinates_no_inertia_labels': len(schema['names']) == 108
            and not any('inertia' in name for name in schema['names']),
        '16_balanced_physical_factors': len(schema['groups']) == 16
            and all(math.isclose(sum(schema['coordinate_weights'][i] for i in group), 1/16)
                    for group in schema['groups'].values()),
        'checkpoint_payload_contract': json.loads(json.dumps(state['heavy_payload_contract'])) == config['heavy_payload_contract'],
        'cpu_archive_float32': config['replay']['weak_archive_storage']['device'] == 'cpu'
            and config['replay']['weak_archive_storage']['dtype'] == 'float32',
        'finite_losses_with_correct_coefficients': all(all(check_loss(row['optimization_train'],
            config['objective_weights'])[k] for k in ('finite', 'term_sum_matches', 'negative_term_absent')) for row in rows),
        'finite_validation_metrics': all(all(math.isfinite(v) for v in row['fixed_probe'].values()) for row in rows),
        'all_encoder_levels_receive_gradients': all(all(
            math.isfinite(row['collector_memory']['gradient_norm_' + key])
            and row['collector_memory']['gradient_norm_' + key] > 0
            for key in ('chunk_encoder', 'long_encoder', 'final_encoder')) for row in rows),
    }
    command = command_for(Path(plan['output_dir']))
    checks['formal_eight_gpus_16384'] = '--nproc-per-node=8' in command and command[command.index('--num-envs')+1] == '16384'
    checks['formal_no_update_cap'] = '--until-user-stop' in command and '--stop-after-updates' not in command
    checks['formal_full_dataset'] = command[command.index('--motion-path')+1] == plan['dataset']['root']
    checks['formal_logging_interval_preserved'] = int(command[command.index('--log-interval')+1]) == baseline['optimization']['diagnostics_interval']
    paths = [*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('tests/*.py'), ROOT/'pyproject.toml']
    result = {'passed': all(checks.values()), 'verified_at': time.time(), 'checks': checks,
              'source_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in paths},
              'smoke_checkpoint': str(run / 'last.pt'), 'payload_audit': payload,
              'smoke_peak_torch_gpu_reserved_gib': max(r['collector_memory']['gpu_peak_reserved_gib'] for r in rows),
              'nominal_pair_numerical_audit': config.get('nominal_pair_numerical_audit'),
              'formal_command': command}
    (RUN / 'launch_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'checks': checks,
                      'torch_gpu_peak_reserved_gib': result['smoke_peak_torch_gpu_reserved_gib']}, indent=2))
    assert result['passed'], [k for k,v in checks.items() if not v]
    plan['readiness']['single_gpu_16384_smoke_verified'] = True
    plan['status'] = 'ready_for_eight_gpu_launch'
    (RUN / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
