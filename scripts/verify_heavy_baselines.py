"""Audit the isolated baseline smoke, resume, evaluation and export artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from intact_tracking.limb_context_sampling import state_digest
from intact_tracking.memory350_checkpoint import embedded_tracker
from intact_tracking.heavy_baseline_deploy import HeavyBaselinePolicy


def read(path):
    return json.loads(path.read_text())


def verify(root):
    summary = {'scope': 'functional verification, 4 ranks x 64 environments; not formal training',
               'formal_4x8192_validated': False, 'arms': {}, 'evaluations': {}, 'exports': {}}
    for arm, method in (('rma', 'rma_teacher'), ('any', 'any2track')):
        first, second = root/f'{arm}_4rank', root/f'{arm}_4rank_resume'
        before = torch.load(first/'checkpoint_final.pt', map_location='cpu', weights_only=False)
        restored = torch.load(second/'checkpoint_resume.pt', map_location='cpu', weights_only=False)
        assert state_digest(before['rsl_rl']) == state_digest(restored['rsl_rl'])
        report = read(second/'run_config.json')
        assert report['method'] == method and report['sampling_reset_generation'] == 1
        audits = report['sampling_resume_audits']
        assert len(audits) == 4 and all(row['reset_to_priors'] and not row['restored']
            and row['failure_prior_max_error'] == 0 and row['visit_prior_max_error'] == 0
            and row['pending_failure_count'] == 0 and row['pending_visit_count'] == 0
            and row['ema_last_iteration'] is None for row in audits)
        for directory, count in ((first, 2), (second, 3)):
            completion = read(directory/'completion.json')
            assert completion['completed_updates'] == count and completion['complete']
            assert completion['distributed_parameter_agreement']['passed']
            assert completion['distributed']['world_size'] == 4
            assert completion['distributed']['num_envs_per_rank'] == 64
            state = torch.load(directory/'checkpoint_final.pt', map_location='cpu', weights_only=False)
            embedded_tracker(state)  # Includes exact frozen-weight checksum.
            if arm == 'any':
                assert state['world_optimizer_steps'] == count*20
                assert completion['distributed_parameter_agreement']['world_model_state_sha256']
        summary['arms'][arm] = {'passed': True, 'refresh_boundary': 'test 2 -> 3',
                                'resume_full_state_sha256': state_digest(restored['rsl_rl']),
                                'input_audit': report['input_audit']}
        package = root/f'{arm}_onnx'
        export = read(package/'policy.json')
        assert export['validation']['passed']
        client = HeavyBaselinePolicy(package)
        observation = np.zeros(8199, dtype=np.float32)
        observation[0] = 1.
        options = {}
        if arm == 'rma':
            schema = export['physical_input_schema']
            options['physics_parameters'] = (np.array(schema['lower'])+np.array(schema['upper']))/2
        for _ in range(3):
            assert np.isfinite(client.step(observation, **options)).all()
        summary['exports'][arm] = {'passed': True, 'runtime_three_steps_finite': True,
            'action_max_abs_error': max(row[0] for row in export['validation']['max_absolute_errors_by_case'])}
    inherited = read(root/'any_4rank_inherit/run_config.json')
    assert inherited['sampling_reset_generation'] == 1
    assert all(row['restored'] and not row.get('reset_to_priors', False) for row in inherited['sampling_resume_audits'])
    assert read(root/'any_4rank_inherit/completion.json')['completed_updates'] == 4
    summary['ordinary_resume_inherits_sampler'] = True
    for physics in ('nominal', 'hdr'):
        teacher, adapter = (read(root/f'{arm}_{physics}.json') for arm in ('rma', 'any'))
        for key in ('physics_world_fingerprints', 'query_initial_state_sha256', 'start_frames', 'horizons'):
            assert teacher[key] == adapter[key], key
        assert teacher['evaluation_diagnostics']['query_force_sha256'] == adapter['evaluation_diagnostics']['query_force_sha256']
        assert all(value == (physics == 'nominal') for value in teacher['is_nominal'] + adapter['is_nominal'])
        summary['evaluations'][physics] = {'paired_physics_initial_state_and_forces': True,
                                          'all_worlds_use_requested_physics': True,
                                          'motions': teacher['motions'], 'episodes': teacher['episodes'],
                                          'steps': teacher['max_steps']}
    summary['passed'] = True
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.runtime/heavy_baselines_validation'))
    args = parser.parse_args()
    report = verify(args.root)
    (args.root/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
