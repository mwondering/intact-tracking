from copy import deepcopy
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('method', ['rma_teacher', 'any2track'])
def test_uniform_configuration_is_unlimited_without_stage_transition(method):
    # configure() installs experiment globals: isolate them from other tests.
    import subprocess
    import sys
    source = '''
from intact_tracking.cli import heavy_baseline_train as cli
args = cli.build_parser().parse_args(['--method', %r, '--output-dir', 'unused'])
cli.configure(args)
assert args.motion_sampling == 'uniform' and args.stage == 'continuous_uniform'
assert args.until_user_stop and not args.reset_adaptive_sampling and not args.resume_uniform_sampling
metadata = {'research_source_sha256': {}, 'dataset': {'motion_count': 220480,
    'manifest_sha256': '59b8e336c152e4773133bcdcd86cf86fd6545c3c0e6cc00ce67912d107c740ac'}}
cli.adapt_metadata(metadata, args, None)
assert metadata['sampling_reset_generation'] == 0
assert metadata['baseline_contract']['training_schedule'] == 'continuous_until_user_stop'
assert metadata['baseline_contract']['stage1_boundary_completed_updates'] is None
assert metadata['baseline_contract']['stage2_first_completed_update'] is None
''' % method
    subprocess.run([sys.executable, '-c', source], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('count', [1999, 2000, 2001, 2500, 8000])
def test_uniform_controller_never_switches_or_resets_at_2000(tmp_path, monkeypatch, count):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_144000_heavy_baselines as controller
    assert controller.select_stage(tmp_path, 'uniform') == ('continuous_uniform', None, False)
    output = tmp_path/'continuous_uniform'
    output.mkdir()
    checkpoint = output/'checkpoint_final.pt'
    torch.save({'completed_updates': count, 'residual_policy': {
        'baseline_stage': 'continuous_uniform', 'sampling_reset_generation': 0,
        'motion_sampling': {'active_mode': 'uniform'}}}, checkpoint)
    stage, parent, reset = controller.select_stage(tmp_path, 'uniform')
    assert (stage, parent, reset) == ('continuous_uniform', checkpoint, False)
    command = controller.command_for('rma_teacher', stage, output, resume=parent)
    assert command[command.index('--motion-sampling')+1] == 'uniform'
    assert '--reset-adaptive-sampling' not in command and '--resume-uniform-sampling' not in command


def test_uniform_refuses_adaptive_checkpoint_and_reset(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_144000_heavy_baselines as controller
    output = tmp_path/'continuous_uniform'
    output.mkdir()
    torch.save({'residual_policy': {'baseline_stage': 'stage1', 'sampling_reset_generation': 0,
                                   'motion_sampling': {'active_mode': 'adaptive'}}}, output/'checkpoint_final.pt')
    with pytest.raises(ValueError, match='own checkpoint'):
        controller.select_stage(tmp_path, 'uniform')
    with pytest.raises(ValueError, match='no planned sampler reset'):
        controller.command_for('any2track', 'continuous_uniform', output, reset=True)


@pytest.mark.parametrize('method', ['rma_teacher', 'any2track'])
def test_uniform_metadata_continues_past_previous_boundary(method):
    from intact_tracking.cli import heavy_baseline_train as cli
    args = cli.build_parser().parse_args(['--method', method, '--output-dir', 'unused', '--bounded-smoke'])
    metadata = {'research_source_sha256': {}}
    cli.adapt_metadata(metadata, args, None)
    before = deepcopy(metadata)
    cli.adapt_metadata(metadata, args, {'completed_updates': 8000, 'residual_policy': before})
    assert metadata['sampling_reset_generation'] == 0
    assert metadata['baseline_contract']['stage1_boundary_completed_updates'] is None
