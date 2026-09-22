"""Persistent residual PPO health observer; never stops or alters a healthy trainer."""

import argparse
from collections import deque
import fcntl
import json
import math
from pathlib import Path
import signal
import time

from monitor_memory350_nominal_direction import atomic_json, json_file, process_identity


def recent_metrics(path, limit=64):
    if not path.exists():
        return []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        start = max(0, stream.tell() - 2 * 1024 * 1024)
        stream.seek(start)
        if start:
            stream.readline()
        lines = stream.read().split(b'\n')[:-1]
    return list(deque((json.loads(line) for line in lines if line.strip()), maxlen=limit))


def nonfinite(value, prefix=''):
    if isinstance(value, dict):
        return [name for key, item in value.items() for name in nonfinite(item, prefix + '/' + key)]
    if isinstance(value, (list, tuple)):
        return [name for i, item in enumerate(value) for name in nonfinite(item, prefix + '/' + str(i))]
    return [prefix] if isinstance(value, float) and not math.isfinite(value) else []


def worker_pids(parent, output):
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry/'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            if int(fields[1]) != parent or fields[0] in ('Z', 'X', 'x'):
                continue
            command = (entry/'cmdline').read_bytes().split(b'\0')
            modules = (b'intact_tracking.cli.memory350_proprio_native_policy_train',
                       b'intact_tracking.cli.memory350_proprio_heavy_policy_train')
            if any(module in command for module in modules) and str(output).encode() in command:
                result.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(result)


def observe(root, *, stall_seconds=300, initialization_seconds=1200):
    reference = json_file(root/'latest_training_process.json')
    record = json_file(Path(reference['process_record']))
    identity = process_identity(record['pid'], record['process_start_ticks'])
    output = Path(record['output'])
    rows = recent_metrics(output/'metrics.jsonl')
    latest = rows[-1] if rows else {}
    now = time.time()
    workers = worker_pids(identity['pid'], output) if identity['live'] else []
    issues, warnings = [], []
    if not identity['live']:
        issues.append('training_process_not_live')
    elif rows and len(workers) != len(record['physical_gpus']):
        issues.append('training_worker_count_changed')
    age = now - latest.get('unix_time', record['started_at'])
    if age > (stall_seconds if rows else initialization_seconds):
        issues.append('no_recent_training_update')
    for row in rows:
        invalid = nonfinite(row)
        if invalid:
            issues.append({'nonfinite_update': row['completed_updates'], 'fields': invalid})
    loss = latest.get('loss', {})
    if rows:
        if record.get('motion_sampling') == 'uniform':
            observed_sampling = latest.get('motion_sampling', {})
            if (observed_sampling.get('adaptive_enabled') is not False
                    or observed_sampling.get('failure_rewind_enabled') is not False):
                issues.append('uniform_training_sampling_contract_changed')
        mode = record.get('latent_input_mode')
        if mode in ('learned', 'zero'):
            if loss.get('latent_input_is_zero') != float(mode == 'zero'):
                issues.append('latent_input_mode_changed')
            if mode == 'zero' and any(loss.get(key) != 0 for key in (
                    'dynamics_latent_rms', 'latent_zero_action_delta_rms', 'latent_shuffle_action_delta_rms')):
                issues.append('baseline_received_nonzero_latent')
        expected_aux = record.get('dr_aux_coef', .05)
        required = ('residual_action_rms', 'residual_action_abs_max')
        if expected_aux > 0:
            required += ('AuxDR/loss',)
        elif loss.get('AuxDR/enabled') != 0 or 'AuxDR/loss' in loss:
            issues.append('baseline_auxiliary_loss_enabled')
        for name in required:
            if name not in loss:
                issues.append('missing_metric:' + name)
        if loss.get('residual_output_bounded') != 0:
            issues.append('residual_bounding_enabled')
        if abs(loss.get('AuxDR/coefficient', -1) - expected_aux) > 1e-8:
            issues.append('auxiliary_coefficient_changed')
        if any(name.startswith(('AuxDR/kp_', 'AuxDR/kd_', 'AuxDR/armature_')) for name in loss):
            issues.append('excluded_motor_supervision_present')
        if record.get('dr_aux_payload_targets') == 'mass':
            if any(name.startswith('AuxDR/payload_com_') for name in loss):
                issues.append('excluded_payload_com_supervision_present')
            if expected_aux > 0 and sum(name.startswith('AuxDR/payload_mass_') for name in loss) != 8:
                issues.append('missing_payload_mass_metrics')
        if loss.get('nominal_physics_max_error', 0) > 1e-6:
            issues.append('nominal_physics_changed')
        if latest.get('action_std', 0) <= 0:
            issues.append('nonpositive_action_std')
        elif latest['action_std'] < .03:
            warnings.append('low_action_std_review_exploration')
        if len(rows) >= 20 and all(row['loss'].get('residual_action_rms') == 0 for row in rows[-20:]):
            warnings.append('residual_mean_still_zero')
        # Finite PPO losses alone do not imply healthy policy learning. A
        # persistent short-episode regime deserves inspection even if its
        # return just crossed zero; that alone is not a recovery of tracking.
        if latest['completed_updates'] >= 32 and len(rows) >= 8 and all(
            row.get('mean_episode_length') is not None and 0 < row['mean_episode_length'] < 50
            for row in rows[-8:]
        ):
            warnings.append('persistent_short_episodes')
            if all(row.get('mean_reward') is not None and row['mean_reward'] < 0 for row in rows[-8:]):
                warnings.append('persistent_short_episodes_with_negative_returns')
        if loss.get('AuxDR/shared_gradient_ratio_valid') and loss.get('AuxDR/shared_gradient_ratio', 0) > 1:
            warnings.append('auxiliary_hidden_gradient_exceeds_ppo_gradient')
    selected = {name: loss[name] for name in (
        'entropy', 'value', 'surrogate', 'AuxDR/loss', 'AuxDR/weighted_loss', 'AuxDR/history_weight_mean',
        'AuxDR/shared_gradient_ratio', 'AuxDR/policy_kl', 'residual_action_rms', 'residual_action_abs_max',
        'residual_to_base_rms_ratio', 'latent_shuffle_action_delta_rms', 'latent_zero_action_delta_rms',
        'dynamics_latent_rms', 'latent_input_is_zero',
        'AuxDR/com_x_mae_m', 'AuxDR/com_y_mae_m', 'AuxDR/com_z_mae_m', 'AuxDR/friction_mae_coefficient',
    ) if name in loss}
    selected.update({k:v for k,v in loss.items() if k.startswith('AuxDR/payload_')})
    return {'checked_at': now, 'status': 'needs_attention' if issues else 'warning' if warnings else 'healthy' if rows else 'initializing',
            'issues': issues, 'warnings': warnings, 'training_process': identity, 'worker_pids': workers,
            'process_record': reference['process_record'], 'output': str(output), 'completed_updates': latest.get('completed_updates', 0),
            'seconds_since_update': age, 'action_std': latest.get('action_std'), 'mean_reward': latest.get('mean_reward'),
            'mean_episode_length': latest.get('mean_episode_length'), 'metrics': selected,
            'training_update_limit': record.get('maximum_updates'), 'automatic_training_control': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError('poll-seconds must be positive')
    root = args.run_root.resolve()
    directory = root/'monitor'
    directory.mkdir(exist_ok=True)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (directory/'observer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while running:
            try:
                report = observe(root)
            except Exception as error:
                report = {'checked_at': time.time(), 'status': 'monitor_error',
                          'issues': [f'{type(error).__name__}: {error}']}
            atomic_json(directory/'health.json', report)
            print(json.dumps({key: report.get(key) for key in ('checked_at', 'status', 'completed_updates', 'issues', 'warnings')}), flush=True)
            if args.once:
                break
            deadline = time.monotonic() + args.poll_seconds
            while running and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
