"""Start v2 MLP/MoE arms with raw tracker action supplied to actor and critic."""

import os
import json
import math
from pathlib import Path
import signal
import time

import run_memory350_online_moe_ppo as launcher

original_command = launcher.moe_command
original_comparison = launcher.update_comparison


def last_complete_record(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 131072))
        data = stream.read()
    lines = data.split(b'\n')[:-1]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def monitor_comparison(root):
    report = {'checked_at':time.time(), 'arms':{}, 'issues':[]}
    try:
        original_comparison(root)
        report['comparison_updated'] = True
    except json.JSONDecodeError:
        # An evaluator may still be appending its last JSONL record. Retry on
        # the next heartbeat instead of interrupting both healthy experiments.
        report['comparison_updated'] = False
    for fusion in launcher.ASSIGNMENTS:
        directory = root / 'ppo' / f'{fusion}_121'
        try:
            config = json.loads((directory / 'run_config.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            config = None
        latest = last_complete_record(directory / 'metrics.jsonl')
        endpoint = last_complete_record(directory / 'endpoint_eval_metrics.jsonl')
        row = {'latest_update':latest['completed_updates'] if latest else None,
               'latest_metric_time':latest.get('unix_time') if latest else None,
               'latest_endpoint_update':endpoint['completed_updates'] if endpoint else None}
        if config:
            audit = config['input_audit']
            row['critic_action_dim'] = audit.get('critic_tracker_action_input_dim')
            row['independent_actor_critic'] = audit.get('actor_critic_parameters_shared') is False
            if row['critic_action_dim'] != 29 or not row['independent_actor_critic']:
                report['issues'].append(f'{fusion}: critic action or independent encoder invariant failed')
        if latest:
            if not all(math.isfinite(float(v)) for v in latest['loss'].values()):
                report['issues'].append(f'{fusion}: nonfinite training metric')
            if fusion == 'concat':
                row['router'] = {k:v for k,v in latest['loss'].items() if k.startswith('router_') and 'fraction_' not in k}
                if latest['loss']['router_center_updates'] != latest['completed_updates']:
                    report['issues'].append('MoE: center update counter differs from completed PPO updates')
        report['arms'][fusion] = row
    report['ok'] = not report['issues']
    folder = root / 'health'
    folder.mkdir(exist_ok=True)
    launcher.write_json(folder / 'latest.json', report)
    previous = getattr(monitor_comparison, 'last_history_time', 0)
    if report['checked_at'] - previous >= 60:
        with (folder / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(report) + '\n')
        monitor_comparison.last_history_time = report['checked_at']
    if report['issues']:
        raise RuntimeError('; '.join(report['issues']))


def critic_action_command(root, fusion, context, **kwargs):
    command = original_command(root, fusion, context, **kwargs)
    index = command.index("intact_tracking.cli.memory350_moe_policy_train")
    command[index] = "intact_tracking.cli.memory350_moe_critic_action_train"
    return command


def stop_rank_workers(parent):
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (path / 'cmdline').read_bytes()
            if int(stat[1]) == parent and b'intact_tracking.cli.memory350_moe_critic_action_train' in command:
                os.kill(int(path.name), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass


def main():
    launcher.moe_command = critic_action_command
    launcher.stop_rank_workers = stop_rank_workers
    launcher.update_comparison = monitor_comparison
    launcher.main()


if __name__ == '__main__':
    main()
