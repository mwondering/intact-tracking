"""One-off paired mixed-DR warm evaluation; never controls the live trainers."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from run_limb_context_experiment import ROOT, process_environment
from monitor_memory350_nominal_direction import process_identity
from intact_tracking.memory350_policy_checkpoint_eval import digest, evaluation_environment, validate_result
from intact_tracking.memory350_policy_results import compare

RUN = ROOT / 'runs/limb_context_20260917_soft20000_history5_uniform_ppo'
MANIFEST = ROOT / 'runs/limb_context_20260912_memory350_response_window_ablation/protocols/periodic_motions.txt'
PRIOR = RUN / 'manual_evaluation/mixture_warm_20260918_024942'
METRICS = [('局部 body', 'common_error_body_pos'), ('joint', 'common_error_joint_pos'),
           ('全局 body', 'common_error_body_pos_global'), ('全局 anchor', 'common_error_anchor_pos_global')]


def read(path):
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt == 2:
                raise
            time.sleep(.1)


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def training_status():
    result = {}
    for arm in ('concat', 'baseline'):
        record = read(RUN / f'ppo_{arm}_train_only_process.json')
        identity = process_identity(record['pid'], record['start_ticks'])
        cfg = read(RUN / 'ppo' / f'{arm}_121_train_only/run_config.json')
        try:
            progress = read(RUN / 'ppo' / f'{arm}_121_train_only/progress.json')
        except (FileNotFoundError, json.JSONDecodeError):
            progress = {'temporarily_unavailable': True}
        result[arm] = {'pid': record['pid'], 'start_ticks': record['start_ticks'],
                       'live': identity['live'], 'progress': progress,
                       'periodic_evaluation_disabled': not cfg.get('periodic_evaluation')}
    return result


def encoder_status(run):
    record = read(run / 'train_process.json')
    start = record.get('start_ticks', record.get('process_start_ticks'))
    identity = process_identity(record['pid'], start)
    return {'pid': record['pid'], 'start_ticks': start, 'live': identity['live'], 'run_root': str(run)}


def select_latest():
    # Capture the file list once, so a new save cannot move the evaluation target.
    paths = {arm: sorted((RUN / 'ppo' / f'{arm}_121_train_only').glob('checkpoint_*.pt'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
             for arm in ('concat', 'baseline')}
    selected = []
    for arm in paths:
        path = paths[arm][0]
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        update = state['completed_updates']
        del state
        selected.append({'arm': arm, 'completed_updates': update, 'checkpoint': str(path)})
    common_paths = {
        arm: {int(p.stem.rsplit('_', 1)[1]) + 1: p for p in files
              if p.stem.rsplit('_', 1)[1].isdigit()}
        for arm, files in paths.items()
    }
    # Graceful-stop checkpoints need not coincide with the other arm's save interval.
    shared = common_paths['concat'].keys() & common_paths['baseline'].keys()
    latest_common_limit = min(job['completed_updates'] for job in selected)
    common = max(update for update in shared if update <= latest_common_limit)
    for arm in paths:
        if next(job for job in selected if job['arm'] == arm)['completed_updates'] == common:
            continue
        path = common_paths[arm][common]
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if state['completed_updates'] != common:
            raise ValueError('The matched checkpoint update differs from its saved metadata')
        del state
        selected.append({'arm': arm, 'completed_updates': common, 'checkpoint': str(path)})
    counts = {'concat': 0, 'baseline': 0}
    for job in selected:
        job['label'] = f"{job['arm']}_u{job['completed_updates']:06d}"
        job['checkpoint_sha256'] = digest(job['checkpoint'])
        job['gpu'] = {'concat': [2, 3], 'baseline': [6, 7]}[job['arm']][counts[job['arm']]]
        counts[job['arm']] += 1
    return selected, common


def analyze(out, manifest):
    inputs, loaded = {}, {}
    protocol = {'seed': 20001, 'steps': 1000, 'warmup_steps': 500,
                'cases': {'mixture_warm': {'masses': None, 'memory_start': 'warm'}}}
    files = MANIFEST.read_text().splitlines()

    def load(label, path, job=None):
        row = read(path)
        with np.load(path.with_suffix('.traces.npz')) as data:
            assert data['lengths'].tolist() == row['episode_lengths']
            assert data['metric_names'].tolist() == row['metric_names']
            traces = data['all_metrics'].copy()
        assert traces.shape == (512, 1000, 20) and np.isfinite(traces).all()
        assert row['policy_precision'] == 'fp32' and row['latent_intervention'] == 'correct'
        assert row['physics']['runtime_audit']['nominal_population']['count'] == 52
        assert row['physics']['independent_nominal_mixture']['probability'] == .5
        masses = np.asarray(row['actual_limb_masses_kg'])
        assert (masses >= 0).all() and (masses <= [2.5, 2.5, 4, 4]).all()
        if job:
            assert digest(job['checkpoint']) == job['checkpoint_sha256']
            validate_result(row, protocol, files, job['checkpoint_sha256'], job['completed_updates'], 'mixture_warm')
        loaded[label] = (row, traces)
        inputs[label] = {'result': str(path), 'result_sha256': digest(path),
                         'trace_sha256': digest(path.with_suffix('.traces.npz')),
                         'checkpoint': row['checkpoint'], 'checkpoint_sha256': row['checkpoint_sha256'],
                         'completed_updates': row['completed_training_updates'], 'failures': int(sum(row['failed'])),
                         'coverage': row['coverage_fraction'], 'warmup': row['warmup'],
                         'query_initial_state_sha256': row['query_initial_state_sha256']}

    for job in manifest['jobs']:
        load(job['label'], Path(job['output']), job)
    for arm in ('concat', 'baseline'):
        load(f'{arm}_u001301', PRIOR / f'{arm}_u001301.json')
        load(f'{arm}_u000000', RUN / 'ppo' / f'{arm}_121_warm_bounded/endpoint_eval/update_000000/mixture_warm.json')
    common = manifest['common_update']
    latest = {arm: max((j for j in manifest['jobs'] if j['arm'] == arm), key=lambda j: j['completed_updates'])['label']
              for arm in ('concat', 'baseline')}
    pairs = {'same_update': (f'baseline_u{common:06d}', f'concat_u{common:06d}'),
             'latest_at_selection': (latest['baseline'], latest['concat'])}
    for job in manifest['jobs']:
        pairs[job['label'] + '_vs_initial'] = (job['arm'] + '_u000000', job['label'])
        pairs[job['label'] + '_since_u1301'] = (job['arm'] + '_u001301', job['label'])
    comparisons = {}
    for name, (reference, candidate) in pairs.items():
        a, ta = loaded[reference]
        b, tb = loaded[candidate]
        comparisons[name] = {'reference_label': reference, 'candidate_label': candidate,
                             'metrics': compare(a, b, ta, tb)}
    result = {'created_at': time.time(), 'protocol': manifest['protocol'], 'common_update': common,
              'inputs': inputs, 'comparisons': comparisons,
              'uncertainty': 'Paired bootstrap over 512 motions, 2000 repeats; one training seed. Simulator contact trajectories need not be bitwise identical.',
              'checks': {'checkpoint_hashes_unchanged': True, 'paired_worlds_motions_query_states': True,
                         'warmup_protocol_paired': True, 'full_nominal_worlds': 52, 'limb_limits_kg': [2.5, 2.5, 4, 4]}}
    write(out / 'comparison.json', result)
    lines = ['混合 DR、warm、512条motion，预热500步、正式评测最多1000步，seed 20001。',
             '52个完整nominal环境；其余环境每项静态DR参数50% nominal / 50%范围采样，手部/小腿负载上限各2.5/4 kg。',
             '同一motion取两组共同存活时段，再等权平均。误差变化负数表示改善。', '']
    for name in ('same_update', 'latest_at_selection'):
        c = comparisons[name]
        title = '同轮数主对照' if name == 'same_update' else '开始测试时各自最新checkpoint（轮数可能不同）'
        lines += [f"{title}：{c['reference_label']} → {c['candidate_label']}", '',
                  '| 指标 | baseline | latent | latent误差变化 | 95% CI |', '|---|---:|---:|---:|---:|']
        for title, key in METRICS:
            m = c['metrics'][key]
            lo, hi = m['reduction_percent_ci95']
            lines.append(f"| {title} | {m['reference']:.6f} | {m['candidate']:.6f} | {-m['reduction_percent']:+.2f}% | [{-hi:+.2f}%, {-lo:+.2f}%] |")
        a, b = inputs[c['reference_label']], inputs[c['candidate_label']]
        lines += ['', f"失败数：baseline {a['failures']}/512，latent {b['failures']}/512。", '']
    for suffix, title in (('_vs_initial', '相对各自初始策略'), ('_since_u1301', '相对各自上次u1301')):
        lines += [title, '', '| 组别 | 局部body | joint | 全局body | 全局anchor |', '|---|---:|---:|---:|---:|']
        for job in manifest['jobs']:
            c = comparisons[job['label'] + suffix]
            changes = [f"{-c['metrics'][key]['reduction_percent']:+.2f}%" for _, key in METRICS]
            lines.append('| ' + job['label'] + ' | ' + ' | '.join(changes) + ' |')
        lines.append('')
    lines += ['只有一个训练种子，置信区间仅反映motion配对采样不确定性。完整区间、覆盖率、失败率百分点差及输入哈希见comparison.json。',
              '本次为独立评测；训练持续运行，定期评测保持关闭。', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'report': str(out / 'report.md'), 'comparisons': {
        k: {name: v['metrics'][name] for _, name in METRICS} for k, v in comparisons.items()
        if k in ('same_update', 'latest_at_selection')}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-stopped-baseline', action='store_true',
                        help='Evaluate the explicitly stopped baseline while preserving the live encoder and latent PPO')
    args = parser.parse_args()
    before = training_status()
    if not before['concat']['live'] or any(not row['periodic_evaluation_disabled'] for row in before.values()):
        raise RuntimeError('Expected live latent PPO and disabled periodic evaluations')
    encoder_before = None
    if not before['baseline']['live']:
        if not args.allow_stopped_baseline:
            raise RuntimeError('A stopped baseline requires --allow-stopped-baseline')
        stopped = read(RUN / 'baseline_stopped.json')
        if before['baseline']['progress']['completed_updates'] != stopped['completed_updates']:
            raise RuntimeError('Baseline progress differs from its explicit stop record')
        encoder_before = encoder_status(Path(stopped['encoder_run']))
        if not encoder_before['live']:
            raise RuntimeError('Expected the replacement encoder trainer to remain live')
    jobs, common = select_latest()
    assert len(MANIFEST.read_text().splitlines()) == 512
    out = RUN / 'manual_evaluation' / ('mixture_warm_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'prepared', 'created_at': time.time(), 'common_update': common, 'jobs': jobs,
                'training_before': before, 'encoder_before': encoder_before,
                'protocol': {'case': 'mixture_warm', 'motions': 512,
                'manifest': str(MANIFEST), 'manifest_sha256': digest(MANIFEST), 'seed': 20001,
                'steps': 1000, 'warmup_steps': 500, 'memory_start': 'warm', 'global_metrics': True}}
    children = {}
    try:
        for job in jobs:
            job['output'], job['log'] = str(out / (job['label'] + '.json')), str(out / (job['label'] + '.log'))
            command = [str(ROOT / '.venv/bin/python'), '-B', '-u', '-m',
                       'intact_tracking.cli.memory350_history5_policy_eval', '--checkpoint', job['checkpoint'],
                       '--motion-manifest', str(MANIFEST), '--motion-path', '/data_zcy/wxy/motion_data_correct/motion_data_full',
                       '--output', job['output'], '--seed', '20001', '--steps', '1000', '--memory-start', 'warm',
                       '--warmup-steps', '500', '--global-metrics']
            job['command'] = command
            with open(job['log'], 'w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=evaluation_environment(job['gpu'], process_environment()),
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children[job['label']] = child
            job.update(pid=child.pid, start_ticks=process_identity(child.pid)['start_ticks'])
        manifest['status'] = 'running'
        write(out / 'processes.json', manifest)
        print(json.dumps({'started': str(out), 'common_update': common,
                          'jobs': [{k: j[k] for k in ('label', 'pid', 'gpu')} for j in jobs]}), flush=True)
        started = time.monotonic()
        while any(p.poll() is None for p in children.values()):
            progress = {}
            for job in jobs:
                code = children[job['label']].poll()
                if code not in (None, 0):
                    raise RuntimeError(f"{job['label']} failed with exit {code}; see {job['log']}")
                last = None
                for line in Path(job['log']).read_text(errors='replace').splitlines():
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict) and ('step' in row or 'failure_rate' in row):
                            last = {k: row[k] for k in ('step', 'active', 'failed', 'failure_rate') if k in row}
                    except ValueError:
                        pass
                progress[job['label']] = {'exit_code': code, 'progress': last}
            print(json.dumps({'elapsed_seconds': round(time.monotonic() - started), 'evaluations': progress}), flush=True)
            if time.monotonic() - started > 7200:
                raise TimeoutError('One-off evaluation exceeded two hours')
            time.sleep(30)
        assert all(p.returncode == 0 for p in children.values())
        assert digest(MANIFEST) == manifest['protocol']['manifest_sha256']
        analyze(out, manifest)
        after = training_status()
        for arm in before:
            assert after[arm]['pid'] == before[arm]['pid'] and after[arm]['start_ticks'] == before[arm]['start_ticks']
            assert after[arm]['live'] == before[arm]['live'] and after[arm]['periodic_evaluation_disabled']
        encoder_after = encoder_status(Path(encoder_before['run_root'])) if encoder_before else None
        if encoder_before:
            assert encoder_after == encoder_before
        manifest.update(status='complete', completed_at=time.time(), training_after=after,
                        encoder_after=encoder_after,
                        evaluator_exit_codes={k: v.returncode for k, v in children.items()})
        write(out / 'processes.json', manifest)
        print(json.dumps({'complete': str(out / 'report.md'), 'training': after}), flush=True)
    except BaseException as error:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        manifest.update(status='failed', error=repr(error), failed_at=time.time())
        write(out / 'processes.json', manifest)
        raise


if __name__ == '__main__':
    main()
