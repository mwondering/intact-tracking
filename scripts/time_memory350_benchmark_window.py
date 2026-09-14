"""Bounded, audited pause of the owned continuation for uncontended GPU timing.

Compilation/warmup happens before this controller. An independent watchdog
resumes exactly the recorded worker PIDs after at most 45 seconds, including if
the controller itself is killed. It never terminates/restarts training.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/limb_context_20260912_memory350_encoder2x_nominal50_weakpairs'


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return int(stat[19])


def resume(workers):
    for pid, ticks in workers:
        try:
            if identity(pid) == ticks:
                os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ready-file', required=True)
    parser.add_argument('--audit-file', required=True)
    parser.add_argument('--watchdog-workers')
    args = parser.parse_args()
    if args.watchdog_workers:
        time.sleep(45)
        resume(json.loads(args.watchdog_workers))
        return
    ready = Path(args.ready_file)
    if not ready.exists() or Path(str(ready) + '.go').exists():
        raise RuntimeError('Benchmark must be ready and timing must not already have begun')
    state = json.loads((RUN / 'continuation_005000_010000/state.json').read_text())
    if state['status'] != 'continuation_training' or state.get('active_evaluation'):
        raise RuntimeError('Expected active owned training and no milestone evaluation')
    parent = state['job']['pid']
    assert identity(parent) == state['job']['process_start_ticks']
    workers = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ')
            if int(stat[1]) == parent and b'intact_tracking.cli.forward_memory_weak_pairs_train' in cmd:
                workers.append((int(p.name), int(stat[19])))
        except (FileNotFoundError, ProcessLookupError):
            pass
    assert len(workers) == 4, workers
    audit = dict(workers=workers, parent=parent, started_at=time.time(), attempts=[])
    audit_path = Path(args.audit_file)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for attempt in range(3):
            watchdog = subprocess.Popen([sys.executable, __file__, '--ready-file', str(ready),
                                         '--audit-file', str(audit_path), '--watchdog-workers', json.dumps(workers)],
                                        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stamp = time.monotonic()
            for pid, ticks in workers:
                assert identity(pid) == ticks
                os.kill(pid, signal.SIGSTOP)
            record = dict(attempt=attempt, paused_at=time.time(), watchdog_pid=watchdog.pid)
            audit['attempts'].append(record)
            audit_path.write_text(json.dumps(audit, indent=2) + '\n')
            time.sleep(2.5)
            utilization = subprocess.check_output(['nvidia-smi', '--id=4,5,6,7',
                '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            record['utilization_after_drain'] = utilization
            idle = all(int(line.split(',')[1]) <= 15 for line in utilization.strip().splitlines())
            if not idle:
                resume(workers)
                record['resumed_at'] = time.time()
                record['skipped_reason'] = 'GPU still busy, possibly a pending collective'
                watchdog.terminate()
                watchdog.wait()
                print(json.dumps(record), flush=True)
                time.sleep(2)
                continue
            record['timing_started_at'] = time.time()
            ready.with_name(ready.name + '.go').write_text('go\n')
            while not Path(str(ready) + '.done').exists() and time.monotonic() - stamp < 40:
                time.sleep(.1)
            record['completed_within_window'] = Path(str(ready) + '.done').exists()
            record['done_text'] = (Path(str(ready)+'.done').read_text() if record['completed_within_window'] else None)
            resume(workers)
            record['resumed_at'] = time.time()
            record['pause_seconds'] = time.monotonic() - stamp
            watchdog.terminate()
            watchdog.wait()
            print(json.dumps(record), flush=True)
            if not record['completed_within_window']:
                raise RuntimeError('Timing exceeded bounded pause; training was resumed')
            break
        else:
            raise RuntimeError('No idle GPU window obtained after three bounded attempts')
    finally:
        resume(workers)
        audit['finished_at'] = time.time()
        audit['final_process_states'] = {str(pid): Path(f'/proc/{pid}/stat').read_text().split()[2] for pid, _ in workers}
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
