"""Launch the approved RMA student on local GPUs 4-7, with a required full-scale preflight."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from monitor_memory350_nominal_direction import process_identity
from run_limb_context_experiment import process_environment
from run_memory350_scale_nominal_stage1 import gpu_status

ROOT=Path(__file__).resolve().parents[1]
PYTHON=str(ROOT/'.venv/bin/python')
RUN_ROOT=ROOT/'runs/144000-exp-heavy/baselines/rma_student_uniform'
TEACHER=ROOT/'runs/144000-exp-heavy/baselines/rma_teacher_uniform/continuous_uniform/checkpoint_9000.pt'
TEACHER_HASH='fbd26367b3cc014f2d7c2fa181bb7801912fefe017f4dee68286d5e4f53ec9e2'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def command_for(output,teacher,*,smoke=False,resume=None,iterations=None):
    result=[PYTHON,'-B','-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        '--max-restarts=0','-m','intact_tracking.cli.heavy_rma_student_train',
        '--teacher-checkpoint',str(teacher),'--output-dir',str(output)]
    if smoke:
        result+=['--bounded-smoke','--iterations',str(iterations or 3),'--save-interval','1','--motion-file',
                 str(ROOT/'runs/144000_exp/smoke_motions/walk1_subject1.motion.npz')]
    if resume:result+=['--resume',str(resume)]
    return result


def launch(root,output,teacher,*,smoke=False,resume=None,iterations=None):
    cards=gpu_status([4,5,6,7])
    if any(c['processes'] or c['free_mib']<60000 for c in cards):
        raise RuntimeError('GPUs 4-7 must be free; other trainers were left running')
    command=command_for(output,teacher,smoke=smoke,resume=resume,iterations=iterations)
    env=process_environment()
    env.update(CUDA_VISIBLE_DEVICES='4,5,6,7',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        WANDB_MODE='disabled' if smoke else 'online',WANDB_RESUME='allow',
        WANDB_RUN_ID='heavy-rma-student-'+hashlib.sha256(str(output).encode()).hexdigest()[:10])
    key=ROOT/'.runtime/limb_context/wandb_api_key'
    if key.exists():env['WANDB_API_KEY']=key.read_text().strip()
    log=root/(output.name+f'_{time.time_ns()}.log')
    with log.open('ab') as f:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record={**process_identity(child.pid),'command':command,'output':str(output),'log':str(log),
        'gpus':[4,5,6,7],'started_at':time.time(),'wandb_id':env['WANDB_RUN_ID'],'smoke':smoke,
        'maximum_updates':iterations or 3 if smoke else None}
    record_path=root/('smoke_process.json' if smoke else 'active_process.json')
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    def stop(*_):
        children=Path(f'/proc/{child.pid}/task/{child.pid}/children')
        if children.exists():
            for pid in children.read_text().split():
                try:os.kill(int(pid),signal.SIGTERM)
                except ProcessLookupError:pass
    handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:code=child.wait()
    finally:
        for s,h in handlers.items():signal.signal(s,h)
    record.update(returncode=code,ended_at=time.time(),live=False)
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    if code:raise RuntimeError(f'Student exited {code}; inspect {log}')
    return json.loads((output/'completion.json').read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--root',type=Path,default=RUN_ROOT)
    args=p.parse_args();root=args.root.resolve()
    if not root.is_relative_to(ROOT):raise ValueError('Output must remain in repository')
    if not args.run and not args.smoke:
        print(json.dumps(command_for(root/'continuous_uniform',root/'teacher_checkpoint_9000.pt')));return
    root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        teacher=root/'teacher_checkpoint_9000.pt'
        if not teacher.exists():
            if digest(TEACHER)!=TEACHER_HASH:raise ValueError('Source teacher changed')
            temporary=teacher.with_suffix('.tmp');shutil.copyfile(TEACHER,temporary);temporary.rename(teacher)
        if digest(teacher)!=TEACHER_HASH:raise ValueError('Frozen teacher changed')
        if args.smoke:
            output=root/f'smoke_4gpu8192_{time.time_ns()}'
            first=launch(root,output,teacher,smoke=True)
            if not first['complete'] or not first['distributed_parameter_agreement']['passed']:
                raise RuntimeError('Initial smoke failed')
            second=launch(root,output,teacher,smoke=True,resume=output/'checkpoint_final.pt',iterations=4)
            if not second['complete'] or second['completed_updates']!=4 or not second['distributed_parameter_agreement']['passed']:
                raise RuntimeError('Resume smoke failed')
            cfg=json.loads((output/'run_config.json').read_text())
            (root/'preflight.json').write_text(json.dumps({'passed':True,'output':str(output),
                'source_sha256':cfg['research_source_sha256'],'teacher_sha256':TEACHER_HASH,
                'completed_updates':4,'resume_verified':True,'agreement':second['distributed_parameter_agreement']},indent=2)+'\n')
            return
        check=json.loads((root/'preflight.json').read_text())
        if not check['passed'] or any(digest(ROOT/k)!=v for k,v in check['source_sha256'].items()):
            raise RuntimeError('Run full-scale preflight for current code first')
        output=root/'continuous_uniform'
        checkpoints=list(output.glob('checkpoint_*.pt')) if output.exists() else []
        resume=max(checkpoints,key=lambda p:p.stat().st_mtime_ns) if checkpoints else None
        launch(root,output,teacher,resume=resume)


if __name__=='__main__':main()
