"""Launch the authorized fixed-prototype versus PPO-router load experiment."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/limb_context_20260916_payload256_top5_moe'
CONTEXT = ROOT/'runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/update_035857.pt'
MOTIONS = '/data_zcy/wxy/motion_data_correct/motion_data_full'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('smoke','train'),required=True)
    args = parser.parse_args()
    prototype = RUN/'calibration/prototypes.pt'
    if not prototype.is_file(): raise RuntimeError('Complete real calibration before launching either arm')
    if args.stage == 'train':
        for arm in ('baseline','latent'):
            completion = json.loads((RUN/f'smoke_{arm}'/'completion.json').read_text())
            if completion['completed_updates'] != 2 or not completion['distributed_parameter_agreement']['passed']:
                raise RuntimeError(f'{arm} has not passed the two-rank real-physics smoke test')
    env_base = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                    PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',WANDB_RESUME='allow')
    for key,name in {'TMPDIR':'tmp','MPLCONFIGDIR':'matplotlib','XDG_CACHE_HOME':'cache','WARP_CACHE_PATH':'warp',
                     'TORCHINDUCTOR_CACHE_DIR':'inductor','WANDB_CACHE_DIR':'wandb_cache','WANDB_DIR':'wandb',
                     'CUDA_CACHE_PATH':'cuda','WANDB_CONFIG_DIR':'wandb_config','WANDB_DATA_DIR':'wandb_data',
                     'SP_TRACKING_MULTIMOTION_MANIFEST_DIR':'manifests','MJLAB_BOOTSTRAP_DEBUG_DIR':'bootstrap'}.items():
        directory=ROOT/'.runtime/limb_context'/name
        directory.mkdir(parents=True,exist_ok=True)
        env_base[key]=str(directory)
    if args.stage == 'train':
        env_base['WANDB_API_KEY']=(ROOT/'.runtime/limb_context/wandb_api_key').read_text().strip()
    for arm in ('baseline','latent'):
        smoke = args.stage == 'smoke'
        name = ('smoke_' if smoke else '')+arm
        out = RUN/name
        if out.exists() or (RUN/f'{name}_process.json').exists(): raise FileExistsError(out)
        gpus = ([1,2] if arm=='baseline' else [4,5]) if smoke else ([0,1,2,3] if arm=='baseline' else [4,5,6,7])
        mode = 'learned' if arm=='baseline' else 'latent'
        run_id='payload256-'+hashlib.sha256(str(out).encode()).hexdigest()[:12]
        command=[str(ROOT/'.venv/bin/python'),'-B','-u','-m','torch.distributed.run','--standalone',
                 f'--nproc-per-node={len(gpus)}','--max-restarts=0','-m','intact_tracking.cli.payload_prototype_train',
                 '--route-mode',mode,'--prototype-file',str(prototype),'--output-dir',str(out),
                 '--training-ranks',str(len(gpus)),'--num-envs','8192','--rollout-steps','24',
                 '--epochs','5','--mini-batches','4','--actor-lr','0.0001','--critic-lr','0.0005',
                 '--seed','121','--save-interval','2' if smoke else '250','--policy-precision','fp32',
                 '--wandb-mode','offline' if smoke else 'online','--wandb-group',RUN.name,
                 '--wandb-name',f'payload256-top5-{arm}-4x8192']
        if arm=='latent': command.extend(['--context-checkpoint',str(CONTEXT)])
        if smoke:
            old=json.loads((ROOT/'runs/limb_context_20260916_tracker_film_obs_latent_4x8192_cold/preflight_8192/run_config.json').read_text())
            command.extend(['--motion-file',old['arguments']['motion_file'],'--iterations','2'])
        else:
            command.extend(['--motion-path',MOTIONS,'--until-user-stop'])
        env=dict(env_base,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),WANDB_RUN_ID=run_id,
                 WANDB_MODE='offline' if smoke else 'online')
        log_path=RUN/f'{name}.log'
        with log_path.open('xb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                      stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record={'pid':process.pid,'command':command,'gpus':gpus,'wandb_run_id':run_id,
                'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'output':str(out),'log':str(log_path),'unbounded':not smoke,
                'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__),*sorted((ROOT/'src/intact_tracking').glob('payload_prototype*.py')),
                                           *sorted((ROOT/'src/intact_tracking/cli').glob('payload_prototype*.py'))]}}
        (RUN/f'{name}_process.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:record[k] for k in ('pid','gpus','output','wandb_run_id')}),flush=True)


if __name__ == '__main__':
    main()
