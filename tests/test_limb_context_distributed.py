import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rsl_rl.utils.logger import Logger

from intact_tracking.distributed import DistributedContext
from intact_tracking.limb_context_distributed import (
    DistributedResidualPPO, SynchronizedDecayVecNorm, audit_rank_agreement,
    main_process_call,
)
from intact_tracking.residual_policy import DecayVecNorm, ResidualPPO
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.cli.limb_context_train import attach_json_logger


class SmallCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(2, 1)
        self.obs_normalizer = SynchronizedDecayVecNorm(2, decay=.9)

    def forward(self, obs):
        return torch.zeros(obs.shape[0], 1)

    def get_hidden_state(self):
        return None

    def reset(self, **kwargs):
        pass


def distributed_worker(rank, directory):
    directory = Path(directory)
    dist.init_process_group('gloo', init_method=(directory / 'rendezvous').as_uri(), rank=rank, world_size=2)
    context = DistributedContext(rank, rank, 2, torch.device('cpu'), 'gloo')
    try:
        torch.manual_seed(121)
        actor, critic = torch.nn.Linear(2, 1), SmallCritic()
        normalizer = critic.obs_normalizer
        reference = DecayVecNorm(2, decay=.9)
        # A restored prior must be decayed once, not added once per rank.
        for norm in (normalizer, reference):
            norm.sum.copy_(torch.tensor([12., -4.]))
            norm.ssq.copy_(torch.tensor([40., 30.]))
            norm.count.fill_(8)
        for step in range(2):
            batches = [torch.tensor([[1., 3.], [2., 6.]]) + 20*r + step for r in range(2)]
            normalizer.update(batches[rank])
            reference.update(torch.cat(batches))
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(normalizer.state_dict()[key], value)

        algorithm = object.__new__(DistributedResidualPPO)
        algorithm.actor, algorithm.critic = actor, critic
        algorithm.gpu_world_size, algorithm.is_multi_gpu = 2, True
        algorithm.normalize_advantage_per_mini_batch = False
        algorithm.gamma, algorithm.lam = .99, .95
        rewards = torch.tensor([1., 2.]) if rank == 0 else torch.tensor([10., 20.])
        algorithm.storage = SimpleNamespace(num_transitions_per_env=1,
            rewards=rewards.reshape(1,2,1), values=torch.zeros(1,2,1),
            returns=torch.zeros(1,2,1), dones=torch.ones(1,2,1))
        algorithm.compute_returns(torch.zeros(2,2))
        full = torch.tensor([1., 2., 10., 20.])
        torch.testing.assert_close(algorithm.storage.advantages.reshape(-1),
                                   ((full-full.mean())/full.std())[rank*2:rank*2+2])

        optimizer = torch.optim.SGD([*actor.parameters(), *critic.parameters()], lr=.1)
        before = actor.weight.detach().clone()
        for parameter in [*actor.parameters(), *critic.parameters()]:
            parameter.grad = torch.full_like(parameter, rank + 1.)
        ResidualPPO.reduce_parameters(algorithm)
        optimizer.step()
        torch.testing.assert_close(actor.weight, before - .15)
        runner = SimpleNamespace(alg=algorithm, completed_learning_updates=1,
                                 is_distributed=True, device='cpu', stop_requested=(rank==0))
        assert ResidualOnPolicyRunner._collective_stop_requested(runner)
        agreement = audit_rank_agreement(runner, context)
        assert agreement['passed'] and agreement['world_size'] == 2

        # Use the actual RSL buffer collection path, including writer=None on rank 1.
        cfg = {'num_steps_per_env':24, 'algorithm':{'rnd_cfg':None}}
        logger = Logger(str(directory), cfg, {}, 2, True, 2, rank, 'cpu')
        logger.log = lambda **kwargs: logger.ep_extras.clear()
        runner.logger, runner.cfg, runner.env = logger, cfg, SimpleNamespace(num_envs=2)
        attach_json_logger(runner, directory, context)
        logger.process_env_step(rewards, torch.ones(2), {'log':{'metric':float(rank+1)}})
        assert len(logger.rewbuffer)==2 and logger.writer is None
        logger.log(it=0, collect_time=rank+1., learn_time=.5, loss_dict={'value':rank+1.},
                   action_std=torch.ones(2), learning_rate=.001)
        assert not logger.ep_extras
        context.barrier()
        records = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        assert len(records)==1
        record=records[0]
        assert record['mean_reward']==8.25 and record['mean_episode_length']==1
        assert record['episode_window_count_across_ranks']==4
        assert record['loss']['value']==1.5 and record['episode_metrics']['metric']==1.5
        assert record['collect_seconds']==2 and record['global_transitions']==96
        with pytest.raises(RuntimeError, match='PermissionError: example'):
            main_process_call(context, lambda: (_ for _ in ()).throw(PermissionError('example')))
        # Router buffers are not trainable parameters, but must agree too.
        from intact_tracking.memory350_online_kmeans import OnlineKMeansRouter
        actor.residual_mlp = torch.nn.Module()
        actor.residual_mlp.router = OnlineKMeansRouter()
        critic.mlp.router = OnlineKMeansRouter()
        for router in (actor.residual_mlp.router, critic.mlp.router):
            router.initialized.fill_(True)
            router.update_count.fill_(1)
        agreement = audit_rank_agreement(runner, context)
        assert agreement['ranks'][0]['actor_router_sha256'] is not None
        if rank == 1:
            actor.residual_mlp.router.centers.add_(1)
        with pytest.raises(RuntimeError, match='actor_router_sha256'):
            audit_rank_agreement(runner, context)
        actor.residual_mlp.router.load_state_dict(critic.mlp.router.state_dict())
        critic.mlp.router.centers.add_(1)
        with pytest.raises(RuntimeError, match='Actor and critic router states differ'):
            audit_rank_agreement(runner, context)
        critic.mlp.router.load_state_dict(actor.residual_mlp.router.state_dict())
        # Divergent state must not receive a successful completion marker.
        if rank==1:
            critic.obs_normalizer.sum.add_(1)
        with pytest.raises(RuntimeError, match='critic_normalizer'):
            audit_rank_agreement(runner, context)
    finally:
        dist.destroy_process_group()


def test_collective_normalization_gradients_logging_and_stop(tmp_path):
    mp.spawn(distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_jobs_require_two_gpus_and_exclude_old_initialization_or_scale(tmp_path):
    spec=importlib.util.spec_from_file_location('four_gpu_scheduler', Path(__file__).parents[1]/'scripts/run_limb_context_experiment.py')
    scheduler=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    jobs=scheduler.jobs(tmp_path)
    for job in [j for j in jobs if j['phase']=='ppo']:
        assert job['gpus_needed']==2
        assert '--nproc-per-node=2' in job['command']
        assert job['command'][job['command'].index('--num-envs')+1]=='8192'
        assert 'stage1' in job['dependencies']
    old=tmp_path/'baseline_121'
    old.mkdir()
    (old/'completion.json').write_text(json.dumps({'complete':True,'completed_updates':5000}))
    baseline=next(j for j in jobs if j['name']=='baseline_121')
    assert not scheduler.job_completion(baseline)[0]
    output=Path(baseline['output'])
    output.mkdir(parents=True)
    (output/'completion.json').write_text((old/'completion.json').read_text())
    assert not scheduler.job_completion(baseline)[0]
    warm={'complete':True,'completed_updates':5000,
          'distributed':{'world_size':2,'num_envs_per_rank':8192,'global_num_envs':16384},
          'distributed_parameter_agreement':{'passed':True,'world_size':2}}
    (output/'completion.json').write_text(json.dumps(warm))
    assert not scheduler.job_completion(baseline)[0]
    warm['initialization_protocol']=scheduler.PPO_INITIALIZATION
    (output/'completion.json').write_text(json.dumps(warm))
    assert scheduler.job_completion(baseline)[0]
    baseline['gpu_pool']=[0,1]
    assert scheduler.select_job_gpus(baseline,[2,3,4,5,6,7])==[]
    assert scheduler.select_job_gpus(baseline,list(range(8)))==[0,1]
    film=next(j for j in jobs if j['name']=='film_121')
    assert scheduler.select_job_gpus(film,list(range(8)))==[2,3]
    assert scheduler.select_job_gpus(film,[0,1,4,5,6,7])==[]
    encoder=next(j for j in jobs if j['name']=='stage1')
    assert encoder['dependencies']==['smoke_stage1']
    assert scheduler.select_job_gpus(encoder,list(range(8)))==[0,1,2,3]


def test_four_gpu_layout_uses_all_eight_cards_and_only_matched_baseline_film(tmp_path):
    spec=importlib.util.spec_from_file_location('new_four_gpu_scheduler', Path(__file__).parents[1]/'scripts/run_limb_context_experiment.py')
    scheduler=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    (tmp_path/'experiment_layout.json').write_text(json.dumps({
        'gpus_per_policy':4,'num_envs_per_gpu':8192,'allowed_gpus':list(range(8)),
        'paired_seeds':[121,122,123],'controls':[],'reuse_stage1':True}))
    queue=scheduler.jobs(tmp_path)
    encoder=next(j for j in queue if j['name']=='stage1')
    assert set(j['name'] for j in queue)=={
        'stage1','smoke_baseline','smoke_film',
        *(f'{fusion}_{seed}' for fusion in ('baseline','film') for seed in (121,122,123))}
    for job in queue:
        if job['phase']!='ppo':
            assert job.get('reused') and job['dependencies']==[]
            continue
        assert '--nproc-per-node=4' in job['command']
        assert job['command'][job['command'].index('--training-terminations')+1]=='no_ee_body_pos'
        assert job['gpus_needed']==4
        expected=[0,1,2,3] if 'baseline' in job['name'] else [4,5,6,7]
        assert scheduler.select_job_gpus(job,list(range(8)))==expected
        assert scheduler.select_job_gpus(job,expected[:-1])==[]
    job=next(j for j in queue if j['name']=='film_121')
    output=Path(job['output']);output.mkdir(parents=True)
    result={'complete':True,'completed_updates':5000,'initialization_protocol':scheduler.PPO_INITIALIZATION,
            'distributed':{'world_size':4,'num_envs_per_rank':8192,'global_num_envs':32768},
            'distributed_parameter_agreement':{'passed':True,'world_size':4}}
    (output/'completion.json').write_text(json.dumps(result))
    assert scheduler.job_completion(job)[0]
    result['distributed']['global_num_envs']=16384
    (output/'completion.json').write_text(json.dumps(result))
    assert not scheduler.job_completion(job)[0]
    assert not scheduler.stage2_authorized(tmp_path)
    gate=tmp_path/'stage2_release.json'
    gate.write_text(json.dumps({'approved':True}))
    assert not scheduler.stage2_authorized(tmp_path)
    stage1=tmp_path/'stage1'
    stage1.mkdir()
    (stage1/'best.pt').write_bytes(b'context model')
    decision={'approved':True,'decision_source':'explicit_user_convergence_decision',
              'context_sha256':__import__('hashlib').sha256(b'context model').hexdigest()}
    gate.write_text(json.dumps(decision))
    assert scheduler.stage2_authorized(tmp_path)
    (stage1/'best.pt').write_bytes(b'changed context model')
    assert not scheduler.stage2_authorized(tmp_path)
    # Neither an old cap nor an automatic plateau may complete manual stage 1.
    complete=stage1/'completion.json'
    complete.write_text(json.dumps({'completed_updates':8000,'hit_cap':True,'converged':True,
                                     'stopped':False,'stopping_mode':'automatic_budget_or_plateau'}))
    assert not scheduler.job_completion(encoder)[0]
    complete.write_text(json.dumps({'completed_updates':10000,'hit_cap':False,'converged':False,
                                     'stopped':True,'stopping_mode':'until_user_stop'}))
    assert not scheduler.job_completion(encoder)[0]
    decision.update(context_sha256=__import__('hashlib').sha256(b'changed context model').hexdigest(),
                    stage1_completed_updates=10000)
    gate.write_text(json.dumps(decision))
    assert scheduler.job_completion(encoder)[0]
