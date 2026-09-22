from copy import deepcopy

import torch

from test_heavy_baselines import models  # shared frozen-tracker fixture
from intact_tracking.distributed import DistributedContext
from intact_tracking.heavy_rma_teacher import PHYSICS_GROUP
from intact_tracking.heavy_rma_student import (
    RMAStudentActor, RMAHistoryEncoder, ProprioHistory, HistoryBank, EMBEDDING_GROUP,
)
from intact_tracking.rma_student_distillation import Distiller, frozen_digest, accumulate_episode_logs


class SmallStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.control=torch.nn.Linear(3,3).requires_grad_(False)
        self.adaptation=RMAHistoryEncoder()


def test_episode_logs_keep_python_scalars_and_sparse_tensor_metrics():
    totals, counts = {}, {}
    reward = torch.tensor([1., 3.], requires_grad=True)
    accumulate_episode_logs(totals, counts, {
        'Metrics/motion/error_joint_pos': .4,
        'Episode_Termination/anchor_pos': 3,
        'Episode_Reward/joint_pos_tracking': reward,
        'description': 'ignored',
    })
    accumulate_episode_logs(totals, counts, {'Metrics/motion/error_joint_pos': .2})
    assert set(totals) == {'Metrics/motion/error_joint_pos', 'Episode_Termination/anchor_pos',
                           'Episode_Reward/joint_pos_tracking'}
    assert abs(totals['Metrics/motion/error_joint_pos']/counts['Metrics/motion/error_joint_pos']-.3) < 1e-7
    assert totals['Episode_Termination/anchor_pos'] == 3
    assert counts['Episode_Termination/anchor_pos'] == 1
    assert totals['Episode_Reward/joint_pos_tracking'].item() == 2
    assert not totals['Episode_Reward/joint_pos_tracking'].requires_grad


def distributed_reference_worker(rank,init_file,output):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2)
    try:
        torch.manual_seed(123)
        actor=SmallStudent()
        dist=DistributedContext(rank,rank,2,torch.device('cpu'),'gloo')
        trainer=Distiller(actor,dist,mini_batches=1,microbatch=4)
        bank,targets=reference_batch(rank)
        trainer.update(bank,targets)
        assert trainer.audit()['passed']
        if rank==0:torch.save(actor.state_dict(),output)
    finally:torch.distributed.destroy_process_group()


def reference_batch(rank=None):
    g=torch.Generator().manual_seed(789)
    frames=torch.randn(4,52,122,generator=g)
    targets=torch.randn(4,64,generator=g)
    selection=slice(None) if rank is None else slice(2*rank,2*rank+2)
    bank=HistoryBank(4 if rank is None else 2,2,'cpu')
    bank.frames.copy_(frames[selection]);bank.counts.fill_(50)
    return bank,targets[selection]


def test_two_rank_gradient_and_normalizer_match_single_global_batch(tmp_path):
    torch.multiprocessing.spawn(distributed_reference_worker,args=(str(tmp_path/'gloo_init'),str(tmp_path/'distributed.pt')),nprocs=2,join=True)
    # Match the worker thread setting; near-zero gradients amplified by Adam's
    # epsilon otherwise also measure CPU reduction-kernel differences.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(123)
        actor=SmallStudent();dist=DistributedContext(0,0,1,torch.device('cpu'),None)
        trainer=Distiller(actor,dist,mini_batches=1,microbatch=8)
        trainer.update(*reference_batch())
        actual=torch.load(tmp_path/'distributed.pt',weights_only=True)
        for name,want in actor.state_dict().items():
            torch.testing.assert_close(actual[name],want,atol=2e-6,rtol=2e-5)
    finally:
        torch.set_num_threads(threads)


def student_from(teacher, obs):
    groups={'actor':['features'],'critic':['priv']}
    actor=RMAStudentActor(obs,groups,'actor',29,tracker_checkpoint='unused.pt',
        tracker_state_dict=teacher.tracker.state_dict(),tracker_actor_kwargs={},tracker_obs_groups=groups,
        residual_hidden_dims=(16,8),initialization_seed=10128,residual_output_mode='unbounded',residual_scale=1.)
    actor.initialize_from_teacher(teacher)
    return actor


def test_actor_privilege_removal_teacher_alignment_and_freezing(models):
    teacher,_,obs=models('rma_teacher')
    with torch.no_grad():teacher.residual_mlp[-1].weight.normal_(std=.2)
    student=student_from(teacher,obs)
    latent=teacher.dr_encoder(obs[PHYSICS_GROUP]).detach()
    clean=obs.select('features')
    clean[EMBEDDING_GROUP]=latent
    torch.testing.assert_close(student(clean),teacher(obs),atol=0,rtol=0)
    poisoned=clean.clone();poisoned[PHYSICS_GROUP]=torch.full_like(obs[PHYSICS_GROUP],float('nan'))
    poisoned['priv']=torch.full_like(obs['priv'],float('nan'))
    torch.testing.assert_close(student(poisoned),student(clean),atol=0,rtol=0)
    before=frozen_digest(student)
    student.train()
    assert not student.tracker.training and not student.residual_mlp.training
    z=student.adaptation(torch.randn(8,50,122),torch.full((8,),50))
    (z-latent).square().mean().backward()
    assert all(p.grad is None for n,p in student.named_parameters() if not n.startswith('adaptation.'))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.adaptation.parameters())
    assert sum(p.numel() for p in student.adaptation.parameters())==44608
    assert frozen_digest(student)==before


def test_history_reconstruction_causality_padding_and_resets():
    live=ProprioHistory(3,'cpu'); bank=HistoryBank(3,7,'cpu')
    for t in range(55):live.append(torch.full((3,122),float(t)))
    bank.start(live)
    expected=[]
    for step in range(7):
        expected.append((live.frames.clone(),live.count.clone()))
        live.append(torch.full((3,122),float(55+step)),torch.tensor([step==1,step==4,False]))
        bank.append(step,live)
    encoder=RMAHistoryEncoder()
    for step,(history,count) in enumerate(expected):
        reconstructed,actual_count,envs=bank.batch(torch.arange(3)*7+step)
        torch.testing.assert_close(actual_count,count)
        torch.testing.assert_close(encoder(reconstructed,actual_count),encoder(history,count))
        assert reconstructed[:,-1].eq(float(54+step)).all()
    assert expected[2][1].tolist()==[1,50,50]
    assert expected[5][1].tolist()==[4,1,50]
    history,count=expected[2]
    history[0,:-1]=float('nan')
    assert torch.isfinite(encoder(history,count)).all()
    assert bank.frames.numel()==3*57*122


def test_microbatch_accumulation_and_exact_learning_resume(models,tmp_path):
    teacher,_,obs=models('rma_teacher')
    a=student_from(teacher,obs); b=student_from(teacher,obs)
    dist=DistributedContext(0,0,1,torch.device('cpu'),None)
    one=Distiller(a,dist,mini_batches=2,microbatch=16)
    micro=Distiller(b,dist,mini_batches=2,microbatch=3)
    history=ProprioHistory(8,'cpu')
    for _ in range(50):history.append(torch.randn(8,122))
    bank=HistoryBank(8,4,'cpu'); bank.start(history)
    for step in range(4):
        history.append(torch.randn(8,122));bank.append(step,history)
    targets=teacher.dr_encoder(obs[PHYSICS_GROUP]).detach()
    one.update(bank,targets); micro.update(bank,targets)
    for x,y in zip(one.encoder.parameters(),micro.encoder.parameters()):
        torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
    path=tmp_path/'checkpoint.pt'
    one.save(path,source_state={'inference_bundle_version':1,'frozen_tracker':{}},cfg={},
             metadata={'teacher_source':{'sha256':'test'}},teacher_encoder=teacher.dr_encoder)
    state=torch.load(path,weights_only=False)
    resumed=Distiller(student_from(teacher,obs),dist,mini_batches=2,microbatch=16)
    resumed.resume(state)
    one.update(bank,targets); resumed.update(bank,targets)
    for k,v in one.encoder.state_dict().items():
        torch.testing.assert_close(v,resumed.encoder.state_dict()[k],atol=0,rtol=0)
    assert one.optimizer_steps==resumed.optimizer_steps==4
    assert one.audit()['passed'] and resumed.audit()['passed']
