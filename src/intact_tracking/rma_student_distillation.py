"""Online RMA embedding regression with synchronized minibatch updates."""

from contextlib import nullcontext
import copy
import json
from numbers import Real
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from intact_tracking.heavy_rma_student import INPUT_CONTRACT
from intact_tracking.limb_context_distributed import tensor_digest, main_process_call


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temporary, path)


def frozen_digest(actor):
    return tensor_digest((k, v) for k, v in actor.state_dict().items() if not k.startswith('adaptation.'))


def accumulate_episode_logs(totals, counts, info):
    """Preserve mjlab scalar tracking/termination metrics and tensor rewards."""
    for key, value in info.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean()
        elif isinstance(value, Real) and not isinstance(value, bool):
            value = float(value)
        else:
            continue
        totals[key] = totals.get(key, 0.) + value
        counts[key] = counts.get(key, 0) + 1


def rng_state(generator, device):
    return {'shuffle': generator.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state, generator, device):
    generator.set_state(state['shuffle'].cpu()); torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)
    np.random.set_state(state['numpy']); random.setstate(state['python'])


class Distiller:
    def __init__(self, actor, distributed, *, learning_rate=5e-4, mini_batches=4, microbatch=49152, seed=121):
        self.actor, self.distributed = actor, distributed
        self.mini_batches, self.microbatch = mini_batches, microbatch
        self.encoder = actor.adaptation
        self.network = (DistributedDataParallel(self.encoder,
                        device_ids=[distributed.local_rank] if distributed.device.type == 'cuda' else None,
                        broadcast_buffers=False) if distributed.enabled else self.encoder)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.generator = torch.Generator(device=distributed.device).manual_seed(seed+1000003*distributed.rank+73)
        self.completed_updates, self.optimizer_steps = 0, 0
        self.frozen_hash = frozen_digest(actor)

    def update(self, bank, targets):
        n = bank.frames.shape[0]*bank.steps
        if n % self.mini_batches:
            raise ValueError('Each rank must have equally sized effective minibatches')
        self.encoder.normalizer.update(bank.frames[:, 49:49+bank.steps], self.distributed)
        permutation = torch.randperm(n, generator=self.generator, device=bank.frames.device)
        mse_sum = torch.zeros(64, dtype=torch.float64, device=bank.frames.device)
        grad_sum = torch.zeros((), device=bank.frames.device)
        self.network.train()
        for batch in permutation.chunk(self.mini_batches):
            self.optimizer.zero_grad(set_to_none=True)
            pieces = batch.split(self.microbatch)
            for index, ids in enumerate(pieces):
                sync = (self.network.no_sync() if self.distributed.enabled and index < len(pieces)-1
                        else nullcontext())
                with sync:
                    history, count, env_ids = bank.batch(ids)
                    errors = (self.network(history, count)-targets[env_ids]).square()
                    loss = errors.mean()*(len(ids)/len(batch))
                    loss.backward()
                    mse_sum.add_(errors.detach().double().sum(0))
            norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1., error_if_nonfinite=True)
            grad_sum.add_(norm.detach())
            self.optimizer.step(); self.optimizer_steps += 1
        self.network.eval()
        self.completed_updates += 1
        self.distributed.all_reduce_sum(mse_sum)
        mse = mse_sum/(n*self.distributed.world_size)
        self.distributed.all_reduce_sum(grad_sum)
        moments = torch.cat((targets.double().sum(0), targets.double().square().sum(0),
                             targets.new_tensor([len(targets)], dtype=torch.float64)))
        self.distributed.all_reduce_sum(moments)
        variance = (moments[64:128]/moments[-1]-(moments[:64]/moments[-1]).square()).clamp_min(0)
        usable = variance > 1e-8
        nmse = mse/variance.clamp_min(1e-8)
        result = {'Distill/latent_mse': float(mse.mean()),
                  'Distill/gradient_norm': float(grad_sum/(self.mini_batches*self.distributed.world_size)),
                  'Distill/learning_rate': self.optimizer.param_groups[0]['lr'],
                  'Distill/optimizer_steps': self.optimizer_steps,
                  'Distill/latent_nmse': float(nmse[usable].mean()) if usable.any() else 0.,
                  'Distill/r2_valid_dimensions': int(usable.sum())}
        for i in range(64):
            result[f'DistillLatent/{i:02d}_mse'] = float(mse[i])
            if usable[i]:
                result[f'DistillLatent/{i:02d}_nmse'] = float(nmse[i])
                result[f'DistillLatent/{i:02d}_r2'] = float(1-nmse[i])
        return result

    def audit(self):
        actual = frozen_digest(self.actor)
        local = {'rank': self.distributed.rank, 'completed_updates': self.completed_updates,
                 'adaptation_sha256': tensor_digest(self.encoder.state_dict().items()),
                 'frozen_sha256': actual, 'frozen_unchanged': actual == self.frozen_hash,
                 'finite': all(bool(torch.isfinite(p).all()) for p in self.encoder.parameters()),
                 'only_adaptation_trainable': all(n.startswith('adaptation.') for n,p in self.actor.named_parameters() if p.requires_grad)}
        rows = self.distributed.all_gather_object(local)
        if (any(not r[k] for r in rows for k in ('finite','frozen_unchanged','only_adaptation_trainable'))
                or any(len({r[k] for r in rows}) != 1 for k in ('completed_updates','adaptation_sha256','frozen_sha256'))):
            raise RuntimeError(f'RMA student parameter audit failed: {rows}')
        return {'passed': True, 'world_size': self.distributed.world_size, 'ranks': rows}

    def save(self, path, *, source_state, cfg, metadata, teacher_encoder):
        agreement = self.audit()
        states = self.distributed.all_gather_object(rng_state(self.generator, self.distributed.device))
        def publish():
            payload = {
                'actor_state_dict': {k:v.detach().cpu() for k,v in self.actor.state_dict().items()},
                'optimizer_state_dict': self.optimizer.state_dict(),
                'teacher_dr_encoder_state_dict': {k:v.detach().cpu() for k,v in teacher_encoder.state_dict().items()},
                'teacher_source': metadata['teacher_source'], 'frozen_control_sha256': self.frozen_hash,
                'inference_bundle_version': source_state['inference_bundle_version'],
                'frozen_tracker': copy.deepcopy(source_state['frozen_tracker']),
                'cfg': cfg, 'residual_policy': copy.deepcopy(metadata), 'input_contract': INPUT_CONTRACT,
                'completed_updates': self.completed_updates, 'iter': self.completed_updates,
                'optimizer_steps': self.optimizer_steps, 'rng_states': states,
                'motion_sampling_state': None, 'distributed_parameter_agreement': agreement,
            }
            temporary = str(path)+'.tmp'
            torch.save(payload, temporary); os.replace(temporary, path)
        main_process_call(self.distributed, publish)
        return agreement

    def resume(self, state):
        if state['input_contract'] != INPUT_CONTRACT:
            raise ValueError('Resume changed history input contract')
        self.actor.load_state_dict(state['actor_state_dict'], strict=True)
        if frozen_digest(self.actor) != state['frozen_control_sha256']:
            raise ValueError('Frozen controller checksum mismatch')
        self.frozen_hash = state['frozen_control_sha256']
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.completed_updates, self.optimizer_steps = state['completed_updates'], state['optimizer_steps']
        if len(state['rng_states']) != self.distributed.world_size:
            raise ValueError('Resume must preserve rank count')
        restore_rng(state['rng_states'][self.distributed.rank], self.generator, self.distributed.device)
