"""Collect a frozen heavy encoder under two policies and disjoint motion views.

Each process owns its simulator and empty interaction memory. World physics is
seeded identically in every arm; all history comes from the selected policy.
Motion identity is fixed per world within a view, including episode resets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from omegaconf import OmegaConf

from intact_tracking import memory350_heavy_policy as heavy
from intact_tracking import memory350_native_policy as native
from intact_tracking.cli.memory350_proprio_native_policy_eval import (
    EvaluationWrapper, paired_environment, evaluation_world_metadata,
)
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.limb_context_sampling import state_digest
from intact_tracking.memory350_checkpoint import embedded_tracker, load_policy_context
from intact_tracking.memory350_inference import Memory350Inference
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.memory350_proprio_inputs import current_proprio
from intact_tracking.memory350_tracker_action_policy import TrackerActionResidualActor
from intact_tracking.residual_dr_aux import capture_dr_aux_targets


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class MatchedForcePulse:
    """Give exogenous forces their own RNG and reapply after simulator resets."""

    def __init__(self, pulse, seed):
        self._pulse, self._seed = pulse, seed

    def __getattr__(self, name):
        return getattr(self._pulse, name)

    def __call__(self, env, env_ids, **kwargs):
        with torch.random.fork_rng(devices=[torch.device(env.device).index or 0]):
            torch.manual_seed(self._seed + int(env.common_step_counter))
            self._pulse(env, env_ids, **kwargs)
        force = self._pulse.current_force_w.unsqueeze(1)
        self._pulse.asset.write_external_wrench_to_sim(
            forces=force, torques=torch.zeros_like(force), body_ids=self._pulse.body_ids)


@torch.inference_mode()
def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    configure_policy_precision('fp32')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    meta = state['residual_policy']
    agent = OmegaConf.to_container(state['cfg'].agent, resolve=True)
    if meta['dr_profile'] != heavy.PROFILE or agent['actor']['latent_input_mode'] != 'learned':
        raise ValueError('Expected a learned-latent heavy residual checkpoint')
    schema = agent['actor']['dr_aux_schema']
    tracker = embedded_tracker(state)
    if tracker is None:
        raise ValueError('Use a portable checkpoint with validated frozen dependencies')
    context = load_policy_context(state, device=args.device)
    files = [str(Path(p).resolve()) for p in Path(args.manifest).read_text().splitlines() if p.strip()]
    if len(set(files)) != len(files) or len(files) < 2:
        raise ValueError('Need distinct motion IDs for two views')
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS['terrain_height_offset'] = native.FlatTerrainHeightOffset
    _seed_everything(args.seed)
    prepared = prepare_rollout(checkpoint_file=meta['tracker_checkpoint'], num_envs=args.num_envs,
        motion_file=None, motion_path=meta['motion_path'], checkpoint_config=tracker['cfg'])
    cfg = prepared.env
    physics = heavy.configure_evaluation_physics(cfg, args.seed, profile=heavy.PROFILE, physics=args.physics)
    command_cfg = cfg.commands['motion']
    command_cfg.motion_manifest_file = str(Path(args.manifest).resolve())
    command_cfg.resample_on_motion_end = True
    command_cfg.resampling_time_range = (1e9, 1e9)
    command_cfg.init_noise, command_cfg.pose_range, command_cfg.velocity_range = {}, {}, {}
    command_cfg.joint_position_range = (0., 0.)
    cfg.seed, cfg.auto_reset = args.seed, True
    cfg.episode_length_s = 500 * cfg.decimation * cfg.sim.mujoco.timestep
    _seed_everything(args.seed)
    env = paired_environment(ManagerBasedRlEnv, physics_factory=heavy.environment_factory,
                             cfg=cfg, device=args.device)
    started = time.monotonic()
    try:
        audit = heavy.audit_physics(env, physics)
        command = env.command_manager.get_term('motion')
        if tuple(command.motion_files) != tuple(files):
            raise ValueError('Motion catalog changed')
        fixed_motion = getattr(args, 'fixed_motion_index', None)
        if fixed_motion is None:
            ids = (torch.arange(args.num_envs, device=env.device) + args.view * (len(files)//2)) % len(files)
        else:
            if not 0 <= fixed_motion < len(files):
                raise ValueError('Fixed motion index is outside the loaded catalog')
            # A full world x motion design: every world executes this same
            # motion, and independent processes collect the other motions.
            ids = torch.full((args.num_envs,), fixed_motion, device=env.device, dtype=torch.long)
        rng = np.random.default_rng(args.seed + 100)
        lengths = command.motion.file_lengths[ids].cpu().numpy()
        starts = torch.as_tensor((rng.random(args.num_envs) * np.maximum(lengths-351, 1)).astype(np.int64), device=env.device)

        def sample(self, env_ids):
            self.motion_idx[env_ids] = ids[env_ids]
            self.motion_length[env_ids] = self.motion.file_lengths[ids[env_ids]]
            self.time_steps[env_ids] = starts[env_ids]

        command._uniform_sampling = MethodType(sample, command)
        _seed_everything(args.seed + 1000000)
        wrapped = EvaluationWrapper(env, prepared.clip_actions, context,
            latent_history_frames=5, dr_aux_schema=schema, latent_input_mode='learned')
        obs = wrapped.get_observations()
        kwargs = copy.deepcopy(agent['actor'])
        kwargs.pop('class_name')
        kwargs['tracker_state_dict'] = tracker['actor_state_dict']
        actor = TrackerActionResidualActor(obs, agent['obs_groups'], 'actor', wrapped.num_actions, **kwargs).to(env.device)
        actor.load_state_dict(state['actor_state_dict'], strict=True)
        actor.eval().requires_grad_(False)
        _seed_everything(args.seed + 2000000)
        obs, _ = wrapped.reset()
        if wrapped.context.memory.total_chunks.any() or wrapped.context.memory.short_count.any():
            raise RuntimeError('Every policy/view must start with completely empty memory')
        initial_state = {'qpos': tensor_sha(env.sim.data.qpos), 'qvel': tensor_sha(env.sim.data.qvel)}
        world_metadata = evaluation_world_metadata(env)
        pulse_cfg = env.event_manager.get_term_cfg('push_robot')
        pulse_cfg.func = MatchedForcePulse(pulse_cfg.func, args.seed + 7000000 + args.view * 10000)
        # These controller quantities otherwise change after policy-dependent
        # resets. Hold each world's initially sampled nuisance physics fixed.
        action_term = env.action_manager.get_term('joint_pos')
        fixed_controller = {key:getattr(action_term,key).clone() for key in ('delay','alpha','joint_offset')}
        original_action_reset = action_term.reset

        def reset_action(self, env_ids=None):
            original_action_reset(env_ids)
            selected = slice(None) if env_ids is None else env_ids
            for key,value in fixed_controller.items():
                getattr(self,key)[selected] = value[selected]

        action_term.reset = MethodType(reset_action, action_term)
        original_env_reset = env._reset_idx

        def reset_worlds(self, env_ids):
            # Reset-only random draws must not shift all worlds' sensor noise.
            with torch.random.fork_rng(devices=[torch.device(self.device).index or 0]):
                return original_env_reset(env_ids)

        env._reset_idx = MethodType(reset_worlds, env)
        truth = capture_dr_aux_targets(env, schema).cpu().numpy()
        before = state_digest(actor.state_dict())
        encoder_before = state_digest(context.encoder.state_dict())
        metadata = {'arguments': vars(args), 'checkpoint_sha256': sha(args.checkpoint),
            'completed_updates': state['completed_updates'], 'context_path': context.path,
            'context_sha256': context.sha256, 'tracker_sha256': tracker['source_sha256'],
            'manifest_sha256': sha(args.manifest), 'schema': schema,
            'physics_audit': audit, 'initial_state_sha256': initial_state,
            'world_metadata': world_metadata, 'motion_files': files,
            'fixed_motion_ids': ids.cpu().tolist(), 'starts': starts.cpu().tolist(),
            'history_contract': 'Empty short and long memory at start; selected policy generates every interaction. One fixed motion ID per world/view, repeated on reset; long memory retained across resets within that motion. Complete 50+300 histories marked explicitly.',
            'pairing_contract': 'Same static physical worlds and initial controller draws in all arms, same starts/state within each motion view. Force clocks do not reset on failures. Controller delay/smoothing/joint offset held at paired initial draws; reset RNG isolated from sensor noise; force RNG isolated and active force reapplied after resets; policy-dependent resets recorded.',
            'scope': 'New physical seed; existing training-catalog motions, not held-out motion families. Deterministic checkpoint policies; no optimizer.',
            'complete': False}
        save_json(output / 'metadata.json', metadata)
        samples = []
        failures = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
        resets = torch.zeros_like(failures)
        action_sq = torch.zeros(args.num_envs, device=env.device, dtype=torch.float64)
        residual_sq = torch.zeros_like(action_sq)
        numerical = None
        for step in range(1, args.steps + 1):
            _seed_everything(args.seed + 3000000 + args.view * 10000 + step)
            if args.policy == 'tracker':
                action = actor._base_features_and_action(obs)[1]
                residual = torch.zeros_like(action)
            else:
                action = actor(obs)
                base_action = actor._base_features_and_action(obs)[1]
                residual = action-base_action
            action_sq += action.square().mean(-1).double()
            residual_sq += residual.square().mean(-1).double()
            env.action_manager.get_term('joint_pos').record_policy_mean(action)
            obs, _, dones, _ = wrapped.step(action)
            failures += env.reset_terminated.long()
            resets += dones.long()
            if not torch.equal(command.motion_idx, ids):
                raise RuntimeError('History crossed into another motion ID')
            if step >= args.sample_start and (step-args.sample_start) % args.sample_interval == 0:
                bank = wrapped.context.memory
                short, short_valid = bank.ordered_short()
                long, long_valid = bank.read_chunks()
                full = short_valid.all(1) & long_valid.all(1)
                raw = torch.cat((long.flatten(1,2), short), 1)
                valid = torch.cat((long_valid.repeat_interleave(bank.chunk_steps, 1), short_valid), 1)
                normalized = wrapped.context._normalize(raw)
                count = valid.sum(1).clamp_min(1)[:, None]
                mean = (normalized * valid[..., None]).sum(1) / count
                variance = ((normalized-mean[:, None]).square()*valid[..., None]).sum(1)/count
                z = wrapped.latent_history.values[:, -1].clone()
                if not torch.isfinite(z).all():
                    raise RuntimeError('Nonfinite latent')
                # Content hashes audit that different views do not share exact histories.
                raw_cpu = raw.cpu().numpy()
                hashes = np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in raw_cpu])
                samples.append({'z':z.cpu().numpy(), 'full':full.cpu().numpy(),
                    'history_features':torch.cat((mean, variance.sqrt()),-1).cpu().numpy(),
                    'state':current_proprio(obs).cpu().numpy(),
                    'short_count':bank.short_count.cpu().numpy().copy(),
                    'long_count':(bank.total_chunks-bank.session_start).clamp_max(bank.long_chunks).cpu().numpy(),
                    'motion':command.motion_idx.cpu().numpy().copy(),
                    'motion_step':command.time_steps.cpu().numpy().copy(),
                    'failures':failures.cpu().numpy().copy(), 'resets':resets.cpu().numpy().copy(),
                    'history_sha256':hashes, 'step':np.asarray(step)})
                if numerical is None and full.any():
                    direct = Memory350Inference(context, args.num_envs, use_bfloat16=True)
                    direct.memory = bank
                    reference = direct.encode()
                    numerical = {'cached_direct_rms': float((reference[full]-z[full]).square().mean().sqrt()),
                        'cached_direct_cosine':float(torch.nn.functional.cosine_similarity(reference[full], z[full]).mean())}
                    if numerical['cached_direct_cosine'] < .999:
                        raise RuntimeError('Cached latent does not match direct frozen encoding')
            if step % 100 == 0:
                print(json.dumps({'policy':args.policy, 'view':args.view, 'step':step,
                    'seconds':round(time.monotonic()-started,1), 'failures':int(failures.sum()),
                    'samples':len(samples)}), flush=True)
        if not samples:
            raise ValueError('No requested sample steps')
        for key,value in fixed_controller.items():
            torch.testing.assert_close(getattr(action_term,key), value, rtol=0, atol=0)
        final_truth = capture_dr_aux_targets(env, schema).cpu().numpy()
        np.testing.assert_array_equal(truth, final_truth)
        if before != state_digest(actor.state_dict()) or encoder_before != state_digest(context.encoder.state_dict()):
            raise RuntimeError('Frozen parameters changed')
        np.savez_compressed(output/'latents.npz', **{key:np.stack([row[key] for row in samples]) for key in samples[0]},
            truth_normalized=truth, names=np.asarray(schema['names']),
            action_rms=(action_sq/args.steps).sqrt().cpu().numpy(),
            residual_rms=(residual_sq/args.steps).sqrt().cpu().numpy())
        diagnostics = wrapped.evaluation_diagnostics
        diagnostics.pop('query_force_prefix_sha256')
        metadata.update(complete=True, numerical_check=numerical, optimizer_steps=0,
            actor_and_encoder_unchanged=True, final_physics_audit=native.audit_native_runtime(env),
            failure_events=int(failures.sum()), failure_worlds=int((failures>0).sum()),
            resets=int(resets.sum()), seconds=time.monotonic()-started, force_diagnostics=diagnostics)
        save_json(output/'metadata.json', metadata)
        print(json.dumps({'complete':True,'output':str(output),'seconds':metadata['seconds']}),flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--policy', choices=('tracker','residual'), required=True)
    p.add_argument('--view', type=int, required=True)
    p.add_argument('--fixed-motion-index', type=int,
                   help='Assign this same catalog motion to every world in this view')
    p.add_argument('--physics', choices=('hdr','nominal'), default='hdr')
    p.add_argument('--num-envs', type=int, default=512)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--sample-start', type=int, default=350)
    p.add_argument('--sample-interval', type=int, default=50)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.view < 0:
        p.error('View index must be nonnegative')
    if args.sample_interval < 1 or not 1 <= args.sample_start <= args.steps:
        p.error('Require a positive interval and a sample start within the rollout')
    run(args)
