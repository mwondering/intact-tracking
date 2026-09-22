"""Evaluate the saved residual DR head on new HDR worlds, without fitting it."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from omegaconf import OmegaConf

from intact_tracking import memory350_heavy_policy as heavy, memory350_native_policy as native
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.memory350_checkpoint import embedded_tracker, load_policy_context
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.memory350_proprio_policy import ProprioNativePolicyWrapper
from intact_tracking.memory350_tracker_action_policy import TrackerActionResidualActor
from intact_tracking.limb_context_sampling import state_digest


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(prediction, truth, valid, names, lower, upper):
    counts = valid.sum(0)
    included = counts > 0
    if not included.all():
        raise ValueError('Every sampled world must contribute full-history predictions')
    error = prediction - truth[None]
    mse = np.where(valid[..., None], error**2, 0).sum(0) / counts[:, None]
    mae = np.where(valid[..., None], np.abs(error), 0).sum(0) / counts[:, None]
    mean_prediction = np.where(valid[..., None], prediction, 0).sum(0) / counts[:, None]
    variance = truth.var(0)
    center = (lower + upper) / 2
    rows = []
    for i, name in enumerate(names):
        pooled = error[..., i][valid]
        covariance = np.mean((truth[:, i]-truth[:, i].mean()) * (mean_prediction[:, i]-mean_prediction[:, i].mean()))
        rows.append({'name': name, 'mae': float(mae[:, i].mean()),
            'rmse': float(np.sqrt(mse[:, i].mean())),
            'r2': float(1-mse[:, i].mean()/variance[i]),
            'mean_bias': float((mean_prediction[:, i]-truth[:, i]).mean()),
            'absolute_error_p95_sample_weighted': float(np.quantile(np.abs(pooled), .95)),
            'range_midpoint_mae': float(np.abs(truth[:, i]-center[i]).mean()),
            'calibration_slope_world_mean': float(covariance/variance[i]),
            'prediction_outside_physical_range_fraction': float(
                ((prediction[..., i][valid] < lower[i]) | (prediction[..., i][valid] > upper[i])).mean())})
    return rows, mean_prediction, counts


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_policy_precision('fp32')
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    meta = state['residual_policy']
    agent = OmegaConf.to_container(state['cfg'].agent, resolve=True)
    if meta['dr_profile'] != heavy.PROFILE or agent['actor']['latent_input_mode'] != 'learned':
        raise ValueError('Supply a learned-latent heavy residual checkpoint')
    schema = agent['actor']['dr_aux_schema']
    tracker = embedded_tracker(state)
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS['terrain_height_offset'] = native.FlatTerrainHeightOffset
    _seed_everything(args.seed)
    prepared = prepare_rollout(checkpoint_file=meta['tracker_checkpoint'], num_envs=args.num_envs,
        motion_file=None, motion_path=meta['motion_path'], checkpoint_config=tracker['cfg'])
    cfg = prepared.env
    configuration = heavy.configure_evaluation_physics(cfg, args.seed, profile=heavy.PROFILE, physics='hdr')
    cfg.commands['motion'].motion_manifest_file = str(Path(args.manifest).resolve())
    cfg.episode_length_s = 500 * cfg.decimation * cfg.sim.mujoco.timestep
    cfg.seed, cfg.auto_reset = args.seed, True
    context = load_policy_context(state, device='cuda:0')
    _seed_everything(args.seed)
    env = heavy.environment_factory(ManagerBasedRlEnv, cfg=cfg, device='cuda:0')
    try:
        audit = heavy.audit_physics(env, configuration)
        wrapped = ProprioNativePolicyWrapper(env, prepared.clip_actions, context,
            latent_history_frames=5, dr_aux_schema=schema, latent_input_mode='learned')
        obs = wrapped.get_observations()
        kwargs = copy.deepcopy(agent['actor']); kwargs.pop('class_name')
        kwargs['tracker_state_dict'] = tracker['actor_state_dict']
        actor = TrackerActionResidualActor(obs, agent['obs_groups'], 'actor', 29, **kwargs).to(env.device)
        actor.load_state_dict(state['actor_state_dict'], strict=True)
        actor.eval().requires_grad_(False)
        before = state_digest(actor.state_dict())
        active = actor.dr_aux_objective.active
        indices = active.cpu().tolist()
        lower = np.asarray(schema['lower'])[indices]
        upper = np.asarray(schema['upper'])[indices]
        names = [schema['names'][i] for i in indices]
        truth_normalized = wrapped._dr_aux_targets[:, active].cpu().numpy()
        truth = lower + truth_normalized * (upper-lower)
        predictions, validity, sampled_steps = [], [], []
        failures = 0
        started = time.time()
        with torch.inference_mode():
            for step in range(args.warmup_steps+args.sample_steps):
                if step >= args.warmup_steps and (step-args.warmup_steps) % args.sample_interval == 0:
                    predictions.append(actor.predict_dr(obs)[:, active].cpu().numpy())
                    memory = wrapped.context.memory
                    full = wrapped.context.history_valid.all(0) & (memory.short_count >= memory.short_steps)
                    full &= (memory.total_chunks-memory.session_start) >= memory.long_chunks
                    validity.append(full.cpu().numpy())
                    sampled_steps.append(step)
                action = actor(obs)
                env.action_manager.get_term('joint_pos').record_policy_mean(action)
                obs, _, _, _ = wrapped.step(action)
                failures += int(env.reset_terminated.sum())
                if (step+1) % 128 == 0:
                    print(json.dumps({'step':step+1, 'seconds':time.time()-started}), flush=True)
        prediction = lower + np.stack(predictions)*(upper-lower)
        valid = np.stack(validity)
        rows, means, counts = summarize(prediction, truth, valid, names, lower, upper)
        if state_digest(actor.state_dict()) != before:
            raise RuntimeError('Prediction probe changed actor state')
        final_audit = native.audit_native_runtime(env)
        np.savez_compressed(output/'predictions.npz', prediction=prediction, truth=truth,
            full_history=valid, sample_steps=np.asarray(sampled_steps), names=np.asarray(names),
            mean_prediction=means, per_world_counts=counts,
            payload_mass_kg=env.heavy_policy_payload.mass.cpu().numpy())
        report = {'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256':digest(args.checkpoint),
            'completed_updates':state['completed_updates'], 'context_sha256':meta['context_sha256'],
            'arguments':vars(args), 'manifest_sha256':digest(args.manifest), 'output_dimensions':108,
            'supervised_dimensions':len(indices), 'supervised_groups':actor.dr_aux_objective.supervised_groups,
            'physics_audit':audit, 'final_physics_audit':final_audit, 'worlds':args.num_envs,
            'full_history_predictions':int(valid.sum()), 'full_history_fraction':float(valid.mean()),
            'per_world_samples_min':int(counts.min()), 'per_world_samples_max':int(counts.max()),
            'prediction_clipping':False, 'optimizer_steps':0, 'actor_state_unchanged':True,
            'aggregation':'Equal world weights; full short+long history only. R2 uses world-averaged per-step MSE divided by across-world physical target variance.',
            'scope':'New HDR worlds and seeds on sampled training-catalog motions; current deterministic residual policy histories, no decoder fitting. Not held-out-motion or multi-seed generalization.',
            'failure_events':failures, 'parameters':rows, 'seconds':time.time()-started}
        (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'completed_updates':state['completed_updates'],'parameters':rows}),flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--sample-steps', type=int, default=256)
    parser.add_argument('--sample-interval', type=int, default=4)
    run(parser.parse_args())
