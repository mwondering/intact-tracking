"""Bounded real-checkpoint audit; never writes or modifies the training job."""

import argparse
import copy
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from omegaconf import OmegaConf
from rsl_rl.storage import RolloutStorage

from intact_tracking.cli.memory350_proprio_native_policy_train import configure_proprio_physics
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.memory350_inference import load_memory350_checkpoint, Memory350Inference
from intact_tracking.memory350_policy_inference import CachedMemory350Inference
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.memory350_proprio_inputs import current_proprio, control_action
from intact_tracking.memory350_proprio_policy import ProprioNativePolicyWrapper
from intact_tracking.memory350_tracker_action_policy import (
    ACTION_GROUP, TrackerActionResidualActor, TrackerActionCritic, TrackerActionPPO,
)
from intact_tracking import memory350_native_policy as native


def run(args):
    configure_policy_precision('fp32')
    torch.cuda.set_per_process_memory_fraction(.25)
    _seed_everything(219019)
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS['terrain_height_offset'] = native.FlatTerrainHeightOffset
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta = state['residual_policy']
    agent = OmegaConf.to_container(state['cfg'].agent, resolve=True)
    prepared = prepare_rollout(checkpoint_file=meta['tracker_checkpoint'], num_envs=64,
                               motion_file=None, motion_path=meta['motion_path'])
    cfg = prepared.env
    configure_proprio_physics(cfg, 219019, profile=native.PROFILE)
    cfg.commands['motion'].motion_manifest_file = str(Path(args.manifest).resolve())
    cfg.auto_reset = True
    # Exercise real auto-resets repeatedly, while keeping all other termination
    # terms and the production wrapper. These are audits, not performance scores.
    cfg.episode_length_s = 64 * cfg.decimation * cfg.sim.mujoco.timestep
    context = load_memory350_checkpoint(meta['context_checkpoint'], device='cuda:0',
                                       expected_tracker_sha256=meta['tracker_sha256'])
    env = native.environment_factory(ManagerBasedRlEnv, cfg=cfg, device='cuda:0')
    report = {'checkpoint': str(args.checkpoint), 'completed_updates': state['completed_updates'],
              'environments': 64, 'control_steps': 360, 'audit_episode_limit': 64,
              'training_process_modified': False, 'checks': {}, 'measurements': {}}
    try:
        wrapped = ProprioNativePolicyWrapper(env, prepared.clip_actions, context, latent_history_frames=5,
                                             dr_aux_schema=agent['actor'].get('dr_aux_schema'))
        obs = wrapped.get_observations()
        kwargs = copy.deepcopy(agent['actor']); kwargs.pop('class_name')
        actor = TrackerActionResidualActor(obs, agent['obs_groups'], 'actor', 29, **kwargs).to(env.device)
        actor.load_state_dict(state['actor_state_dict'], strict=True)
        kwargs = copy.deepcopy(agent['critic']); kwargs.pop('class_name')
        critic = TrackerActionCritic(obs, agent['obs_groups'], 'critic', 1, **kwargs).to(env.device)
        critic.load_state_dict(state['critic_state_dict'], strict=True)
        storage = RolloutStorage('rl', 64, 24, obs, [29], device=env.device)
        kwargs = copy.deepcopy(agent['algorithm']); kwargs.pop('class_name')
        kwargs.update(num_learning_epochs=1, num_mini_batches=2)
        algorithm = TrackerActionPPO(actor, critic, storage, device=env.device, **kwargs)
        actor.train(); critic.train()
        tracker_before = {k: v.detach().cpu().clone() for k, v in actor.tracker.state_dict().items()}
        encoder_before = {k: v.detach().cpu().clone() for k, v in context.encoder.state_dict().items()}
        original_tracker = torch.load(meta['tracker_checkpoint'], map_location='cpu', weights_only=False)
        for k, v in tracker_before.items():
            torch.testing.assert_close(v, original_tracker['actor_state_dict'][k], atol=0, rtol=0)
        del original_tracker, state
        batches, boundaries, max_cache_error, max_log_prob_error = [], 0, 0., 0.
        natural_motion_boundaries = 0
        original_append = wrapped.context.append

        def append(batch):
            batches.append({k: v.detach().clone() for k, v in batch.items()})
            original_append(batch)

        wrapped.context.append = append
        checked_steps = {0, 23, 62, 63, 64, 127, 255, 359}
        with torch.inference_mode():
            for step in range(360):
                if step in (80, 200):
                    # Exercise a motion-end pulse without changing its current
                    # reference pose or ending the environment episode.
                    command = wrapped.motion_command
                    command.motion_length[:16] = command.time_steps[:16] + 1
                before_proprio = current_proprio(obs).clone()
                old = obs.clone()
                action = algorithm.act(obs)
                mean = actor.output_mean.clone()
                if step in checked_steps:
                    direct_obs = obs.select(*actor.tracker.obs_groups).clone()
                    direct = actor.tracker(direct_obs)
                    max_cache_error = max(max_cache_error, float((direct - obs[ACTION_GROUP]).abs().max()))
                    torch.testing.assert_close(direct, obs[ACTION_GROUP], atol=2e-6, rtol=2e-6)
                    features, base = actor._base_features_and_action(obs)
                    expected_mean = base + actor._residual(actor._residual_input(obs, features, base))
                    torch.testing.assert_close(mean, expected_mean, atol=0, rtol=0)
                    torch.testing.assert_close(critic.value_input(obs)[:, -29:], direct, atol=2e-6, rtol=2e-6)
                    torch.testing.assert_close(critic.value_input(obs)[:, -349:-29], obs['dynamics_latent'], atol=0, rtol=0)
                env.action_manager.get_term('joint_pos').record_policy_mean(mean)
                next_obs, reward, dones, extras = wrapped.step(action)
                batch = batches.pop()
                torch.testing.assert_close(batch['robot_state'], before_proprio, atol=0, rtol=0)
                torch.testing.assert_close(batch['next_robot_state'], current_proprio(next_obs), atol=0, rtol=0)
                torch.testing.assert_close(batch['joint_target'], control_action(env), atol=0, rtol=0)
                edge = batch['reset_boundary']
                boundaries += int(edge.sum())
                expected_command = action if prepared.clip_actions is None else action.clamp(-prepared.clip_actions, prepared.clip_actions)
                torch.testing.assert_close(batch['joint_target'][~edge], expected_command[~edge], atol=0, rtol=0)
                # PPO stores obs only after stepping; every original field must
                # remain the same, including the five-frame latent snapshot.
                for key in old.keys():
                    torch.testing.assert_close(obs[key], old[key], atol=0, rtol=0, msg=lambda s: f'Observation alias {key}: {s}')
                new_history = next_obs['dynamics_latent'].reshape(64, 5, 64)
                old_history = old['dynamics_latent'].reshape(64, 5, 64)
                torch.testing.assert_close(new_history[~edge, :-1], old_history[~edge, 1:], atol=0, rtol=0)
                assert not new_history[edge, :-1].any()
                assert not wrapped.context.memory.short_count[edge].any()
                if step in checked_steps:
                    torch.testing.assert_close(new_history[:, -1], wrapped.context.encode(), atol=0, rtol=0)
                motion_boundary = extras['motion_resample_boundary']
                natural_motion_boundaries += int((motion_boundary & ~dones.bool()).sum())
                gae_dones = dones.bool() | motion_boundary
                bootstrap = motion_boundary | extras.get('time_outs', torch.zeros_like(motion_boundary)).bool()
                expected_rewards = reward + algorithm.gamma * algorithm.transition.values.squeeze(-1) * bootstrap
                saved_dones = dones.clone()
                algorithm.process_env_step(next_obs, reward, dones, extras)
                slot = step % 24
                torch.testing.assert_close(dones, saved_dones, atol=0, rtol=0)
                torch.testing.assert_close(storage.dones[slot, :, 0].bool(), gae_dones, atol=0, rtol=0)
                torch.testing.assert_close(storage.rewards[slot, :, 0], expected_rewards, atol=0, rtol=0)
                torch.testing.assert_close(storage.observations['dynamics_latent'][slot], old['dynamics_latent'], atol=0, rtol=0)
                torch.testing.assert_close(storage.observations[ACTION_GROUP][slot], obs[ACTION_GROUP], atol=0, rtol=0)
                obs = next_obs
                if slot == 23:
                    algorithm.compute_returns(obs)
                    if step in (23, 359):
                        for minibatch in storage.mini_batch_generator(2, 1):
                            actor(minibatch.observations, stochastic_output=True)
                            err = float((actor.output_mean - minibatch.old_distribution_params[0]).abs().max())
                            max_cache_error = max(max_cache_error, err)
                            torch.testing.assert_close(actor.output_mean, minibatch.old_distribution_params[0], atol=3e-6, rtol=3e-6)
                            logprob = actor.get_output_log_prob(minibatch.actions)
                            err = float((logprob.flatten() - minibatch.old_actions_log_prob.flatten()).abs().max())
                            max_log_prob_error = max(max_log_prob_error, err)
                            torch.testing.assert_close(logprob.flatten(), minibatch.old_actions_log_prob.flatten(), atol=3e-4, rtol=3e-6)
                    if step != 359:
                        storage.clear()
                if (step + 1) % 120 == 0:
                    print(json.dumps({'step': step + 1, 'boundaries': boundaries}), flush=True)
            # Compare the caching implementation against the full encoder on
            # identical, actual 122D interactions with trained normalization.
            direct = Memory350Inference(context, 64, batch_size=64, use_bfloat16=False)
            direct.memory = wrapped.context.memory
            cached = CachedMemory350Inference(context, 64, batch_size=64, use_bfloat16=False)
            cached.memory = wrapped.context.memory
            full_latent, cached_latent = direct.encode(), cached.encode()
            torch.testing.assert_close(cached_latent, full_latent, atol=2e-5, rtol=2e-5)
            runtime_latent = wrapped.context.encode()
            report['measurements'].update(
                fp32_cached_vs_direct_max_abs=float((full_latent-cached_latent).abs().max()),
                bf16_cached_vs_fp32_direct_rms=float((full_latent-runtime_latent).square().mean().sqrt()),
                fp32_latent_rms=float(full_latent.square().mean().sqrt()),
                tracker_and_minibatch_mean_max_abs_error=max_cache_error,
                minibatch_log_prob_max_abs_error=max_log_prob_error,
                reset_or_motion_boundaries=boundaries, memory=wrapped.latent_metrics,
                natural_motion_boundaries_without_episode_done=natural_motion_boundaries,
                sliding_root_xy_reward=bool(env.command_manager.get_term('motion').cfg.sliding_root_xy_reward),
                root_reward_history_seconds=float(env.command_manager.get_term('motion').reward_root_history_len * env.step_dt))
        # A single local update validates that the real actor/critic receive
        # gradients through all five frames and tracker action inputs.
        assert natural_motion_boundaries > 0
        report['local_update_losses'] = algorithm.update()
        for name, layer in [('actor', actor.residual_mlp.latent_input), ('critic', critic.mlp.latent_input)]:
            grad = layer.weight.grad
            parts = [float(grad[:, i*64:(i+1)*64].abs().sum()) for i in range(5)] + [float(grad[:, 320:].abs().sum())]
            assert all(x > 0 for x in parts)
            report['measurements'][name+'_five_latents_and_action_gradient_l1'] = parts
        for module, saved in [(actor.tracker, tracker_before), (context.encoder, encoder_before)]:
            assert not module.training and not any(p.requires_grad or p.grad is not None for p in module.parameters())
            for k, v in module.state_dict().items():
                torch.testing.assert_close(v.detach().cpu(), saved[k], atol=0, rtol=0)
        report['checks'] = dict.fromkeys([
            'tracker_equals_original_checkpoint', 'sensor_action_next_sensor_alignment',
            'command_is_after_composition_before_sp_filtering', 'no_observation_mutation_before_storage',
            'history5_temporal_order_and_boundary_clear', 'actor_and_critic_current_tracker_action',
            'ppo_storage_and_shuffled_minibatch_alignment', 'pre_update_log_prob_reproducibility',
            'spv53_motion_boundary_gae_and_current_value_bootstrap',
            'cached_encoder_matches_direct_encoder', 'actor_critic_receive_all_appended_inputs',
            'tracker_and_encoder_remain_frozen_after_ppo_update'], True)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
