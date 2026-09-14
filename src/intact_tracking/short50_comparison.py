"""Strict reference-physics, normalization and validation controls for Short50."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import torch

from intact_tracking.data import ForwardPredictorNormalizationStats


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference_contract(reference, actual):
    root = Path(reference).resolve()
    source = json.loads((root/'run_config.json').read_text())
    fields = ('checkpoint_file', 'motion_path', 'num_envs', 'seed', 'validation_worlds',
              'warmup_steps', 'batch_size', 'micro_batch_size', 'gradient_steps_per_update',
              'rollout_steps_per_update', 'replay_capacity', 'replay_sampling', 'amp_dtype',
              'context_history_steps', 'history_steps', 'positive_offset_steps',
              'transformer_dim', 'transformer_depth', 'transformer_heads', 'context_dim',
              'context_depth', 'context_heads', 'dynamics_latent_dim', 'dropout',
              'model_learning_rate', 'weight_decay', 'recursive_weight', 'representation_weight',
              'representation_relation_weight', 'response_distance_scale', 'updates',
              'until_user_stop', 'continuation_min_learning_rate', 'tracker_dr_plus_limb_payload',
              'limb_payload_only', 'nominal_fraction', 'payload', 'stochastic_policy',
              'randomize_initial_episode_phase')
    mismatched = {k: (source['arguments'].get(k), actual['arguments'].get(k)) for k in fields
                  if source['arguments'].get(k) != actual['arguments'].get(k)}
    for field in ('loss', 'objective_weights', 'optimization', 'distributed',
                  'tracker_checkpoint_sha256', 'dr_profile', 'episode_length_control_steps'):
        if source.get(field) != actual.get(field):
            mismatched[field] = 'configuration differs'
    for a, b in zip(source['dataset']['runtime_audits_by_rank'], actual['dataset']['runtime_audits_by_rank'], strict=True):
        for key in ('motion_count', 'motion_file_list_sha256', 'loaded_frames', 'training_worlds',
                    'validation_worlds', 'num_envs', 'episode_length_control_steps', 'dr_profile'):
            if a[key] != b[key]:
                mismatched[f'rank{a["rank"]}/{key}'] = (a[key], b[key])
        for key in ('sampled_mass_sha256', 'actual_mass_sha256', 'original_events', 'observation_corruption'):
            if a['physics'][key] != b['physics'][key]:
                mismatched[f'rank{a["rank"]}/physics/{key}'] = 'physics differs'
    if mismatched:
        raise ValueError(f'Short50 matched control changed settings: {mismatched}')
    return {'reference_dir': str(root), 'reference_config_sha256': sha256(root/'run_config.json'),
            'normalization_sha256': sha256(root/'normalization.json'),
            'shared_normalization_and_validation': True, 'physics_and_training_controls_passed': True,
            'model_input': 'only 50 completed within-episode interactions; no long memory',
            'initialization': 'fresh random shared predictor/encoder weights match Memory350 at the same seed',
            'collector': 'same Memory350 collector/replay and positive eligibility; long fields never enter the model',
            'paired_checkpoint_updates': [100, 500, 1000, 3000, 5000, 7500]}


def reference_normalization(reference, world_ids):
    value = json.loads((Path(reference)/'normalization.json').read_text())
    value = {key: tuple(item) if isinstance(item, list) else item for key, item in value.items()}
    result = ForwardPredictorNormalizationStats(**value)
    if tuple(result.world_ids) != tuple(world_ids):
        raise ValueError('Reference normalization worlds differ from training worlds')
    return result


def reference_probes(reference, output, rank, num_envs, validation_worlds, normalization, device):
    batches, evidence = [], {}
    for prefix in ('validation', 'validation_broad'):
        name = f'{prefix}_rank_{rank}.pt'
        source = Path(reference)/name
        batch = torch.load(source, map_location=device, weights_only=False)
        world = batch['world_id']
        if not bool(((world >= rank*num_envs+num_envs-validation_worlds) & (world < (rank+1)*num_envs)).all()):
            raise ValueError('Reference probe contains non-validation world IDs')
        for key in ('state_mean', 'state_std', 'action_mean', 'action_std', 'delta_mean', 'delta_std'):
            expected = torch.as_tensor(getattr(normalization, key), device=device, dtype=batch[key].dtype)
            if not torch.equal(batch[key], expected):
                raise ValueError(f'Reference validation normalization differs for {key}')
        target = Path(output)/name
        shutil.copy2(source, target)
        digest = sha256(source)
        if digest != sha256(target):
            raise RuntimeError('Copied fixed validation changed')
        evidence[name] = digest
        batches.append(batch)
    return *batches, evidence
