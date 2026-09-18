"""Compare frozen soft encoders on the same saved, held-out simulator queries."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

from analyze_memory350 import cluster_ci, error_metrics, sha256
from intact_tracking.forward_predictor_objective import _normalized_state_error
from intact_tracking.memory350_model import Memory350Config, Memory350Predictor
from intact_tracking.memory350_nominal_dr_rank import rank_view_valid


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def evaluate(state, validation, output, name, hashes):
    model = Memory350Predictor(Memory350Config(**state['model_config'])).eval().requires_grad_(False)
    model.load_state_dict(state['model'])
    parts, centers = [], []
    fields = ('history_state', 'history_action', 'history_next_state', 'history_valid',
              'memory_interactions', 'memory_valid')
    with torch.inference_mode():
        for rank in range(4):
            path = validation / f'validation_broad_rank_{rank}.pt'
            full = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            hashes[path.name] = sha256(path)
            for key in ('state_mean', 'state_std', 'action_mean', 'action_std'):
                torch.testing.assert_close(full[key], torch.tensor(state['normalization'][key],
                                           dtype=full[key].dtype), atol=0, rtol=0)
            n = len(full['state'])
            for start in range(0, n, 64):
                batch = {k: (v[start:start + 64] if v.ndim and len(v) == n else v)
                         for k, v in full.items()}
                raw = model.context_encoder(*(batch[k] for k in fields))
                unit = torch.nn.functional.normalize(raw.float(), dim=-1).numpy()
                unchanged = _normalized_state_error(
                    batch['state'][:, :1].expand_as(batch['state'][:, 1:]), batch['state'][:, 1:],
                    batch['state_mean'], batch['state_std'], batch['delta_std']).square().mean(-1).numpy()
                parts.append({'z': unit, 'mse': error_metrics(model, batch, raw)[:, -1],
                    'denominator': unchanged[:, -1], 'world': batch['world_id'].numpy(),
                    'nominal': batch['is_nominal'].numpy(),
                    'full': (batch['history_valid'].all(1) & batch['memory_valid'].all(1)).numpy(),
                    'valid': batch['label_response_valid'].numpy(),
                    'response': batch['label_response'].square().mean((1, 2)).sqrt().numpy()})
            path = validation / f'validation_rank_{rank}.pt'
            batch = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            hashes[path.name] = sha256(path)
            valid = rank_view_valid(batch)
            pair = []
            for prefix in ('', 'weak_'):
                values = [batch[prefix + k][valid] for k in fields]
                encoded = [model.context_encoder(*(a[start:start + 64] for a in values))
                           for start in range(0, int(valid.sum()), 64)]
                pair.append(torch.nn.functional.normalize(torch.cat(encoded).float(), dim=-1).numpy())
            centers.append({'z': (pair[0] + pair[1]) * .5, 'parameters': batch['dr_metric'][valid].numpy(),
                            'world': batch['world_id'][valid].numpy(),
                            'session': batch['physics_session'][valid].numpy()})
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError('Nonfinite held-out encoder result')
    np.savez_compressed(output / f'{name}_heldout.npz', **data)
    anchor = np.asarray(state['nominal_direction_anchor']['direction'])
    radius = np.linalg.norm(data['z'] - anchor, axis=-1)
    nominal = data['full'] & data['nominal']
    dr = data['full'] & ~data['nominal'] & data['valid']
    scale = state['loss_config']['response_distance_scale']
    target = 2 * data['response'] / (data['response'] + scale)
    report = {'nominal_anchor_rms': float(np.sqrt(np.mean(radius[nominal] ** 2))),
              'ab_target_mae': float(np.abs(radius[dr] - target[dr]).mean()),
              'response_radius_spearman': float(spearmanr(radius[dr], data['response'][dr]).statistic),
              'nominal_samples': int(nominal.sum()), 'full_dr_samples': int(dr.sum())}
    for group, mask in [('dr', ~data['nominal']), ('nominal', data['nominal'])]:
        report[group + '_nmse'] = float(data['mse'][mask].sum() / data['denominator'][mask].sum())
    center = {k: np.concatenate([p[k] for p in centers]) for k in centers[0]}
    _, inverse = np.unique(np.column_stack((center['world'], center['session'])), axis=0, return_inverse=True)
    counts = np.bincount(inverse)

    def pool(x):
        pooled = np.zeros((len(counts), x.shape[-1]))
        np.add.at(pooled, inverse, x)
        return pooled / counts[:, None]

    latent, theta = pool(center['z']), pool(center['parameters'])
    i, j = np.triu_indices(len(latent), 1)
    distance = np.linalg.norm(latent[i] - latent[j], axis=-1)
    physical = np.linalg.norm(theta[i] - theta[j], axis=-1)
    report.update(center_parameter_spearman=float(spearmanr(physical, distance).statistic),
                  world_centers=len(latent), between_center_distance=float(distance.mean()),
                  within_center_rms=float(np.sqrt(np.square(center['z'] - latent[inverse]).sum(-1).mean())))
    return report, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'heldout.json').exists():
        raise FileExistsError('Preserve the existing evaluation')
    torch.set_num_threads(4)
    paths = {'reference': args.reference, 'soft': args.candidate}
    states = {k: torch.load(p, map_location='cpu', weights_only=False, mmap=True) for k, p in paths.items()}
    for field in ('normalization', 'nominal_direction_anchor', 'model_config', 'dr_metric_schema'):
        if states['reference'][field] != states['soft'][field]:
            raise ValueError(f'Encoder comparison contract differs: {field}')
    if any(s['loss_config']['response_distance_scale'] != .6 for s in states.values()):
        raise ValueError('This comparison requires AB scale 0.6')
    result = {'complete': False, 'checkpoints': {k: {'path': str(p.resolve()),
              'update': states[k]['update'], 'sha256': sha256(p)} for k, p in paths.items()},
              'validation_directory': str(args.validation.resolve()), 'validation_sha256': {},
              'precision': 'cpu float32', 'same_queries': True, 'models': {}}
    saved = {}
    for name, state in states.items():
        result['models'][name], saved[name] = evaluate(state, args.validation, args.output, name,
                                                     result['validation_sha256'])
        print(json.dumps({name: result['models'][name]}, allow_nan=False), flush=True)
    reference, candidate = saved['reference'], saved['soft']
    for key in reference.keys() - {'z', 'mse'}:
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f'Held-out queries changed: {key}')
    comparisons = {}
    for group, mask in [('dr', ~reference['nominal']), ('nominal', reference['nominal'])]:
        ratio = float(candidate['mse'][mask].sum() / reference['mse'][mask].sum())
        comparisons[group] = {'candidate_error_change_percent': 100 * (ratio - 1),
            'candidate_to_reference_error_ratio': ratio,
            'paired_world_bootstrap_ratio_ci95': cluster_ci(reference['world'], reference['mse'],
                                                           candidate['mse'], mask)}
    result.update(complete=True, prediction_comparison=comparisons)
    write(args.output / 'heldout.json', result)
    print('complete', flush=True)


if __name__ == '__main__':
    main()
