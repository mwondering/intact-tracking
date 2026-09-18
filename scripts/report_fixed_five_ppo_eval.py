"""Summarize paired fixed-DR expert/baseline evaluation with uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_summary(expert, baseline, environment_id, weights=None):
    """Environment-cluster standard error retains dependence across motion trials."""
    unique, inverse = np.unique(environment_id, return_inverse=True)
    weights = np.ones(len(expert))/len(expert) if weights is None else weights/weights.sum()
    delta = expert - baseline
    mean_delta = float(weights @ delta)
    e, b = float(weights @ expert), float(weights @ baseline)
    cluster_sum = np.bincount(inverse, weights=weights * (delta - mean_delta))
    variance = (len(unique) / (len(unique) - 1)) * (cluster_sum @ cluster_sum)
    halfwidth = 1.96 * np.sqrt(variance)
    return {'expert': e, 'baseline': b,
            'difference': mean_delta,
            'difference_95ci_environment_clustered': [mean_delta-halfwidth, mean_delta+halfwidth],
            'relative_change_percent': float(100*(e/b-1)) if b else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    root = args.directory
    data = []
    metadata = []
    for directory in sorted(root.glob('rank_*')):
        meta = json.loads((directory / 'result.json').read_text())
        rank = meta['rank']
        metadata.append(meta)
        assert meta['physics_matches_training'] and meta['paired_initial_states_verified']
        for trial in range(meta['arguments']['trials']):
            with np.load(directory / f'trial_{trial:02d}.npz') as row:
                n = len(row['class_id'])
                names = row['metric_names'].tolist()
                value = {'class_id': row['class_id'], 'env_id': rank*n + np.arange(n),
                         'motion_id': rank*meta['arguments']['motions'] + row['motion_id']}
                for arm_id, arm in enumerate(['expert','baseline']):
                    sl = slice(arm_id*n, (arm_id+1)*n)
                    count, horizon = row['counts'][sl], row['horizon'][sl]
                    value[f'{arm}/reward_per_planned_step'] = row['returns'][sl]/horizon
                    value[f'{arm}/reward_per_observed_step'] = row['returns'][sl]/np.maximum(count,1)
                    value[f'{arm}/episode_return'] = row['returns'][sl]
                    value[f'{arm}/failure_rate'] = row['failures'][sl].astype(float)
                    value[f'{arm}/coverage'] = count/horizon
                    for j, name in enumerate(names):
                        value[f'{arm}/{name}'] = row['totals'][sl,j]/np.maximum(count,1)
                        value[f'{arm}/matched_prefix/{name}'] = row['common_totals'][sl,j]/np.maximum(row['common_counts'],1)
                data.append(value)
    values = {key:np.concatenate([row[key] for row in data]) for key in data[0]}
    metrics = [key.removeprefix('expert/') for key in values if key.startswith('expert/')]
    training_counts = np.asarray(metadata[0]['training_global_class_counts'])
    sample_counts = np.bincount(values['class_id'], minlength=4)
    overall_weights = (training_counts / sample_counts)[values['class_id']]
    groups = {}
    for group in ['overall',0,1,2,3]:
        select = np.ones(len(values['class_id']),dtype=bool) if group=='overall' else values['class_id']==group
        groups[str(group)] = {'environments': int(len(np.unique(values['env_id'][select]))),
                              'paired_trials': int(select.sum()),
                              'metrics':{key:paired_summary(values[f'expert/{key}'][select],values[f'baseline/{key}'][select],values['env_id'][select], overall_weights[select] if group=='overall' else None) for key in metrics}}
    # A second clustering axis quantifies shared-motion uncertainty separately.
    for group in ['overall',0,1,2,3]:
        select = np.ones(len(values['class_id']),dtype=bool) if group=='overall' else values['class_id']==group
        for key in metrics:
            result = paired_summary(values[f'expert/{key}'][select],values[f'baseline/{key}'][select],values['motion_id'][select], overall_weights[select] if group=='overall' else None)
            groups[str(group)]['metrics'][key]['difference_95ci_motion_clustered'] = result['difference_95ci_environment_clustered']
    result = {'checkpoint_update':metadata[0]['completed_training_updates'],
              'protocol':metadata[0]['protocol'], 'scope':metadata[0]['scope'],
              'pairing':metadata[0]['pairing'], 'policy':metadata[0]['policy'],
              'groups':groups,
              'overall_weighting':{'training_class_counts':training_counts.tolist(), 'sampled_trials_per_class':sample_counts.tolist()},
              'uncertainty':'Paired difference, normal 95% intervals separately clustered by DR world and by motion; not a joint two-way confidence interval.',
              'reward':'Original reward; reward_per_planned_step divides return by planned horizon, treating remaining steps after failure as zero.',
              'errors':'Failure-truncated per-trial means; matched_prefix metrics use equal duration until either paired arm stops.',
              'macro_class_means': {key:{arm:float(np.mean([groups[str(i)]['metrics'][key][arm] for i in range(4)])) for arm in ['expert','baseline']} for key in metrics}}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    keys=['reward_per_planned_step','failure_rate','coverage','error_body_pos','error_joint_pos','error_anchor_pos_global']
    for group,row in groups.items():
        print(group,row['environments'],json.dumps({k:row['metrics'][k] for k in keys}))


if __name__=='__main__':
    main()
