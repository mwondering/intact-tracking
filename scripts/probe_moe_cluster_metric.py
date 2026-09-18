"""Offline partition controls on held-out DR worlds; never reroute a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.vq import kmeans2

from intact_tracking.memory350_online_kmeans import fit_kmeans, squared_distances
from probe_moe_latent_payload import load_case, world_metrics


def membership_readout(fit_ids, test_ids, train_targets, test_targets, worlds, k):
    means = np.stack([train_targets[fit_ids == i].mean(0) if np.any(fit_ids == i)
                      else train_targets.mean(0) for i in range(k)])
    prediction = means[test_ids]
    result = world_metrics(test_targets[:, :4], prediction[:, :4], worlds)
    result['r2_first_nine_coordinates'] = (1-((prediction-test_targets)**2).mean(0)/test_targets.var(0)).tolist()
    result['test_cluster_counts'] = np.bincount(test_ids, minlength=k).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = root/'cluster_metric_probe'
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    cases = {seed: load_case(root, 'sample', seed) for seed in (30401, 30402)}
    report = {'protocol': 'Offline sample-action trajectories; fit on one DR seed, test on the other, then reverse. No policy execution or expert-head reassignment with new centers.',
              'partition_fit': '768 whole worlds from training seed, 32 points per world; 3 KMeans++ restarts, 60 Lloyd iterations; select by training inertia only',
              'readout': 'Predict each DR parameter using the training mean of its assigned cluster; R2 on independent held-out seed',
              'whitening': 'Fit covariance on training latents only, floor eigenvalues at 1% of largest, whiten then unit-normalize; unsupervised',
              'oracle': 'True four-limb masses / their full ranges; illustrative upper reference for load-oriented 16-way partitions, not a deployable latent router',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'folds': []}
    for train_seed, test_seed in ((30402, 30401), (30401, 30402)):
        a, b = cases[train_seed], cases[test_seed]
        order = np.random.default_rng(887).permutation(len(a['raw_dr']))
        fit = np.isin(a['worlds'], order[:768])
        x, t = a['latent'][fit].astype(np.float32), b['latent'].astype(np.float32)
        train_targets, test_targets = a['raw_dr'][a['worlds'][fit], :9], b['raw_dr'][b['worlds'], :9]
        variants = {'saved16': membership_readout(a['routes'][fit], b['routes'], train_targets, test_targets, b['worlds'], 16)}
        with np.load(root/f'sample_seed{train_seed}'/'traces.npz') as f:
            saved = f['centers']
        variants['saved16']['test_unit_latent_inertia'] = float(((t-saved[b['routes']])**2).sum(-1).mean())
        xm = x.mean(0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(x.T))
        transform = eigenvectors/np.sqrt(np.maximum(eigenvalues, eigenvalues.max()*.01))[None, :]
        whiten_fit, whiten_test = (x-xm)@transform, (t-xm)@transform
        whiten_fit /= np.linalg.norm(whiten_fit, axis=-1, keepdims=True)
        whiten_test /= np.linalg.norm(whiten_test, axis=-1, keepdims=True)
        for name, fit_x, test_x, k in [('fresh16', x, t, 16), ('fresh64', x, t, 64),
                                     ('whiten16', whiten_fit, whiten_test, 16)]:
            fit_tensor, test_tensor = torch.from_numpy(fit_x.astype(np.float32)), torch.from_numpy(test_x.astype(np.float32))
            centers, inertia = fit_kmeans(fit_tensor, k, seed=731, restarts=3, iterations=60)
            fit_ids = squared_distances(fit_tensor, centers).argmin(-1).numpy()
            distances = squared_distances(test_tensor, centers)
            test_ids = distances.argmin(-1).numpy()
            values = membership_readout(fit_ids, test_ids, train_targets, test_targets, b['worlds'], k)
            values['training_objective'] = inertia
            values['test_objective'] = float(distances.min(-1).values.mean())
            if name != 'whiten16':
                values['test_unit_latent_inertia'] = values['test_objective']
            variants[name] = values
            print(json.dumps({'test_seed': test_seed, 'partition': name,
                              'r2_loads': values['r2_per_limb'], 'objective': values['test_objective']}), flush=True)
        # This reference uses actual weights and is explicitly not an inferred router.
        fit_load = train_targets[:, :4]/np.asarray([2.5, 2.5, 4., 4.])
        test_load = test_targets[:, :4]/np.asarray([2.5, 2.5, 4., 4.])
        best = None
        for seed in (731, 732, 733):
            centers, ids = kmeans2(fit_load, 16, iter=60, minit='++', rng=np.random.default_rng(seed))
            inertia = np.square(fit_load-centers[ids]).sum(-1).mean()
            if best is None or inertia < best[0]:
                best = inertia, centers, ids
        _, centers, ids = best
        test_ids = np.square(test_load[:, None]-centers[None]).sum(-1).argmin(-1)
        variants['oracle_load16'] = membership_readout(ids, test_ids, train_targets, test_targets, b['worlds'], 16)
        report['folds'].append({'train_seed': train_seed, 'test_seed': test_seed, 'variants': variants})
        (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    averaged = {}
    for name in report['folds'][0]['variants']:
        averaged[name] = {}
        for key in ('r2_per_limb', 'mae_kg_per_limb', 'r2_first_nine_coordinates', 'test_unit_latent_inertia'):
            if key in report['folds'][0]['variants'][name]:
                averaged[name][key] = np.mean([fold['variants'][name][key] for fold in report['folds']], axis=0).tolist()
    report['mean_of_two_direction_metrics'] = averaged
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# 在线中心、重新聚类与距离度量对照', '',
             '全部为离线分组分析，没有将新中心交给已训练的expert执行。使用sample动作轨迹，训练/测试DR seed完全分离，交换方向重复。每个分组仅以训练侧簇内平均重量预测测试侧重量。', '',
             '| 分组方式 | 左手R² | 右手R² | 左小腿R² | 右小腿R² | 原空间测试inertia |',
             '|---|---:|---:|---:|---:|---:|']
    for name, values in averaged.items():
        inertia = values.get('test_unit_latent_inertia')
        lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in values['r2_per_limb'])+' | '+('—' if inertia is None else f'{inertia:.4f}')+' |')
    lines.extend(['', 'saved16为checkpoint实际路由；fresh16/fresh64在当前轨迹上按原单位latent距离重新拟合16/64个中心；whiten16仅用训练latent协方差白化后重新归一化，再拟合16个中心；oracle_load16直接按真实四肢重量/各自范围分组，仅作参考。', '',
                  '白化改变了距离，因此其目标值不能与原空间inertia比较。Oracle更换了路由所依据的信息，也可能牺牲其他DR参数的分辨率。聚类结果更能区分负载，不等于已证明控制性能更好。', ''])
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps(averaged), flush=True)


if __name__ == '__main__':
    main()
