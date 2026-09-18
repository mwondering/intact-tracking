"""Decode limb masses from observed latents, holding out whole DR worlds/seeds."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from intact_tracking.moe_routing_diagnostics import collect_latent_samples


def world_metrics(y, prediction, worlds):
    unique, inverse = np.unique(worlds, return_inverse=True)
    counts = np.bincount(inverse)
    targets, predictions, squared, absolute = [np.zeros((len(unique), 4)) for _ in range(4)]
    for target, value in ((targets, y), (predictions, prediction),
                          (squared, (prediction-y)**2), (absolute, np.abs(prediction-y))):
        np.add.at(target, inverse, value)
        target /= counts[:, None]
    variance = targets.var(0)
    return {
        'worlds': len(unique), 'samples': len(y),
        'r2_per_limb': (1-squared.mean(0)/variance).tolist(),
        'mae_kg_per_limb': absolute.mean(0).tolist(),
        'rmse_kg_per_limb': np.sqrt(squared.mean(0)).tolist(),
        'temporal_mean_prediction_r2_per_limb': (1-((predictions-targets)**2).mean(0)/variance).tolist(),
    }


def ridge_fit(x, y, alpha):
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-6)
    ym = y.mean(0)
    features = (x-xm)/xs
    coef = np.linalg.solve(features.T@features+alpha*np.eye(x.shape[1]), features.T@(y-ym))
    return lambda test: ((test-xm)/xs)@coef+ym


def tuned_ridge(x, y, fit, validation, worlds, test):
    scores = []
    for alpha in (.01, .1, 1., 10., 100., 1000., 10000.):
        predict = ridge_fit(x[fit], y[fit], alpha)
        score = np.mean(world_metrics(y[validation], predict(x[validation]), worlds[validation])['r2_per_limb'])
        scores.append((float(score), alpha))
    score, alpha = max(scores)
    predict = ridge_fit(x[fit], y[fit], alpha)
    return predict(test), {'alpha': alpha, 'validation_mean_r2': score,
                           'validation_grid': [{'alpha': a, 'mean_r2': s} for s, a in scores]}


def mlp_fit(x, y, fit, validation, worlds, test, *, seed, epochs):
    torch.manual_seed(seed)
    xm, xs = x[fit].mean(0), np.maximum(x[fit].std(0), 1e-6)
    ym, ys = y[fit].mean(0), np.maximum(y[fit].std(0), 1e-6)
    a = torch.from_numpy(((x[fit]-xm)/xs).astype(np.float32))
    b = torch.from_numpy(((y[fit]-ym)/ys).astype(np.float32))
    val = torch.from_numpy(((x[validation]-xm)/xs).astype(np.float32))
    model = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_score, best_epoch, best = -np.inf, 0, None
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(a))
        for ids in order.split(1024):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(a[ids])-b[ids]).square().mean()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                prediction = model(val).numpy()*ys+ym
            metrics = world_metrics(y[validation], prediction, worlds[validation])
            score = float(np.mean(metrics['r2_per_limb']))
            history.append({'epoch': epoch, 'validation_mean_r2': score})
            if score > best_score:
                best_score, best_epoch, best = score, epoch, copy.deepcopy(model.state_dict())
            if epoch % 25 == 0:
                print(json.dumps({'stage': 'mlp', 'epoch': epoch, 'best_epoch': best_epoch,
                                  'best_validation_mean_r2': best_score}), flush=True)
            if epoch >= 30 and epoch-best_epoch >= 25:
                break
    model.load_state_dict(best)
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(((test-xm)/xs).astype(np.float32))).numpy()*ys+ym
    return prediction, {'architecture': [64, 128, 128, 4], 'activation': 'SiLU',
                        'best_epoch': best_epoch, 'stopped_epoch': epoch,
                        'validation_mean_r2': best_score, 'validation_history': history}


def load_case(root, mode, seed):
    directory = root/f'{mode}_seed{seed}'
    meta = json.loads((directory/'result.json').read_text())
    assert meta['static_DR_unchanged_verified'] and meta['actual_actor_dispatch_verified_every_step']
    with np.load(directory/'traces.npz') as f:
        trace = {key: f[key] for key in f.files}
    samples = collect_latent_samples(trace)
    worlds = samples['worlds']
    counts = np.bincount(worlds)
    # Equal world sampling makes all regressions and validation scores world-balanced.
    assert np.all(counts == 32), counts.tolist()
    return {**samples, 'y': trace['raw_dr'][worlds, :4].astype(np.float64),
            'raw_dr': trace['raw_dr'], 'context_sha256': meta['context_sha256'],
            'checkpoint_sha256': meta['checkpoint_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.root)
    output = root/'payload_decoder'
    output.mkdir(exist_ok=True)
    report = {'protocol': {'targets': ['left_hand_kg', 'right_hand_kg', 'left_shin_kg', 'right_shin_kg'],
                          'test_split': 'opposite independent DR/interaction seed; no shared worlds',
                          'fit_validation_split': '768/256 whole training-seed worlds; validation selects alpha/epoch',
                          'samples': '32 full-memory steady samples per world, same selection as cluster analysis',
                          'metrics': 'instantaneous squared/absolute errors averaged equally over worlds; R2 uses held-out-world target variance',
                          'temporal_metric': 'average predictions over 32 times per world, then evaluate; not an instantaneous decoder',
                          'limit': 'Successful decoding demonstrates available information; decoder failure cannot prove information absence.'},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'modes': {}}
    for mode in ('mean', 'sample'):
        cases = {seed: load_case(root, mode, seed) for seed in (30401, 30402)}
        assert cases[30401]['checkpoint_sha256'] == cases[30402]['checkpoint_sha256']
        assert cases[30401]['context_sha256'] == cases[30402]['context_sha256']
        folds = []
        for train_seed, test_seed in ((30402, 30401), (30401, 30402)):
            train, test = cases[train_seed], cases[test_seed]
            # Check that no complete physical world occurs in both DR banks.
            train_keys = {row.tobytes() for row in train['raw_dr']}
            assert not any(row.tobytes() in train_keys for row in test['raw_dr'])
            world_order = np.random.default_rng(887).permutation(len(train['raw_dr']))
            fit = np.isin(train['worlds'], world_order[:768])
            validation = ~fit
            y = train['y']
            predictions = {'constant': np.broadcast_to(y[fit].mean(0), test['y'].shape).copy()}
            metadata = {}
            group_mean = np.stack([y[fit & (train['routes'] == expert)].mean(0)
                                   if np.any(fit & (train['routes'] == expert)) else y[fit].mean(0)
                                   for expert in range(16)])
            predictions['expert_id'] = group_mean[test['routes']]
            print(json.dumps({'mode': mode, 'train_seed': train_seed, 'test_seed': test_seed,
                              'stage': 'fit_decoders'}), flush=True)
            for name, raw in (('unit_latent_ridge', False), ('raw_latent_ridge', True)):
                x = train['latent'].astype(np.float64)
                t = test['latent'].astype(np.float64)
                if raw:
                    x *= train['raw_norm'][:, None]
                    t *= test['raw_norm'][:, None]
                predictions[name], metadata[name] = tuned_ridge(x, y, fit, validation, train['worlds'], t)
            predictions['unit_latent_mlp'], metadata['unit_latent_mlp'] = mlp_fit(
                train['latent'], y, fit, validation, train['worlds'], test['latent'],
                seed=train_seed, epochs=args.epochs)
            metrics = {name: world_metrics(test['y'], prediction, test['worlds'])
                       for name, prediction in predictions.items()}
            row = {'train_seed': train_seed, 'test_seed': test_seed, 'models': metrics, 'fit': metadata}
            folds.append(row)
            np.savez_compressed(output/f'{mode}_train{train_seed}_test{test_seed}.npz',
                                targets=test['y'], worlds=test['worlds'], **predictions)
            (output/f'{mode}_train{train_seed}_test{test_seed}.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
            print(json.dumps({'mode': mode, 'test_seed': test_seed,
                              'r2': {name: value['r2_per_limb'] for name, value in metrics.items()}}), flush=True)
        averaged = {}
        for model in folds[0]['models']:
            averaged[model] = {metric: np.mean([fold['models'][model][metric] for fold in folds], axis=0).tolist()
                              for metric in ('r2_per_limb', 'mae_kg_per_limb', 'rmse_kg_per_limb',
                                             'temporal_mean_prediction_r2_per_limb')}
        report['modes'][mode] = {'folds': folds, 'mean_of_two_direction_metrics': averaged}
        (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# 从实际 MoE 轨迹的 latent 解码四肢负载', '',
             '按完整 DR world 划分训练/验证，使用另一组独立 DR seed 的全部1024个world测试，并交换两个seed重复。训练侧768个world拟合、256个world选择超参数；每world取32个充分记忆、稳定窗口的latent。所有模型使用相同数据，policy和encoder均不更新。', '',
             'R²=1−均方预测误差/测试负载方差。1表示完全预测，0相当于测试集均值，负值表示更差。这不是分类识别率。下表为两个交换方向R²的平均。', '',
             '| 动作 | 输入/解码器 | 左手 | 右手 | 左小腿 | 右小腿 |', '|---|---|---:|---:|---:|---:|']
    for mode in ('mean', 'sample'):
        for name, values in report['modes'][mode]['mean_of_two_direction_metrics'].items():
            lines.append('| '+mode+' | '+name+' | '+' | '.join(f'{v:.3f}' for v in values['r2_per_limb'])+' |')
    lines.extend(['', 'expert_id使用训练侧各expert的平均重量；ridge输入完整64维单位latent或原始latent；MLP输入单位latent，结构64→128→128→4，验证集选早停轮次。负载标签只用于这个离线探针，不反馈给policy或encoder。', '',
                  '成功解码能证明信息可提取；线性和MLP都失败也不能证明完全没有信息。测试轨迹来自实际MoE控制，不能据此把原因进一步唯一归结到encoder训练目标、轨迹分布或记忆维护。逐fold指标、kg误差、时间平均后的结果及预测值保存在同目录。', ''])
    (output/'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
