"""Separate encoder gradients of the frozen u35857 objective without optimizer updates."""
from pathlib import Path
import gc
import hashlib
import json
import time

import numpy as np
import torch

from intact_tracking.memory350_model import Memory350Config, Memory350Predictor
from intact_tracking.memory350_dr_centers import DRCenterLossConfig, DRCenterObjective


class CapturedObjective(DRCenterObjective):
    def _representation_terms(self, *args, **kwargs):
        result = super()._representation_terms(*args, **kwargs)
        self.center_graph = result[0]*self.loss_config.representation_weight
        return result

    def _extra_representation_loss(self, *args, **kwargs):
        result = super()._extra_representation_loss(*args, **kwargs)
        self.positive_graph = result[0]
        return result


def dot(a, b):
    return float(torch.dot(a, b))


def grad_stats(gradients):
    norms = {name: float(value.norm()) for name, value in gradients.items()}
    cos = {a+'__'+b: dot(gradients[a], gradients[b])/max(norms[a]*norms[b], 1e-30)
           for a, b in [('prediction', 'center'), ('prediction', 'positive'), ('center', 'positive')]}
    gp, gc, gs = (gradients[name] for name in ('prediction', 'center', 'positive'))
    alternatives = []
    for cw, pw in ((.4, .2), (.8, .2), (1.6, .2), (.8, .4)):
        total = gp+(cw/.4)*gc+(pw/.2)*gs
        alternatives.append({'center_weight': cw, 'positive_weight': pw,
                             'total_encoder_gradient_norm': float(total.norm()),
                             'descent_alignment_with_current_weighted_center_gradient': dot(total, gc)/max(dot(gc, gc), 1e-30),
                             'descent_alignment_with_prediction_gradient': dot(total, gp)/max(dot(gp, gp), 1e-30)})
    return {'norms': norms, 'cosines': cos,
            'weighted_center_to_prediction_norm_ratio': norms['center']/max(norms['prediction'], 1e-30),
            'weighted_positive_to_prediction_norm_ratio': norms['positive']/max(norms['prediction'], 1e-30),
            'algebraic_weight_alternatives_no_updates': alternatives}


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    root = Path('runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192')
    out = Path('runs/limb_context_20260916_dr_center_gradient_balance')
    out.mkdir(exist_ok=False)
    cp = root/'update_035857.pt'
    state = torch.load(cp, map_location='cpu', weights_only=False, mmap=True)
    model = Memory350Predictor(Memory350Config(**state['model_config'])).cuda().train()
    model.load_state_dict(state['model'], strict=True)
    # Freezing predictor parameters preserves the derivative through it to encoder inputs.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('context_encoder.'))
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    objective = CapturedObjective(model, DRCenterLossConfig(**state['loss_config']))
    assert state['loss_config']['representation_weight'] == .4
    assert state['loss_config']['dr_positive_weight'] == .2
    report = {
        'checkpoint': str(cp), 'checkpoint_update': state['update'],
        'protocol': 'All eight saved broad validation batches, 512 histories each; split into original training microbatches of 256. GPU6 BF16 forward/autograd, dropout=0, same objective and recursive weight 0.5. Capture differentiable components from original implementation and obtain separate raw gradients on shared encoder parameters only. No optimizer, model update, new rollout or training. Frozen validation is not the current training stream; raw gradient norms are not AdamW parameter-step magnitudes.',
        'loss_config': state['loss_config'], 'microbatches': [], 'validation_sources': []}
    with cp.open('rb') as stream:
        report['checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    accumulated = {}
    sizes = np.array([value.numel() for _, value in parameters])
    offsets = np.r_[0, np.cumsum(sizes)]
    groups = {'projection': [], 'chunk': [], 'memory': [], 'final': []}
    for i, (name, _) in enumerate(parameters):
        short = name.removeprefix('context_encoder.')
        group = 'projection' if short.startswith('interaction_projection') else 'chunk' if short.startswith('chunk_') else 'memory' if short.startswith('memory_') else 'final'
        groups[group].append(i)
    for rank in range(8):
        path = root/f'validation_broad_rank_{rank}.pt'
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        report['validation_sources'].append({'path': str(path), 'sha256': digest})
        source = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        assert len(source['world_id']) == 512
        for start in (0, 256):
            batch = {key: (value[start:start+256] if value.ndim and value.shape[0] == 512 else value).cuda()
                     for key, value in source.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = objective(batch, recursive_weight=.5, compute_metrics=False, validate_batch=False)
            graphs = {'prediction': result['loss']-objective.center_graph-objective.positive_graph,
                      'center': objective.center_graph, 'positive': objective.positive_graph}
            assert torch.allclose(graphs['prediction'].detach(), result['prediction_loss'], atol=1e-6, rtol=1e-5)
            gradients = {}
            for name, graph in graphs.items():
                grads = torch.autograd.grad(graph, [p for _, p in parameters], retain_graph=True, allow_unused=True)
                flat = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
                                  for (_, p), g in zip(parameters, grads)])
                assert torch.isfinite(flat).all()
                gradients[name] = flat
                if name not in accumulated:
                    accumulated[name] = flat.clone()
                else:
                    accumulated[name].add_(flat)
            entry = {'rank': rank, 'offset': start, 'losses': {key: float(value.detach()) for key, value in graphs.items()},
                     'eligible_centers': float(result['dr_center_valid_worlds']),
                     'center_pairs': float(result['dr_center_pairs']), 'positive_pairs': float(result['dr_positive_pairs']),
                     'encoder': grad_stats(gradients), 'parts': {}}
            for group, indices in groups.items():
                selected = {name: torch.cat([g[offsets[i]:offsets[i+1]] for i in indices]) for name, g in gradients.items()}
                entry['parts'][group] = grad_stats(selected)
            # An independent gradient of the total validates component extraction once.
            if rank == 0 and start == 0:
                grads = torch.autograd.grad(result['loss'], [p for _, p in parameters], allow_unused=True)
                total = torch.cat([(torch.zeros_like(p) if g is None else g).float().flatten() for (_, p), g in zip(parameters, grads)])
                reconstructed = sum(gradients.values())
                error = float((total-reconstructed).norm()/total.norm().clamp_min(1e-12))
                entry['bf16_total_gradient_relative_reconstruction_error'] = error
                assert error < .05, error
                del total, reconstructed
            report['microbatches'].append(entry)
            print(json.dumps({'rank': rank, 'offset': start, 'losses': entry['losses'], 'encoder': entry['encoder'],
                              'elapsed_seconds': time.monotonic()-started}), flush=True)
            objective.center_graph = objective.positive_graph = None
            del result, graphs, gradients, selected, batch, grads, flat, graph
            gc.collect()
        del source
    report['averaged_encoder_gradient'] = grad_stats({name: value/16 for name, value in accumulated.items()})
    report['microbatch_medians'] = {
        'weighted_center_to_prediction_norm_ratio': float(np.median([r['encoder']['weighted_center_to_prediction_norm_ratio'] for r in report['microbatches']])),
        'weighted_positive_to_prediction_norm_ratio': float(np.median([r['encoder']['weighted_positive_to_prediction_norm_ratio'] for r in report['microbatches']])),
        'prediction_center_cosine': float(np.median([r['encoder']['cosines']['prediction__center'] for r in report['microbatches']]))}
    report['model_parameters_unchanged'] = all(torch.equal(value.detach().cpu(), state['model'][name]) for name, value in model.state_dict().items())
    assert report['model_parameters_unchanged']
    report['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
    report['elapsed_seconds'] = time.monotonic()-started
    (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('microbatch_medians', 'averaged_encoder_gradient', 'model_parameters_unchanged', 'elapsed_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
