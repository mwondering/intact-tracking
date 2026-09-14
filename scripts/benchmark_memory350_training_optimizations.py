"""Reproducible training-speed ablations, separate from all live experiments.

Example: CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/benchmark_memory350_training_optimizations.py --output-dir runs/memory350_speed_benchmark_20260912 --mode network
"""

import argparse
from contextlib import nullcontext
import gc
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from intact_tracking.memory350_scale_model import Memory350ScaleConfig, Memory350ScalePredictor
from intact_tracking.memory350_weak_pairs import WeakPairLossConfig
from intact_tracking.memory350_bank import InteractionMemory
from memory350_training_benchmark_components import BenchmarkWeakObjective, DeferredStatsMemory, encode_local_shared

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/limb_context_20260912_memory350_encoder2x_nominal50_weakpairs'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def log(event, **values):
    print(json.dumps(dict(event=event, unix_time=time.time(), **values)), flush=True)


def local_shifts(a, p, av, pv):
    found = torch.zeros(len(a), dtype=torch.bool)
    shift = torch.zeros(len(a), dtype=torch.long)
    for delta in (0, -1, 1):
        indices = torch.arange(a.shape[1]) + delta
        inside = (indices >= 0) & (indices < a.shape[1])
        match = (p[:, inside] == a[:, indices[inside]]).flatten(1).all(1)
        match &= (pv[:, inside] == av[:, indices[inside]]).all(1)
        take = match & ~found
        shift[take] = delta
        found |= match
    if not found.all():
        raise ValueError(f'{int((~found).sum())} saved local pairs cannot be shared exactly')
    return shift


def load_batch(output):
    source = RUN / 'stage1_8192/validation_rank_0.pt'
    raw = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    b = {k: (v[:256].clone() if v.ndim > 0 and v.shape[0] == 512 else v.clone()) for k, v in raw.items()}
    b['local_chunk_shift'] = local_shifts(b['memory_interactions'], b['positive_memory_interactions'],
                                         b['memory_valid'], b['positive_memory_valid'])
    # Fixed validation predates the weak archive. Use recorded full histories
    # solely as extra shape-realistic timing inputs; these artificial labels are
    # NOT a representation/prediction evaluation or new training data.
    q = torch.load(ROOT / 'runs/latent_cluster_probe_memory350_nominal50_u5000_20260911/common/queries/query_001600.pt',
                   map_location='cpu', weights_only=False, mmap=True)
    ids = (q['short_valid'].all(1) & q['long_valid'].all(1)).nonzero().flatten()[:256]
    assert len(ids) == 256
    short, long = q['short'][ids], q['long'][ids]
    def normalize(x):
        return torch.cat(((x[..., :71] - b['state_mean']) / b['state_std'],
                          (x[..., 71:100] - b['action_mean']) / b['action_std'],
                          (x[..., 100:] - b['state_mean']) / b['state_std']), -1)
    valid = torch.zeros(256, dtype=torch.bool)
    valid[(~b['is_nominal']).nonzero().flatten()[:70]] = True
    short = normalize(short).masked_fill(~valid[:, None, None], 0)
    long = normalize(long).masked_fill(~valid[:, None, None, None], 0)
    b.update(weak_history_state=short[..., :71], weak_history_action=short[..., 71:100],
             weak_history_next_state=short[..., 100:], weak_history_valid=valid[:, None].expand(-1, 50),
             weak_memory_interactions=long, weak_memory_valid=valid[:, None].expand(-1, 30),
             weak_pair_valid=valid, weak_world_id=b['world_id'].clone(), weak_motion_id=b['motion_id'] + 10000000,
             physics_session=torch.zeros(256, dtype=torch.long), weak_session=torch.zeros(256, dtype=torch.long),
             weak_age_steps=torch.full((256,), 800, dtype=torch.long))
    description = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       batch_size=256, views=3, weak_view_count=70,
                       warning='Real saved anchor/local pair; weak timing views/labels are constructed, not valid evaluation pairs.',
                       chunk_shift_counts={str(k): int((b['local_chunk_shift'] == k).sum()) for k in (-1, 0, 1)},
                       baseline_chunk_encodings=256*90, shared_chunk_encodings=256*61,
                       full_anchor_dr=int((~b['is_nominal'] & b['history_valid'].all(1) & b['memory_valid'].all(1)).sum()))
    write_json(output / 'batch.json', description)
    return b


def compare_tensors(actual, expected):
    # Long FP32 reductions can even produce cosine > 1 for 21M gradients.
    a, b = actual.double().flatten(), expected.double().flatten()
    d = a - b
    return dict(max_abs=float(d.abs().max()), relative_l2=float(d.norm() / b.norm().clamp_min(1e-20)),
                cosine=float(torch.nn.functional.cosine_similarity(a[None], b[None])),
                finite=bool(torch.isfinite(a).all()))


def synchronize_measure(fn):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    wall = time.perf_counter()
    fn()
    end.record()
    end.synchronize()
    return dict(wall_ms=1000*(time.perf_counter()-wall), cuda_ms=start.elapsed_time(end))


def summarize(samples):
    values = [x['wall_ms'] for x in samples]
    return dict(wall_median_ms=float(np.median(values)), wall_mean_ms=float(np.mean(values)),
                wall_p10_ms=float(np.quantile(values, .1)), wall_p90_ms=float(np.quantile(values, .9)),
                cuda_median_ms=float(np.median([x['cuda_ms'] for x in samples])), samples=samples)


def network(args):
    output = Path(args.output_dir)
    cp = torch.load(RUN / 'stage1_8192/update_005000.pt', map_location='cpu', weights_only=False, mmap=True)
    batch = {k: v.cuda() for k, v in load_batch(output).items()}
    model = Memory350ScalePredictor(Memory350ScaleConfig(**cp['model_config'])).cuda().train()
    model.load_state_dict(cp['model'])
    objective = BenchmarkWeakObjective(model, WeakPairLossConfig(**cp['loss_config']))
    original_forward = model.forward
    original_attention_forward = model.transformer.forward
    assert torch.equal(model.causal_mask, torch.ones_like(model.causal_mask).triu(diagonal=1))
    hinted_attention_forward = partial(original_attention_forward, is_causal=True)
    encoder = model.context_encoder
    def eager_views(hs, ha, hn, hv, mem, mv, shift):
        return encoder(hs, ha, hn, hv, mem, mv)
    def shared_views(hs, ha, hn, hv, mem, mv, shift):
        return encode_local_shared(encoder, hs, ha, hn, hv, mem, mv, shift)
    compiled_predictor = None
    functions = {'baseline': (None, original_forward, False),
                 'shared_chunks': (shared_views, original_forward, False),
                 'causal_hint': (None, original_forward, True),
                 'shared_hint': (shared_views, original_forward, True)}
    if not args.no_compile:
        # Fullgraph prevents silently accepting eager fallback inside these
        # fixed-shape neural regions. Recursive transition state remains eager.
        options = {'triton.cudagraphs': False, 'emulate_precision_casts': args.emulate_precision_casts}
        compiled_predictor = torch.compile(original_forward, fullgraph=True, dynamic=False, options=options)
        functions.update(compiled=(torch.compile(eager_views, fullgraph=True, dynamic=False, options=options), compiled_predictor, True),
                         shared_compiled=(torch.compile(shared_views, fullgraph=True, dynamic=False, options=options), compiled_predictor, True))
    latest = {}
    def select(name):
        objective.view_function, model.forward, hint = functions[name]
        model.transformer.forward = hinted_attention_forward if hint else original_attention_forward
    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = objective(batch, compute_metrics=False, validate_batch=False)
            (loss['loss'] * .25).backward()
        latest['loss'] = loss
    results = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                   checkpoint_update=cp['update'], scope='one microbatch forward + backward; no optimizer, DDP, replay or simulation',
                   emulate_precision_casts=args.emulate_precision_casts,
                   measurement='randomized interleaved, live GPU co-tenancy unless pause controller is explicitly used',
                   warmup={}, correctness={}, timing={})
    reference = None
    ready = []
    for name in functions:
        log('warmup_start', variant=name)
        select(name)
        start = time.perf_counter()
        try:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            gradients = torch.cat([p.grad.detach().float().flatten() for p in model.parameters()]).cpu()
            losses = {k: float(v.detach()) for k, v in latest['loss'].items()}
            if reference is None:
                reference = (gradients, losses)
            results['correctness'][name] = dict(gradient=compare_tensors(gradients, reference[0]), losses=losses,
                                                loss_relative_difference=abs(losses['loss']/reference[1]['loss'] - 1))
            results['warmup'][name] = dict(seconds=time.perf_counter()-start, passed=True,
                                           max_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            ready.append(name)
            del gradients
        except Exception as exc:
            results['warmup'][name] = dict(seconds=time.perf_counter()-start, passed=False,
                                           error=str(exc), traceback=traceback.format_exc())
            log('variant_error', variant=name, error=str(exc)[:1500])
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        write_json(output / 'network.json', results)
        log('warmup_complete', variant=name, result=results['warmup'][name])
    optimizer = None
    if args.update_repeats:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0., weight_decay=.001, fused=True)
        # Initialize states before timing. lr=0 holds the weights fixed across
        # performance variants while executing the full optimizer kernel.
        select('baseline')
        step()
        optimizer.step()
        torch.cuda.synchronize()
    def training_update():
        parameters = list(model.parameters())
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            accumulated = {}
            for _ in range(4):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = objective(batch, compute_metrics=False, validate_batch=False)
                    (result['loss'] * .25).backward()
                for key in result:
                    weighted = result[key].detach().float() * .25
                    accumulated[key] = accumulated.get(key, torch.zeros_like(weighted)) + weighted
            torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
            optimizer.step()
    if args.ready_file:
        Path(args.ready_file).write_text('ready\n')
        gate = Path(args.ready_file + '.go')
        log('waiting_for_timing_gate', path=str(gate))
        while not gate.exists():
            time.sleep(.1)
    samples = {name: [] for name in ready}
    rng = random.Random(8192)
    for iteration in range(args.repeats):
        order = list(ready)
        rng.shuffle(order)
        for name in order:
            select(name)
            samples[name].append(synchronize_measure(step))
        if iteration % 5 == 0:
            log('timing', iteration=iteration, medians={n: float(np.median([s['wall_ms'] for s in ss])) for n,ss in samples.items()})
    results['timing'] = {k: summarize(v) for k,v in samples.items()}
    if args.update_repeats:
        updates = {name: [] for name in ready}
        for iteration in range(args.update_repeats):
            order = list(ready)
            rng.shuffle(order)
            for name in order:
                select(name)
                updates[name].append(synchronize_measure(training_update))
        results['training_update'] = {k: summarize(v) for k,v in updates.items()}
        results['training_update_scope'] = ('4 optimizer steps x 4 accumulated microbatches of 256; '
                                             'includes scalar accumulation, finiteness check and fused AdamW; '
                                             'lr=0, fixed repeated batch; excludes collection, replay, DDP and evaluation')
    results['compile_counters'] = {k: dict(v) for k,v in torch._dynamo.utils.counters.items()}
    write_json(output / 'network.json', results)
    if args.ready_file:
        Path(args.ready_file + '.done').write_text('done\n')
    log('network_complete', timing={k: v['wall_median_ms'] for k,v in results['timing'].items()})


def memory(args):
    output = Path(args.output_dir)
    n = 8064
    original = InteractionMemory(n, device='cuda', archive_chunks=75)
    deferred = DeferredStatsMemory(n, device='cuda', archive_chunks=75)
    # Steady staggered world ages: each step has ~0.67% reset boundaries, as well
    # as ordinary 10-step commits and partially flushed histories.
    phase = torch.arange(n, device='cuda').remainder(150)
    interaction = torch.randn(n, 171, device='cuda')
    for bank in (original, deferred):
        bank.short.copy_(interaction[:, None].expand(-1, 50, -1))
        bank.short_count.copy_(phase.clamp_max(50))
        bank.short_cursor.copy_(phase.remainder(50))
        bank.pending_count.copy_((phase - 50).clamp_min(0).remainder(10))
        bank.pending.copy_(interaction[:, None].expand(-1, 10, -1))
    boundaries = [(phase + step).remainder(150) == 149 for step in range(150)]
    results = dict(scope='InteractionMemory.finish_step only; 8064 training worlds, 75 archived chunks, 0.67% staggered reset/step; identical state for both variants',
                   measurement='interleaved under live GPU co-tenancy unless paused', timing={})
    def run(bank, start):
        for i in range(5):
            bank.finish_step(interaction, boundaries[(start+i) % 150])
    for start in range(0, 20, 5):
        run(original, start); run(deferred, start)
    for key, value in vars(original).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(deferred, key), rtol=0, atol=0)
    assert original.metrics() == deferred.metrics()
    results['initial_state_and_counters_exact'] = True
    if args.ready_file:
        Path(args.ready_file).write_text('ready\n')
        while not Path(args.ready_file + '.go').exists():
            time.sleep(.1)
    samples = {k: [] for k in ('baseline', 'deferred_stats')}
    rng = random.Random(8192)
    banks = {'baseline': original, 'deferred_stats': deferred}
    for iteration in range(args.repeats):
        order = list(banks)
        rng.shuffle(order)
        for name in order:
            samples[name].append(synchronize_measure(lambda: run(banks[name], 20 + iteration*5)))
    for key, value in vars(original).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(deferred, key), rtol=0, atol=0)
    assert original.metrics() == deferred.metrics()
    results['final_state_and_counters_exact'] = True
    results['timing'] = {k: summarize(v) for k,v in samples.items()}
    # Show actual synchronization counts rather than infer them from source.
    results['profiler'] = {}
    for name, bank in banks.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            run(bank, 20+args.repeats*5)
        results['profiler'][name] = {e.key: dict(count=e.count, self_cpu_time_us=e.self_cpu_time_total)
                                    for e in profile.key_averages()
                                    if any(k in e.key for k in ('item', 'local_scalar', 'nonzero', 'Synchronize'))}
    write_json(output / 'memory.json', results)
    if args.ready_file:
        Path(args.ready_file + '.done').write_text('done\n')
    log('memory_complete', results=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=('network', 'memory'), required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--emulate-precision-casts', action='store_true')
    parser.add_argument('--update-repeats', type=int, default=0)
    parser.add_argument('--ready-file')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(8192)
    torch.set_float32_matmul_precision('high')
    torch.cuda.set_per_process_memory_fraction(.40)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log('start', mode=args.mode, pid=__import__('os').getpid())
    try:
        (network if args.mode == 'network' else memory)(args)
    except BaseException:
        write_json(Path(args.output_dir) / (args.mode + '_error.json'), dict(traceback=traceback.format_exc()))
        if args.ready_file:
            Path(args.ready_file + '.done').write_text('error\n')
        raise


if __name__ == '__main__':
    main()
