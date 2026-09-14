"""Read-only checkpoint inference benchmarks while the main training continues."""

from __future__ import annotations

from contextlib import nullcontext
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from intact_tracking.memory350_inference import Memory350Inference, load_memory350_checkpoint
from intact_tracking.residual_context import DynamicsContextInference, load_frozen_context_checkpoint


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/limb_context_20260909_memory350/inference_benchmark_20260910'


def measure(fn, repeats=12, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, gpu = [], []
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        start_event.record()
        value = fn()
        end_event.record()
        end_event.synchronize()
        wall.append((time.perf_counter()-start)*1000)
        gpu.append(start_event.elapsed_time(end_event))
        del value
    return {'wall_median_ms': float(np.median(wall)), 'wall_p90_ms': float(np.quantile(wall, .9)),
            'wall_min_ms': min(wall), 'cuda_event_median_ms': float(np.median(gpu)),
            'incremental_peak_allocated_mib': (torch.cuda.max_memory_allocated()-allocated)/2**20,
            'repeats': repeats, 'wall_samples_ms': wall}


def repeat_rows(value, n):
    return value.index_select(0, torch.arange(n, device=value.device) % len(value))


def chunk_encode(encoder, raw, valid):
    b, c, s, _ = raw.shape
    clean = raw.masked_fill(~valid[..., None, None], 0)
    tokens = encoder.interaction_projection(clean).reshape(b*c, s, -1)
    tokens = torch.cat((encoder.chunk_cls.expand(b*c, -1, -1), tokens), dim=1)
    hidden = encoder.chunk_encoder(tokens + encoder.chunk_position)
    return encoder.chunk_output(hidden[:, 0]).reshape(b, c, -1).masked_fill(~valid[..., None], 0)


def long_encode(encoder, chunks, valid):
    b = chunks.shape[0]
    tokens = torch.cat((encoder.memory_cls.expand(b, -1, -1), chunks), dim=1)
    padding = torch.cat((torch.zeros(b, 1, dtype=torch.bool, device=valid.device), ~valid), dim=1)
    hidden = encoder.memory_encoder(tokens + encoder.memory_position, src_key_padding_mask=padding)
    return encoder.memory_output(hidden[:, 0]).masked_fill(~valid.any(1, keepdim=True), 0)


def final_encode(encoder, short_raw, short_valid, memory, memory_valid):
    short = encoder.interaction_projection(short_raw.masked_fill(~short_valid[..., None], 0))
    short = short.masked_fill(~short_valid[..., None], 0)
    b = len(short)
    tokens = torch.cat((encoder.cls_token.expand(b, -1, -1), memory[:, None]+encoder.memory_type, short), dim=1)
    padding = torch.cat((torch.zeros(b, 1, dtype=torch.bool, device=short.device),
                         ~memory_valid.any(1, keepdim=True), ~short_valid), dim=1)
    hidden = encoder.transformer(tokens + encoder.position, src_key_padding_mask=padding)
    return encoder.output(hidden[:, 0])


def run():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0'
    paths = {'memory350': ROOT/'runs/limb_context_20260909_memory350/stage1_8192/update_007500.pt',
             'context200': ROOT/'runs/limb_context_20260909_context200/stage1/update_007500.pt'}
    checkpoints = {'memory350': load_memory350_checkpoint(paths['memory350'], device=device),
                   'context200': load_frozen_context_checkpoint(paths['context200'], device=device)}
    batches = {}
    for name in checkpoints:
        b = torch.load(paths[name].parent/'validation_rank_0.pt', map_location='cpu', weights_only=False)
        # Fully populated history avoids making empty memory appear computationally cheap.
        mask = b['history_valid'].all(1)
        if name == 'memory350':
            mask &= b['memory_valid'].all(1)
        ids = mask.nonzero().flatten()
        assert len(ids)
        batches[name] = {k: v.index_select(0, ids).to(device) for k, v in b.items()
                         if k in ['state','history_state','history_action','history_valid','history_next_state',
                                  'memory_interactions','memory_valid']}
    result = {'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
              'precision': 'BF16 autocast, FP32 stored weights/history',
              'concurrent_training': True, 'device': 'physical GPU 0',
              'scope': 'encoder and online history processing; excludes tracker, actor, critic, predictor, simulator',
              'checkpoint_paths': {k: str(v) for k,v in paths.items()},
              'parameters': {k: sum(p.numel() for p in c.encoder.parameters()) for k,c in checkpoints.items()},
              'measurements': {}, 'started_at': time.time()}

    def save():
        (OUT/'benchmark.json').write_text(json.dumps(result, indent=2)+'\n')

    with torch.inference_mode():
        for n in [1, 512, 4096, 8192]:
            for name in ['context200', 'memory350']:
                cp = checkpoints[name]
                b = {k: repeat_rows(v, n) for k,v in batches[name].items()}
                key = f'{name}_envs{n}'
                record = {}
                def pure():
                    outputs = []
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        for start in range(0, n, 512):
                            sl = slice(start, start+512)
                            if name == 'memory350':
                                z = cp.encoder(b['history_state'][sl], b['history_action'][sl], b['history_next_state'][sl],
                                               b['history_valid'][sl], b['memory_interactions'][sl], b['memory_valid'][sl])
                            else:
                                z = cp.encoder(b['history_state'][sl], b['history_action'][sl], b['state'][sl, 0], b['history_valid'][sl])
                            outputs.append(z.float())
                    return torch.cat(outputs)
                record['encoder_only_same_chunk512'] = measure(pure)
                if name == 'memory350':
                    inference = Memory350Inference(cp, n)
                    raw_short = torch.cat((b['history_state']*cp.state_std+cp.state_mean,
                                           b['history_action']*cp.action_std+cp.action_mean,
                                           b['history_next_state']*cp.state_std+cp.state_mean), dim=-1)
                    m = b['memory_interactions']
                    raw_long = torch.cat((m[..., :71]*cp.state_std+cp.state_mean,
                                          m[..., 71:100]*cp.action_std+cp.action_mean,
                                          m[..., 100:]*cp.state_std+cp.state_mean), dim=-1)
                    bank = inference.memory
                    bank.short.copy_(raw_short); bank.short_count.fill_(50); bank.short_cursor.zero_()
                    bank.chunks.copy_(raw_long); bank.total_chunks.fill_(30)
                    bank.chunk_stamp.copy_(torch.arange(30, device=device)[None])
                    record['persistent_history_mib'] = bank.storage_bytes/2**20
                    record['online_encode'] = measure(inference.encode)
                    # Normalized->raw->normalized changes tiny FP32 rounding, so use tolerance.
                    online = inference.encode()
                    direct = pure()
                    record['online_vs_direct_latent_cosine'] = float(torch.nn.functional.cosine_similarity(online, direct).mean())
                    assert record['online_vs_direct_latent_cosine'] > .999
                    if n == 8192:
                        episode = torch.zeros(n, dtype=torch.long, device=device)
                        steps = torch.zeros_like(episode)
                        calls = [0]
                        def append():
                            boundary = torch.arange(n, device=device) % 100 == calls[0] % 100
                            inference.append({'episode_id': episode, 'episode_step': steps,
                                              'motion_id': episode, 'motion_step': steps,
                                              'robot_state': raw_short[:, -1, :71],
                                              'joint_target': raw_short[:, -1, 71:100],
                                              'next_robot_state': raw_short[:, -1, 100:],
                                              'reset_boundary': boundary})
                            steps.add_(1); steps.masked_fill_(boundary, 0); episode.add_(boundary.long())
                            calls[0] += 1
                        record['append_with_one_percent_resets'] = measure(append, repeats=20, warmup=5)
                    del inference, bank, raw_short, raw_long, m, online, direct
                else:
                    inference = DynamicsContextInference(cp, num_envs=n, device=device)
                    inference.history_state.copy_((b['history_state']*cp.state_std+cp.state_mean).transpose(0, 1))
                    inference.history_action.copy_((b['history_action']*cp.action_std+cp.action_mean).transpose(0, 1))
                    inference.history_valid.fill_(True)
                    current = b['state'][:, 0]*cp.state_std+cp.state_mean
                    record['persistent_history_mib'] = sum(v.numel()*v.element_size() for v in
                                                          (inference.history_state, inference.history_action, inference.history_valid))/2**20
                    record['online_encode'] = measure(lambda: inference.encode(current))
                    del inference, current
                result['measurements'][key] = record
                save()
                print(json.dumps({'case':key, 'encoder_ms':record['encoder_only_same_chunk512']['wall_median_ms'],
                                  'online_ms':record['online_encode']['wall_median_ms'],
                                  'history_mib':record['persistent_history_mib']}), flush=True)
                del b
                gc.collect(); torch.cuda.empty_cache()

        cp = checkpoints['memory350']
        b = {k: repeat_rows(v, 512) for k,v in batches['memory350'].items()}
        e = cp.encoder
        short = torch.cat((b['history_state'], b['history_action'], b['history_next_state']), dim=-1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            chunks = chunk_encode(e, b['memory_interactions'], b['memory_valid'])
            memory = long_encode(e, chunks, b['memory_valid'])
            direct = e(b['history_state'], b['history_action'], b['history_next_state'], b['history_valid'], b['memory_interactions'], b['memory_valid'])
            cached = final_encode(e, short, b['history_valid'], memory, b['memory_valid'])
            torch.testing.assert_close(cached, direct, rtol=0, atol=0)
            stages = {
                'all_30_chunk_encoders': lambda: chunk_encode(e, b['memory_interactions'], b['memory_valid']),
                'long_attention_on_cached_chunks': lambda: long_encode(e, chunks, b['memory_valid']),
                'short_plus_cached_memory_final_encoder': lambda: final_encode(e, short, b['history_valid'], memory, b['memory_valid']),
            }
            result['stages_batch512'] = {k: measure(fn, repeats=20) for k,fn in stages.items()}
        result['cached_stage_output_exactly_matches'] = True
        del b, chunks, memory, direct, cached, short, stages
        gc.collect(); torch.cuda.empty_cache()
        # Interleave old/new native paths so contiguous training phases cannot
        # dominate a comparison made from separate short timing blocks.
        result['interleaved_native'] = {}
        rng = np.random.default_rng(350)
        for n in [1, 8192]:
            old_cp = checkpoints['context200']
            old_b = {k: repeat_rows(v, n) for k,v in batches['context200'].items()}
            old = DynamicsContextInference(old_cp, num_envs=n, device=device)
            old.history_state.copy_((old_b['history_state']*old_cp.state_std+old_cp.state_mean).transpose(0,1))
            old.history_action.copy_((old_b['history_action']*old_cp.action_std+old_cp.action_mean).transpose(0,1))
            old.history_valid.fill_(True)
            old_current = old_b['state'][:,0]*old_cp.state_std+old_cp.state_mean
            del old_b
            cp = checkpoints['memory350']
            b = {k: repeat_rows(v, n) for k,v in batches['memory350'].items()}
            new = Memory350Inference(cp, n)
            new.memory.short.copy_(torch.cat((b['history_state']*cp.state_std+cp.state_mean,
                                              b['history_action']*cp.action_std+cp.action_mean,
                                              b['history_next_state']*cp.state_std+cp.state_mean),dim=-1))
            m = b['memory_interactions']
            new.memory.chunks.copy_(torch.cat((m[...,:71]*cp.state_std+cp.state_mean,
                                               m[...,71:100]*cp.action_std+cp.action_mean,
                                               m[...,100:]*cp.state_std+cp.state_mean),dim=-1))
            new.memory.short_count.fill_(50); new.memory.total_chunks.fill_(30)
            new.memory.chunk_stamp.copy_(torch.arange(30,device=device)[None])
            del b, m
            functions = {'context200':lambda: old.encode(old_current), 'memory350':new.encode}
            for fn in functions.values():
                for _ in range(3):fn()
            observations = {key:[] for key in functions}
            for _ in range(40):
                for key in rng.permutation(list(functions)):
                    observations[key].append(measure(functions[key],repeats=1,warmup=0)['wall_median_ms'])
            result['interleaved_native'][str(n)] = {
                key:{'wall_median_ms':float(np.median(v)), 'wall_p90_ms':float(np.quantile(v,.9)),
                     'wall_mean_ms':float(np.mean(v)), 'repeats':len(v),'wall_samples_ms':v}
                for key,v in observations.items()}
            print(json.dumps({'interleaved_envs':n,'medians_ms':{
                k:float(np.median(v)) for k,v in observations.items()}}),flush=True)
            del old,new,old_current,functions
            gc.collect();torch.cuda.empty_cache()
            save()
        result['finished_at'] = time.time()
        save()
    print(json.dumps({'finished':True, 'out':str(OUT)}), flush=True)


if __name__ == '__main__':
    run()
