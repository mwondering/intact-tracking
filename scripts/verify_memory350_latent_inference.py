"""Reproduce saved validation and verify the online raw-memory encoder route."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_memory350 import encode, error_metrics
from intact_tracking.forward_predictor_objective import _normalized_state_error
from intact_tracking.memory350_inference import Memory350Inference, load_memory350_checkpoint
from intact_tracking.memory350_model import Memory350Config, Memory350Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model = Memory350Predictor(Memory350Config(**state['model_config']))
    model.load_state_dict(state['model'], strict=True)
    model = model.to('cuda:0').eval().requires_grad_(False)
    frozen = load_memory350_checkpoint(args.checkpoint, device='cuda:0')
    rows = []
    with torch.inference_mode():
        for rank in range(4):
            batch = {k: v.to('cuda:0') for k, v in torch.load(
                args.checkpoint.parent / f'validation_broad_rank_{rank}.pt',
                map_location='cpu', weights_only=False).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = encode(model, batch, batch['memory_interactions'], batch['memory_valid'])
                squared_error = error_metrics(model, batch, z)
                no_change = batch['state'][:, :1].expand_as(batch['state'][:, 1:])
                denominator = _normalized_state_error(no_change, batch['state'][:, 1:],
                    batch['state_mean'], batch['state_std'], batch['delta_std']).float().square().mean(-1)
            dr = ~batch['is_nominal'].bool()
            nmse = float(squared_error[dr.cpu().numpy(), -1].sum() / denominator[dr, -1].sum().item())
            online = Memory350Inference(frozen, len(z), batch_size=256, use_bfloat16=True)
            c, bank = frozen, online.memory
            short = torch.cat((batch['history_state'] * c.state_std + c.state_mean,
                               batch['history_action'] * c.action_std + c.action_mean,
                               batch['history_next_state'] * c.state_std + c.state_mean), dim=-1)
            normalized_long = batch['memory_interactions']
            long = torch.cat((normalized_long[..., :71] * c.state_std + c.state_mean,
                              normalized_long[..., 71:100] * c.action_std + c.action_mean,
                              normalized_long[..., 100:] * c.state_std + c.state_mean), dim=-1)
            for w in range(len(z)):
                short_count = int(batch['history_valid'][w].sum())
                long_count = int(batch['memory_valid'][w].sum())
                bank.short_count[w] = short_count
                bank.short_cursor[w] = short_count % 50
                bank.short[w, :short_count] = short[w, batch['history_valid'][w]]
                bank.total_chunks[w] = long_count
                bank.chunks[w, :long_count] = long[w, batch['memory_valid'][w]]
                bank.chunk_stamp[w, :long_count] = torch.arange(long_count, device='cuda:0')
            restored = online.encode()
            cosine = torch.nn.functional.cosine_similarity(z.float(), restored)
            row = {'rank': rank, 'samples': len(z), 'five_step_nmse': nmse,
                   'dr_samples': int(dr.sum()), 'nominal_samples': int((~dr).sum()),
                   'online_vs_direct_unit_rms': float((torch.nn.functional.normalize(z.float(),dim=-1) - torch.nn.functional.normalize(restored,dim=-1)).square().sum(-1).mean().sqrt()),
                   'online_vs_direct_mean_cosine': float(cosine.mean()),
                   'online_vs_direct_min_cosine': float(cosine.min())}
            assert row['online_vs_direct_mean_cosine'] > .9999
            rows.append(row)
            print(json.dumps(row), flush=True)
            del batch, online, restored
    logs = [json.loads(s) for s in (args.checkpoint.parent / 'metrics.jsonl').read_text().splitlines() if s.strip()]
    recorded = next(r['fixed_probe']['dr_five_step_nmse'] for r in logs if r['update'] == state['update'])
    measured = float(np.mean([r['five_step_nmse'] for r in rows]))
    relative = measured / recorded - 1
    assert abs(relative) < .01
    result = {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': frozen.sha256,
              'rows': rows, 'recorded_five_step_nmse': recorded,
              'reproduced_five_step_nmse': measured, 'relative_difference': relative,
              'online_roundtrip': 'Saved normalized history denormalized into raw online memory bank, then encoded with the exact collector inference class.',
              'passed': True}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
