"""Correctness gates for isolated speed prototypes, not new training losses."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from memory350_training_benchmark_components import DeferredStatsMemory, encode_local_shared
from intact_tracking.memory350_bank import InteractionMemory
from intact_tracking.memory350_model import HierarchicalContextEncoder, Memory350Config


def test_shared_chunk_gradients_include_both_views_and_mask_partial_histories():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    encoder = HierarchicalContextEncoder(Memory350Config(
        context_dim=16, context_heads=4, context_depth=1, chunk_depth=1, memory_depth=1)).double()
    batch = 6
    shifts = torch.tensor([-1, 0, 1, -1, 0, 1])
    history = torch.randn(3 * batch, 50, 171, dtype=torch.double)
    hv = torch.ones(3 * batch, 50, dtype=torch.bool)
    hv[0] = False
    hv[1, :20] = False
    hv[2*batch:2*batch+2] = False
    pool = torch.randn(batch, 34, 10, 171, dtype=torch.double)
    anchor = pool[:, 1:31]
    positive = torch.stack([pool[i, 1+int(s):31+int(s)] for i, s in enumerate(shifts)])
    validity = torch.ones(batch, 34, dtype=torch.bool)
    validity[0] = False
    validity[1, :12] = False
    validity[2, :15] = False
    av = validity[:, 1:31]
    pv = torch.stack([validity[i, 1+int(s):31+int(s)] for i, s in enumerate(shifts)])
    mv = torch.cat((av, pv, torch.ones_like(av)))
    memory = torch.cat((anchor, positive, torch.randn_like(anchor))).masked_fill(~mv[..., None, None], 0)
    args = (history[..., :71], history[..., 71:100], history[..., 100:], hv, memory, mv)
    weight = torch.randn(3 * batch, 64, dtype=torch.double)
    expected = encoder(*args)
    (expected * weight).sum().backward()
    gradients = {k: p.grad.clone() for k, p in encoder.named_parameters()}
    encoder.zero_grad(set_to_none=True)
    actual = encode_local_shared(encoder, *args, shifts)
    (actual * weight).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    for name, p in encoder.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name], rtol=1e-10, atol=1e-10)
    # A later parameter update must change the encoding: there is no stale cache.
    with torch.no_grad():
        encoder.output[1].weight.add_(torch.randn_like(encoder.output[1].weight) * .001)
    updated = encode_local_shared(encoder, *args, shifts)
    assert not torch.equal(updated, actual)
    torch.testing.assert_close(updated, encoder(*args), rtol=1e-11, atol=1e-11)


def test_deferred_statistics_preserve_resets_invalidations_and_raw_history():
    torch.set_num_threads(2)
    torch.manual_seed(717)
    reference, actual = InteractionMemory(8), DeferredStatsMemory(8)
    for step in range(165):
        interaction = torch.randn(8, 171)
        boundary = (torch.arange(8) + step).remainder(23) == 22
        for bank in (reference, actual):
            if step == 98:
                bank.invalidate(torch.arange(8).remainder(3) == 0)
            bank.finish_step(interaction, boundary)
    for name, value in vars(reference).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(actual, name), rtol=0, atol=0)
    assert reference.metrics() == actual.metrics()
    for a, b in zip(reference.read_chunks(), actual.read_chunks()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
