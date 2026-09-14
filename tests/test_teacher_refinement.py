from types import SimpleNamespace

import torch
from torch import nn

from intact_tracking.adaptation_policy import additional_refinement, make_refinement_mlp


def test_zero_refinement_preserves_teacher_and_only_new_branch_learns():
    torch.manual_seed(601)
    source = nn.Sequential(nn.Linear(5, 8), nn.ELU(), nn.Linear(8, 3)).requires_grad_(False)
    actor = SimpleNamespace(refinement_mlp=make_refinement_mlp(5, 3, 1.0), refinement_scale=1.0)
    values = torch.randn(7, 5)
    before = {key: value.clone() for key, value in source.state_dict().items()}
    expected = source(values).tanh() * 0.25
    torch.testing.assert_close(expected + additional_refinement(actor, values), expected, atol=0, rtol=0)
    optimizer = torch.optim.Adam(actor.refinement_mlp.parameters(), lr=1e-3)
    loss = (expected + additional_refinement(actor, values) - 0.2).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert additional_refinement(actor, values).count_nonzero() > 0
    assert all(torch.equal(value, before[key]) for key, value in source.state_dict().items())
    assert all(p.grad is None for p in source.parameters())


def test_disabled_refinement_registers_no_random_weights():
    state = torch.get_rng_state().clone()
    assert make_refinement_mlp(5, 3, 0.0) is None
    assert torch.equal(state, torch.get_rng_state())
