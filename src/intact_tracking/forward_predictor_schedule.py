"""Extend a running cosine decay while preserving its current learning rate."""

import math
from itertools import count


def training_update_indices(first_update, planned_updates, until_user_stop=False):
    return count(first_update) if until_user_stop else range(first_update, planned_updates + 1)


def advance_predictor_schedule(scheduler, continuation_floor=None):
    """Keep the existing decay, then a positive constant LR for manual stopping."""
    if continuation_floor is None:
        scheduler.step()
        return
    if continuation_floor <= 0:
        raise ValueError("Unbounded training requires a positive learning rate floor")
    if scheduler.last_epoch < scheduler.T_max:
        scheduler.step()
    else:
        # CosineAnnealingLR would otherwise turn upward past its old endpoint.
        scheduler.last_epoch += 1
        scheduler._step_count += 1
    for group in scheduler.optimizer.param_groups:
        group["lr"] = max(group["lr"], continuation_floor)
    scheduler._last_lr = [group["lr"] for group in scheduler.optimizer.param_groups]
    scheduler.continuation_min_learning_rate = continuation_floor


def extend_cosine_schedule(scheduler, target_steps):
    previous_target = scheduler.T_max
    if target_steps < previous_target:
        raise ValueError("Resuming may extend the optimizer budget, but cannot shorten it")
    if target_steps == previous_target:
        return None
    step = scheduler.last_epoch
    if step >= previous_target:
        raise ValueError("Extend the budget before the previous cosine schedule reaches its endpoint")
    learning_rates = [group["lr"] for group in scheduler.optimizer.param_groups]
    old_bases = list(scheduler.base_lrs)
    factor = (1 + math.cos(math.pi * step / target_steps)) / 2
    scheduler.T_max = target_steps
    scheduler.base_lrs = [scheduler.eta_min + (lr - scheduler.eta_min) / factor
                         for lr in learning_rates]
    for group, base in zip(scheduler.optimizer.param_groups, scheduler.base_lrs):
        group["initial_lr"] = base
    # CosineAnnealingLR uses the current optimizer LR recursively. Rescaling
    # base_lrs also makes its closed form and subsequent resumes consistent.
    return {"previous_target_steps": previous_target, "target_steps": target_steps,
            "anchor_optimizer_step": step, "anchor_learning_rates": learning_rates,
            "previous_base_lrs": old_bases, "extended_base_lrs": list(scheduler.base_lrs),
            "contract": "preserve the current LR and decay monotonically to the new endpoint"}
