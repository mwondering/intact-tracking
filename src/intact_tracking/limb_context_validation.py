"""Disjoint physical worlds for validation; no validation statistics in training."""

import math


class SplitWorldReplay:
    def __init__(self, training, validation):
        self.training, self.validation = training, validation
        self.collect_validation = True

    def add_step(self, batch):
        n = self.training.num_worlds
        self.training.add_step({k: v[:n] for k, v in batch.items()})
        if self.collect_validation:
            self.validation.add_step({k: v[n:] for k, v in batch.items()})


def validation_plateau(history, update, window=1000, minimum=2000,
                       diagnostics=("latent_positive_cosine", "latent_response_correlation")):
    if update < minimum:
        return False
    old = [r for r in history if r["update"] <= update - window]
    recent = [r for r in history if r["update"] > update - window]
    if not old or len(recent) < 5:
        return False
    before = min(r["fixed_probe"]["dr_five_step_nmse"] for r in old)
    after = min(r["fixed_probe"]["dr_five_step_nmse"] for r in history)
    improvement = (before - after) / max(before, 1e-8)
    stable = all(
        all(math.isfinite(r["fixed_probe"][k]) for r in recent)
        and max(r["fixed_probe"][k] for r in recent) - min(r["fixed_probe"][k] for r in recent) < .1
        for k in diagnostics
    )
    return improvement < .01 and stable
