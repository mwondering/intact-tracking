"""Small rank-zero Weights & Biases adapter for online training."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


class WandbLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        is_main: bool,
        project: str,
        output_dir: Path,
        config: dict[str, Any],
        entity: str | None = None,
        group: str | None = None,
        name: str | None = None,
        tags: tuple[str, ...] = (),
        mode: str = "online",
    ) -> None:
        self.run = None
        if not enabled or not is_main:
            return
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as error:
            raise RuntimeError(
                "W&B logging is enabled but wandb is not installed; install the project "
                "dependencies or pass --no-wandb"
            ) from error
        self.run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=name,
            tags=list(tags),
            mode=mode,
            dir=str(output_dir),
            config=config,
        )
        self.run.define_metric("update")
        self.run.define_metric("optimizer_steps")
        self.run.define_metric("train/*", step_metric="update")
        self.run.define_metric("tracking/*", step_metric="update")
        self.run.define_metric("optimization/*", step_metric="update")
        self.run.define_metric("replay/*", step_metric="update")

    @property
    def id(self) -> str | None:
        return None if self.run is None else str(self.run.id)

    @property
    def url(self) -> str | None:
        return None if self.run is None else str(self.run.url)

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        if self.run is not None:
            self.run.log(payload, step=step)

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None

