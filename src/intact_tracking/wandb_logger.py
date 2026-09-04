"""Weights & Biases adapters used by the standalone and RSL-RL trainers."""

from __future__ import annotations

import importlib
import os
import pathlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rsl_rl.utils.log_writer import LogWriter
from torch.utils.tensorboard import SummaryWriter


class RslWandbLogWriter(SummaryWriter, LogWriter):
    """RSL-RL W&B writer compatible with current W&B releases.

    RSL-RL 5.4's bundled writer passes ``Settings(start_method="thread")``.
    That setting was removed from newer W&B versions, so constructing the
    bundled writer raises before training starts.  W&B now manages its own
    service startup, therefore no explicit start method is needed.
    """

    def __init__(self, log_dir: str, project_name: str) -> None:
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as error:
            raise RuntimeError(
                "W&B logging is enabled but wandb is not installed; install the project "
                "dependencies or pass --logger tensorboard"
            ) from error

        super().__init__(log_dir, flush_secs=10)
        self._wandb = wandb
        self.logged_videos: set[str] = set()
        try:
            self._wandb.init(
                project=project_name,
                entity=os.environ.get("WANDB_USERNAME"),
                name=os.path.basename(os.path.normpath(log_dir)),
                config={"log_dir": log_dir},
            )
        except Exception:
            SummaryWriter.close(self)
            raise

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self._wandb.log({tag: scalar_value}, step=global_step)

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        self._wandb.config.update({"train_cfg": train_cfg})
        if hasattr(env_cfg, "to_dict"):
            serialized_env_cfg = env_cfg.to_dict()
        else:
            try:
                serialized_env_cfg = asdict(env_cfg)
            except TypeError:
                serialized_env_cfg = env_cfg
        self._wandb.config.update({"env_cfg": serialized_env_cfg})

    def save_model(self, model_path: str, it: int) -> None:
        del it
        self._wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        self._wandb.save(path, base_path=os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        if video.name in self.logged_videos:
            return
        self._wandb.log(
            {"video": self._wandb.Video(str(video), format="mp4")},
            step=it,
        )
        self.logged_videos.add(video.name)

    def stop(self) -> None:
        SummaryWriter.close(self)
        self._wandb.finish()


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
        self.run.define_metric("fixed_probe/*", step_metric="update")
        self.run.define_metric("tracking/*", step_metric="update")
        self.run.define_metric("optimization/*", step_metric="update")
        self.run.define_metric("replay/*", step_metric="update")
        self.run.define_metric("rollout/*", step_metric="update")

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
