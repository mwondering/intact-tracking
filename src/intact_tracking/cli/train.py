"""Train context-conditioned tracking INTACT from rollout shards."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from intact_tracking.data import RolloutWindowDataset, split_world_ids
from intact_tracking.model import SIGReg, TrackingINTACT, TrackingINTACTConfig
from intact_tracking.objective import INTACTLossConfig, intact_objective


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Stop each epoch after this many optimizer steps (intended for smoke tests).",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        help="Stop validation after this many batches (intended for smoke tests).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--effect-steps", type=int, default=5)
    parser.add_argument(
        "--query-transitions", "--horizon", dest="query_transitions", type=int, default=5
    )
    parser.add_argument("--context-chunk-steps", type=int, default=5)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--allow-padded-context", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--encoder-hidden-dim", type=int, default=512)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--forward-history", type=int, default=3)
    parser.add_argument("--forward-depth", type=int, default=6)
    parser.add_argument("--forward-heads", type=int, default=8)
    parser.add_argument("--actor-hidden-dim", type=int, default=1024)
    parser.add_argument("--actor-depth", type=int, default=3)
    parser.add_argument("--sigreg-projections", type=int, default=1024)
    parser.add_argument("--forward-weight", type=float, default=1.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument("--physical-weight", type=float, default=0.1)
    parser.add_argument("--goal-weight", type=float, default=0.05)
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _scalar_metrics(output: dict[str, torch.Tensor]) -> dict[str, float]:
    names = (
        "loss",
        "forward_loss",
        "sigreg_loss",
        "action_loss",
        "physical_nll",
        "goal_nll",
        "physical_mae",
        "goal_mae",
        "forward_nmse",
        "forward_target_variance",
        "weighted_forward_loss",
        "weighted_sigreg_loss",
        "weighted_physical_nll",
        "weighted_goal_nll",
        "physical_log_std",
        "goal_log_std",
        "latent_mean_abs",
        "latent_rms",
        "latent_std_mean",
        "latent_std_min",
        "latent_std_max",
        "latent_collapsed_fraction",
    )
    return {name: float(output[name].detach()) for name in names}


def _average(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    return {name: sum(item[name] for item in metrics) / len(metrics) for name in metrics[0]}


@torch.inference_mode()
def _evaluate(
    model: TrackingINTACT,
    loader: DataLoader,
    loss_config: INTACTLossConfig,
    sigreg: SIGReg,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    results = []
    for batch_index, batch in enumerate(loader):
        output = intact_objective(
            model, _to_device(batch, device), loss_config=loss_config, sigreg=sigreg
        )
        results.append(_scalar_metrics(output))
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return _average(results)


def run(args: argparse.Namespace) -> Path:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    for name in ("max_train_batches", "max_validation_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive when provided")
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    splits = split_world_ids(
        manifest["world_ids"],
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    dataset_kwargs = {
        "effect_steps": args.effect_steps,
        "query_transitions": args.query_transitions,
        "context_chunk_steps": args.context_chunk_steps,
        "sample_stride": args.sample_stride,
        "context_tokens": 16,
        "require_full_context": not args.allow_padded_context,
    }
    statistics_dataset = RolloutWindowDataset(
        manifest_path, world_ids=splits["train"], **dataset_kwargs
    )
    statistics = statistics_dataset.compute_normalization()
    train_dataset = RolloutWindowDataset(
        manifest_path,
        world_ids=splits["train"],
        normalization=statistics,
        **dataset_kwargs,
    )
    validation_dataset = RolloutWindowDataset(
        manifest_path,
        world_ids=splits["validation"],
        normalization=statistics,
        **dataset_kwargs,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, **loader_kwargs
    )

    dims = train_dataset.dimensions
    model_config = TrackingINTACTConfig(
        observation_dim=dims.observation,
        proprio_dim=dims.proprio,
        action_dim=dims.action,
        effect_steps=args.effect_steps,
        context_chunk_steps=args.context_chunk_steps,
        context_tokens=16,
        embed_dim=args.embed_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        context_depth=args.context_depth,
        context_heads=args.context_heads,
        forward_history=args.forward_history,
        forward_depth=args.forward_depth,
        forward_heads=args.forward_heads,
        forward_mlp_dim=4 * args.embed_dim,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_depth=args.actor_depth,
    )
    loss_config = INTACTLossConfig(
        forward_weight=args.forward_weight,
        sigreg_weight=args.sigreg_weight,
        physical_weight=args.physical_weight,
        goal_weight=args.goal_weight,
    )
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = TrackingINTACT(model_config).to(device)
    sigreg = SIGReg(num_proj=args.sigreg_projections).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "last.pt").exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir / 'last.pt'}")
    statistics.to_json(output_dir / "normalization.json")
    run_config = {
        "method": "INTACT",
        "architecture_version": model_config.architecture_version,
        "training_architecture": {
            "forward": "LeWM-style causal Forward Predictor",
            "forward_transition": "effect_steps raw controls: z[t] -> z[t+effect_steps]",
            "physical_intent": "attached z[t+effect_steps] - z[t]",
            "goal_intent": "stop-gradient z_ref[t+effect_steps] - z[t]",
            "intent_actor": "one shared four-slot Gaussian actor; one 29-D action",
            "policy_action_steps": 1,
            "context_tokens": 16,
            "context_injection": "shared latent FiLM; no fifth actor slot",
        },
        "arguments": vars(args),
        "model": asdict(model_config),
        "loss": asdict(loss_config),
        "world_splits": splits,
        "dataset_sizes": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_metrics = []
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            output = intact_objective(
                model,
                _to_device(batch, device),
                loss_config=loss_config,
                sigreg=sigreg,
            )
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            train_metrics.append(_scalar_metrics(output))
            if args.max_train_batches is not None and batch_index + 1 >= args.max_train_batches:
                break
        if not train_metrics:
            raise RuntimeError(
                "The training loader produced no batches; reduce batch-size or collect more data"
            )
        scheduler.step()
        validation_metrics = _evaluate(
            model,
            validation_loader,
            loss_config,
            sigreg,
            device,
            max_batches=args.max_validation_batches,
        )
        if not validation_metrics:
            raise RuntimeError("The validation loader produced no batches")
        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": _average(train_metrics),
            "validation": validation_metrics,
            "train_batches": len(train_metrics),
            "validation_batches": min(
                len(validation_loader),
                args.max_validation_batches or len(validation_loader),
            ),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True))
        checkpoint = {
            "architecture_version": model_config.architecture_version,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": asdict(model_config),
            "loss_config": asdict(loss_config),
            "world_splits": splits,
            "normalization": asdict(statistics),
        }
        torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint, output_dir / "last.pt")
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n"
        )
    return output_dir / "last.pt"


def main() -> None:
    checkpoint = run(build_parser().parse_args())
    print(checkpoint)


if __name__ == "__main__":
    main()
