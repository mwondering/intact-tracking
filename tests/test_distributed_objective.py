from __future__ import annotations

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from intact_tracking.distributed import DistributedContext
from intact_tracking.model import SIGReg, TrackingINTACT, TrackingINTACTConfig
from intact_tracking.objective import INTACTLossConfig, TrackingINTACTObjective
from intact_tracking.residual_policy import ResidualPPO


def _batch(rank: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(200 + rank)
    batch_size = 2
    horizon = 3
    return {
        "observation": torch.randn(batch_size, horizon + 1, 4, generator=generator),
        "goal_observation": torch.randn(batch_size, horizon, 4, generator=generator),
        "forward_action": torch.randn(batch_size, horizon, 4, generator=generator),
        "action": torch.randn(batch_size, horizon, 2, generator=generator),
        "previous_action": torch.randn(batch_size, horizon, 2, generator=generator),
        "context": torch.randn(batch_size, 16, 10, generator=generator),
        "context_mask": torch.ones(batch_size, 16, dtype=torch.bool),
    }


def _ddp_worker(rank: int, world_size: int, store_path: str, result_path: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(7)
        config = TrackingINTACTConfig(
            observation_dim=4,
            proprio_dim=3,
            action_dim=2,
            effect_steps=2,
            context_chunk_steps=2,
            context_tokens=16,
            embed_dim=8,
            encoder_hidden_dim=16,
            context_depth=1,
            context_heads=2,
            forward_history=2,
            forward_depth=1,
            forward_heads=2,
            forward_mlp_dim=16,
            actor_hidden_dim=16,
            actor_depth=1,
        )
        model = TrackingINTACT(config)
        objective = TrackingINTACTObjective(
            model,
            loss_config=INTACTLossConfig(),
            sigreg=SIGReg(num_proj=8),
        )
        ddp = DistributedDataParallel(objective, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses: list[float] = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = ddp(_batch(rank))
            output["loss"].backward()
            optimizer.step()
            losses.append(float(output["loss"].detach()))

        flattened = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        gathered = [torch.empty_like(flattened) for _ in range(world_size)]
        dist.all_gather(gathered, flattened)
        if rank == 0:
            torch.save(
                {
                    "maximum_parameter_difference": max(
                        float((candidate - gathered[0]).abs().max()) for candidate in gathered[1:]
                    ),
                    "losses": losses,
                },
                result_path,
            )
    finally:
        dist.destroy_process_group()


def test_objective_wrapper_synchronizes_complete_model_with_ddp(tmp_path: Path) -> None:
    world_size = 2
    store_path = str(tmp_path / "ddp_store")
    result_path = str(tmp_path / "ddp_result.pt")
    mp.spawn(
        _ddp_worker,
        args=(world_size, store_path, result_path),
        nprocs=world_size,
        join=True,
    )
    result = torch.load(result_path, map_location="cpu", weights_only=True)
    assert result["maximum_parameter_difference"] == 0.0
    assert len(result["losses"]) == 2
    assert torch.isfinite(torch.tensor(result["losses"])).all()


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _distributed_context_worker(
    rank: int,
    world_size: int,
    port: int,
    result_path: str,
) -> None:
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
        }
    )
    context = DistributedContext.initialize(requested_backend="gloo")
    try:
        result = {
            "sum": context.sum_integers({"value": rank + 1})["value"],
            "all_true": context.all_true(rank == 0),
            "broadcast": context.broadcast_object("rank-zero" if rank == 0 else None),
            "identity": (context.rank, context.local_rank, context.world_size),
            "device": str(context.device),
        }
        gathered = context.all_gather_object(result)
        if context.is_main:
            torch.save(gathered, result_path)
    finally:
        context.close()


def test_distributed_context_uses_torchrun_environment_and_collectives(
    tmp_path: Path,
) -> None:
    world_size = 2
    result_path = str(tmp_path / "context_result.pt")
    mp.spawn(
        _distributed_context_worker,
        args=(world_size, _available_port(), result_path),
        nprocs=world_size,
        join=True,
    )
    results = torch.load(result_path, map_location="cpu", weights_only=True)
    assert [result["identity"] for result in results] == [(0, 0, 2), (1, 1, 2)]
    assert all(result["sum"] == 3 for result in results)
    assert all(result["all_true"] is False for result in results)
    assert all(result["broadcast"] == "rank-zero" for result in results)
    assert all(result["device"] == "cpu" for result in results)


def _residual_parameter_broadcast_worker(
    rank: int,
    world_size: int,
    store_path: str,
    result_path: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        actor = torch.nn.Module()
        actor.register_parameter(
            "residual",
            torch.nn.Parameter(torch.full((3,), float(rank + 1))),
        )
        actor.register_parameter(
            "frozen_tracker",
            torch.nn.Parameter(torch.tensor([20.0 + rank]), requires_grad=False),
        )
        critic = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            critic.weight.fill_(10.0 + rank)

        algorithm = ResidualPPO.__new__(ResidualPPO)
        algorithm._raw_actor = actor
        algorithm._raw_critic = critic
        algorithm.broadcast_parameters()

        trainable = torch.cat((actor.residual.detach(), critic.weight.detach().reshape(-1)))
        gathered_trainable = [torch.empty_like(trainable) for _ in range(world_size)]
        gathered_frozen = [torch.empty_like(actor.frozen_tracker) for _ in range(world_size)]
        dist.all_gather(gathered_trainable, trainable)
        dist.all_gather(gathered_frozen, actor.frozen_tracker.detach())
        if rank == 0:
            torch.save(
                {
                    "maximum_trainable_difference": max(
                        float((candidate - gathered_trainable[0]).abs().max())
                        for candidate in gathered_trainable[1:]
                    ),
                    "frozen_values": [float(value.item()) for value in gathered_frozen],
                    "tensor_count": algorithm.last_parameter_broadcast_tensor_count,
                },
                result_path,
            )
    finally:
        dist.destroy_process_group()


def test_residual_parameter_sync_uses_tensor_collective_and_skips_frozen_tracker(
    tmp_path: Path,
) -> None:
    world_size = 2
    store_path = str(tmp_path / "residual_broadcast_store")
    result_path = str(tmp_path / "residual_broadcast_result.pt")
    mp.spawn(
        _residual_parameter_broadcast_worker,
        args=(world_size, store_path, result_path),
        nprocs=world_size,
        join=True,
    )
    result = torch.load(result_path, map_location="cpu", weights_only=True)
    assert result["maximum_trainable_difference"] == 0.0
    assert result["frozen_values"] == [20.0, 21.0]
    assert result["tensor_count"] == 2
