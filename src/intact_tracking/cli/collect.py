"""Command-line entry point for repository-owned MJLab rollout collection."""

from __future__ import annotations

import argparse

from intact_tracking.rollout import MjlabCollectorConfig, collect_mjlab_rollouts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--output-dir", required=True)
    motion = parser.add_mutually_exclusive_group(required=True)
    motion.add_argument("--motion-path")
    motion.add_argument("--motion-file")
    parser.add_argument("--task-id")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--transitions", type=int, default=100_000)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument("--include-disturbances", action="store_true")
    parser.add_argument("--world-session-steps", type=int, default=3_000)
    parser.add_argument("--world-id-offset", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = collect_mjlab_rollouts(MjlabCollectorConfig(**vars(args)))
    print(manifest)


if __name__ == "__main__":
    main()
