"""Launch paired grid256 PPO with current deterministic tracker action inputs."""

import run_memory350_compressed_adaptive_ppo as launcher
from run_memory350_grid256_ppo import grid_command


def action_command(root, fusion, context):
    command = grid_command(root, fusion, context)
    index = command.index("intact_tracking.cli.memory350_compressed_policy_train")
    command[index] = "intact_tracking.cli.memory350_action_policy_train"
    return command


def main():
    launcher.scratch_command = action_command
    launcher.main()


if __name__ == "__main__":
    main()
