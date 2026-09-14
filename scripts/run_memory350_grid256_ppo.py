"""Launch both compressed PPO arms on the shared 256-profile load grid."""

import run_memory350_compressed_adaptive_ppo as launcher

_scratch_command = launcher.scratch_command


def grid_command(root, fusion, context):
    return _scratch_command(root, fusion, context) + ["--dr-sampling", "grid256_shared"]


def main():
    launcher.scratch_command = grid_command
    launcher.main()


if __name__ == "__main__":
    main()
