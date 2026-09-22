"""Export a proprio122/history5 residual checkpoint as policy.onnx + policy.json."""

import argparse
import json

import torch

from intact_tracking.memory350_onnx_export import export_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(export_policy(args.checkpoint, args.output_dir, run_name=args.run_name)), flush=True)


if __name__ == "__main__":
    main()
