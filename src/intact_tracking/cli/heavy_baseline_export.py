"""Export a portable heavy baseline checkpoint to ONNX/JSON."""

import argparse
import json

from intact_tracking.heavy_baseline_export import export_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(export_policy(args.checkpoint, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
