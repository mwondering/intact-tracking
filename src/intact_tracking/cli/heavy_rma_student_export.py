import argparse
import json
import torch
from intact_tracking.rma_student_export import export_policy


def main():
    p=argparse.ArgumentParser(description='Export self-contained RMA student ONNX + JSON')
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(1)
    print(json.dumps(export_policy(args.checkpoint,args.output)),flush=True)


if __name__=='__main__':main()
