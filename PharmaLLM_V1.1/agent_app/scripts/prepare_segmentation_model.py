#!/usr/bin/env python3
"""Prepare metadata and a symlink for the user's checkpoint; no torch/GPU inference."""

import argparse
from pathlib import Path
from pharma_agent.checkpoint_metadata import prepare_model

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=Path("../model/checkpoint_best.pth"))
parser.add_argument("--output", type=Path, default=Path("../model/femoralhead-t1"))
args = parser.parse_args()
model = prepare_model(args.checkpoint, args.output)
print(f"已准备 {args.output}；模型 {model['id']}；输入 T1；fold {model['fold']}。未加载网络或执行推理。")
