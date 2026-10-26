# Training

Use a CUDA/PyTorch environment compatible with the target A800 cluster.

```bash
conda create -n pharma-llm-train python=3.10 -y
conda activate pharma-llm-train
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install gpustat tensorboard
pip install -e ".[dev]"
```

The Phase 4 training command will use:

```bash
torchrun --nproc_per_node=4 -m pharma_llm.training.sft \
  --model-config configs/model.yaml \
  --train-config configs/train_sft.yaml
```

