# CIFAR-10 GFD-KL Training

Use Python 3.10 or newer. For GPU training, install a CUDA-enabled PyTorch build compatible with your system, then install the dependencies:

```bash
cd cifar10_gfd_kl_vmf
python -m pip install -r requirements.txt
```

Prepare these local files before training. The program does not download data or weights:

- CIFAR-10: the data root must contain the complete `cifar-10-batches-py` directory.
- DINOv3: the timm-format `vit_base_patch16_dinov3.lvd1689m` weights in `model.safetensors`.

Start training:

```bash
python train.py \
  --data-root /path/to/cifar10 \
  --dino-weights /path/to/model.safetensors \
  --out runs/gfd_kl_vmf
```

Defaults are one CUDA GPU, seed 42, and 50,000 updates. Use `--device cuda:0`, `--seed 42`, and `--steps 50000` to set the device, seed, and final step. Method parameters are in `config.yaml`; use `--config /path/to/config.yaml` for another configuration. Use `--device cpu` for CPU execution. For a short training check, add `--steps 50`.

The default feature shapes use accelerated kernels on NVIDIA GPUs with compute capability 8.0 or newer and Triton 3.x. Other cases use the PyTorch implementation. The first invocation can include compilation.

Repeating the same command automatically restores `latest.pt` from the output directory. To choose a checkpoint explicitly:

```bash
python train.py \
  --data-root /path/to/cifar10 \
  --dino-weights /path/to/model.safetensors \
  --out runs/gfd_kl_vmf \
  --resume runs/gfd_kl_vmf/state_0005000.pt \
  --steps 50000
```

Keep the configuration, seed, DINO weights, and execution backend unchanged when resuming. `--steps` is the total final step, not the number of additional updates. Choose a new output directory if an existing nonempty directory has no recoverable checkpoint.

The output directory contains:

- `metrics.jsonl`: training metrics at the first step, every 20 steps, and the final step.
- `samples_*.png`: sample grids every 1,000 steps and at the final step.
- `state_*.pt`: model, EMA, optimizer, progress, and random states every 5,000 steps and at the final step.
- `latest.pt`: the latest complete checkpoint.
