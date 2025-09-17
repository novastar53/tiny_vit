# Tiny ViT

A compact, educational Vision Transformer (TinyViT) implemented with JAX and Flax's nnx API.

<img src="robot.jpg" style="width: 50%; height: auto;" />

Overview
--------
This small project demonstrates a minimal ViT-style image classifier written for JAX. It includes:

- A lightweight Transformer-based model (`TinyViT`) in `src/tiny_vit/model.py`.
- A self-attention implementation in `src/tiny_vit/attn.py` and a simple GLU feed-forward in `src/tiny_vit/glu.py`.
- Training and example scripts in `src/tiny_vit/train.py` and `src/tiny_vit/main.py`.
- Dataloaders for CIFAR-10 (and a placeholder for ImageNet) in `src/tiny_vit/dataloaders/`.

Quick start
-----------
Requirements: Python, JAX, Flax (nnx), Optax, TensorFlow, and TensorFlow Datasets. See `pyproject.toml`.

Run a quick model sanity check:

1. From the `tiny_vit/` directory, run the example in `src/tiny_vit/main.py`:

   python -m src.tiny_vit.main

2. To run the simple training loop (CPU/GPU dependent):

   python -m src.tiny_vit.train

Files of interest
-----------------
- `src/tiny_vit/model.py` — TinyViT model and block definitions.
- `src/tiny_vit/attn.py` — Multi-head causal self-attention wrapper using JAX dot-product attention.
- `src/tiny_vit/glu.py` — Gated linear unit feed-forward block used in residual blocks.
- `src/tiny_vit/train.py` — Minimal training loop using Optax and the CIFAR-10 dataloader.
- `src/tiny_vit/dataloaders/cifar10.py` — TFDS-based pipeline that yields JAX arrays.

Notes
-----
- The code is intentionally small and pedagogical — it's a good starting point for experimentation with ViT architectures in JAX.
- The CIFAR-10 dataloader will download data via TFDS the first time you run it.
- If you plan to run on multiple devices with pmap, set `DataConfig.shard_for_pmap=True` and scale `batch_size` accordingly.

License
-------
See the repository LICENSE (if present).
