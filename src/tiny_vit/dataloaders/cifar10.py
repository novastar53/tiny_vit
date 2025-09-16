# cifar10_jax_input.py
# CIFAR-10 → JAX dataloader
# - No augmentation
# - No resize (keeps 32x32 RGB)
# - Normalizes to CIFAR-10 stats
# - Yields jax.numpy arrays
# - Optional per-device sharding for pmap

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

# CIFAR-10 channel statistics (train split)
CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465], dtype=jnp.float32)
CIFAR10_STD  = jnp.array([0.2023, 0.1994, 0.2010], dtype=jnp.float32)

@dataclass
class DataConfig:
    batch_size: int = 128
    num_epochs: Optional[int] = None    # None = repeat forever
    shuffle: bool = True
    shuffle_buffer: int = 50_000        # full CIFAR-10 train set
    drop_last: bool = True
    data_dir: Optional[str] = None      # tfds data dir (optional)
    seed: int = 0
    one_hot: bool = False
    shard_for_pmap: bool = False        # True → shape (n_devices, per_device_bs, ...)
    prefetch: int = 2                   # tf.data prefetch buffer

    # Augmentations (reference: Keras preprocessing snippet)
    resize_to: Optional[int] = 72          # None = no resize; 72 to match reference
    augment: bool = True                   # apply random aug on train split
    rotation_factor: float = 0.02
    zoom_height: float = 0.2
    zoom_width: float = 0.2

def _to_float_and_normalize(img_uint8: tf.Tensor) -> tf.Tensor:
    """uint8 [0,255] → float32 normalized by CIFAR-10 stats."""
    x = tf.cast(img_uint8, tf.float32) / 255.0  # to [0,1]
    mean = tf.constant(np.array(CIFAR10_MEAN), dtype=tf.float32)
    std  = tf.constant(np.array(CIFAR10_STD), dtype=tf.float32)
    return (x - mean) / std

def _build_keras_layers(cfg: DataConfig, is_train: bool):
    """
    Build Keras preprocessing/augmentation layers (channel-last) to mirror the reference:
      - Normalization (we already normalize by CIFAR stats in _to_float_and_normalize, so we omit it here)
      - Resizing(72,72)
      - RandomRotation(0.02)  [train only]
      - RandomZoom(0.2, 0.2)  [train only]
    Returns (resize_layer, aug_layer_or_None)
    """
    # Resizing for both train/test if configured
    resize_layer = None
    if cfg.resize_to is not None:
        resize_layer = tf.keras.layers.Resizing(cfg.resize_to, cfg.resize_to)

    aug_layer = None
    if is_train and cfg.augment:
        aug_layer = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(factor=cfg.rotation_factor),
            tf.keras.layers.RandomZoom(height_factor=cfg.zoom_height, width_factor=cfg.zoom_width),
        ])
    return resize_layer, aug_layer

def _prep_example(example: dict,
                  one_hot: bool = False,
                  resize_layer: Optional[tf.keras.layers.Layer] = None,
                  aug_layer: Optional[tf.keras.layers.Layer] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Map a TFDS example to (image, label) with optional resize/augment + normalization."""
    # example["image"]: [32,32,3] uint8, example["label"]: scalar int64
    x = tf.cast(example["image"], tf.float32) / 255.0  # [H,W,C] float32 in [0,1]

    # Resize (both train/test) to match the reference if configured
    if resize_layer is not None:
        x = resize_layer(x)

    # Random augmentation (train only)
    if aug_layer is not None:
        # Keras preprocessing layers behave according to 'training' flag
        x = aug_layer(x, training=True)

    # Now normalize to CIFAR stats (channel-last)
    mean = tf.constant(np.array(CIFAR10_MEAN), dtype=tf.float32)
    std  = tf.constant(np.array(CIFAR10_STD), dtype=tf.float32)
    x = (x - mean) / std

    # Label
    y = tf.cast(example["label"], tf.int32)
    if one_hot:
        y = tf.one_hot(y, depth=10, dtype=tf.float32)  # [10]

    # Convert to channels-first for many JAX ViT impls (optional).
    x = tf.transpose(x, [2, 0, 1])  # [C,H,W]
    return x, y

def _build_tfds_pipeline(split: str, cfg: DataConfig) -> tf.data.Dataset:
    ds = tfds.load(
        "cifar10",
        split=split,
        shuffle_files=cfg.shuffle if split == "train" else False,
        data_dir=cfg.data_dir,
        as_supervised=False,
        with_info=False,
    )

    is_train = (split == "train" or split.startswith("train"))
    resize_layer, aug_layer = _build_keras_layers(cfg, is_train=is_train)

    if cfg.shuffle and split == "train":
        ds = ds.shuffle(cfg.shuffle_buffer, seed=cfg.seed, reshuffle_each_iteration=True)

    ds = ds.map(lambda ex: _prep_example(ex, cfg.one_hot, resize_layer, aug_layer),
                num_parallel_calls=tf.data.AUTOTUNE)

    # set batch size; for sharding we’ll reshape later in Python/jax
    ds = ds.batch(cfg.batch_size, drop_remainder=cfg.drop_last)

    # repeat for multiple epochs (or forever if None)
    ds = ds.repeat(cfg.num_epochs) if cfg.num_epochs is not None else ds.repeat()

    ds = ds.prefetch(cfg.prefetch)
    return ds

def _to_jax_batches(ds: tf.data.Dataset) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield (images, labels) as jnp arrays, shapes [B,C,H,W] and [B] / [B,10]."""
    for batch in tfds.as_numpy(ds):
        x_np, y_np = batch  # numpy arrays
        # ensure contiguous arrays for faster device put
        x = jnp.asarray(np.ascontiguousarray(x_np))
        y = jnp.asarray(np.ascontiguousarray(y_np))
        yield x, y

def _shard_batch_for_pmap(x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reshape batch to (n_devices, per_device_bs, ...) for pmap."""
    n_dev = jax.local_device_count()
    b = x.shape[0]
    assert b % n_dev == 0, f"Batch size {b} must be divisible by local_device_count {n_dev}."
    per_dev = b // n_dev
    x = x.reshape(n_dev, per_dev, *x.shape[1:])
    y = y.reshape(n_dev, per_dev, *y.shape[1:])
    return x, y

def make_dataloader(split: str, cfg: DataConfig) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Create an iterator of (x, y) batches as jnp arrays.
    - x: [B,C,H,W] float32 normalized (or [n_dev, per_dev, C,H,W] if shard_for_pmap)
    - y: [B] int32 or [B,10] float32 one-hot (same leading dims as x)
    """
    assert split in {"train", "test", "validation"} or split.startswith("train") or split.startswith("test")
    ds = _build_tfds_pipeline(split, cfg)

    for x, y in _to_jax_batches(ds):
        if cfg.shard_for_pmap:
            x, y = _shard_batch_for_pmap(x, y)
        yield x, y

# ---------------------------
# Demo / quick sanity check
# ---------------------------
if __name__ == "__main__":
    # Example usage:
    cfg = DataConfig(
        batch_size=128,
        num_epochs=1,         # iterate once through the split
        shuffle=True,
        drop_last=True,
        seed=42,
        one_hot=False,
        shard_for_pmap=False, # set True if using pmap
        data_dir=None,        # or point to a TFDS cache dir
    )

    print("Building train dataloader…")
    train_iter = make_dataloader("train", cfg)

    x, y = next(train_iter)
    print("✅ Got a batch.")
    print(f"x: shape={x.shape}, dtype={x.dtype}  (expect [B,3,H,W] with H=W={cfg.resize_to or 32})")
    print(f"y: shape={y.shape}, dtype={y.dtype}  (expect [B])")
    print(f"min/max x after norm: {x.min().item():.3f}, {x.max().item():.3f}")

    # If you plan to pmap:
    # cfg.shard_for_pmap = True
    # cfg.batch_size = 128 * jax.local_device_count()
    # then the shapes become [n_dev, per_dev, 3, 32, 32]