# imagenet1k.py
# ImageNet‑1k → JAX dataloader (GCS-backed file loader)
# - Reads images directly from a Google Cloud Storage bucket structure:
#     gs://<bucket>/<base_dir>/train/<class_name>/*.JPEG
#     gs://<bucket>/<base_dir>/validation/<class_name>/*.JPEG
# - Train: RandomResizedCrop + optional horizontal flip (tf.image ops)
# - Eval : Resize(shorter=eval_resize) -> CenterCrop(image_size)
# - Normalizes to ImageNet stats
# - Yields jax.numpy arrays (channels-first [B,C,H,W])
# - Optional per-device sharding for pmap
#
# Notes
# -----
# • Requires TensorFlow's GCS filesystem support (included in standard TF builds).
# • Class → id mapping is derived from the subfolder names under the TRAIN directory
#   (sorted lexicographically). You can override via `class_index_path` (a JSON with
#   {"n01440764": 0, ...}). Ensure TRAIN and VAL use the same class subfolder names.

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, Dict, List

import os
import json
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf

# ImageNet channel statistics (standard)
IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
IMAGENET_STD  = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)

@dataclass
class DataConfig:
    # I/O & batching
    batch_size: int = 256
    num_epochs: Optional[int] = None       # None = repeat forever
    shuffle: bool = True
    shuffle_buffer: int = 50_000           # large buffer for large dataset
    drop_last: bool = True
    seed: int = 0
    one_hot: bool = False
    shard_for_pmap: bool = False           # True → (n_devices, per_device_bs, ...)

    # GCS layout
    gcs_base: str = "gs://my-bucket/imagenet"  # base prefix
    train_dirname: str = "train"
    val_dirname: str = "validation"
    class_index_path: Optional[str] = None  # path to JSON mapping {class_name: id}
    image_extensions: Tuple[str, ...] = ("*.JPEG", "*.jpg", "*.jpeg", "*.png")

    # Aug / sizes
    image_size: int = 224                  # final train/eval crop size
    eval_resize: int = 256                 # shorter side resize before center crop (eval)
    augment: bool = True                   # enable train augments
    random_flip: bool = True               # horizontal flip on train
    # RandomResizedCrop controls (scale and aspect ratio ranges)
    rrc_scale_min: float = 0.08
    rrc_scale_max: float = 1.0
    rrc_ratio_min: float = 3/4
    rrc_ratio_max: float = 4/3


# -----------------------
# Utilities: class mapping
# -----------------------

def _list_class_names(gcs_split_dir: str) -> List[str]:
    # List immediate subdirectories in the split dir (class names)
    gfile = tf.io.gfile
    entries = gfile.listdir(gcs_split_dir)
    classes = []
    for name in entries:
        if name in (".", ".."):
            continue
        full = gcs_split_dir.rstrip("/") + "/" + name
        try:
            if gfile.isdir(full):
                classes.append(name)
        except tf.errors.NotFoundError:
            # Skip dangling entries
            pass
    classes.sort()
    return classes


def _load_or_build_class_index(cfg: DataConfig, split: str = "train") -> Dict[bytes, int]:
    # Load mapping if provided, else derive from TRAIN split directory structure
    if cfg.class_index_path:
        with tf.io.gfile.GFile(cfg.class_index_path, "r") as f:
            mapping = json.load(f)  # {str: int}
        # Ensure bytes keys for TF lookup table
        return {k.encode("utf-8"): int(v) for k, v in mapping.items()}

    train_dir = cfg.gcs_base.rstrip("/") + "/" + cfg.train_dirname
    class_names = _list_class_names(train_dir)
    mapping = {name.encode("utf-8"): i for i, name in enumerate(class_names)}
    return mapping


# -----------------------
# Image transforms
# -----------------------

def _normalize_imagenet(x_hwcn: tf.Tensor) -> tf.Tensor:
    mean = tf.constant(np.array(IMAGENET_MEAN), dtype=tf.float32)
    std  = tf.constant(np.array(IMAGENET_STD), dtype=tf.float32)
    return (x_hwcn - mean) / std


def _center_crop(resized: tf.Tensor, out_h: int, out_w: int) -> tf.Tensor:
    h = tf.shape(resized)[0]
    w = tf.shape(resized)[1]
    off_h = tf.maximum(0, (h - out_h) // 2)
    off_w = tf.maximum(0, (w - out_w) // 2)
    return tf.image.crop_to_bounding_box(resized, off_h, off_w, out_h, out_w)


def _random_resized_crop(x: tf.Tensor, cfg: DataConfig) -> tf.Tensor:
    H = tf.shape(x)[0]
    W = tf.shape(x)[1]
    area = tf.cast(H * W, tf.float32)
    scale = tf.random.uniform([], cfg.rrc_scale_min, cfg.rrc_scale_max)
    log_ratio_min = tf.math.log(tf.constant(cfg.rrc_ratio_min, tf.float32))
    log_ratio_max = tf.math.log(tf.constant(cfg.rrc_ratio_max, tf.float32))
    aspect = tf.exp(tf.random.uniform([], log_ratio_min, log_ratio_max))
    target_area = scale * area
    crop_h = tf.cast(tf.round(tf.sqrt(target_area / aspect)), tf.int32)
    crop_w = tf.cast(tf.round(tf.sqrt(target_area * aspect)), tf.int32)

    def _fallback():
        shorter = tf.minimum(H, W)
        scale = tf.cast(cfg.image_size, tf.float32) / tf.cast(shorter, tf.float32)
        new_h = tf.cast(tf.round(tf.cast(H, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(W, tf.float32) * scale), tf.int32)
        y = tf.image.resize(x, [new_h, new_w], method='bilinear')
        return _center_crop(y, cfg.image_size, cfg.image_size)

    def _do_crop():
        max_off_h = tf.maximum(0, H - crop_h)
        max_off_w = tf.maximum(0, W - crop_w)
        off_h = tf.random.uniform([], 0, max_off_h + 1, dtype=tf.int32)
        off_w = tf.random.uniform([], 0, max_off_w + 1, dtype=tf.int32)
        y = tf.image.crop_to_bounding_box(x, off_h, off_w, crop_h, crop_w)
        return tf.image.resize(y, [cfg.image_size, cfg.image_size], method='bilinear')

    valid = (crop_h > 0) & (crop_w > 0) & (crop_h <= H) & (crop_w <= W)
    return tf.cond(valid, _do_crop, _fallback)


# -----------------------
# Parsing files from GCS
# -----------------------

def _build_lookup_table(class_to_id: Dict[bytes, int]) -> tf.lookup.StaticHashTable:
    keys = tf.constant(list(class_to_id.keys()), dtype=tf.string)
    values = tf.constant([class_to_id[k] for k in class_to_id.keys()], dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), default_value=-1
    )
    return table


def _extract_class_from_path(path: tf.Tensor) -> tf.Tensor:
    # path like gs://bucket/base/train/<class>/<file>.JPEG
    parts = tf.strings.split(path, '/')
    # class dir is the penultimate component
    return parts[-2]


def _decode_and_transform(path: tf.Tensor, table: tf.lookup.StaticHashTable, cfg: DataConfig, is_train: bool):
    # read bytes
    img_bytes = tf.io.read_file(path)
    # decode (JPEG/PNG)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]

    if is_train and cfg.augment:
        img = _random_resized_crop(img, cfg)
        if cfg.random_flip:
            img = tf.image.random_flip_left_right(img)
    else:
        # eval resize -> center crop
        H = tf.shape(img)[0]; W = tf.shape(img)[1]
        shorter = tf.minimum(H, W)
        scale = tf.cast(cfg.eval_resize, tf.float32) / tf.cast(shorter, tf.float32)
        new_h = tf.cast(tf.round(tf.cast(H, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(W, tf.float32) * scale), tf.int32)
        img = tf.image.resize(img, [new_h, new_w], method='bilinear')
        img = _center_crop(img, cfg.image_size, cfg.image_size)

    img = _normalize_imagenet(img)

    # label from folder name
    cls_str = _extract_class_from_path(path)
    label = table.lookup(cls_str)

    if cfg.one_hot:
        label = tf.one_hot(label, depth=1000, dtype=tf.float32)

    # CHW for JAX
    img = tf.transpose(img, [2, 0, 1])
    return img, label


def _file_dataset_for_split(split: str, cfg: DataConfig) -> tf.data.Dataset:
    base = cfg.gcs_base.rstrip('/')
    split_dir = base + '/' + (cfg.train_dirname if split.startswith('train') else cfg.val_dirname)

    # Build class mapping and lookup table
    class_to_id = _load_or_build_class_index(cfg, split='train')
    table = _build_lookup_table(class_to_id)

    # Build glob patterns for all extensions
    patterns = [split_dir + '/*/' + ext for ext in cfg.image_extensions]
    files = tf.data.Dataset.from_tensor_slices(patterns)
    files = files.interleave(
        lambda p: tf.data.Dataset.list_files(p, shuffle=cfg.shuffle if split.startswith('train') else False, seed=cfg.seed),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    if cfg.shuffle and split.startswith('train'):
        files = files.shuffle(cfg.shuffle_buffer, seed=cfg.seed, reshuffle_each_iteration=True)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_slack = True
    files = files.with_options(options)

    ds = files.map(lambda p: _decode_and_transform(p, table, cfg, is_train=split.startswith('train')),
                   num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(cfg.batch_size, drop_remainder=cfg.drop_last)
    ds = ds.repeat(cfg.num_epochs) if cfg.num_epochs is not None else ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _to_jax_batches(ds: tf.data.Dataset) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    for batch in ds.as_numpy_iterator():
        x_np, y_np = batch
        x = jnp.asarray(np.ascontiguousarray(x_np))
        y = jnp.asarray(np.ascontiguousarray(y_np))
        yield x, y


def _shard_batch_for_pmap(x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    - y: [B] int32 or [B,1000] float32 one-hot (same leading dims as x)
    Splits: 'train' or 'validation' based on GCS directory names.
    """
    assert split in {"train", "validation"} or split.startswith("train") or split.startswith("validation")
    ds = _file_dataset_for_split(split, cfg)

    for x, y in _to_jax_batches(ds):
        if cfg.shard_for_pmap:
            x, y = _shard_batch_for_pmap(x, y)
        yield x, y


# ---------------------------
# Demo / quick sanity check
# ---------------------------
if __name__ == "__main__":
    cfg = DataConfig(
        gcs_base="gs://YOUR_BUCKET/imagenet",
        batch_size=256,
        num_epochs=1,
        shuffle=True,
        drop_last=True,
        seed=42,
        one_hot=False,
        shard_for_pmap=False,
        image_size=224,
        eval_resize=256,
        augment=True,
    )

    print("Building GCS ImageNet-1k train dataloader… (ensure bucket paths are correct)")
    train_iter = make_dataloader("train", cfg)
    x, y = next(train_iter)
    print("✅ Got a batch.")
    print(f"x: shape={x.shape}, dtype={x.dtype}  (expect [B,3,{cfg.image_size},{cfg.image_size}])")
    print(f"y: shape={y.shape}, dtype={y.dtype}  (expect [B])")
    print(f"min/max x after norm: {x.min().item():.3f}, {x.max().item():.3f}")
