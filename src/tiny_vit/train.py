import jax
import jax.numpy as jnp
import flax.nnx as nnx

import optax

from model import Config, TinyViT
from dataloaders.cifar10 import DataConfig, make_dataloader

config = Config()
rngs = nnx.Rngs(default=0)

m = TinyViT(config, rngs)


def loss_fn(m, x, labels):
    logits = m(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, logits


tx = optax.adamw(1e-3, weight_decay=1e-4)
optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)


@nnx.jit
def step_fn(m, optimizer, x, y):
   (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m, x, y) 
   optimizer.update(m, grads)
   preds = jnp.argmax(logits, axis=-1)
   acc = (preds == y).mean()
   return loss, acc


cfg = DataConfig(
    batch_size=246,
    num_epochs=100,         # iterate once through the split
    shuffle=True,
    drop_last=True,
    seed=42,
    resize_to=72,
    augment=True,          # random rotation + zoom on train
    rotation_factor=0.02,
    zoom_height=0.2,
    zoom_width=0.2,
    one_hot=False,
    shard_for_pmap=False, # set True if using pmap
    data_dir=None,        # or point to a TFDS cache dir
)

print("Building train dataloader…")
m.train()

for e in range(100):
    train_iter = make_dataloader("train", cfg)
    for x, y in train_iter:
        logits = m(x)
        loss, acc = step_fn(m, optimizer, x, y)
    print(loss, acc)

