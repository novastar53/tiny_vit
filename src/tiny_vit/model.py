from typing import Literal
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx


from attn import CausalSelfAttention
from glu import GLU

@dataclass
class Config:
    image_size: int = 32
    patch_size: int = 4
    grid_size: int = image_size // patch_size
    num_patches: int = grid_size * grid_size
    in_channels: int = 3
    n_classes: int = 10

    n_layer: int = 8
    embed_dim: int = 192
    hidden_dim: int = 192 * 4

    n_heads: int = 3
    sdpa_implementation: Literal["xla", "cudnn", "slow"] = (
        "xla"  # self-attention kernel implementation
    )



class Block(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.ln1 = nnx.LayerNorm(config.embed_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(config.embed_dim, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs)
        self.glu = GLU(config, rngs)
    

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.glu(self.ln2(x))
        return x


class TinyViT(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.patch_embed = nnx.Linear(
            config.patch_size * config.patch_size * config.in_channels,
            config.embed_dim,
            rngs=rngs)
        self.head = nnx.Linear(
            config.embed_dim,
            config.n_classes,
            rngs=rngs
        )
        self.pos = nnx.Param(jnp.zeros((1, config.grid_size * config.grid_size, config.embed_dim)))
        self.h = [ Block(config, rngs) for _ in range(config.n_layer) ]


    def __call__(self, x):
        B, nC, D, D = x.shape
        x = x.reshape(B, nC, self.config.grid_size, self.config.patch_size, 
                      self.config.grid_size, self.config.patch_size)
        x = jnp.transpose(x, (0, 2, 4, 1, 3, 5))
        x = x.reshape(B, self.config.grid_size * self.config.grid_size, nC * self.config.patch_size * self.config.patch_size)
        x = self.patch_embed(x) + self.pos.value
        for i in range(self.config.n_heads):
            x = self.h[i](x)
        x = x.mean(axis=1)
        y = self.head(x)
        return y