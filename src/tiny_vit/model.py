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

    n_layer: int = 4
    embed_dim: int = 8



class Block(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.ln1 = nnx.LayerNorm()
        self.ln2 = nnx.LayerNorm()
        self.attn = CausalSelfAttention(config, rngs)
        self.glu = GLU(config, rngs)
    

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.glu(self.ln2(x))


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


    def __call__(self, x):
        B, nC, D, D = x.shape
        x = x.reshape(B, nC, self.config.grid_size, self.config.patch_size, 
                      self.config.grid_size, self.config.patch_size)
        x = jnp.transpose(x, (0, 2, 4, 1, 3, 5))
        x = x.reshape(B, self.config.grid_size * self.config.grid_size, nC * self.config.patch_size * self.config.patch_size)
        x = self.patch_embed(x)
        x = x.mean(axis=1)
        y = self.head(x)
        return y