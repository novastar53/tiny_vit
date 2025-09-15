import jax
import jax.numpy as jnp
import flax.nnx as nnx


class TinyViT(nnx.Module):
    def __init__(self, config):
        self.config = config
    
    def __call__(self, x):
        return x