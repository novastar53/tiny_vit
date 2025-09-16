import jax
import jax.numpy as jnp

import flax.nnx as nnx

class GLU(nnx.Module):
    def __init__(self, config, rngs):
        self.config = config
        self.c_fc = nnx.Linear(config.embed_dim, config.hidden_dim, rngs=rngs)
        self.gate = nnx.Linear(config.embed_dim, config.hidden_dim, rngs=rngs)
        self.c_proj = nnx.Linear(config.hidden_dim, config.embed_dim, rngs=rngs)
    

    def __call__(self, x):
        h = self.c_fc(x)
        g = self.gate(x)
        g = nnx.silu(g)
        h = g * h
        y = self.c_proj(h)
        return y