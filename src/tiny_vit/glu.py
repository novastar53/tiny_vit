import jax
import jax.numpy as jnp

import flax.nnx as nnx

class GLU(nnx.Module):
    def __init__(self, config, rngs):
        self.config = config
    

    def __call__(self, x):
        return x