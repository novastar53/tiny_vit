import jax
import jax.numpy as jnp

import flax.nnx as nnx

class CausalSelfAttention(nnx.Module):
    def __init__(self, config, rngs):
        self.config = config
        self.qkv = nnx.Linear(config.embed_dim, 3 * config.embed_dim, rngs=rngs)


    def __call__(self, x):
        qkv = self.qkv(x)
        q, k, v = nnx.split(qkv, 3, axis=-1)
        q = q.reshape(-1, config.n_heads, config.embed_dim // config.n_heads)
        k = k.reshape(-1, config.n_heads, config.embed_dim // config.n_heads)
        v = v.reshape(-1, config.n_heads, config.embed_dim // config.n_heads)
        y = jax.nn.dot_product_attention(
                q,
                k,
                v,
                implementation=self.config.sdpa_implementation
        )
        return y