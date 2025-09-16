import jax
import jax.numpy as jnp

import flax.nnx as nnx

class CausalSelfAttention(nnx.Module):
    def __init__(self, config, rngs):
        self.config = config
        self.qkv = nnx.Linear(config.embed_dim, 3 * config.embed_dim, rngs=rngs)


    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.config.n_heads, self.config.embed_dim // self.config.n_heads)
        k = k.reshape(B, T, self.config.n_heads, self.config.embed_dim // self.config.n_heads)
        v = v.reshape(B, T, self.config.n_heads, self.config.embed_dim // self.config.n_heads)
        y = jax.nn.dot_product_attention(
                q,
                k,
                v,
                implementation=self.config.sdpa_implementation
        )
        y = y.reshape(B, T, C)
        return y