import jax
import jax.numpy as jnp
import flax.nnx as nnx

from model import TinyViT, Config


def main():
    rngs = nnx.Rngs(default=42)
    m = TinyViT(Config(), rngs)
    B, nC, D = 16, 3, 32
    x = jax.random.normal(
        jax.random.key(0),
        (B, nC, D, D)
    ) 
    logits = m(x)

if __name__ == "__main__":
    main()