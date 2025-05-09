import jax
import jax.numpy as jnp
from functools import partial
import math


# @partial(jax.remat, policy=jax.checkpoint_policies.nothing_saveable)
def f(x):
    return jnp.sin(x) + jnp.cos(x)


def attn(q, k, v):
    S = (q / math.sqrt(k.shape[-1])) @ k.transpose((0, 2, 1))
    P = jax.nn.softmax(S, axis=-1)
    return P @ v
    

def attn_simple(x):
    return attn(x, x, x)


class MinimalModel:
    @staticmethod
    def construct():
        weights = {
            'linear': jax.random.normal(jax.random.PRNGKey(0), (4, 8)),
        }
        return weights
    
    @staticmethod
    def forward(weights, x):
        x = jnp.dot(x, weights['linear'])
        return x

    @staticmethod
    def loss(weights, x, y):
        # input = jnp.ones((4,))
        # output = jax.random.normal(jax.random.PRNGKey(0), (8,))
        prediction = MinimalModel.forward(weights, x)
        return jnp.mean((prediction - y) ** 2)


if __name__ == "__main__":
    # inp = jnp.ones((1, 4, 8))
    # out, f_derivative = jax.vjp(attn_simple, inp)

    # f_jaxpr = jax.make_jaxpr(attn_simple)(inp)
    # f_derivative_jaxpr = jax.make_jaxpr(f_derivative)(inp)

    # print("JAXPR representation of f:")
    # print(f_jaxpr)
    # print(f_derivative_jaxpr)

    weights = MinimalModel.construct()

    loss_and_grad = jax.value_and_grad(MinimalModel.loss)

    x = jax.random.normal(jax.random.PRNGKey(0), (4,))
    y = jax.random.normal(jax.random.PRNGKey(1), (8,))

    model_jaxpr = jax.make_jaxpr(loss_and_grad)(weights, x, y)
    print(model_jaxpr)