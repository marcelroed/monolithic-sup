import jax
import jax.numpy as jnp
from functools import partial
import math
from jax import lax


# @partial(jax.remat, policy=jax.checkpoint_policies.nothing_saveable)
def f(x):
    return jnp.sin(x) + jnp.cos(x)


def attn(q, k, v):
    S = (q / math.sqrt(k.shape[-1])) @ k.transpose((0, 2, 1))
    P = jax.nn.softmax(S, axis=-1)
    return P @ v
    

def attn_simple(x):
    return attn(x, x, x)


LR = 0.01

class MinimalModel:
    @staticmethod
    def construct():
        weights = {
            'linear1': jax.random.normal(jax.random.PRNGKey(0), (4, 8)),
            'linear2': jax.random.normal(jax.random.PRNGKey(1), (8, 8)),
        }
        return weights
    
    @staticmethod
    def forward(weights, x):
        x = jnp.dot(x, weights['linear1'])
        x = jax.nn.relu(x)
        x = jnp.dot(x, weights['linear2'])
        return x

    @staticmethod
    def loss(weights, x, y):
        # input = jnp.ones((4,))
        # output = jax.random.normal(jax.random.PRNGKey(0), (8,))
        prediction = MinimalModel.forward(weights, x)
        return jnp.mean((prediction - y) ** 2)
    
    @staticmethod
    def single_update(weights, x , y):
        loss, grad = jax.value_and_grad(MinimalModel.loss)(weights, x, y)
        weights = jax.tree.map(lambda w, g: w - LR * g, weights, grad)
        return weights


def mse_fun(x, y):
    return jnp.mean((x - y) ** 2)

def linear(w, x):
    return jnp.dot(w, x)

def cross_entropy_loss(y_pred, y_true):
    # y_pred: (batch_size, num_classes)
    # y_true: (batch_size,) indices of the true classes
    return -jnp.sum(y_true * jnp.log(y_pred + 1e-10)) / y_pred.shape[0]


def softmax(x, axis=-1):
    x_max = jnp.max(x, axis=axis, keepdims=True)
    unnormalized = jnp.exp(x - lax.stop_gradient(x_max))
    denom = jnp.sum(unnormalized, axis=axis, keepdims=True)
    # denom = checkpoint_name(denom, "denom")
    return unnormalized / denom


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _softmax(x, axis: int | tuple[int, ...] | None = -1, where=None, initial=-jnp.inf):
    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x_safe = x if where is None else jnp.where(where, x, initial)
    unnormalized = jnp.exp(x_safe - x_max)
    result = unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True, dtype=jnp.float32).astype(x.dtype)
    if where is not None:
        result = jnp.where(where, result, 0)
    return result

@_softmax.defjvp
def _softmax_jvp(axis, primals, tangents):
    (x, where, initial), (x_dot, _, _) = primals, tangents
    y = _softmax(x, axis, where, initial)
    return y, y * (x_dot - jnp.sum(y * x_dot, axis, where=where, keepdims=True, dtype=jnp.float32)).astype(x_dot.dtype)


mse_value_and_grad = jax.value_and_grad(mse_fun)

def show_function(fun, *args, save_as: str = None):
    jitted = jax.jit(fun)
    traced = jitted.trace(*args)
    lowered = traced.lower()
    print("Lowered HLO:")
    print(lowered.as_text())
    compiled = lowered.compile()
    print("Compiled HLO:")
    print(compiled.as_text())
    if save_as:
        for i, hlo_module in enumerate(compiled._executable.xla_executable.hlo_modules()):
            for j, computation in enumerate(hlo_module.computations()):
                computation.render_html(f'{save_as}_{i}_{j}')


if __name__ == "__main__":
    x = jax.random.normal(jax.random.PRNGKey(3), (4,))
    y = jax.random.normal(jax.random.PRNGKey(4), (8,))
    z = jax.random.normal(jax.random.PRNGKey(5), (8,))

    ## ATTN
    # inp = jnp.ones((1, 4, 8))
    # out, f_derivative = jax.vjp(attn_simple, inp)

    # f_jaxpr = jax.make_jaxpr(attn_simple)(inp)
    # f_derivative_jaxpr = jax.make_jaxpr(f_derivative)(inp)

    # print("JAXPR representation of f:")
    # print(f_jaxpr)
    # print(f_derivative_jaxpr)

    ## Actual model
    weights = MinimalModel.construct()

    loss_and_grad = jax.value_and_grad(MinimalModel.loss)

    # ONLY MSE
    show_function(mse_value_and_grad, y, z, save_as='mse_value_and_grad')

    # SINGLE LINEAR
    # out, linear_vjp = jax.vjp(linear, weights['linear1'], y)
    # show_function(linear_vjp, x, save_as='linear_vjp')

    out, linear_vjp = jax.vjp(linear, weights['linear1'], y)
    show_function(linear_vjp, x, save_as='linear_vjp')

    input_softmax = jax.random.normal(jax.random.PRNGKey(0), (10, 15))
    out, softmax_vjp = jax.vjp(_softmax, input_softmax)
    show_function(softmax_vjp, input_softmax, save_as='softmax_vjp')


    # model_jaxpr = jax.make_jaxpr(MinimalModel.single_update)(weights, x, y)
    # print(model_jaxpr)
    # jitted = jax.jit(MinimalModel.single_update)
    # traced = jitted.trace(weights, x, y)
    # lowered = traced.lower()
    # compiled = lowered.compile()
    # print(compiled.as_text())
    # breakpoint()