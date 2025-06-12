# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import math
import argparse
from pathlib import Path

import numpy as np
from dataset import download_dataset
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops, BufferType, BufferValue
from max import nn
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
import torch
import random
from tqdm import trange, tqdm
import wandb

from dataclasses import dataclass

def update_weight(w, dw, lr):
    return ops.sub(w, ops.mul(lr, dw))


def relu_fwd(x, constant_zero):
    y = ops.relu(x)
    grad_mask = ops.greater_equal(x, constant_zero)
    return y, grad_mask


def relu_bwd(dy, grad_mask, constant_zero):
    dx = ops.select(grad_mask, dy, constant_zero)
    return dx

from max.graph.value import TensorValue

def split_heads(x: TensorValue, n_heads: int) -> TensorValue:
    assert len(x.shape) == 3, f"Expected 3D tensor, got {x.shape}"
    assert int(x.shape[-1]) % n_heads == 0, f"Last dimension {x.shape[-1]} must be divisible by n_heads {n_heads}"

    batch_dim = x.shape[0]
    seq_len = x.shape[1]
    head_dim = x.shape[2] // n_heads
    # Reshape to (batch, n_heads, seq_len, head_dim)
    x = x.reshape((batch_dim, seq_len, n_heads, head_dim))
    x = ops.transpose(x, 1, 2)  # Move n_heads to second dimension
    x = x.reshape((batch_dim * n_heads, seq_len, head_dim))
    return x


def join_heads(x: TensorValue, n_heads: int) -> TensorValue:
    assert len(x.shape) == 3, f"Expected 3D tensor, got {x.shape}"
    assert int(x.shape[0]) % n_heads == 0, f"First dimension {x.shape[0]} must be divisible by n_heads {n_heads}"

    batch_size = x.shape[0] // n_heads
    seq_len = x.shape[1]
    head_dim = x.shape[2]
    # Reshape to (batch, n_heads, seq_len, head_dim)
    x = x.reshape((batch_size, n_heads, seq_len, head_dim))
    x = ops.transpose(x, 1, 2)  # Move seq_len to second dimension
    x = x.reshape((batch_size, seq_len, n_heads * head_dim))
    return x


@dataclass
class Attention:
    neginf: TensorValue
    constant_zero: TensorValue
    attn_mask: TensorValue
    n_heads: int
    head_dim: int
    p: TensorValue | None = None
    q: TensorValue | None = None
    k: TensorValue | None = None
    v: TensorValue | None = None

    def __call__(
        self,
        Q: TensorValue,
        K: TensorValue,
        V: TensorValue,
    ) -> TensorValue:
        # Broadcast the attention mask across heads.
        # Do so in the graph so that the broadcast can be fused into downstream
        # ops.
        batch, seq_len, d = Q.shape

        Q, K, V = (
            split_heads(x, self.n_heads) for x in (Q, K, V)
        )

        self.q = Q; self.k = K; self.v = V

        attn_mask = self.attn_mask.broadcast_to(
            (
                batch * self.n_heads,
                seq_len,
                seq_len,
            )
        )
        self.attn_mask = attn_mask

        scale = math.sqrt(1.0 / self.head_dim)
        scores = Q @ ops.transpose(K, -2, -1)
        # Note, the graph compiler currently requires the order of operands
        # to be `scores * scale` in order to pattern match the fused attention
        # operator.
        # scores = ops.softmax(scores * scale + attn_mask)
        scores = ops.select(self.attn_mask, scores, self.neginf)
        scores = ops.softmax(scores * scale)


        self.p = scores

        return join_heads(scores @ V, n_heads=self.n_heads)

    def backward(
        self,
        dO: TensorValue,
        lr: TensorValue,
    ):
        dO = split_heads(dO, self.n_heads)
        dV = ops.matmul(ops.transpose(self.p, -2, -1), dO)
        dP = ops.matmul(dO, ops.transpose(self.v, -2, -1))
        
        dS = ops.add(
            ops.mul(ops.broadcast_to(-ops.sum(ops.mul(dP, self.p), axis=-1), self.p.shape), self.p),
            ops.mul(dP, self.p)
        )

        dS = ops.select(self.attn_mask, dS, self.constant_zero)

        dQ = ops.matmul(dS, self.k) * (1.0 / math.sqrt(self.head_dim))
        dK = ops.matmul(ops.transpose(dS, -1, -2), self.q) * (1.0 / math.sqrt(self.head_dim))
        return tuple(map(lambda x: join_heads(x, n_heads=self.n_heads), (dQ, dK, dV)))



@dataclass
class Linear:
    weight: TensorValue

    # Saved for bwd
    x_activation: TensorValue | None = None

    def __call__(self, x: TensorValue) -> TensorValue:
        if len(x.shape) > 2:
            batch_dims = x.shape[:-1]
            x = x.reshape((-1, x.shape[-1]))
        else:
            batch_dims = None

        self.x_activation = x

        y = ops.matmul(x, ops.transpose(self.weight, -2, -1))

        if batch_dims is not None:
            y = y.reshape((*batch_dims, y.shape[-1]))
        return y

    
    def backward(self, dy: TensorValue, lr: TensorValue) -> tuple[TensorValue, TensorValue]:
        if len(dy.shape) > 2:
            batch_dims = dy.shape[:-1]
            dy = dy.reshape((-1, dy.shape[-1]))
        else:
            batch_dims = None
        
        dw = ops.matmul(
            ops.transpose(dy, -2, -1),
            self.x_activation,
        )
        dx = ops.matmul(
            dy,
            self.weight,
        )
        w_new = update_weight(self.weight, dw, lr)
        self.weight = w_new

        if batch_dims is not None:
            dx = dx.reshape((*batch_dims, dx.shape[-1]))

        return dx, w_new

@dataclass
class SelfAttention:
    Wq: Linear
    Wk: Linear
    Wv: Linear
    Wo: Linear
    attn: Attention
    def __init__(self, wq, wk, wv, wo, neginf, constant_zero, attn_mask, n_heads, head_dim):
        self.attn = Attention(
            neginf=neginf, constant_zero=constant_zero, 
            attn_mask=attn_mask, n_heads=n_heads, head_dim=head_dim
        )
        self.Wq = Linear(wq)
        self.Wk = Linear(wk)
        self.Wv = Linear(wv)
        self.Wo = Linear(wo)

    def __call__(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        o = self.attn(q, k, v)
        o = self.Wo(o)

        return o
    
    def backward(self, do, lr):
        do, Wo_new = self.Wo.backward(do, lr)
        dQ, dK, dV = self.attn.backward(do, lr)
        dx1, Wq_new = self.Wq.backward(dQ, lr)
        dx2, Wk_new = self.Wk.backward(dK, lr)
        dx3, Wv_new = self.Wv.backward(dV, lr)
        dx = ops.add(ops.add(dx1, dx2), dx3)
        return dx, Wq_new, Wk_new, Wv_new, Wo_new
    


def linear_fwd(x, w):
    y = ops.matmul(x, ops.transpose(w, -2, -1))
    return y


def linear_bwd(x, w, dy, lr):
    dw = ops.matmul(
        ops.transpose(dy, -2, -1),
        x,
    )
    dx = ops.matmul(
        dy,
        w,
    )
    w_new = update_weight(w, dw, lr)

    return dx, w_new


def attn_fwd(q, k, v, scale, attn_mask):
    attn_mask = attn_mask.broadcast_to()

def self_attn_fwd(x, wq, wk, wv, wo, scale, attn_mask):
    saved_for_backward = (x, wq, wk, wv, scale)
    q = linear_fwd(x, wq)
    k = linear_fwd(x, wk)
    v = linear_fwd(x, wv)
    o, attn_saved = attn_fwd(q, k, v, scale, attn_mask)
    final_o = linear_fwd(o, wo)
    return final_o, (saved_for_backward, attn_saved)


def cross_entropy_fwdbwd(logits, target):
    if len(logits.shape) != 2:
        batch_shape = logits.shape[:-1]
        logits = logits.reshape((-1, logits.shape[-1]))
    else:
        batch_shape = None
    assert len(logits.shape) == 2, f'Expected 2d logits, got {logits.shape}'
    assert len(target.shape) == 1, f'Expected 1d targets, got {target.shape}'
    loss_val, grad_val = ops.custom(
        name="cross_entropy",
        values=[logits, target],
        out_types=[
            # mean loss scalar
            TensorType(
                logits.tensor.dtype,
                shape=target.tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            # grad has same shape as logits
            TensorType(
                logits.tensor.dtype,
                shape=logits.tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        # parameters={"algorithm": algorithm},  # if you want to select variants
    )

    if batch_shape is not None:
        grad_val = grad_val.reshape((*batch_shape, grad_val.shape[-1]))

    return loss_val, grad_val



def train_loop(
    learning_rate: float,
    session: InferenceSession,
    device: Device,
) -> Tensor:
    dtype = DType.float32

    # Create driver tensors from the input arrays, and move them to the
    # accelerator.
    # dout_tensor = Tensor.from_numpy(x_points).to(device)
    # x_activation_tensor = Tensor.from_numpy(y_points).to(device)
    # weight_tensor = Tensor.from_numpy(weight).to(device)
    B, D = 1, 2048
    V = 64
    T = 256
    input_embedding_tensor = Tensor.from_numpy(np.random.normal(size=(B, T, D)).astype(np.float32)).to(device)
    weight_tensor = Tensor.from_numpy(np.random.normal(size=(V, D)).astype(np.float32) * 0.02).to(device)
    # target_tensor = Tensor.from_numpy(np.random.randint(0, D, size=(B,)).astype(np.int32)).to(device)
    target_tensor = Tensor.from_numpy(np.ones((B, T)).astype(np.int32)).to(device)

    qi = np.arange(T)[:, None]
    ki = np.arange(T)[None, :]

    Wq_tensor, Wk_tensor, Wv_tensor, Wo_tensor = (
        Tensor.from_numpy(np.random.normal(size=(D, D)).astype(np.float32) * 0.02).to(device)
        for _ in range(4)
    )

    attn_mask_tensor = Tensor.from_numpy(qi >= ki).to(device)

    mojo_kernels = Path(__file__).parent / "operations"

    # Configure our simple one-operation graph.
    with Graph(
        "linear_bwd_graph",
        input_types=[
            TensorType(  # input_embedding
                dtype,
                shape=input_embedding_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(  # weight
                dtype,
                shape=weight_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(  # W_q
                dtype,
                shape=Wq_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(  # W_k
                dtype,
                shape=Wk_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(  # W_v
                dtype,
                shape=Wv_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(  # W_o
                dtype,
                shape=Wo_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(  # target
                DType.int32,
                shape=target_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(  # attn_mask
                DType.bool,
                shape=attn_mask_tensor.shape,
                device=DeviceRef.from_device(device),
            )
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        # dout_tensor_value, x_activation_tensor_value, weight_tensor_value = graph.inputs
        input_embedding, weight, wq, wk, wv, wo, target, attn_mask = graph.inputs
        lr = ops.constant(learning_rate, weight_tensor.dtype, weight.tensor.device)
        scale = ops.constant(1.0 / np.sqrt(D), weight_tensor.dtype, weight.tensor.device)
        neginf = ops.constant(-1e9, weight_tensor.dtype, weight.tensor.device)

        # attn_mask = ops.select(attn_mask, 0, neginf)


        self_attn = SelfAttention(
            wq, wk, wv, wo, neginf=neginf, constant_zero=ops.constant(0.0, dtype, weight.tensor.device),
            attn_mask=attn_mask,
            n_heads=4, head_dim=D // 4,
        )
        a1 = self_attn(input_embedding)

        linear1 = Linear(weight)
        # logits = linear_fwd(input_embedding, weight)
        logits = linear1(a1)

        loss, dlogits = cross_entropy_fwdbwd(
            logits, target.reshape((B * T,))
        )
        # dlogits = dlogits.reshape((B, T, V))
        dlogits = ops.mul(dlogits, ops.constant(1.0 / B, weight_tensor.dtype, weight.tensor.device))

        # B, D = 64, 128
        # V = 32
        # T = 256

        # _dx, new_weight = linear_bwd(input_embedding, weight, dlogits, lr)
        dx, new_weight = linear1.backward(dlogits, lr)

        dx, *new_attention_weights = self_attn.backward(dx, lr)

        # _dx, new_weight = linear_bwd(input_embedding, weight, dlogits, lr)

        graph.output(loss, new_weight, *new_attention_weights)
        # The matrix multiplication custom operation takes in two matrices and
        # produces a result, with the specific algorithm that is used chosen
        # via compile-time parameterization.
        # res = ops.custom(
        #     name="line",
        #     values=[x_value, y_value],
        #     out_types=[
        #         TensorType(
        #             dtype=x_value.tensor.dtype,
        #             shape=[1, 1],
        #             device=DeviceRef.from_device(device),
        #         ),
        #         TensorType(
        #             dtype=x_value.tensor.dtype,
        #             shape=[x_value.tensor.shape[0], y_value.tensor.shape[1]],
        #             device=DeviceRef.from_device(device),
        #         ),
        #     ],
        #     parameters={"algorithm": algorithm},
        # )
        # lr_const = ops.constant(-learning_rate, weight_tensor.dtype, weight_tensor_value.device)
        # dweight_tensor_value = ops.matmul(
        #     ops.transpose(x_activation_tensor_value, 0, 1),
        #     dout_tensor_value,
        # )
        # dx_tensor_value = ops.matmul(
        #     dout_tensor_value,
        #     weight_tensor_value,
        # )
        # weight_tensor_value = ops.add(
        #     weight_tensor_value, ops.mul(lr_const, dweight_tensor_value)
        # )
        # graph.output(weight_tensor_value, dx_tensor_value)
        # graph.output(res[1])

    # Compile the graph.
    print("Compiling...")
    model = session.load(graph)

    # Perform the calculation on the target device.
    print("Executing...")
    pbar = tqdm()
    step = 0
    # wandb.init(project="mojo")
    attention_weights = (Wq_tensor, Wk_tensor, Wv_tensor, Wo_tensor)
    while True:
        loss, weight_tensor, *attention_weights = model.execute(
            input_embedding_tensor, weight_tensor, *attention_weights, target_tensor, attn_mask_tensor,
        )
        loss_to_log = loss.to(CPU()).to_numpy().mean()
        pbar.set_postfix({"loss": loss_to_log})
        # wandb.log({"loss": loss_to_log}, step=step)
        step += 1

    # print(f"{weight_tensor=} {dx.shape=}")
    # Copy values back to the CPU to be read.
    assert isinstance(loss, Tensor)
    assert isinstance(weight_tensor, Tensor)
    return loss.to(CPU()), weight_tensor.to(CPU())


def numpy_mse(y_points, y_hat_points):
    loss = (y_points - y_hat_points) ** 2
    grad = -2 * (y_points - y_hat_points) / (y_points.shape[0] * y_points.shape[1])
    return loss, grad
    # return (x_points - y_points) ** 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run train loop')
    parser.add_argument('--dir', type=str, default='fineweb10B', help='Local directory (default: fineweb10B)')
    args = parser.parse_args()
    download_dataset(args.dir)

    seed = 13
    random.seed(seed)
    np.random.seed(seed)
    BATCH_SIZE = 16
    K = 64

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    print(device)

    # Set up an inference session for running the graph.
    session = InferenceSession(devices=[device])

    # Fill the input matrices with random values.
    y = np.random.uniform(size=(BATCH_SIZE, K)).astype(np.float32)
    y_hat = np.random.uniform(size=(BATCH_SIZE, K)).astype(np.float32)

    print(f"{y.shape=} {y_hat.shape=}")

    print(numpy_mse(y, y_hat))
    print([x.shape for x in numpy_mse(y, y_hat)])

    # # First, perform the matrix multiplication in NumPy.

    # print("Expected result:")
    # print(a @ b)
    # print()
    weight = np.random.uniform(size=(K, K)).astype(np.float32)

    # assert accelerator_count() > 0
    #     # Then, test the various versions of matrix multiplication operations.
    loss, weight = train_loop(0.001, session, device)
    # print("Naive matrix multiplication:")
    # print(loss.to_numpy())
    # print(loss.shape)
    # print(weight.to_numpy())
    # print(weight.shape)
    # print()
