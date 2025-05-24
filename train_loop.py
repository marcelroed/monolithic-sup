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

@dataclass
class Attention:
    p: TensorValue | None = None
    q: TensorValue | None = None
    k: TensorValue | None = None
    v: TensorValue | None = None
    attn_mask: TensorValue | None = None
    head_dim: int = None



    def __call__(
        self,
        Q: TensorValue,
        K: TensorValue,
        V: TensorValue,
        attn_mask: TensorValue,
        n_heads: int,
        head_dim: int,
        neginf: TensorValue,
        constant_zero: TensorValue,
    ) -> TensorValue:
        self.q = Q; self.k = K; self.v = V
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.neginf = neginf
        self.constant_zero = constant_zero
        # Broadcast the attention mask across heads.
        # Do so in the graph so that the broadcast can be fused into downstream
        # ops.
        batch, _, seq_len, post_seq_len = attn_mask.shape
        attn_mask = attn_mask.broadcast_to(
            (
                batch,
                n_heads,
                seq_len,
                post_seq_len,
            )
        )
        self.attn_mask = attn_mask

        scale = math.sqrt(1.0 / head_dim)
        scores = Q @ ops.transpose(K, -2, -1)
        # Note, the graph compiler currently requires the order of operands
        # to be `scores * scale` in order to pattern match the fused attention
        # operator.
        # scores = ops.softmax(scores * scale + attn_mask)
        scores = ops.select(attn_mask, scores)
        scores = ops.softmax(scores * scale)

        self.p = scores

        return scores @ V

    def backward(
        self,
        dO: TensorValue,
        lr: TensorValue,
    ):
        dV = ops.matmul(ops.transpose(self.p, 2, 3), dO)
        dP = ops.matmul(dO, ops.transpose(self.v, -1, -2))
        
        dS = ops.add(
            ops.mul(ops.broadcast_to(-ops.sum(ops.mul(dP, self.p), axis=-1), self.p.shape), self.p),
            ops.mul(dP, self.p)
        )

        dS = ops.select(self.attn_mask, dS, self.constant_zero)

        dQ = ops.matmul(dS, self.k) * (1.0 / math.sqrt(self.head_dim))
        dK = ops.matmul(ops.transpose(dS, -1, -2), self.q) * (1.0 / math.sqrt(self.head_dim))
        return dQ, dK, dV



@dataclass
class Linear:
    weight: TensorValue

    # Saved for bwd
    x_activation: TensorValue | None = None

    def __call__(self, x: TensorValue) -> TensorValue:
        self.x_activation = x
        return ops.matmul(x, ops.transpose(self.weight))
    
    def backward(self, dy: TensorValue, lr: TensorValue) -> tuple[TensorValue, TensorValue]:
        dw = ops.matmul(
            ops.transpose(dy, 0, 1),
            self.x_activation,
        )
        dx = ops.matmul(
            dy,
            self.weight,
        )
        w_new = update_weight(self.weight, dw, lr)
        self.weight = w_new

        return dx

@dataclass
class SelfAttention:
    Wq: Linear
    Wk: Linear
    Wv: Linear
    Wo: Linear
    attn: Attention

    def __call__(self, x, attn_mask, n_heads, head_dim, neginf, constant_zero):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        o = self.attn(q, k, v, attn_mask, n_heads, head_dim, neginf, constant_zero,)
        o = self.Wo(o)

        return o
    
    def backward(self, do, lr):
        do = self.Wo.backward(do, lr)
        dq, dk, dv = self.attn.backward(do, lr)
        dx = ops.add(ops.add(self.Wq.backward(dq, lr), self.Wk.backward(dk, lr)), self.Wv.backward(dv, lr))
        return dx
    


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
    B, D = 64, 128
    V = 32
    T = 256
    input_embedding_tensor = Tensor.from_numpy(np.random.normal(size=(B, T, D)).astype(np.float32)).to(device)
    weight_tensor = Tensor.from_numpy(np.random.normal(size=(V, D)).astype(np.float32) * 0.02).to(device)
    # target_tensor = Tensor.from_numpy(np.random.randint(0, D, size=(B,)).astype(np.int32)).to(device)
    target_tensor = Tensor.from_numpy(np.ones((B, T)).astype(np.int32)).to(device)

    qi = np.arange(T)[:, None]
    ki = np.arange(T)[None, :]

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
            TensorType(  # target
                DType.int32,
                shape=(B, T),
                device=DeviceRef.from_device(device),
            ),
            TensorType(  # attn_mask
                DType.bool,
                shape=(T, T),
                device=DeviceRef.from_device(device),
            )
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        # dout_tensor_value, x_activation_tensor_value, weight_tensor_value = graph.inputs
        input_embedding, weight, target, attn_mask = graph.inputs
        lr = ops.constant(learning_rate, weight_tensor.dtype, weight.tensor.device)
        scale = ops.constant(1.0 / np.sqrt(D), weight_tensor.dtype, weight.tensor.device)
        neginf = ops.constant(-1e9, weight_tensor.dtype, weight.tensor.device)

        attn_mask = ops.select(attn_mask, 0, neginf)

        logits = linear_fwd(input_embedding, weight).reshape((B * T, V))

        loss, dlogits = cross_entropy_fwdbwd(
            logits, target.reshape((B * T,))
        )
        dlogits = dlogits.reshape((B, T, V))
        dlogits = ops.mul(dlogits, ops.constant(1.0 / B, weight_tensor.dtype, weight.tensor.device))

        # B, D = 64, 128
        # V = 32
        # T = 256
        
        _dx, new_weight = linear_bwd(
            input_embedding, weight, dlogits, lr
        )

        _dx, new_weight = linear_bwd(input_embedding, weight, dlogits, lr)

        graph.output(loss, new_weight)
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
    while True:
        loss, weight_tensor = model.execute(
            input_embedding_tensor, weight_tensor, target_tensor, attn_mask_tensor,
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
