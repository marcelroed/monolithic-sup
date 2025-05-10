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

from pathlib import Path

import numpy as np
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
import wandb
from tqdm import tqdm

from dataclasses import dataclass


@dataclass
class Linear:
    weight: torch.Tensor
    bias: torch.Tensor

    def __call__(self, x):
        pass


def run_mymse(
    x_points: NDArray[np.float32],
    y_points: NDArray[np.float32],
    algorithm: str,
    session: InferenceSession,
    device: Device,
) -> Tensor:
    dtype = DType.float32

    # Create driver tensors from the input arrays, and move them to the
    # accelerator.
    x_tensor = Tensor.from_numpy(x_points).to(device)
    y_tensor = Tensor.from_numpy(y_points).to(device)

    mojo_kernels = Path(__file__).parent / "operations"

    # Configure our simple one-operation graph.
    with Graph(
        "mymse_graph",
        input_types=[
            TensorType(
                dtype,
                shape=x_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=y_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        x_value, y_value = graph.inputs
        # The matrix multiplication custom operation takes in two matrices and
        # produces a result, with the specific algorithm that is used chosen
        # via compile-time parameterization.
        res = ops.custom(
            name="mymse",
            values=[x_value, y_value],
            out_types=[
                TensorType(
                    dtype=x_value.tensor.dtype,
                    shape=[1, 1],
                    device=DeviceRef.from_device(device),
                ),
                TensorType(
                    dtype=x_value.tensor.dtype,
                    shape=[x_value.tensor.shape[0], y_value.tensor.shape[1]],
                    device=DeviceRef.from_device(device),
                ),
            ],
            parameters={"algorithm": algorithm},
        )
        graph.output(*res)
        # graph.output(res[1])

    # Compile the graph.
    print("Compiling...")
    model = session.load(graph)

    # Perform the calculation on the target device.
    print("Executing...")
    loss, grad = model.execute(x_tensor, y_tensor)

    # print(f"{loss=} {grad.shape=}")
    # # Copy values back to the CPU to be read.
    # assert isinstance(loss, Tensor)
    # assert isinstance(grad, Tensor)
    # return loss.to(CPU()), grad.to(CPU())

    loss = loss.to_numpy()
    grad = grad.to_numpy()

    return loss, grad


def numpy_mse(y_points, y_hat_points):
    loss = (y_points - y_hat_points) ** 2
    grad = -2 * (y_points - y_hat_points) / (y_points.shape[0] * y_points.shape[1])
    return loss, grad
    # return (x_points - y_points) ** 2


def main_old():
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

    # assert accelerator_count() > 0
    #     # Then, test the various versions of matrix multiplication operations.
    loss, grad = run_mymse(y, y_hat, "naive", session, device)
    print("Naive matrix multiplication:")
    print(loss.to_numpy())
    print(loss.shape)
    print(grad.to_numpy())
    print(grad.shape)
    print()


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    BATCH_SIZE = 16
    K = 64
    np_y = np.random.uniform(size=(BATCH_SIZE, K)).astype(np.float32)
    np_x = np.random.uniform(size=(BATCH_SIZE, K)).astype(np.float32)

    device = CPU() if accelerator_count() == 0 else Accelerator()
    print(f"{device=}")

    session = InferenceSession(devices=[device])

    mojo_kernels = Path(__file__).parent / "operations"

    dtype = DType.float32
    with Graph(
        "mymse_graph",
        input_types=[
            TensorType(
                dtype,
                shape=(BATCH_SIZE, K),
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=(BATCH_SIZE, K),
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        x_value, y_value = graph.inputs
        res = ops.custom(
            name="mymse",
            values=[x_value, y_value],
            out_types=[
                TensorType(
                    dtype=x_value.tensor.dtype,
                    shape=[1, 1],
                    device=DeviceRef.from_device(device),
                ),
                TensorType(
                    dtype=x_value.tensor.dtype,
                    shape=[x_value.tensor.shape[0], y_value.tensor.shape[1]],
                    device=DeviceRef.from_device(device),
                ),
            ],
            parameters={"algorithm": "naive"},
        )
        graph.output(*res)

    print("Compiling...")
    model = session.load(graph)

    step = 0
    LR = 0.1
    MU = 0.9
    wandb.init(project="mojo")
    pbar = tqdm()
    while True:
        np_loss = ((np_y - np_x) ** 2).mean()
        pbar.set_postfix({"loss": np_loss})
        wandb.log({"train_loss": np_loss}, step=step)

        mojo_x = Tensor.from_numpy(np_x).to(device)
        mojo_y = Tensor.from_numpy(np_y).to(device)
        _, mojo_grad = model.execute(mojo_x, mojo_y)

        np_grad = mojo_grad.to_numpy()
        if step == 0:
            np_moment = np_grad
        else:
            np_moment = MU * np_moment + (1 - MU) * np_grad

        np_x = np_x - LR * np_moment
        step += 1
        pbar.update(1)


if __name__ == "__main__":
    main()
