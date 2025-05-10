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
    weight: NDArray[np.float32],
    learning_rate: float,
    session: InferenceSession,
    device: Device,
) -> Tensor:
    dtype = DType.float32

    # Create driver tensors from the input arrays, and move them to the
    # accelerator.
    dout_tensor = Tensor.from_numpy(x_points).to(device)
    x_activation_tensor = Tensor.from_numpy(y_points).to(device)
    weight_tensor = Tensor.from_numpy(weight).to(device)

    mojo_kernels = Path(__file__).parent / "operations"

    # Configure our simple one-operation graph.
    with Graph(
        "linear_bwd_graph",
        input_types=[
            TensorType(  # dout
                dtype,
                shape=dout_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(  # x_activation
                dtype,
                shape=x_activation_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(  # weight
                dtype,
                shape=(x_activation_tensor.shape[1], dout_tensor.shape[1]),
                device=DeviceRef.from_device(device),
            )
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        dout_tensor_value, x_activation_tensor_value, weight_tensor_value = graph.inputs
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
        lr_const = ops.constant(-learning_rate, weight_tensor.dtype, weight_tensor_value.device)

        dweight_tensor_value = ops.matmul(
            ops.transpose(x_activation_tensor_value, 0, 1),
            dout_tensor_value,
        )
        dx_tensor_value = ops.matmul(
            dout_tensor_value,
            weight_tensor_value,
        )
        weight_tensor_value = ops.add(
            weight_tensor_value, ops.mul(lr_const, dweight_tensor_value)
        )
        graph.output(weight_tensor_value, dx_tensor_value)
        # graph.output(res[1])

    # Compile the graph.
    print("Compiling...")
    model = session.load(graph)

    # Perform the calculation on the target device.
    print("Executing...")
    dweight, dx = model.execute(dout_tensor, x_activation_tensor, weight_tensor)

    print(f"{dweight=} {dx.shape=}")
    # Copy values back to the CPU to be read.
    assert isinstance(dweight, Tensor)
    assert isinstance(dx, Tensor)
    return dweight.to(CPU()), dx.to(CPU())

def numpy_mse(y_points, y_hat_points):
    loss = (y_points - y_hat_points) ** 2
    grad = - 2 * (y_points - y_hat_points) / (y_points.shape[0] * y_points.shape[1])
    return loss, grad
    # return (x_points - y_points) ** 2

if __name__ == "__main__":
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
    dx, dweight = run_mymse(y, y_hat, weight, 0.01, session, device)
    print("Naive matrix multiplication:")
    print(dx.to_numpy())
    print(dx.shape)
    print(dweight.to_numpy())
    print(dweight.shape)
    print()