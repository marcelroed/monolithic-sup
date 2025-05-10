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

from dataclasses import dataclass

@dataclass
class Linear:
    weight: torch.Tensor
    bias: torch.Tensor

    def __call__(self, x):
        pass



def mse(
    x: NDArray[np.float32],
    target: NDArray[np.float32],
    algorithm: str,
    session: InferenceSession,
    device: Device,
) -> Tensor:
    dtype = DType.float32

    # Create driver tensors from the input arrays, and move them to the
    # accelerator.
    x_tensor = Tensor.from_numpy(x).to(device)
    target_tensor = Tensor.from_numpy(target).to(device)

    mojo_kernels = Path(__file__).parent / "operations"

    # Configure our simple one-operation graph.
    with Graph(
        "mse_graph",
        input_types=[
            TensorType(
                dtype,
                shape=x_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=target_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        x_value, target_value = graph.inputs
        # The matrix multiplication custom operation takes in two matrices and
        # produces a result, with the specific algorithm that is used chosen
        # via compile-time parameterization.
        output = ops.custom(
            name="mse_fwdbwd",
            values=[x_value, target_value],
            out_types=[
                TensorType(
                    dtype=x_value.tensor.dtype,
                    shape=[1, 1],
                    device=DeviceRef.from_device(device),
                ),
                TensorType(
                    dtype=x_value.tensor.dtype,
                    shape=[x_value.tensor.shape[0], x_value.tensor.shape[1]],
                    device=DeviceRef.from_device(device),
                )
            ],
            # parameters={"algorithm": algorithm},
        )[1].tensor
        graph.output(output)

    # Compile the graph.
    print("Compiling...")
    model = session.load(graph)

    # Perform the calculation on the target device.
    print("Executing...")
    result = model.execute(x_tensor, target_tensor)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    print(result)
    print("SUCCEEDED!")
    return result.to(CPU())


if __name__ == "__main__":
    M = 256
    K = 256
    N = 256

    batch_size = 16
    d_model = 16

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    print(device)

    # Set up an inference session for running the graph.
    session = InferenceSession(devices=[device])

    # Generate points on a plane in 16d space
    x_points = np.random.uniform(size=(batch_size, d_model)).astype(np.float32)
    target_points = np.random.uniform(size=(batch_size, d_model)).astype(np.float32)
    
    # a = np.random.uniform(size=(M, K)).astype(np.float32)
    # b = np.random.uniform(size=(K, N)).astype(np.float32)

    # First, perform the matrix multiplication in NumPy.
    print(f'{x_points=}\n')

    print(f"{target_points=}\n")

    # print("Expected result:")
    # print(a @ b)
    # print()

    if accelerator_count() > 0:
        # Then, test the various versions of matrix multiplication operations.
        naive_result = mse(x_points, target_points, "naive", session, device)
        print("MSE RSEULT")
        print(naive_result.to_numpy())
        print()
        exit()
    else:
        print("ya dun goofed")