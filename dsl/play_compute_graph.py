import numpy as np
from max.nn.linear import Linear
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Graph, ops


matrix = np.random.rand(256, 128, dtype=np.float32)

with Graph("test_graph") as graph:
    tensor = ops.constant(matrix, dtype=DType.bfloat16, device=DeviceRef.GPU())

    linear_layer = Linear(
        in_dim=256,
        out_dim=128,
        dtype=DType.float32,
        device=DeviceRef.GPU(),
        name="linear",
        has_bias=False,
    )

    output = linear_layer(tensor)
    print(output)