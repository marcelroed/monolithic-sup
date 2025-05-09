from gpu import warp
from gpu.host import Dim
from gpu.id import block_idx, thread_idx, block_dim
from max.driver import Accelerator, accelerator, cpu, Tensor
from sys import exit, has_accelerator
from layout import LayoutTensor, Layout, print_layout
from math import ceildiv
import gpu


alias float_dtype = DType.float32
alias tensor_rank = 2
alias vector_size = 100


fn weighted_sum[
    a_layout: Layout,
    weight_layout: Layout,
    out_layout: Layout,
    row_tile_size: Int = 64,
    col_tile_size: Int = 128,
](
    a: LayoutTensor[float_dtype, a_layout, MutableAnyOrigin],
    weight: LayoutTensor[float_dtype, weight_layout, MutableAnyOrigin],
    out: LayoutTensor[float_dtype, out_layout, MutableAnyOrigin],
):
    """The calculation to perform across the vector on the GPU."""

    alias size = out_layout.size()  # Force compile-time evaluation.
    tid = block_dim.x * block_idx.x + thread_idx.x
    if tid < size:
        var out_val = 0.0
        for i in range(a.dim(1)):
            out[tid] += a[tid, i] * weight[i]

fn cpu_reference[
    a_layout: Layout,
    weight_layout: Layout,
    out_layout: Layout,
](
    a: LayoutTensor[float_dtype, a_layout, MutableAnyOrigin],
    weight: LayoutTensor[float_dtype, weight_layout, MutableAnyOrigin],
    out: LayoutTensor[float_dtype, out_layout, MutableAnyOrigin],
):
    for row in range(a.dim(0)):
        for col in range(a.dim(1)):
            out[row] += a[row, col] * weight[row, col]

def main():
    if not has_accelerator():
        print("A GPU is required to run this program")
        exit()

    host_device = cpu()
    gpu_device = accelerator()

    alias rows = 2048
    alias cols = 2048

    # Allocate the two input tensors on the host.
    a_tensor = Tensor[float_dtype, 2]((rows, cols), host_device)
    weight_tensor = Tensor[float_dtype, 1]((cols,), host_device)

    # Fill them with initial values.
    for i in range(rows):
        for j in range(cols):
            a_tensor[i, j] = Float32(i + 45 * j)
    for j in range(cols):
        weight_tensor[j] = Float32(j * 0.5)

    print("a_tensor:", a_tensor)
    print("weight_tensor:", weight_tensor)

    # Move the input tensors to the accelerator.
    a_tensor = a_tensor.move_to(gpu_device)
    weight_tensor = weight_tensor.move_to(gpu_device)

    # Allocate the output tensor on the accelerator.
    out_tensor = Tensor[float_dtype, 1](rows, gpu_device)

    # Create a LayoutTensor for each tensor.
    a_layout_tensor = a_tensor.to_layout_tensor()
    weight_layout_tensor = weight_tensor.to_layout_tensor()
    out_layout_tensor = out_tensor.to_layout_tensor()

    print_layout(weight_layout_tensor.layout)

    # Compile the kernel function to run on the GPU.
    gpu_function = Accelerator.compile[
        weighted_sum[
            a_layout_tensor.layout,
            weight_layout_tensor.layout,
            out_layout_tensor.layout,
        ]
    ](gpu_device)

    # Calculate the number of thread blocks needed by dividing the vector size
    # by the block size and rounding up.
    num_blocks = ceildiv(vector_size, block_size)

    # Invoke the kernel function.
    gpu_function(
        gpu_device,
        a_layout_tensor,
        weight_layout_tensor,
        out_layout_tensor,
        grid_dim=Dim(num_blocks),
        block_dim=Dim(block_size),
    )

    # Move the output tensor back onto the CPU so that we can read the results.
    out_tensor = out_tensor.move_to(host_device)

    print("out_tensor:", out_tensor)


alias block_size = 32

fn vector_addition[
    lhs_layout: Layout,
    rhs_layout: Layout,
    out_layout: Layout,
](
    lhs: LayoutTensor[float_dtype, lhs_layout, MutableAnyOrigin],
    rhs: LayoutTensor[float_dtype, rhs_layout, MutableAnyOrigin],
    out: LayoutTensor[float_dtype, out_layout, MutableAnyOrigin],
):
    """The calculation to perform across the vector on the GPU."""

    alias size = out_layout.size()  # Force compile-time evaluation.
    tid = block_dim.x * block_idx.x + thread_idx.x
    if tid < size:
        out[tid] = lhs[tid] + rhs[tid]