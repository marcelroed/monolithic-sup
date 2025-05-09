from gpu import warp
from gpu.host import Dim
from gpu.id import block_idx, thread_idx, block_dim
from max.driver import Accelerator, accelerator, cpu, Tensor
from sys import exit, has_accelerator
from layout import LayoutTensor, Layout, print_layout
from math import ceildiv
import gpu



alias float_dtype = DType.bfloat16
alias tensor_rank = 2
alias vector_size = 100


fn flash_attention[
    q_layout: Layout,
    k_layout: Layout,
    v_layout: Layout,
    o_layout: Layout,
    Q_tile_size: Int = 32,
    K_tile_size: Int = 32,
    Dim: Int = 32,
](
    q: LayoutTensor[float_dtype, q_layout, MutableAnyOrigin],
    k: LayoutTensor[float_dtype, k_layout, MutableAnyOrigin],
    v: LayoutTensor[float_dtype, v_layout, MutableAnyOrigin],
    o: LayoutTensor[float_dtype, o_layout, MutableAnyOrigin],
):
    """The calculation to perform across the vector on the GPU."""
    q_tile_idx = block_idx.x
    batch_idx = block_idx.y
    
    q_tile = q.tile[1, Q_tile_size, Dim](batch_idx, q_tile_idx)[0]

    tid = block_dim.x * block_idx.x + thread_idx.x
    # if tid < size:
    #     var out_val = 0.0
    #     for i in range(a.dim(1)):
    #         out[tid] += a[tid, i] * weight[i]

fn cpu_reference[
    a_layout: Layout,
    weight_layout: Layout,
    out_layout: Layout,
](
    a: LayoutTensor[float_dtype, a_layout, MutableAnyOrigin],
    weight: LayoutTensor[float_dtype, weight_layout, MutableAnyOrigin],
    out: LayoutTensor[float_dtype, out_layout, MutableAnyOrigin],
):
    for row in range(a.dim[0]()):
        for col in range(a.dim[1]()):
            out[row] += a[row, col] * weight[row, col]

def main():
    if not has_accelerator():
        print("A GPU is required to run this program")
        exit()

    host_device = cpu()
    gpu_device = accelerator()

    alias batch_size = 1
    alias n_queries = 8
    alias n_keys = 8
    alias dim = 32

    # Allocate the two input tensors on the host.
    q_tensor = Tensor[float_dtype, 2]((batch_size, n_queries, dim), host_device)
    k_tensor = Tensor[float_dtype, 2]((batch_size, n_keys, dim), host_device)
    v_tensor = Tensor[float_dtype, 2]((batch_size, n_keys, dim), host_device)


    # Fill them with initial values.
    # for i in range(rows):
    #     for j in range(cols):
    #         a_tensor[i, j] = Float32(i + 45 * j)
    # for j in range(cols):
    #     weight_tensor[j] = Float32(j * 0.5)

    # Move the input tensors to the accelerator.
    q_tensor = q_tensor.move_to(gpu_device)
    k_tensor = k_tensor.move_to(gpu_device)
    v_tensor = v_tensor.move_to(gpu_device)

    # Allocate the output tensor on the accelerator.
    o_tensor = Tensor[float_dtype, 3]((batch_size, n_queries, dim), gpu_device)

    # Create a LayoutTensor for each tensor.
    q_layout_tensor = q_tensor.to_layout_tensor()
    k_layout_tensor = k_tensor.to_layout_tensor()
    v_layout_tensor = v_tensor.to_layout_tensor()
    o_layout_tensor = o_tensor.to_layout_tensor()


    # Compile the kernel function to run on the GPU.
    gpu_function = Accelerator.compile[
        flash_attention[
            q_layout_tensor.layout,
            k_layout_tensor.layout,
            v_layout_tensor.layout,
            o_layout_tensor.layout,
        ]
    ](gpu_device)

    # Calculate the number of thread blocks needed by dividing the vector size
    # by the block size and rounding up.
    num_blocks = ceildiv(vector_size, block_size)

    # Invoke the kernel function.
    gpu_function(
        gpu_device,
        q_layout_tensor,
        k_layout_tensor,
        v_layout_tensor,
        o_layout_tensor,
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