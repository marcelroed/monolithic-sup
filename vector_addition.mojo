from gpu.host import Dim
from gpu.id import block_idx, thread_idx, block_dim
from max.driver import Accelerator, Tensor, accelerator, cpu
from sys import exit, has_accelerator
from layout import LayoutTensor, Layout
from math import ceildiv

alias float_dtype = DType.float32
alias tensor_rank = 1
alias vector_size = 100

def main():
    if not has_accelerator():
        print("A GPU is required to run this program")
        exit()

    host_device = cpu()
    gpu_device = accelerator()

    # Allocate the two input tensors on the host.
    lhs_tensor = Tensor[float_dtype, tensor_rank](vector_size, host_device)
    rhs_tensor = Tensor[float_dtype, tensor_rank](vector_size, host_device)

    # Fill them with initial values.
    for i in range(vector_size):
        lhs_tensor[i] = Float32(i)
        rhs_tensor[i] = Float32(i * 0.5)

    print("lhs_tensor:", lhs_tensor)
    print("rhs_tensor:", rhs_tensor)

    # Move the input tensors to the accelerator.
    lhs_tensor = lhs_tensor.move_to(gpu_device)
    rhs_tensor = rhs_tensor.move_to(gpu_device)

    # Allocate the output tensor on the accelerator.
    out_tensor = Tensor[float_dtype, tensor_rank](vector_size, gpu_device)

    # Create a LayoutTensor for each tensor.
    lhs_layout_tensor = lhs_tensor.to_layout_tensor()
    rhs_layout_tensor = rhs_tensor.to_layout_tensor()
    out_layout_tensor = out_tensor.to_layout_tensor()

    # Compile the kernel function to run on the GPU.
    gpu_function = Accelerator.compile[
        vector_addition[
            lhs_layout_tensor.layout,
            rhs_layout_tensor.layout,
            out_layout_tensor.layout,
        ]
    ](gpu_device)

    # Calculate the number of thread blocks needed by dividing the vector size
    # by the block size and rounding up.
    num_blocks = ceildiv(vector_size, block_size)

    # Invoke the kernel function.
    gpu_function(
        gpu_device,
        lhs_layout_tensor,
        rhs_layout_tensor,
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