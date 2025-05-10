import compiler
from gpu import WARP_SIZE, barrier, block_dim, block_idx, thread_idx
from gpu.host import DeviceBuffer, DeviceContext
from gpu.memory import async_copy_wait_all
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
)
from layout.math import outer_product_acc
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import TensorCore
from math import ceildiv
from memory import UnsafePointer
from runtime.asyncrt import DeviceContextPtr
from sys.info import simdwidthof
from tensor import InputTensor, ManagedTensorSlice, OutputTensor
from utils.index import Index
from testing import assert_equal

fn my_mse_kern[
    dtype: DType,
    y_layout: Layout,
    y_hat_layout: Layout,
    out_loss_layout: Layout,
    out_grad_layout: Layout,
    BM: Int,
    BN: Int,
](
    y: LayoutTensor[dtype, y_layout, MutableAnyOrigin],
    y_hat: LayoutTensor[dtype, y_hat_layout, MutableAnyOrigin],
    out_loss: LayoutTensor[dtype, out_loss_layout, MutableAnyOrigin],
    out_grad: LayoutTensor[dtype, out_grad_layout, MutableAnyOrigin],
):
    var B = y.dim[0]()
    var D = y.dim[1]()

    var mean_factor = 1.0 / (B * D)


    # Calculate the column and row indices for each thread.
    var row = block_dim.x * block_idx.x + thread_idx.x
    var col = block_dim.y * block_idx.y + thread_idx.y

    # var dst_reg: out_grad.element_type = 0.0

    # Iterate over the K dimension to compute the dot product.
    if row < B and col < D:
        var out_grad_reg = y[row, col] - y_hat[row, col]
        # Grad is scaled by 2 / (B * D)
        out_grad[row, col] = ((2.0 * mean_factor).cast[dtype]()) * out_grad_reg

    # var M = y.dim[0]()
    # var N = y_hat.dim[1]()
    # var K = y_hat.dim[0]()

    # # Calculate the column and row indices for each thread.
    # var row = block_dim.x * block_idx.x + thread_idx.x
    # var col = block_dim.y * block_idx.y + thread_idx.y

    # # Initialize a register to accumulate the result for this thread.
    # var dst_reg: out_loss.element_type = 0

    # # Iterate over the K dimension to compute the dot product.
    # if row < M and col < N:
    #     for k_index in range(K):
    #         # Multiply the elements and accumulate the result.
    #         dst_reg = dst_reg + y[row, k_index] * y_hat[k_index, col]

    # # Write the final accumulated result to the output matrix.
    # out_grad[row, col] = dst_reg


@compiler.register("mymse")
struct MyMse[algorithm: StaticString]:
    """
    The central custom operation that dispatches to multiple different
    matrix multiplication implementations, depending on target hardware and
    selected algorithm.
    """

    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        out_loss: OutputTensor[rank=2],
        out_grad: OutputTensor[rank=2],
        y: InputTensor[type = out_grad.type, rank = out_grad.rank],
        y_hat: InputTensor[type = out_grad.type, rank = out_grad.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        @parameter
        assert_equal(target, "gpu")
        y_layout = y.to_layout_tensor()
        y_hat_layout = y_hat.to_layout_tensor()
        out_loss_layout = out_loss.to_layout_tensor()
        out_grad_layout = out_grad.to_layout_tensor()

        M = y_layout.shape[0]()
        N = y_hat_layout.shape[1]()

        gpu_ctx = ctx.get_device_context()

        # Zero out the memory in the outbound tensor.
        # gpu_ctx.enqueue_memset(
        #     DeviceBuffer[out_loss.type](
        #         gpu_ctx,
        #         rebind[UnsafePointer[Scalar[out_loss.type]]](out_loss_layout.ptr),
        #         1,
        #         owning=False,
        #     ),
        #     0,
        # )

        # gpu_ctx.enqueue_memset(
        #     DeviceBuffer[out_grad.type](
        #         gpu_ctx,
        #         rebind[UnsafePointer[Scalar[out_grad.type]]](out_grad_layout.ptr),
        #         M * N,
        #         owning=False,
        #     ),
        #     0,
        # )
            # We support several compile-time variants for the matrix
            # multiplication calculation:
            # - "naive": A naive matrix multiplication using LayoutTensors.
            # - "coalescing": Matrix multiplication with memory coalescing
            #   optimizations.
            # - "tiled": Matrix multiplication using a tiling strategy.
            # - "tiled_register": Matrix multiplication using shared memory
            #   and register tiling .
            # - "block_tiled": Matrix multiplication using a 2D block tiling
            #   strategy.
            # - "block_tiled_vectorized": Matrix multiplication using a
            #   further-optimized 2D block tiling strategy.
            # - "tensor_core": Matrix multiplication using Tensor Cores.
            # In each case, the specific matrix multiplication function is
        # compiled and enqueued to run on the GPU.
        alias BM = 32
        alias BN = 32
        gpu_ctx.enqueue_function[
            my_mse_kern[
                out_loss.type,
                y_layout.layout,
                y_hat_layout.layout,
                out_loss_layout.layout,
                out_grad_layout.layout,
                BM,
                BN,
            ]
        ](
            y_layout,
            y_hat_layout,
            out_loss_layout,
            out_grad_layout,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BN, BM),
        )