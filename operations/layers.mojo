import compiler
from gpu import WARP_SIZE, barrier, block_dim, block_idx, thread_idx, warp
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
from memory import stack_allocation
from gpu.memory import AddressSpace
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    lane_id,
    warp_id,
    syncwarp,
    thread_idx,
)


@always_inline
fn block_reduce[
    type: DType, max_warps_per_block: Int
](val: Scalar[type]) -> Scalar[type]:
    var m2_shared = stack_allocation[
        max_warps_per_block, type, address_space = AddressSpace.SHARED
    ]()
    var m2_broadcast = stack_allocation[
        1, type, address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    for i in range(tid, max_warps_per_block, block_dim.x):
        m2_shared[i] = 0

    if tid == 0:
        m2_broadcast[0] = 0

    barrier()

    var warp_m2 = warp.sum(val)

    var warp_id = warp.broadcast(tid // WARP_SIZE)
    var lane_idx = lane_id()

    if lane_idx == 0:
        m2_shared[warp_id] = warp_m2
    barrier()

    if warp_id == 0 and lane_idx < max_warps_per_block:
        var block_m2 = warp.lane_group_sum[num_lanes=max_warps_per_block](
            m2_shared[lane_idx]
        )
        if lane_idx == 0:
            m2_broadcast[0] = block_m2
    barrier()
    return m2_broadcast[0]


fn mse_kernel_gpu[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    out_loss_layout: Layout,
    out_grad_layout: Layout,
](
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    out_loss: LayoutTensor[dtype, out_loss_layout, MutableAnyOrigin],
    out_grad: LayoutTensor[dtype, out_grad_layout, MutableAnyOrigin],
):

    var B = a.dim[0]()
    var D = a.dim[1]()

    var mean_factor = 1.0 / (B * D)


    # Calculate the column and row indices for each thread.
    var row = block_dim.x * block_idx.x + thread_idx.x
    var col = block_dim.y * block_idx.y + thread_idx.y

    # var dst_reg: out_grad.element_type = 0.0

    # Iterate over the K dimension to compute the dot product.
    if row < B and col < D:
        var out_grad_reg = a[row, col] - b[row, col]
        # Grad is scaled by 2 / (B * D)
        out_grad[row, col] = ((2.0 * mean_factor).cast[dtype]()) * out_grad_reg

        # # Square and then reduce the result to get the loss.
        # out_grad_reg = out_grad_reg * out_grad_reg

        # # Start reducing
        # out_grad


    # Write the final accumulated result to the output matrix.
    # c[row, col] = dst_reg


@compiler.register("mse_fwdbwd")
struct MSEFwdBwd:
    """
    The central custom operation that dispatches to multiple different
    matrix multiplication implementations, depending on target hardware and
    selected algorithm.
    """

    @staticmethod
    fn execute(
        out_loss: OutputTensor[rank=2],
        out_grad: OutputTensor[rank=2],
        a: InputTensor[type=out_loss.type, rank=2],
        b: InputTensor[type=out_loss.type, rank=2],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        a_layout = a.to_layout_tensor()
        b_layout = b.to_layout_tensor()
        out_loss_layout = out_loss.to_layout_tensor()
        out_grad_layout = out_grad.to_layout_tensor()

        B = a_layout.shape[0]()
        D = a_layout.shape[1]()

        gpu_ctx = ctx.get_device_context()

        # Zero out the memory in the outbound tensor.
        # gpu_ctx.enqueue_memset(
        #     DeviceBuffer[out_loss.type](
        #         gpu_ctx,
        #         rebind[UnsafePointer[Scalar[out.type]]](out_layout.ptr),
        #         M * N,
        #         owning=False,
        #     ),
        #     0,
        # )

        gpu_ctx.enqueue_function[
            mse_kernel_gpu[
                out_loss.type,
                a_layout.layout,
                b_layout.layout,
                out_loss_layout.layout,
                out_grad_layout.layout,

                # out_loss.type,
                # a_layout.layout,
                # b_layout.layout,
                # out_loss_layout.layout,
                # out_grad_layout.layout,
                # out_loss: DType,
                # a_layout: Layout,
                # b_layout: Layout,
                # out_loss_layout: Layout,
                # out_grad_layout: Layout,
            ]
        ](
            a_layout,
            b_layout,
            out_loss_layout,
            out_grad_layout,
            grid_dim=(ceildiv(B, 32), ceildiv(D, 32)),
            block_dim=(32, 32),
        )

        # alias BM = 32
        # alias BN = 32
        # gpu_ctx.enqueue_function[
        #     naive_matrix_multiplication[
        #         out.type,
        #         a_layout.layout,
        #         b_layout.layout,
        #         out_layout.layout,
        #         BM,
        #         BN,
        #     ]
        # ](
        #     a_layout,
        #     b_layout,
        #     out_layout,
        #     grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        #     block_dim=(BN, BM),
        # )