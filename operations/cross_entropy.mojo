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
from math import ceildiv, exp, log
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


@always_inline
fn cross_entropy_kernel_gpu[
    dtype: DType,
    a_layout:  Layout,        # logits  [B, D]
    b_layout:  Layout,        # labels  [B, D]  (one‑hot rows)
    loss_layout: Layout,      # per instance  [B]
    grad_layout: Layout,      # grad    [B, D]
    max_warps_per_block: Int = 8,
](
    logits: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    labels: LayoutTensor[DType.int32, b_layout, MutableAnyOrigin],
    out_loss: LayoutTensor[dtype, loss_layout, MutableAnyOrigin],
    dlogits: LayoutTensor[dtype, grad_layout, MutableAnyOrigin],
):
    # Shapes
    var B = logits.dim[0]()
    var D = logits.dim[1]()
    var row = block_dim.y * block_idx.y + thread_idx.y
    if row >= B:
        return
    var lane = lane_id()

    var m_i = logits[row, lane]  # max of current tile
    var m_new = logits[row, lane]  # max of current tile
    var l_i = exp(m_i - m_i)  # running scaled sum

    var count = 0
    for col in range(lane+WARP_SIZE, D, WARP_SIZE):
        var x = logits[row, col]

        if x > m_i:
            m_new = x

        # update the running scaled sum  (log‑sum‑exp recurrence)
        var scaled_l = exp(m_i - m_new) * l_i
        var add_l    = exp(x - m_new)

        l_i = scaled_l + add_l
        m_i = m_new
        count += 1

    var lane_lse = log(l_i) + m_i
    var warp_max = warp.lane_group_max_and_broadcast[num_lanes=WARP_SIZE](lane_lse)
    var lane_l_i = exp(lane_lse - warp_max)
    var total_l_i = warp.lane_group_sum_and_broadcast[num_lanes=WARP_SIZE](lane_l_i)
    var lse = log(total_l_i) + warp_max

    out_loss[row] = lse - logits[row, Int(labels[row])]

    # Compute the gradient
    for col in range(lane, D, WARP_SIZE):
        var grad = exp(logits[row, col] - lse)
        if col == Int(labels[row]):
            grad -= 1
        dlogits[row, col] = grad

    #if lane == 0:
    #    print("row ", row, " ", lse_sum)



@compiler.register("cross_entropy")
struct CrossEntropy[algorithm: StaticString]:
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
        out_loss: OutputTensor[rank=1],
        dlogits: OutputTensor[type=out_loss.type, rank=2],
        logits: InputTensor[type=out_loss.type, rank=2],
        labels: InputTensor[type=DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        @parameter
        if target == "gpu":
            logits_layout = logits.to_layout_tensor()
            labels_layout = labels.to_layout_tensor()
            out_grad_layout = dlogits.to_layout_tensor()
            out_loss_layout = out_loss.to_layout_tensor()


            gpu_ctx = ctx.get_device_context()
            B = logits_layout.dim[0]()
            D = logits_layout.dim[1]()

            # Zero out the memory in the outbound tensor.
            gpu_ctx.enqueue_memset(
                DeviceBuffer[dlogits.type](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[dlogits.type]]](out_grad_layout.ptr),
                    B*D,
                    owning=False,
                ),
                0,
            )

            gpu_ctx.enqueue_memset(
                DeviceBuffer[out_loss.type](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[out_loss.type]]](out_loss_layout.ptr),
                    1,
                    owning=False,
                ),
                0,
            )

            # Launch the kernel
            alias blockX = 32
            alias blockY = 8
            alias rows_per_block = blockY

            gpu_ctx.enqueue_function[
                cross_entropy_kernel_gpu[
                    out_loss.type,
                    logits_layout.layout,
                    labels_layout.layout,
                    out_loss_layout.layout,
                    out_grad_layout.layout,
                ]
            ](
                logits_layout,
                labels_layout,
                out_loss_layout,
                out_grad_layout,
                grid_dim=(
                    1,
                    ceildiv(B, rows_per_block),
                ),
                block_dim=(blockX, blockY),
            )

