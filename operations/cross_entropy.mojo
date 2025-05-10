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
    loss_layout: Layout,      # scalar  []
    grad_layout: Layout,      # grad    [B, D]
    max_warps_per_block: Int = 8,
](
    logits: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    labels: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    out_loss: LayoutTensor[dtype, loss_layout, MutableAnyOrigin],
    out_grad: LayoutTensor[dtype, grad_layout, MutableAnyOrigin],
):

    # Shapes
    var B = logits.dim[0]()
    var D = logits.dim[1]()
    var inv_B = (1.0 / B).cast[dtype]()

    # Row / column this thread is responsible for
    var row = block_dim.x * block_idx.x + thread_idx.x
    var col = block_dim.y * block_idx.y + thread_idx.y
    if row >= B or col >= D:
        return

    # ---------------------------------------------------------------------
    # 1) Row‑wise max for numerical stability
    # ---------------------------------------------------------------------
    # Gather max over the row with a warp‑wide reduction
    var local_max = logits[row, col]
    for j in range(col, D, block_dim.y):
        if logits[row, j] > local_max:
            local_max = logits[row, j]

    var row_max = warp.max(local_max)

    barrier()  # synchronise before re‑using registers

    # ---------------------------------------------------------------------
    # 2) Row‑wise exponentials and soft‑max denominator
    # ---------------------------------------------------------------------
    var local_sum = exp(logits[row, col] - row_max)
    for j in range(col+1, D, block_dim.y):
        local_sum += exp(logits[row, j] - row_max)
    var row_sum = warp.sum(local_sum)     # warp‑reduce intrinsic :contentReference[oaicite:3]{index=3}
    var inv_row_sum = 1.0 / row_sum

    barrier()

    # ---------------------------------------------------------------------
    # 3) Compute probabilities, gradient and loss term
    # ---------------------------------------------------------------------
    var prob = exp(logits[row, col] - row_max) * inv_row_sum
    var grad_val = prob - labels[row, col]             # ∂L/∂logits = p − y
    out_grad[row, col] = grad_val * inv_B              # mean over batch

    var loss_elem = -labels[row, col] * log(prob)
    # Warp reduction to obtain per‑row loss contribution
    var row_loss = warp.sum(loss_elem) * inv_B

    # ---------------------------------------------------------------------
    # 4) Block‑level reduction to scalar and atomic add
    # ---------------------------------------------------------------------
    #var block_loss = block_reduce[dtype, max_warps_per_block](row_loss)
    #if thread_idx.x == 0 and thread_idx.y == 0:
    #    atomic_add(out_loss.ptr, block_loss)       # global atomic add


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
        out_loss: OutputTensor[rank=2],
        out_grad: OutputTensor[type=out_loss.type, rank=2],
        logits: InputTensor[type=out_loss.type, rank=2],
        labels: InputTensor[rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        @parameter
        if target == "gpu":
            logits_layout = logits.to_layout_tensor()
            labels_layout = labels.to_layout_tensor()
            out_grad_layout = out_grad.to_layout_tensor()
            out_loss_layout = out_loss.to_layout_tensor()


            gpu_ctx = ctx.get_device_context()
            B = logits_layout.dim[0]()
            D = logits_layout.dim[1]()

            # Zero out the memory in the outbound tensor.
            gpu_ctx.enqueue_memset(
                DeviceBuffer[out_grad.type](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[out_grad.type]]](out_grad_layout.ptr),
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
            alias blockY = 32

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
                    ceildiv(B, blockX),
                    ceildiv(D, blockY),
                ),
                block_dim=(blockX, blockY),
            )

