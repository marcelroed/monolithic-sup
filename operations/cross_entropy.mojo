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
from layout.math import outer_product_acc, max
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
    var local_max = -1.0e30
    for j in range(col, D, block_dim.y):
        local_max = max(local_max, logits[row, j])
    var row_max = warp.max(local_max)     # warp‑sum intrinsic :contentReference[oaicite:2]{index=2}

    barrier()  # synchronise before re‑using registers

    # ---------------------------------------------------------------------
    # 2) Row‑wise exponentials and soft‑max denominator
    # ---------------------------------------------------------------------
    var local_sum = 0.0
    for j in range(col, D, block_dim.y):
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
    var block_loss = block_reduce[dtype, max_warps_per_block](row_loss)
    #if thread_idx.x == 0 and thread_idx.y == 0:
    #    atomic_add(out_loss.ptr, block_loss)       # global atomic add


