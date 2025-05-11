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
from math import ceildiv, exp, log, sqrt
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
fn adamw_step_kernel_gpu[
    dtype: DType,
    prev_m_layout:  Layout,        # adamW memory 1
    prev_v_layout:  Layout,        # adamW memory 2
    prev_weight_layout: Layout,    # weights
    next_m_layout: Layout,      # output memory 1
    next_v_layout: Layout,      # output memory 2
    next_weight_layout: Layout,  # updated weights
    d_weight_layout: Layout,        # gradients
    t_layout: Layout,
    lr: Scalar[dtype] = 0.001,
    beta1: Scalar[dtype] = 0.9,
    beta2: Scalar[dtype] = 0.999,
    eps: Scalar[dtype] = 1e-8,
    weight_decay: Scalar[dtype] = 0.01,
](
    prev_m: LayoutTensor[dtype, prev_m_layout, MutableAnyOrigin],
    prev_v: LayoutTensor[dtype, prev_v_layout, MutableAnyOrigin],
    prev_weight: LayoutTensor[dtype, prev_weight_layout, MutableAnyOrigin],
    next_m: LayoutTensor[dtype, next_m_layout, MutableAnyOrigin],
    next_v: LayoutTensor[dtype, next_v_layout, MutableAnyOrigin],
    next_weight: LayoutTensor[dtype, next_weight_layout, MutableAnyOrigin],
    d_weight: LayoutTensor[dtype, d_weight_layout, MutableAnyOrigin],
    t: LayoutTensor[DType.int32, t_layout, MutableAnyOrigin],
):
    # Shapes
    var N = prev_m.dim[0]()
    var idx = block_dim.x * block_idx.x + thread_idx.x
    if idx >= N:
        return

    # AdamW step
    var m_i = prev_m[idx]
    var v_i = prev_v[idx]
    var w_i = prev_weight[idx]
    var g_i = d_weight[idx]

    m_i = beta1 * m_i + (1 - beta1) * g_i
    v_i = beta2 * v_i + (1 - beta2) * g_i * g_i

    var alpha_i = lr * (sqrt(1 - (beta2**Int(t[0]))) / (1 - beta1**Int(t[0])))
    var udpate = alpha_i * m_i / (sqrt(v_i) + eps) + lr * weight_decay * w_i

    next_m[idx] = m_i
    next_v[idx] = v_i
    next_weight[idx] = w_i - udpate


@compiler.register("adamw")
struct AdamW:

    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        # The output tensor
        next_m: OutputTensor[rank=1],
        next_v: OutputTensor[type=next_m.type,rank=1],
        next_weight: OutputTensor[type=next_m.type,rank=1],
        # The input tensor
        prev_m: InputTensor[type=next_m.type,rank=1],
        prev_v: InputTensor[type=next_m.type,rank=1],
        prev_weight: InputTensor[type=next_m.type,rank=1],
        # The gradient tensor
        d_weight: InputTensor[type=next_m.type,rank=1],
        t: InputTensor[type=DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            prev_m_layout = prev_m.to_layout_tensor()
            prev_v_layout = prev_v.to_layout_tensor()
            prev_weight_layout = prev_weight.to_layout_tensor()
            next_m_layout = next_m.to_layout_tensor()
            next_v_layout = next_v.to_layout_tensor()
            next_weight_layout = next_weight.to_layout_tensor()
            d_weight_layout = d_weight.to_layout_tensor()
            t_layout = t.to_layout_tensor()

            alias blockX = 256
            N = prev_m_layout.dim[0]()
            gpu_ctx.enqueue_function[
                adamw_step_kernel_gpu[
                    prev_m.type,
                    prev_m_layout.layout,
                    prev_v_layout.layout,
                    prev_weight_layout.layout,
                    next_m_layout.layout,
                    next_v_layout.layout,
                    next_weight_layout.layout,
                    d_weight_layout.layout,
                    t_layout.layout,
                ]
            ](
                prev_m_layout,
                prev_v_layout,
                prev_weight_layout,
                next_m_layout,
                next_v_layout,
                next_weight_layout,
                d_weight_layout,
                t_layout,
                grid_dim=(ceildiv(N, blockX),1),
                block_dim=(blockX,1),
            )