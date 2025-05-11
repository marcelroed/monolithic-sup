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
from sup.nn.flash_attention import flash_attention

fn my_flash_fwd[
    dtype: DType,
    q_layout: Layout,
    k_layout: Layout,
    v_layout: Layout,
    mask_layout: Layout,
    out_o_layout: Layout,
    out_l_layout: Layout,
    BM: Int,
    BN: Int,
](
    q: LayoutTensor[dtype, q_layout, MutableAnyOrigin],
    k: LayoutTensor[dtype, k_layout, MutableAnyOrigin],
    v: LayoutTensor[dtype, v_layout, MutableAnyOrigin],
    mask: LayoutTensor[dtype, mask_layout, MutableAnyOrigin],
    out_o: LayoutTensor[dtype, out_o_layout, MutableAnyOrigin],
    out_l: LayoutTensor[dtype, out_l_layout, MutableAnyOrigin],
):
    pass

@compiler.register("my_flash_fwd")
struct FlashAttentionFwd[algorithm: StaticString]:
    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        out_o: OutputTensor[rank=2],
        out_l: OutputTensor[rank=2],
        # out_grad: OutputTensor[rank=2],
        # y: InputTensor[type = out_grad.type, rank = out_grad.rank],
        # y_hat: InputTensor[type = out_grad.type, rank = out_grad.rank],
        q: InputTensor[type = out_o.type, rank = out_o.rank],
        k: InputTensor[type = out_o.type, rank = out_o.rank],
        v: InputTensor[type = out_o.type, rank = out_o.rank],
        mask: InputTensor[type = out_o.type, rank = out_o.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        @parameter
        assert_equal(target, "gpu")
        q_layout = q.to_layout_tensor()
        k_layout = k.to_layout_tensor()
        v_layout = v.to_layout_tensor()
        mask_layout = mask.to_layout_tensor()
        out_o_layout = out_o.to_layout_tensor()
        out_l_layout = out_l.to_layout_tensor()

        M = q_layout.shape[0]()
        N = q_layout.shape[1]()

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
        # gpu_ctx.enqueue_function[
        #     flash_attention[ input_k_fn, input_v_fn, mask_fn ]
        # ](
        # q, k.get_shape(), v.get_shape(), mask.get_shape(), output, scale
        # )