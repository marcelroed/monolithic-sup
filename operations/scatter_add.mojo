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
from os.atomic import Atomic
from gpu.semaphore import Semaphore
from sys import is_amd_gpu, is_gpu, is_nvidia_gpu, sizeof
from sys.intrinsics import llvm_intrinsic, readfirstlane

@always_inline
fn atomic_add[
    type: DType, width: Int
](src_resource: SIMD[DType.uint32, 4], gds_offset: Int32, val: SIMD[type, width]):
    alias bytes = sizeof[type]() * width
    var src_wave_addr_offset: Int32 = 0
    alias glc: Int32 = 0

    @parameter
    if is_nvidia_gpu():
        @parameter
        fn get_inst_name() -> StaticString:
            @parameter
            if bytes == 4:
                return "llvm.nvvm.atomic.load.add.f32"
            else:
                constrained[False, "TODO: Only fp32 supported"]()
                return ""

        @parameter
        fn get_datatype() -> StaticString:
            @parameter
            if type == DType.float32:
                return "f32"
            else:
                constrained[False, "TODO: Only fp32 supported"]()
                return ""

        llvm_intrinsic[
            get_inst_name(),
            NoneType,
            has_side_effect=True,
        ](src_resource, gds_offset, src_wave_addr_offset, glc)
    else:
        constrained[False, "TODO: Only supported on NVIDIA GPU"]()


@always_inline
fn scatter_add_kernel_gpu[
    dtype: DType,
    src_layout: Layout, # B, E, K
    index_layout: Layout, # B, E
    out_layout: Layout, # B, N, K
    max_warps_per_block: Int = 8,
](
    src: LayoutTensor[dtype, src_layout],
    index: LayoutTensor[DType.int32, index_layout],
    out: LayoutTensor[dtype, out_layout, MutableAnyOrigin],
    locks: UnsafePointer[Int32],
):
    # TODO: Remove the cuda-version of the source code
    # const int64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    # const int64_t total  = static_cast<int64_t>(B) * E * K;
    # if (tid >= total) return;
    #
    # // Decode (b,e,k) from flattened tid
    # const int  b =  tid / (E * K);
    # const int  e = (tid / K) % E;
    # const int  k =  tid % K;
    #
    # const int64_t idx = index[b * E + e];           // 0 ≤ idx < N
    # const int64_t out_offset =
    #     (static_cast<int64_t>(b) * N + idx) * K + k;
    # // Native atomicAdd for all types (CC ≥ 6.0)
    # atomicAdd(out + out_offset, src[tid]);

    var B = src.dim[0]()
    var E = src.dim[1]()
    var K = src.dim[2]()
    var N = out.dim[1]()

    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total = B * E * K
    if tid >= total:
        return

    var b = tid // (E * K)
    var e = (tid // K) % E
    var k = tid % K

    var idx = index[b * E + e]
    var out_offset = (b * N + idx) * K + k

    atomic_add[dtype]("TODO: What should go here?", src[tid])
    # Atomic.fetch_add(out + out_offset, src[tid]) # TODO: No works:((
    # alias swizzle_block = src_layout.is_half_float() and b_type.is_half_float() and is_nvidia_gpu()
    # var block_idx_swizzle = block_swizzle(
    #     Index[dtype = DType.uint32](block_idx.x, block_idx.y),
    #     Index[dtype = DType.uint32](grid_dim.x, grid_dim.y),
    # ) if swizzle_block else Index[dtype = DType.uint32](
    #     block_idx.x, block_idx.y
    # )

    # var bid = block_idx_swizzle[
    #     1
    # ] + block_dim.x * block_idx_swizzle[0]

    # var semaphore = Semaphore(locks.offset(bid), thread_idx.x)
    # semaphore.fetch()
    # semaphore.wait(block_idx.z)
    # semaphore.release(lock_flag)

@compiler.register("scatter_add")
struct ScatterAdd:

    @staticmethod
    fn execute[
        target: StaticString,
    ](

   ) raises:
       @parameter
       if target == "gpu":
            pass
        else:
            constrained[False, "TODO: Only supported on GPU"]()
# template <typename scalar_t>
# void scatter_add(const scalar_t* d_src,
#                  const int64_t*  d_index,
#                  scalar_t*       d_out,     // already zero-initialised
#                  int B, int E, int K, int N,
#                  cudaStream_t stream = 0)
# {
#     const int64_t total = static_cast<int64_t>(B) * E * K;
#     constexpr int THREADS = 256;
#     const int BLOCKS = static_cast<int>((total + THREADS - 1) / THREADS);
#     scatter_add_kernel<scalar_t>
#         <<<BLOCKS, THREADS, 0, stream>>>(d_src, d_index, d_out,
#                                          B, E, K, N);
# }

