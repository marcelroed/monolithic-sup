from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import stack_allocation
from utils import Index, IndexList
from flash_attention import flash_attention
from memory import UnsafePointer
from random import rand, seed
import math
from sys import has_accelerator
from max.driver import accelerator, cpu

def build_ndbuffer[
    type: DType,
    rank: Int,
    *,
    static_shape: DimList = DimList.create_unknown[rank](),
](shape: IndexList[rank]) -> NDBuffer[
    type, rank, MutableAnyOrigin, static_shape
]:
    var ptr = UnsafePointer[Scalar[type]].alloc(shape.flattened_length())
    rand(ptr, shape.flattened_length())
    return NDBuffer[type, rank, _, static_shape](ptr, shape)

alias type = DType.float32

def main():
    seed(1337)
    alias batch_size = 1
    alias seq_len = 128
    alias d_k = 128
    alias q_shape = Index(batch_size, seq_len, d_k)
    var q = build_ndbuffer[type, 3](q_shape)
    var k = build_ndbuffer[type, 3](q_shape)
    var v = build_ndbuffer[type, 3](q_shape)

    var mask = build_ndbuffer[type](
        Index(seq_len, seq_len)
    )

    # var output = build_ndbuffer[type, static_shape=output_static_shape](q_shape)
    var output = build_ndbuffer[type](q_shape) # TODO: static shape
    var ref_output = build_ndbuffer[type](q_shape)

    @parameter
    @always_inline
    fn input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return k.load[width=simd_width](rebind[IndexList[k.rank]](idx))

    @parameter
    @always_inline
    fn input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return v.load[width=simd_width](rebind[IndexList[v.rank]](idx))

    @parameter
    @always_inline
    fn mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return mask.load[width=simd_width](rebind[IndexList[mask.rank]](idx))

    alias scale = Float32(1 / math.sqrt(d_k))

    flash_attention[input_k_fn, input_v_fn, mask_fn](
        q, k.get_shape(), v.get_shape(), mask.get_shape(), output, scale
    )

    print(output)
    
