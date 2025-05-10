# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""A vanilla opaque KV Cache optimized attention mechanism."""

from dataclasses import dataclass
from typing import Union

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

from max.nn.kv_cache import ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
# from ..kernels import flash_attention
from max.nn.attention.interfaces import AttentionImpl, AttentionImplQKV
from max.nn.attention import Attention

# def fused_qkv_matmul(
#     kv_params: KVCacheParams,
#     input: TensorValue,
#     wqkv: TensorValue,
#     kv_collection: ContinuousBatchingKVCacheCollection,
#     layer_idx: TensorValue,
#     n_heads: int,
# ) -> TensorValue:
#     """Computes fused query, key and value projections."""
#     if input.dtype != wqkv.dtype:
#         msg = (
#             "expected input and wqkv to have the same dtype, but got"
#             f" {input.dtype} and {wqkv.dtype}, respectively."
#         )
#         raise ValueError(msg)

#     input_rank_expected = 3
#     if input.rank != input_rank_expected:
#         msg = f"expected input to have rank {input_rank_expected}, was {input.rank}"
#         raise ValueError(msg)

#     wqkv_rank_expected = 2
#     if wqkv.rank != wqkv_rank_expected:
#         msg = (
#             f"expected wqkv to have rank {wqkv_rank_expected}, was {wqkv.rank}"
#         )
#         raise ValueError(msg)

#     if layer_idx.dtype != DType.uint32:
#         msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
#         raise ValueError(msg)

#     if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
#         msg = f"unsupported cache strategy for fused_qkv_matmul: {kv_params.cache_strategy}"
#         raise ValueError(msg)

#     cache_strategy_str = kv_params.cache_strategy.kernel_substring()
#     op_name = f"mo.fused_qkv_matmul.padded.{cache_strategy_str}"

#     return ops.inplace_custom(
#         op_name,
#         values=[input, wqkv, kv_collection, layer_idx],
#         out_types=[
#             TensorType(
#                 dtype=input.dtype,
#                 shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
#                 device=input.device,
#             )
#         ],
#         parameters={
#             "num_heads": kv_params.n_kv_heads_per_device,
#             "head_dim": kv_params.head_dim,
#         },
#     )[0].tensor

# @dataclass
# class Attention(AttentionImpl):
#     def __call__(
#         self,
#         x: TensorValue,
#         kv_collection: Union[
#             ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
#         ],
#         **kwargs,
#     ) -> TensorValue:
#         if isinstance(kv_collection, PagedKVCacheCollection):
#             raise ValueError(
#                 "Paged attention not supported for Attention on non-ragged tensors."
#             )

#         if "attention_mask" not in kwargs:
#             raise ValueError("attention_mask not passed as input to Attention")
#         attention_mask = kwargs["attention_mask"]
#         if attention_mask.dtype != x.dtype:
#             msg = (
#                 "expected attention_mask and x to have the same dtype, but got"
#                 f" {attention_mask.dtype} and {x.dtype}, respectively."
#             )
#             raise ValueError(msg)

#         # Get attributes from inputs
#         batch_size, seq_len = x.shape[0], x.shape[1]

#         # Call into fused qkv matmul.
#         xq = fused_qkv_matmul(
#             self.kv_params,
#             input=x,
#             wqkv=self.wqkv,
#             kv_collection=kv_collection,
#             layer_idx=self.layer_idx,
#             n_heads=self.n_heads,
#         )

#         xq = ops.reshape(
#             xq,
#             [
#                 batch_size,
#                 seq_len,
#                 self.n_heads,
#                 self.kv_params.head_dim,
#             ],
#         )

#         # Calculate Flash Attention
#         attn_out = flash_attention(
#             self.kv_params,
#             input=xq,
#             kv_collection=kv_collection,
#             layer_idx=self.layer_idx,
#             attention_mask=attention_mask,
#             valid_lengths=kwargs["valid_lengths"],
#             scale=self.scale,
#         )

#         attn_out = ops.reshape(attn_out, shape=[batch_size, seq_len, -1])

#         return self.wo(attn_out)


# @dataclass
# class AttentionQKV(AttentionImplQKV):
#     def __call__(
#         self,
#         x: TensorValue,
#         kv_collection: Union[
#             ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
#         ],
#         **kwargs,
#     ) -> TensorValue:
#         if isinstance(kv_collection, PagedKVCacheCollection):
#             raise ValueError(
#                 "Paged attention not supported for Attention on non-ragged tensors."
#             )

#         if "attention_mask" not in kwargs:
#             raise ValueError("attention_mask not passed as input to Attention")
#         attention_mask = kwargs["attention_mask"]
#         if attention_mask.dtype != x.dtype:
#             msg = (
#                 "expected attention_mask and x to have the same dtype, but got"
#                 f" {attention_mask.dtype} and {x.dtype}, respectively."
#             )
#             raise ValueError(msg)

#         wqkv = ops.concat((self.wq, self.wk, self.wv)).transpose(0, 1)
#         wqkv = ops.cast(wqkv, x.dtype)

#         # Get attributes from inputs
#         batch_size, seq_len = x.shape[0], x.shape[1]

#         # Call into fused qkv matmul.
#         xq = fused_qkv_matmul(
#             self.kv_params,
#             input=x,
#             wqkv=wqkv,
#             kv_collection=kv_collection,
#             layer_idx=ops.constant(
#                 self.layer_idx, DType.uint32, device=DeviceRef.CPU()
#             ),
#             n_heads=self.n_heads,
#         )

#         xq = ops.reshape(
#             xq,
#             [
#                 batch_size,
#                 seq_len,
#                 self.n_heads,
#                 self.kv_params.head_dim,
#             ],
#         )

#         # Calculate Flash Attention
#         attn_out = flash_attention(
#             self.kv_params,
#             input=xq,
#             kv_collection=kv_collection,
#             layer_idx=ops.constant(
#                 self.layer_idx, DType.uint32, device=DeviceRef.CPU()
#             ),
#             attention_mask=attention_mask,
#             valid_lengths=kwargs["valid_lengths"],
#             scale=self.scale,
#         )

#         attn_out = ops.reshape(attn_out, shape=[batch_size, seq_len, -1])

#         return self.wo(attn_out)

n_heads = 8

attention = Attention(
    # Required parameters:
    n_heads=32,  # Number of attention heads
    kv_params=KVCacheParams(
        head_dim=128,  # Dimension of each attention head
        cache_strategy=CacheStrategy.OPAQUE,  # Must use opaque cache strategy
        dtype=DType.float32  # Data type for the cache
    ),
    layer_idx=0,  # Layer index in the transformer stack
    wqkv=wqkv_weights,  # Combined QKV weight matrix
    wo=output_projection,  # Output projection layer
    scale=1.0  # Optional scaling factor (defaults to 1/sqrt(head_dim))
)