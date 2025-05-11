from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops, BufferType, BufferValue
from max import nn
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
import torch
import random
import wandb
from tqdm import tqdm

from dataclasses import dataclass

import math
from dataclasses import dataclass

from max.graph import TensorValue, ops
from max.nn import LinearV1
from max.nn.layer import Layer

from torch.nn.functional import scaled_dot_product_attention


def attention_fwd(
    self,
    Q: TensorValue,
    K: TensorValue,
    V: TensorValue,
    attn_mask: TensorValue,
    n_heads: int,
    head_dim: int,
    constant_zero: TensorValue,
    neginf: TensorValue
) -> TensorValue:
    # Broadcast the attention mask across heads.
    # Do so in the graph so that the broadcast can be fused into downstream
    # ops.
    batch, _, seq_len, post_seq_len = attn_mask.shape
    attn_mask = attn_mask.broadcast_to(
        (
            batch,
            n_heads,
            seq_len,
            post_seq_len,
        )
    )

    scale = math.sqrt(1.0 / head_dim)
    scores = Q @ ops.transpose(K, -2, -1)
    # Note, the graph compiler currently requires the order of operands
    # to be `scores * scale` in order to pattern match the fused attention
    # operator.
    # scores = ops.softmax(scores * scale + attn_mask)
    scores = ops.select(attn_mask, scores, neginf)
    scores = ops.softmax(scores * scale)

    return scores @ V, scores # scores are P



def attention_bwd(
    P: TensorValue,
    dO: TensorValue,
    Q: TensorValue,
    K: TensorValue,
    V: TensorValue,
    mask: TensorValue,
    head_dim: int,
    constant_zero: TensorValue
):
    dV = ops.matmul(ops.transpose(P, 2, 3), dO)
    dP = ops.matmul(dO, ops.transpose(V, -1, -2))
    
    dS = ops.add(
        ops.mul(ops.broadcast_to(-ops.sum(ops.mul(dP, P), axis=-1), P.shape), P),
        ops.mul(dP, P)
    )

    # if is_causal:
    #     dS = dS.masked_fill(mask.unsqueeze(0).expand_as(dS), 0)
    dS = ops.select(mask, dS, constant_zero)

    # P_m_1_n = ops.reshape(P, (-1, ))
    # dS = dP # TODO: DO THIS

    # TODO: Do mask

    dQ = ops.matmul(dS, K) * (1.0 / math.sqrt(head_dim))
    dK = ops.matmul(ops.transpose(dS, -1, -2), Q) * (1.0 / math.sqrt(head_dim))
    return dQ, dK, dV



def run_sdpa(
    q: NDArray[np.float32],
    k: NDArray[np.float32],
    v: NDArray[np.float32],
    dO: NDArray[np.float32],
    mask: NDArray[np.bool],
    session: InferenceSession,
    device: Device,
    n_heads: int,
    head_dim: int
):
    D = k.shape[-1]
    q_tensor = Tensor.from_numpy(q).to(device)
    k_tensor = Tensor.from_numpy(k).to(device)
    v_tensor = Tensor.from_numpy(v).to(device)
    mask_tensor = Tensor.from_numpy(mask).to(device)
    dO_tensor = Tensor.from_numpy(dO).to(device)

    mojo_kernels = Path(__file__).parent / "operations"

    dtype = DType.float32

    with Graph(
        "sdpa_graph",
        input_types=[
            TensorType(
                dtype,
                shape=q_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=k_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=v_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.bool,
                shape=mask.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=dO_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        q, k, v, mask, dO = graph.inputs

        constant_zero = ops.constant(0.0, DType.bool, q.tensor.device)
        neginf = ops.constant(-1e9, DType.float32, q.tensor.device)
        o, p = attention_fwd(graph, q, k, v, mask, n_heads, head_dim, constant_zero, neginf)
        print("mask",mask)
        print("constzero",constant_zero)

        dQ, dK, dV = attention_bwd(p, dO, q, k, v, mask, head_dim, constant_zero)

        graph.output(o, p, dQ, dK, dV)

    model = session.load(graph)

    o, p, dQ, dK, dV = model.execute(q_tensor, k_tensor, v_tensor, mask_tensor, dO_tensor)

    return o.to(CPU()), p.to(CPU()), dQ.to(CPU()), dK.to(CPU()), dV.to(CPU())

def main():
    batch_size = 16
    n_head = 12
    seq_len = 64
    head_dim = 24

    seed = 13
    random.seed(seed)
    np.random.seed(seed)

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    print(device)

    # Set up an inference session for running the graph.
    session = InferenceSession(devices=[device])

    q = np.random.uniform(size=(batch_size, n_head, seq_len, head_dim)).astype(np.float32)
    k = np.random.uniform(size=(batch_size, n_head, seq_len, head_dim)).astype(np.float32)
    v = np.random.uniform(size=(batch_size, n_head, seq_len, head_dim)).astype(np.float32)
    mask = np.tril(np.ones((batch_size, n_head, seq_len, seq_len))).astype(np.bool)
    dO = np.random.uniform(size=(batch_size, n_head, seq_len, head_dim)).astype(np.float32)

    o, p, dQ, dK, dV = run_sdpa(q, k, v, dO, mask, session, device, n_head, head_dim)
    print(o.to_numpy() - scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v), attn_mask=torch.tensor(mask)).numpy())
    print(np.linalg.norm(o.to_numpy() - scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v), attn_mask=torch.tensor(mask)).numpy()))

    dQ, dK, dV = scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v), attn_mask=torch.tensor(mask)).backward()
    print(dQ, dK, dV)


if __name__ == "__main__":
    main()