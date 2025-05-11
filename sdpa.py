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

# @torch.compile
# def simple_sdpa_backward_torch(ctx, dO: Float[Tensor, "... N_q d"]):
#     Q, K, V, O, L = ctx.saved_tensors
#     N_q = Q.size(-2)
#     N_k = K.size(-2)
#     d = Q.size(-1)
#     is_causal = ctx.is_causal
#     S = einsum(Q, K, "... N_q d, ... N_k d -> ... N_q N_k") / math.sqrt(d)
#     P = (S - L.unsqueeze(-1)).exp()
#     if is_causal:
#         mask = torch.ones(N_q, N_k, dtype=torch.bool, device=P.device).triu(diagonal=1)
#         P = P.masked_fill(mask.unsqueeze(0).expand_as(P), 0)
#     dV = einsum(P, dO, "... N_q N_k, ... N_q d -> ... N_k d")
#     dP = einsum(dO, V, "... N_q d, ... N_k d -> ... N_q N_k")
#     D = einsum(O, dO, "... N_q_o d, ... N_q_do d -> ... N_q_o N_q_do").diagonal(dim1=1, dim2=2)
#     dS = P * (dP - D.unsqueeze(-1))
#     if is_causal:
#         dS = dS.masked_fill(mask.unsqueeze(0).expand_as(dS), 0)
#     dQ = einsum(dS, K, "... N_q N_kv, ... N_kv d -> ... N_q d") / math.sqrt(d)
#     dK = einsum(dS, Q, "... N_q N_k, ... N_q d -> ... N_k d") / math.sqrt(d)
#     return dQ, dK, dV, None


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

    # D = einsum(O, dO, "... N_q_o d, ... N_q_do d -> ... N_q_o N_q_do").diagonal(dim1=1, dim2=2)
    # dS = P * (dP - D.unsqueeze(-1))
    # if is_causal:
    #     dS = dS.masked_fill(mask.unsqueeze(0).expand_as(dS), 0)


# def sdpa_fwd(q, k, v, mask, d_k_inv, neg_inf):
#     s = ops.mul(ops.matmul(q, ops.transpose(k, -1, -2)), d_k_inv)
#     # TODO: mask
#     s = ops.select(mask, s, neg_inf)
#     p = ops.softmax(s)
#     o = ops.matmul(p, v)
#     return o, p


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

    q = torch.tensor(q, requires_grad=True)
    k = torch.tensor(k, requires_grad=True)
    v = torch.tensor(v, requires_grad=True)
    mask = torch.tensor(mask)
    dO = torch.tensor(dO)

    loss = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    loss.backward(dO)
    print(dQ, dK, dV)

    print(q.grad)

    print(np.linalg.norm(q.grad.numpy() - dQ.to_numpy()))
    print(np.linalg.norm(k.grad.numpy() - dK.to_numpy()))
    print(np.linalg.norm(v.grad.numpy() - dV.to_numpy()))

    # print(dQ, dK, dV)



    # print(o.to_numpy())

    # scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v), attn_mask=torch.tensor(mask))
    # print(p.to_numpy())

    # # Fill the input matrices with random values.
    # y = np.random.uniform(size=(BATCH_SIZE, K)).astype(np.float32)
    # y_hat = np.random.uniform(size=(BATCH_SIZE, K)).astype(np.float32)

    # print(f"{y.shape=} {y_hat.shape=}")

    # print(numpy_mse(y, y_hat))
    # print([x.shape for x in numpy_mse(y, y_hat)])

    # # # First, perform the matrix multiplication in NumPy.

    # # print("Expected result:")
    # # print(a @ b)
    # # print()

    # # assert accelerator_count() > 0
    # #     # Then, test the various versions of matrix multiplication operations.
    # loss, grad = run_mymse(y, y_hat, "naive", session, device)
    # print("Naive matrix multiplication:")
    # print(loss.to_numpy())
    # print(loss.shape)
    # print(grad.to_numpy())
    # print(grad.shape)
    # print()

if __name__ == "__main__":
    main()