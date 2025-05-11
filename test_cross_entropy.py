from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# ---- Runtime / driver imports ------------------------------------------------
# These come from the same SDK you used for the mat‑mul example.
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops, BufferType, BufferValue
from max import nn
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
import torch


# ------------------------------------------------------------------------------
# Helper that compiles + runs the Mojo softmax‑cross‑entropy kernel
# ------------------------------------------------------------------------------
def cross_entropy_loss(
    logits: np.ndarray,  # shape [B, C], dtype float32 / float16
    labels: np.ndarray,  # shape [B]   , dtype int32
    algorithm: str,  # passthrough compile‑time flag (if you need it)
    session: InferenceSession,
    device: Device,
) -> Tuple[Tensor, Tensor]:
    """
    Executes the Mojo kernel and returns (mean_loss_scalar, grad_logits) tensors.
    Both results are copied back to the CPU before returning.
    """
    if logits.dtype not in (np.float32, np.float16):
        raise TypeError("logits must be fp32 or fp16")
    if labels.dtype != np.int32:
        raise TypeError("labels must be int32")

    # 1) Move host arrays -> accelerator tensors
    logits_t = Tensor.from_numpy(logits).to(device)
    labels_t = Tensor.from_numpy(labels).to(device)

    # mojo_kernels = Path(__file__).parent / "operations"  # *.mojo lives here
    mojo_kernels = Path("operations")  # *.mojo lives here

    B, V = logits.shape
    dtype = DType.float32
    # 2) Build a tiny graph that wraps the single custom op
    with Graph(
        "test_cross_entropy",
        input_types=[
            TensorType(
                dtype, shape=logits_t.shape, device=DeviceRef.from_device(device)
            ),
            TensorType(
                DType.int32, shape=labels_t.shape, device=DeviceRef.from_device(device)
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        logits_val, labels_val = graph.inputs

        loss_val, grad_val = ops.custom(
            name="cross_entropy",
            values=[logits_val, labels_val],
            out_types=[
                # mean loss scalar
                TensorType(
                    dtype,
                    shape=labels_t.shape,
                    device=DeviceRef.from_device(device),
                ),
                # grad has same shape as logits
                TensorType(
                    dtype,
                    shape=logits_t.shape,
                    device=DeviceRef.from_device(device),
                ),
            ],
            parameters={"algorithm": algorithm},  # if you want to select variants
        )
        graph.output(loss_val, grad_val)

    # 3) Compile & run
    print("⇢ Compiling softmax-xent graph …")
    model = session.load(graph)

    print("⇢ Executing on device …")
    loss_out, grad_out = model.execute(logits_t, labels_t)

    # 4) Bring results back to CPU for inspection
    return loss_out.to(CPU()), grad_out.to(CPU())


# ------------------------------------------------------------------------------
# Quick functional check against PyTorch
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    device = CPU() if accelerator_count() == 0 else Accelerator()
    session = InferenceSession(devices=[device])

    print(device)

    B, D = 1000, 1000
    np.random.seed(0)

    logits_np = np.ones(D, dtype=np.float32).reshape(1, D)
    logits_np = np.tile(logits_np, (B, 1))  # shape [B, D]
    logits_np += np.random.uniform(size=(B, D)).astype(np.float32) * 30
    # print log sum exp of logits

    lse = np.log(
        np.sum(
            np.exp(logits_np - logits_np.max(axis=1, keepdims=True)),
            axis=1,
            keepdims=True,
        )
    ) + logits_np.max(axis=1, keepdims=True)
    # print(lse)

    labels_np = np.random.randint(0, D, size=(B,), dtype=np.int32)

    mojo_loss, mojo_grad = cross_entropy_loss(
        logits_np,
        labels_np,
        algorithm="warp_per_row",
        session=session,
        device=device,
    )

    # PyTorch reference
    torch_logits = torch.tensor(logits_np, device="cuda", requires_grad=True)
    torch_labels = torch.tensor(labels_np, device="cuda", dtype=torch.long)
    torch_loss = torch.nn.functional.cross_entropy(
        torch_logits, torch_labels, reduce=False
    )
    # print("PyTorch loss:")
    # print(torch_loss.cpu().detach().numpy())

    torch_loss.sum().backward()
    torch_grad = torch_logits.grad.cpu().numpy()

    # print("Mojo loss:")
    # print(mojo_loss.to_numpy())

    loss_err = np.linalg.norm(
        mojo_loss.to_numpy() - torch_loss.cpu().detach().numpy()
    ) / np.linalg.norm(mojo_loss.to_numpy())
    grad_err = np.linalg.norm(
        np.abs(mojo_grad.to_numpy() - torch_grad)
    ) / np.linalg.norm(torch_grad)

    print(f"Loss rel‑err  : {loss_err:e}")
    print(f"Grad rel‑err  : {grad_err:e}")
