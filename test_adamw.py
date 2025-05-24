# test_adamw.py
from pathlib import Path
from typing import Tuple

import numpy as np

# ---- Runtime / driver imports ----------------------------------------------
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


# ---------------------------------------------------------------------------
# Helper that compiles + runs the Mojo AdamW kernel
# ---------------------------------------------------------------------------
def adamw_step(
    prev_m: np.ndarray,
    prev_v: np.ndarray,
    prev_w: np.ndarray,
    d_w: np.ndarray,
    step_t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    session: InferenceSession = None,
    device: Device = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Executes the Mojo AdamW kernel (registered as 'adamw') and returns
    (next_m, next_v, next_weight) tensors.  Results are copied back to CPU.
    """
    if session is None or device is None:
        raise ValueError("session and device must be supplied")

    for arr in (prev_m, prev_v, prev_w, d_w):
        if arr.dtype != np.float32:
            raise TypeError("all tensors must be fp32")

    # 1) Move host arrays → accelerator tensors
    pm_t = Tensor.from_numpy(prev_m).to(device)
    pv_t = Tensor.from_numpy(prev_v).to(device)
    pw_t = Tensor.from_numpy(prev_w).to(device)
    dw_t = Tensor.from_numpy(d_w).to(device)
    step_t_np = np.array([step_t], dtype=np.int32)
    step_t_t = Tensor.from_numpy(step_t_np).to(device)

    mojo_kernels = Path("operations")  # where adamw.mojo lives
    dtype = DType.float32
    int_dtype = DType.int32

    # 2) Build a tiny graph that wraps the single custom op
    with Graph(
        "test_adamw",
        input_types=[
            TensorType(dtype, pm_t.shape, device=DeviceRef.from_device(device)),
            TensorType(dtype, pv_t.shape, device=DeviceRef.from_device(device)),
            TensorType(dtype, pw_t.shape, device=DeviceRef.from_device(device)),
            TensorType(dtype, dw_t.shape, device=DeviceRef.from_device(device)),
            TensorType(int_dtype, step_t_t.shape, device=DeviceRef.from_device(device)),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        pm_val, pv_val, pw_val, dw_val, step_val = graph.inputs

        next_m_val, next_v_val, next_w_val = ops.custom(
            name="adamw",
            values=[pm_val, pv_val, pw_val, dw_val, step_val],
            out_types=[
                TensorType(dtype, pm_t.shape, device=DeviceRef.from_device(device)),
                TensorType(dtype, pv_t.shape, device=DeviceRef.from_device(device)),
                TensorType(dtype, pw_t.shape, device=DeviceRef.from_device(device)),
            ],
            # No compile‑time parameters needed for this kernel
        )
        graph.output(next_m_val, next_v_val, next_w_val)

    # 3) Compile & run
    # import ipdb; ipdb.set_trace() # fmt: skip
    model = session.load(graph)
    next_m, next_v, next_w = model.execute(pm_t, pv_t, pw_t, dw_t, step_t_t)

    # 4) Bring results back to CPU
    return next_m.to(CPU()), next_v.to(CPU()), next_w.to(CPU())


# ---------------------------------------------------------------------------
# Quick functional check against a NumPy reference implementation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = CPU() if accelerator_count() == 0 else Accelerator()
    session = InferenceSession(devices=[device])
    print(device)

    N = 4096  # number of parameters
    rng = np.random.default_rng(0)

    prev_m_np = rng.standard_normal(N, dtype=np.float32) * 0.1
    prev_v_np = rng.standard_normal(N, dtype=np.float32) ** 2 * 0.1
    prev_w_np = rng.standard_normal(N, dtype=np.float32)
    grad_np = rng.standard_normal(N, dtype=np.float32)

    # Hyper‑parameters (must match defaults in adamw.mojo)
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 1e-2
    t = 1

    # ---- Mojo ----------------------------------------------------------------
    # import ipdb; ipdb.set_trace()  # fmt: skip
    mojo_m, mojo_v, mojo_w = adamw_step(
        prev_m_np,
        prev_v_np,
        prev_w_np,
        grad_np,
        step_t=t,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=wd,
        session=session,
        device=device,
    )

    # ---- NumPy reference ------------------------------------------------------
    m_ref = beta1 * prev_m_np + (1.0 - beta1) * grad_np
    v_ref = beta2 * prev_v_np + (1.0 - beta2) * grad_np**2

    # bias‑corrected step size as coded in adamw.mojo
    alpha_t = lr * np.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
    update = alpha_t * m_ref / (np.sqrt(v_ref) + eps) + lr * wd * prev_w_np
    w_ref = prev_w_np - update

    # ---- Error metrics --------------------------------------------------------
    l2 = lambda x: np.linalg.norm(x.ravel())
    m_err = l2(mojo_m.to_numpy() - m_ref) / l2(m_ref)
    v_err = l2(mojo_v.to_numpy() - v_ref) / l2(v_ref)
    w_err = l2(mojo_w.to_numpy() - w_ref) / l2(w_ref)

    max_abs_rel = lambda a, b: np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))

    print(
        f"m   rel‑err l2: {m_err:e},  max abs rel‑err: {max_abs_rel(mojo_m.to_numpy(), m_ref):e}"
    )
    print(
        f"v   rel‑err l2: {v_err:e},  max abs rel‑err: {max_abs_rel(mojo_v.to_numpy(), v_ref):e}"
    )
    print(
        f"w   rel‑err l2: {w_err:e},  max abs rel‑err: {max_abs_rel(mojo_w.to_numpy(), w_ref):e}"
    )
