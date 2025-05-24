# test_adamw_comprehensive.py
from pathlib import Path
from typing import Tuple, List, Dict
from tqdm import tqdm
import time
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
        )
        graph.output(next_m_val, next_v_val, next_w_val)

    # 3) Compile & run
    model = session.load(graph)
    next_m, next_v, next_w = model.execute(pm_t, pv_t, pw_t, dw_t, step_t_t)

    # 4) Bring results back to CPU
    return next_m.to(CPU()), next_v.to(CPU()), next_w.to(CPU())


# ---------------------------------------------------------------------------
# NumPy reference implementation
# ---------------------------------------------------------------------------
def adamw_numpy_reference(
    prev_m: np.ndarray,
    prev_v: np.ndarray,
    prev_w: np.ndarray,
    grad: np.ndarray,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy reference implementation of AdamW"""
    m_ref = beta1 * prev_m + (1.0 - beta1) * grad
    v_ref = beta2 * prev_v + (1.0 - beta2) * grad**2

    # Bias-corrected step size
    alpha_t = lr * np.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
    update = alpha_t * m_ref / (np.sqrt(v_ref) + eps) + lr * weight_decay * prev_w
    w_ref = prev_w - update

    return m_ref, v_ref, w_ref


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------
def compute_errors(mojo_result: Tensor, numpy_ref: np.ndarray) -> Dict[str, float]:
    """Compute various error metrics between Mojo result and NumPy reference"""
    mojo_np = mojo_result.to_numpy()

    # L2 relative error
    l2_norm = np.linalg.norm(numpy_ref.ravel())
    l2_rel_err = np.linalg.norm((mojo_np - numpy_ref).ravel()) / (l2_norm + 1e-12)

    # Max absolute error
    max_abs_err = np.max(np.abs(mojo_np - numpy_ref))

    # Max relative error
    max_rel_err = np.max(
        np.abs(mojo_np - numpy_ref) / np.maximum(np.abs(numpy_ref), 1e-12)
    )

    # Mean absolute error
    mean_abs_err = np.mean(np.abs(mojo_np - numpy_ref))

    return {
        "l2_rel": l2_rel_err,
        "max_abs": max_abs_err,
        "max_rel": max_rel_err,
        "mean_abs": mean_abs_err,
    }


def run_single_test(
    N: int,
    t: int,
    session: InferenceSession,
    device: Device,
    rng: np.random.Generator,
    hyperparams: Dict[str, float],
    verbose: bool = True,
) -> Dict[str, float]:
    """Run a single test case with given size and timestep"""
    # Generate random test data
    prev_m = rng.standard_normal(N, dtype=np.float32) * 0.1
    prev_v = rng.standard_normal(N, dtype=np.float32) ** 2 * 0.1
    prev_w = rng.standard_normal(N, dtype=np.float32)
    grad = rng.standard_normal(N, dtype=np.float32)

    # Time the Mojo kernel
    start_time = time.time()
    mojo_m, mojo_v, mojo_w = adamw_step(
        prev_m,
        prev_v,
        prev_w,
        grad,
        step_t=t,
        **hyperparams,
        session=session,
        device=device,
    )
    mojo_time = time.time() - start_time

    # Compute NumPy reference
    start_time = time.time()
    ref_m, ref_v, ref_w = adamw_numpy_reference(
        prev_m, prev_v, prev_w, grad, t, **hyperparams
    )
    numpy_time = time.time() - start_time

    # Compute errors
    m_errors = compute_errors(mojo_m, ref_m)
    v_errors = compute_errors(mojo_v, ref_v)
    w_errors = compute_errors(mojo_w, ref_w)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Test Case: N={N}, t={t}")
        print(f"{'='*60}")
        print(f"Execution Time - Mojo: {mojo_time:.6f}s, NumPy: {numpy_time:.6f}s")
        print(f"Speedup: {numpy_time/mojo_time:.2f}x")
        print(f"\nErrors:")
        print(f"  Momentum (m):")
        print(f"    L2 relative: {m_errors['l2_rel']:.2e}")
        print(f"    Max absolute: {m_errors['max_abs']:.2e}")
        print(f"    Max relative: {m_errors['max_rel']:.2e}")
        print(f"  Variance (v):")
        print(f"    L2 relative: {v_errors['l2_rel']:.2e}")
        print(f"    Max absolute: {v_errors['max_abs']:.2e}")
        print(f"    Max relative: {v_errors['max_rel']:.2e}")
        print(f"  Weights (w):")
        print(f"    L2 relative: {w_errors['l2_rel']:.2e}")
        print(f"    Max absolute: {w_errors['max_abs']:.2e}")
        print(f"    Max relative: {w_errors['max_rel']:.2e}")

    return {
        "N": N,
        "t": t,
        "mojo_time": mojo_time,
        "numpy_time": numpy_time,
        "m_l2_rel": m_errors["l2_rel"],
        "v_l2_rel": v_errors["l2_rel"],
        "w_l2_rel": w_errors["l2_rel"],
        "max_error": max(m_errors["max_rel"], v_errors["max_rel"], w_errors["max_rel"]),
    }


def run_multi_step_test(
    N: int,
    num_steps: int,
    session: InferenceSession,
    device: Device,
    rng: np.random.Generator,
    hyperparams: Dict[str, float],
) -> None:
    """Test multiple optimization steps to verify state propagation"""
    print(f"\n{'='*60}")
    print(f"Multi-Step Test: N={N}, steps={num_steps}")
    print(f"{'='*60}")

    # Initialize states
    m_mojo = rng.standard_normal(N, dtype=np.float32) * 0.1
    v_mojo = rng.standard_normal(N, dtype=np.float32) ** 2 * 0.1
    w_mojo = rng.standard_normal(N, dtype=np.float32)

    m_numpy = m_mojo.copy()
    v_numpy = v_mojo.copy()
    w_numpy = w_mojo.copy()

    # Run multiple steps
    for t in tqdm(range(1, num_steps + 1)):
        grad = rng.standard_normal(N, dtype=np.float32)

        # Mojo step
        m_tensor, v_tensor, w_tensor = adamw_step(
            m_mojo,
            v_mojo,
            w_mojo,
            grad,
            step_t=t,
            **hyperparams,
            session=session,
            device=device,
        )
        m_mojo = m_tensor.to_numpy()
        v_mojo = v_tensor.to_numpy()
        w_mojo = w_tensor.to_numpy()

        # NumPy step
        m_numpy, v_numpy, w_numpy = adamw_numpy_reference(
            m_numpy, v_numpy, w_numpy, grad, t, **hyperparams
        )

    # Check final errors
    final_w_error = np.linalg.norm(w_mojo - w_numpy) / np.linalg.norm(w_numpy)
    print(
        f"Final weight L2 relative error after {num_steps} steps: {final_w_error:.2e}"
    )

    # Verify weights are actually changing
    initial_w = rng.standard_normal(N, dtype=np.float32)
    weight_change = np.linalg.norm(w_numpy - initial_w) / np.linalg.norm(initial_w)
    print(f"Total weight change magnitude: {weight_change:.2%}")


# ---------------------------------------------------------------------------
# Main test suite
# ---------------------------------------------------------------------------
def main():
    # Setup device and session
    device = CPU() if accelerator_count() == 0 else Accelerator()
    session = InferenceSession(devices=[device])
    print(f"Running tests on: {device}")

    # Test hyperparameters
    hyperparams = {
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-2,
    }

    # Test configurations
    test_sizes = [32, 256, 1024, 4096, 16384, 65536]  # Various problem sizes
    test_timesteps = [1, 10, 100, 1000]  # Test bias correction at different t

    # Random seed for reproducibility
    rng = np.random.default_rng(42)

    # Run comprehensive tests
    all_results = []
    tolerance = 1e-5  # Relative error tolerance

    print("\n" + "=" * 60)
    print("ADAMW KERNEL COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print(f"Hyperparameters: {hyperparams}")
    print(f"Error tolerance: {tolerance}")

    # Test 1: Various sizes and timesteps
    for N in test_sizes:
        for t in test_timesteps:
            if N > 16384 and t > 100:  # Skip very large tests
                continue

            result = run_single_test(
                N, t, session, device, rng, hyperparams, verbose=False
            )
            all_results.append(result)

            # Check if test passed
            if result["max_error"] > tolerance:
                print(f"❌ FAILED: N={N}, t={t} - Max error: {result['max_error']:.2e}")
            else:
                print(f"✅ PASSED: N={N}, t={t} - Max error: {result['max_error']:.2e}")

    # Test 2: Multi-step convergence tests
    print("\n" + "=" * 60)
    print("MULTI-STEP CONVERGENCE TESTS")
    print("=" * 60)

    for N in [256, 4096]:
        run_multi_step_test(N, 50, session, device, rng, hyperparams)

    # Test 3: Edge cases
    print("\n" + "=" * 60)
    print("EDGE CASE TESTS")
    print("=" * 60)

    # Test with zero gradients
    print("\nTesting with zero gradients...")
    N = 128
    prev_m = rng.standard_normal(N, dtype=np.float32) * 0.1
    prev_v = rng.standard_normal(N, dtype=np.float32) ** 2 * 0.1
    prev_w = rng.standard_normal(N, dtype=np.float32)
    zero_grad = np.zeros(N, dtype=np.float32)

    mojo_m, mojo_v, mojo_w = adamw_step(
        prev_m,
        prev_v,
        prev_w,
        zero_grad,
        1,
        **hyperparams,
        session=session,
        device=device,
    )

    # With zero gradients, only weight decay should affect weights
    expected_w = prev_w * (1 - hyperparams["lr"] * hyperparams["weight_decay"])
    w_error = np.linalg.norm(mojo_w.to_numpy() - expected_w) / np.linalg.norm(
        expected_w
    )
    print(f"Zero gradient test - Weight error: {w_error:.2e}")

    # Test with very large timestep (extreme bias correction)
    print("\nTesting with large timestep (t=10000)...")
    result = run_single_test(
        256, 10000, session, device, rng, hyperparams, verbose=True
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in all_results if r["max_error"] <= tolerance)
    total = len(all_results)
    print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️  {total - passed} tests failed")

    # Performance summary
    print("\nPerformance Summary:")
    for size in test_sizes[:4]:  # Show first 4 sizes
        size_results = [r for r in all_results if r["N"] == size]
        if size_results:
            avg_speedup = np.mean(
                [r["numpy_time"] / r["mojo_time"] for r in size_results]
            )
            print(f"  N={size}: Average speedup {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
