from gpu.host import Dim
from gpu.id import block_idx, thread_idx
from max.driver import Accelerator, Device, accelerator, cpu
from sys import exit, has_accelerator

fn print_threads():
    """Print thread IDs."""
    print("Block index: [",
        block_idx.x, block_idx.y, block_idx.z,
        "]\tThread index: [",
        thread_idx.x, thread_idx.y, thread_idx.z,
        "]"
    )

def main():
    if not has_accelerator():
        print("A GPU is required to run this program")
        exit()

    host_device = cpu()
    print("Found the CPU device")
    gpu_device = accelerator()
    print("Found the GPU device")

    print_threads_gpu = Accelerator.compile[print_threads](gpu_device)

    print_threads_gpu(gpu_device, grid_dim=Dim(2, 2, 1), block_dim=Dim(64, 4, 2))

    # Required for now to keep the main thread alive until the GPU is done
    Device.wait_for(gpu_device)
    print("Program finished")