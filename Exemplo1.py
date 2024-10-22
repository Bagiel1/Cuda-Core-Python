import numba
from numba import cuda
import numpy as np
import time

# Kernel Definition
@cuda.jit
def increment_kernel(device_array):
    idx = cuda.grid(1)
    if idx < device_array.size:
        device_array[idx] += 1 


def WGPU(n):
    # Build an array in the Host
    host_array = np.random.randint(0,1000, size=n, dtype=np.int32)
    
    # Transfer to GPU
    device_array = cuda.to_device(host_array)

    # Config execution
    threads_per_block = 32
    blocks_per_grid = (device_array.size + (threads_per_block - 1)) // threads_per_block

    # Execute the kernel
    increment_kernel[blocks_per_grid, threads_per_block](device_array)

    # Copy the results back to the host
    result_array = device_array.copy_to_host()

start= time.time()
WGPU(1000000000)
end= time.time() - start
print(end)

def NoGPU(n):
    a= np.random.randint(0,1000, size=n, dtype=np.int32)
    result= a[:]
    for i in range(len(a)):
        result[i] += 1
    return result

start= time.time()
NoGPU(1000000000)
end= time.time() - start
print(end)

