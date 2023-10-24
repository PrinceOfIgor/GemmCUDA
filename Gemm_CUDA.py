import numpy as np
from numba import cuda, float32, jit
import sys
import time


#Kernel function for GPU
@cuda.jit
def cuda_gemm_kernel(A, B, C):
    #Determine thread indices
    i, j = cuda.grid(2)
    #Ensure that they stay within valid bounds
    if i < C.shape[0] and j < C.shape[1]:
        C_value = 0.0
        for k in range(A.shape[1]):
            #Operation
            C_value += A[i, k] * B[k, j]
        #Store in memory
        C[i, j] = C_value

def cuda_gemm(A, B, threadsperblock):

    m, n = A.shape[0], B.shape[1]
    #Send each array to device memory
    A_global = cuda.to_device(A)
    B_global = cuda.to_device(B)
    C_global = cuda.device_array((m, n), dtype=float32)

    #Determine the tiling for the GPU threads
    blockspergrid_x = (m + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    #Initiate the kernel call with the request blocks
    cuda_gemm_kernel[blockspergrid, threadsperblock](A_global, B_global, C_global)
    #Copy to host memory
    C = C_global.copy_to_host()

    return C

@jit(nopython=True)
def naive_matrix_mul_numba(A, B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C

@jit(nopython=True)
def ikj_matrix_mul_numba(A,B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C

def main():

    if len(sys.argv) != 4:
        matrix_size = 256
        print(f"Defaulting to NxN {matrix_size} sized matrices")
        print("To change the size of the matrices to be multiplied, provide it as arguments")
        #Cast to int
        m, n, k = int(matrix_size), int(matrix_size), int(matrix_size)
        print(f"Running with {matrix_size} sized matrices")
    else:
        m, n, k = sys.argv[1:]
        m, n, k = int(m), int(n), int(k)
        print(f"Running with A as {m}x{n} and B as {n}x{k} sized matrices")

    #Initialize values
    #Threads per block of operations, good to be a multiple of 32 according to programming guide
    threadsperblock = (16, 16)
    num_runs = 10
    #Randomized initial matrices
    A = np.random.rand(m, n)
    B = np.random.rand(n, k)

    start = time.time()
    for _ in range(num_runs):
        result = cuda_gemm(A, B, threadsperblock)
    end = time.time()
    cuda_time_numba = end - start

    #naive matrix mult with numba
    start = time.time()
    for _ in range(num_runs):
        naive_matrix_mul_numba(A, B)
    end = time.time()
    naive_time_numba = end - start

    # TODO: time other implementations here
    #ikj matrix mult with numba
    start = time.time()
    for _ in range(num_runs):
        ikj_matrix_mul_numba(A, B)
    end = time.time()
    ikj_time_numba = end - start

    print('naive time numba: {}'.format(naive_time_numba))
    print('ikj time numba: {}'.format(ikj_time_numba))
    print('ikj time numba: {}'.format(cuda_time_numba))

main()



