#Alexandru Barsan
#NOTE CuBLAS is a wrapper call to its own underlying kernel setup code, has been moved out of this routine
#Module containing the CUDA kernels
import numpy as np
from numba import cuda, float32
import math
import sys
import time

#Naive kernel function for GPU
@cuda.jit
def cuda_gemm_kernel(A, B, C):
    #Determine thread indices
    i, j = cuda.grid(2)
    #Ensure that they stay within valid bounds
    if i < C.shape[0] and j < C.shape[1]:
        #Initialize the resultant for each thread as 0
        C_value = 0.0
        for k in range(A.shape[1]):
            #Inner loop iteration
            C_value += A[i, k] * B[k, j]
        #Store in memory
        C[i, j] = C_value

#Kernel function to improve shared global memory access
@cuda.jit
def cuda_gmc_gemm_kernel(A, B, C):
    #Determine thread indices
    i, j = cuda.grid(2)
    #Boundary checking
    if i < C.shape[0] and j < C.shape[1]:
        #Initialize C
        C_value = 0.0
        for k in range(A.shape[1]):
            # Ensure coalesced access by having threads access contiguous elements
            A_element = A[i, k]
            B_element = B[k, j]
            C_value += A_element * B_element

        # Warp synchronization to ensure shared memory usage doesn't cause issues
        cuda.syncthreads()

        # Reduction to calculate the final C_value across all threads in the warp
        for offset in range(1, cuda.blockDim.x):
            temp_i = i - offset
            temp_j = j - offset
            if temp_i >= 0 and temp_j >= 0:
                C_value += C[temp_i, temp_j]

        C[i, j] = C_value
        
@cuda.jit
def cuda_gemm_smc_kernel(A, B, C):
    i, j = cuda.grid(2)
    #Define thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    #define block indices
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    #define block limits
    blockDim_x = cuda.blockDim.x
    blockDim_y = cuda.blockDim.y
    #define grid limits
    gridDim_x = cuda.gridDim.x
    gridDim_y = cuda.gridDim.y

    # Define shared memory for caching
    TILE_DIM = 16
    TILE_SIZE = TILE_DIM * TILE_DIM
    
    #Copy into shared memory
    A_smem = cuda.shared.array((TILE_DIM, TILE_DIM), float32)
    B_smem = cuda.shared.array((TILE_DIM, TILE_DIM), float32)

    C_value = 0.0

    for tile in range(gridDim_x):
        # Cache A and B tiles into shared memory
        x, y = tx, ty
        
        #Boundary checking for threads across x
        if i < A.shape[0] and (tile * TILE_DIM + ty) < A.shape[1]:
            A_smem[tx, ty] = A[i, tile * TILE_DIM + ty]
        else:
            A_smem[tx, ty] = 0.0
            
        #Boundary checking for threads across y   
        if j < B.shape[1] and (tile * TILE_DIM + tx) < B.shape[0]:
            B_smem[tx, ty] = B[tile * TILE_DIM + tx, j]
        else:
            B_smem[tx, ty] = 0.0

        #As with general memory coalescing, sync threads for each warp
        cuda.syncthreads()

        #Final value calculation
        for k in range(TILE_DIM):
            C_value += A_smem[x, k] * B_smem[k, y]
        
        #Synchronize threads again for resultant
        cuda.syncthreads()
    
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = C_value


@cuda.jit
def cuda_gemm_vec_kernel(A, B, C):
    i, j = cuda.grid(2)
    #Define thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    #define block indices
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    #define block limits
    blockDim_x = cuda.blockDim.x
    blockDim_y = cuda.blockDim.y
    #define grid limits
    gridDim_x = cuda.gridDim.x
    gridDim_y = cuda.gridDim.y

    # Define shared memory for caching
    TILE_DIM = 16
    TILE_SIZE = TILE_DIM * TILE_DIM
     #Copy into shared memory
    A_smem = cuda.shared.array((TILE_DIM, TILE_DIM), float32)
    B_smem = cuda.shared.array((TILE_DIM, TILE_DIM), float32)

    # Use float4 for vectorized memory access, double2 is also supported
    float4_type = float32[:4]
    A_smem_vec = cuda.local.array((TILE_DIM // 4, TILE_DIM), float4_type)
    B_smem_vec = cuda.local.array((TILE_DIM, TILE_DIM // 4), float4_type)

    C_value = 0.0

    for tile in range(gridDim_x):
        x, y = tx, ty
        
        #Boundary checking for threads across y
        if i < A.shape[0] and (tile * TILE_DIM + ty) < A.shape[1]:
            A_smem[tx, ty] = A[i, tile * TILE_DIM + ty]
        else:
            A_smem[tx, ty] = 0.0
        
            #Boundary checking for threads across x
        if j < B.shape[1] and (tile * TILE_DIM + tx) < B.shape[0]:
            B_smem[tx, ty] = B[tile * TILE_DIM + tx, j]
        else:
            B_smem[tx, ty] = 0.0

        # Vectorize memory access from A_smem and B_smem
        for k in range(0, TILE_DIM, 4):
            A_smem_vec[x // 4, k] = A_smem[tx, k:k+4]
            B_smem_vec[k, y // 4] = B_smem[k:k+4, ty]
            
        #sync threads for each warp
        cuda.syncthreads()
        
        #Access each value in the vectorized memory and compute
        for k in range(TILE_DIM // 4):
            C_value += math.fmaf(A_smem_vec[k, ty // 4], B_smem_vec[tx // 4, k], C_value)

        cuda.syncthreads()

    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = C_value

#cupy blas doesn't require decorator on overall function call or an explicit kernel setup, called directly in Gemm_CUDA.py
#@cuda.jit
