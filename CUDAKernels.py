import numpy as np
from numba import cuda, float32, jit
import sys
import time


#Module containing the CUDA kernels

#Kernel function for GPU
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