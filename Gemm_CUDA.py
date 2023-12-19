#Alexandru Barsan 2023
#NOTE this will not work properly unless you first installed Anaconda for MKL support, have an Nvidia GPU with CUDA compute compatibility
#PLEASE SEE THE README
#Import all required libraries
import numpy as np
import pandas
from numba import cuda, float32, jit #Used for the actual kernel calls and kernels
from cuda import cuda as cupy #Not to be confused for actual cupy.. cuda python shortening, also strictly to get some lower-level attributes from the card
import cupy as cp #Just for cuBLAS
import sys
import time
import CUDAKernels as ck

#Set up the overall grid limits
def cuda_gemm(A, B, threadsperblock, k_type):

    m, n = A.shape[0], B.shape[1]
    #Send each array to device memory
    A_global = cuda.to_device(A)
    B_global = cuda.to_device(B)
    C_global = cuda.device_array((m, n), np.float32)

    #Determine the tiling for the GPU threads
    blockspergrid_x = (m + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)


    if k_type == "Naive":
        start = time.time()
        #Initiate the kernel call with the request blocks
        ck.cuda_gemm_kernel[blockspergrid, threadsperblock](A_global, B_global, C_global)
        end = time.time()
        print(f"Time for naive kernel: {end-start}")
    elif k_type == "Global Memory Coalescing":
        start = time.time()
        #Initiate the kernel call with the request blocks
        ck.cuda_gmc_gemm_kernel[blockspergrid, threadsperblock](A_global, B_global, C_global)
        end = time.time()
        print(f"Time for GMC kernel: {end-start}")
    elif k_type == "Shared Memory Caching":
        start = time.time()
        #Initiate the kernel call with the request blocks
        ck.cuda_gemm_smc_kernel[blockspergrid, threadsperblock](A_global, B_global, C_global)
        end = time.time()
        print(f"Time for SMC kernel: {end-start}")
    elif k_type == "Vectorized":
        start = time.time()
        #Initiate the kernel call with the request blocks
        ck.cuda_gemm_kernel[blockspergrid, threadsperblock](A_global, B_global, C_global)
        end = time.time()
        print(f"Time for Vectorized kernel: {end-start}")

    #print(C_global.shape)
    #cuda.synchronize()

    #Copy result back from GPU memory to CPU memory
    C = C_global.copy_to_host()


    return C

def naive_matrix_mul(A, B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C


#Naive GEMM using numba JIT
@jit(nopython=True)
def naive_matrix_mul_numba(A, B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C

#Naive GEMM using numba JIT and loop reordering
@jit(nopython=True)
def ikj_matrix_mul_numba(A,B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C

#MKL numpy matrix multiplication
def mkl_gemm(A,B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    #MKL is linked to NumPy already, can just do a dot product directly.
    C = np.dot(A,B)

    return C

#Numba JIT definitely doesn't like this.
#@jit(nopython=True)
#def jit_mkl_gemm(A,B):
#    assert A.shape[1] == B.shape[0]
#    C = np.zeros((A.shape[0], B.shape[1]))
#    #MKL is linked to NumPy already, can just do a dot product directly.
#    C = np.dot(A,B)
#
#    return C

#cuBLAS kernel call
def cublas_gemm(A,B):
    #CuPy CuBLAS handles the CUDA grid configuration and other optimizations implicitly
    m, k = A.shape
    k, n = B.shape
    #Allocate GPU memory, ensure it is float32 for cuBLAS... pull it out into the Gemm_CUDA call to have setup outside of the main blas anyway.
    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)
    C_gpu = cp.zeros((m, n), dtype=cp.float32)

    # Perform GEMM using cuBLAS
    C_gpu = cp.matmul(A_gpu, B_gpu)

    #Transfer the result back to the host (CPU)
    C_cpu_result = cp.asnumpy(C_gpu)

    #Since everything will be passed in, return just C after changing
    return C_cpu_result


def save_trial(trialTimes):

    excel_filename = "trials.xlsx"

    #Check if the file exists first
    try:
        df = pandas.read_excel(excel_filename)
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame, expand with more times as required
        df = pandas.DataFrame(columns=['Trial Name', 'Naive Time', 'Naive Time Numba', 'ikj Time Numba', 'CUDA Time naive', 'CUDA Time Global Memory Coalescing', 'CUDA Time Shared Memory Caching', 'CUDA Time Vectorized', 'MKL Time', 'CuBLAS Time'])

        # Get the current trial name from the last row in the DataFrame (if it exists)
    if not df.empty:
        last_trial = df.iloc[-1]['Trial Name']
        trial_number = int(last_trial.split()[-1])
        trial_name = f'Trial {trial_number + 1}'
    else:
        trial_name = 'Trial 1'

       #Set up the data frame with the passed in times
    new_data = pandas.DataFrame(
        {
        'Trial Name' : [trial_name],
        'Naive Time' : [trialTimes[0]],
        'Naive Time Numba' : [trialTimes[1]],
        'ikj Time Numba' : [trialTimes[2]],
        'CUDA Time naive' : [trialTimes[3]],
        'CUDA Time Global Memory Coalescing' : [trialTimes[4]],
        'CUDA Time Shared Memory Caching' : [trialTimes[5]],
        'CUDA Time Vectorized' : [trialTimes[6]],
        'MKL Time' : [trialTimes[7]],
        'CuBLAS Time' : [trialTimes[8]]
        })


    #Append the data frame to excel as a new row
    df = pandas.concat([df, new_data], ignore_index=True)

    # Save the updated DataFrame to the Excel file
    df.to_excel(excel_filename, index=False)

    print(f'Data saved to {excel_filename}')

def CudaInfo():
    cuda.detect()
    # Initialize CUDA Driver API
    (err,) = cupy.cuInit(0)

    # Get attributes
    err, DEVICE_NAME = cupy.cuDeviceGetName(128, 0)
    DEVICE_NAME = DEVICE_NAME.decode("ascii").replace("\x00", "")

    err, MAX_THREADS_PER_BLOCK = cupy.cuDeviceGetAttribute(
        cupy.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0
    )
    err, MAX_BLOCK_DIM_X = cupy.cuDeviceGetAttribute(
        cupy.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0
    )
    err, MAX_GRID_DIM_X = cupy.cuDeviceGetAttribute(
        cupy.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 0
    )
    err, SMs = cupy.cuDeviceGetAttribute(
        cupy.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0
    )
    err, SHR_MEM = cupy.cuDeviceGetAttribute(
        cupy.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 0
    )
    err, WARPSIZE = cupy.cuDeviceGetAttribute(
        cupy.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0
    )
    err, L2Size = cupy.cuDeviceGetAttribute(
        cupy.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0
    )


    print(f"Device Name: {DEVICE_NAME}")
    print(f"Maximum number of multiprocessors: {SMs}")
    print(f"Maximum number of threads per block: {MAX_THREADS_PER_BLOCK:10}")
    print(f"Maximum number of blocks per grid:   {MAX_BLOCK_DIM_X:10}")
    print(f"Maximum number of threads per grid:  {MAX_GRID_DIM_X:10}")
    print(f"Maximum shared memory per block:  {SHR_MEM} bytes")
    print(f"Warp Size:  {WARPSIZE}")
    print(f"L2 Cache Size:  {L2Size} bytes")



def main():
#Default to 256x256 arrays by default unless arguments are passed in
    if len(sys.argv) != 4:
        matrix_size = 256
        print(f"Defaulting to NxN {matrix_size} sized matrices")
        print("To change the size of the matrices to be multiplied, provide it as arguments (i.e. python Gemm_CUDA.py 4096 4096 4096")
        #Cast to int
        m, n, k = int(matrix_size), int(matrix_size), int(matrix_size)
        print(f"Running with {matrix_size} sized matrices")
    else:
        m, n, k = sys.argv[1:]
        m, n, k = int(m), int(n), int(k)
        print(f"Running with A as {m}x{n} and B as {n}x{k} sized matrices")

    CudaInfo()
    #See if Numpy is properly linked with MKL
    print(np.show_config())

    #Initialize values
    #Threads per block of operations, good to be a multiple of 32 according to programming guide, maximum of 32,32 since 32*32 = 1024, as per device info
    threadsperblock = (16, 16)
    num_runs = 10

    #Randomized initial matrices, numba CUDA works with numpy arrays
    A = np.random.rand(m, n).astype(np.float32)
    B = np.random.rand(n, k).astype(np.float32)
    print("-----------------")

    #Run against the GPU naively
    start = time.time()
    for _ in range(num_runs):
        result = cuda_gemm(A, B, threadsperblock, "Naive")
    end = time.time()
    cuda_time = end - start
    print("-----------------")

    #Run against the GPU with global memory coalesced access
    start = time.time()
    for _ in range(num_runs):
        result = cuda_gemm(A, B, threadsperblock, "Global Memory Coalescing")
    end = time.time()
    cuda_gmc_time = end - start
    print("-----------------")

    #Run against the GPU with shared memory caching
    start = time.time()
    for _ in range(num_runs):
        result = cuda_gemm(A, B, threadsperblock, "Shared Memory Caching")
    end = time.time()
    cuda_smc_time = end - start
    print("-----------------")

    #Run against the GPU with vectorization
    start = time.time()
    for _ in range(num_runs):
        result = cuda_gemm(A, B, threadsperblock, "Vectorized")
    end = time.time()
    cuda_vec_time = end - start
    print("-----------------")


    print("Naive implementation")
     #Naive GEMM
    start = time.time()
    for _ in range(num_runs):
        #print(_)
        naive_matrix_mul(A, B)
    end = time.time()
    naive_time = end - start

    print("Naive implementation with JIT")
    #naive matrix mult with numba
    start = time.time()
    for _ in range(num_runs):
        #print(_)
        naive_matrix_mul_numba(A, B)
    end = time.time()
    naive_time_numba = end - start
    print("Naive implementation with JIT and loop reordering")
    #ikj matrix mult with numba
    start = time.time()
    for _ in range(num_runs):
        #print(_)
        ikj_matrix_mul_numba(A, B)
    end = time.time()
    ikj_time_numba = end - start

    print("MKL implementation")
   #MKL gemm
    start = time.time()
    for _ in range(num_runs):
        #print(_)
        mkl_gemm(A, B)
    end = time.time()
    mkl_gemm_time = end - start

    #Not possible, wanted to see what would happen.
    #MKL JIT gemm
    #start = time.time()
    #for _ in range(num_runs):
        #print(_)
    #    jit_mkl_gemm(A, B)
    #end = time.time()
    #jit_mkl_gemm_time = end - start

    print("CuBLAS implementation")
    #CuBLAS gemm
    start = time.time()
    for _ in range(num_runs):
        #print(_)
        cublas_gemm(A, B)
    end = time.time()
    cublas_gemm_time = end - start

    #Save an array of the trial run
    trialTimes = [naive_time,naive_time_numba,ikj_time_numba,cuda_time, cuda_gmc_time, cuda_smc_time, cuda_vec_time, mkl_gemm_time, cublas_gemm_time]
    #Call the save to excel function, this will save on every successful run through, comment out to just play around
    #save_trial(trialTimes)

    #Overall time
    print('naive time: {}'.format(naive_time))
    print('naive time numba: {}'.format(naive_time_numba))
    print('ikj time numba: {}'.format(ikj_time_numba))
    print('CUDA time : {}'.format(cuda_time))
    print('CUDA GMC time : {}'.format(cuda_gmc_time))
    print('CUDA SMC time : {}'.format(cuda_smc_time))
    print('CUDA Vec time : {}'.format(cuda_vec_time))
    print('MKL Time : {}'.format(mkl_gemm_time))
    #print('MKL JIT Time : {}'.format(jit_mkl_gemm_time))
    print('CuBLAS Time : {}'.format(cublas_gemm_time))


main()



