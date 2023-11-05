import numpy as np
import pandas
from numba import cuda, float32, jit
import sys
import time
import CUDAKernels as ck

def cuda_gemm(A, B, threadsperblock):

    m, n = A.shape[0], B.shape[1]
    #Send each array to device memory
    A_global = cuda.to_device(A)
    B_global = cuda.to_device(B)
    C_global = cuda.device_array((m, n))

    #Determine the tiling for the GPU threads
    blockspergrid_x = (m + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    #Initiate the kernel call with the request blocks
    ck.cuda_gemm_kernel[blockspergrid, threadsperblock](A_global, B_global, C_global)
    #Copy result back from GPU memory to CPU memory
    C = C_global.copy_to_host()

    #TODO: Add other metrics to output
    #TODO: Dig for more info/optimization potentially

    return C

def naive_matrix_mul(A, B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C


#Naive GEMM using numba JIT as lab 3
@jit(nopython=True)
def naive_matrix_mul_numba(A, B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C

#Naive GEMM using numba JIT and loop reordering as lab 3
@jit(nopython=True)
def ikj_matrix_mul_numba(A,B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C

def save_trial(trialTimes):
    
    excel_filename = "trials.xlsx"
    
    #Check if the file exists first
    try:
        df = pandas.read_excel(excel_filename)
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame, expand with more times as required
        df = pandas.DataFrame(columns=['Trial Name', 'Naive Time', 'Naive Time Numba', 'ikj Time Numba', 'CUDA Time Numba naive'])
    
        # Get the current trial name from the last row in the DataFrame (if it exists)
    if not df.empty:
        last_trial = df.iloc[-1]['Trial Name']
        trial_number = int(last_trial.split()[-1])
        trial_name = f'Trial {trial_number + 1}'
    else:
        trial_name = 'Trial 1'   
      
       #Set up the data frame with the passed in times
    new_data = {
        'Trial Name' : trial_name,
        'Naive Time' : trialTimes[0], 
        'Naive Time Numba' : trialTimes[1], 
        'ikj Time Numba' : trialTimes[2], 
        'CUDA Time Numba naive' : trialTimes[3]
    }
    
    #Append the data frame to excel as a new row
    df.append(new_data, ignore_index=True)

    # Save the updated DataFrame to the Excel file
    df.to_excel(excel_filename, index=False)

    print(f'Data saved to {excel_filename}')


def main():
#Taken from lab 3 and played with a bit
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
    #TODO: Run trials and see which is a happy number
    threadsperblock = (16, 16)
    num_runs = 10
    #Randomized initial matrices, numba CUDA works with numpy arrays
    A = np.random.rand(m, n).astype(np.float32)
    B = np.random.rand(n, k).astype(np.float32)

    #Run against the GPU
    start = time.time()
    for _ in range(num_runs):
        result = cuda_gemm(A, B, threadsperblock)
    end = time.time()
    cuda_time_numba = end - start

    #Naive GEMM
    start = time.time()
    for _ in range(num_runs):
        naive_matrix_mul(A, B)
    end = time.time()
    naive_time = end - start


    #naive matrix mult with numba
    start = time.time()
    for _ in range(num_runs):
        naive_matrix_mul_numba(A, B)
    end = time.time()
    naive_time_numba = end - start

    #ikj matrix mult with numba
    start = time.time()
    for _ in range(num_runs):
        ikj_matrix_mul_numba(A, B)
    end = time.time()
    ikj_time_numba = end - start

    #Overall time
    print('naive time: {}'.format(naive_time))
    print('naive time numba: {}'.format(naive_time_numba))
    print('ikj time numba: {}'.format(ikj_time_numba))
    print('CUDA time numba: {}'.format(cuda_time_numba))

main()



