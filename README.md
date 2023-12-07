# GemmCUDA
## Alexandru Barsan 2023 for ECE 718


# Machine Specifications  

CPU: Intel i7-7700K  
GPU: Nvidia Geforce GTX 1080 Ti  
RAM: DDR4 16 GB Memory size 1200 MHz DRAM Freq  

Found 1 CUDA devices  
id 0    b'NVIDIA GeForce GTX 1080 Ti'                              [SUPPORTED]  
                      Compute Capability: 6.1  
                           PCI Device ID: 0  
                              PCI Bus ID: 1  
                                    UUID: GPU-d9ff490e-0dae-b9ff-d914-d72433dae551  
                                Watchdog: Enabled  
                            Compute Mode: WDDM  
             FP32/FP64 Performance Ratio: 32  
Summary:  
        1/1 devices are supported   
Device Name: NVIDIA GeForce GTX 1080 Ti                                                                                   
Maximum number of multiprocessors: 28  
Maximum number of threads per block:       1024  
Maximum number of blocks per grid:         1024  
Maximum number of threads per grid:  2147483647  
Maximum shared memory per block:  49152 bytes  
Warp Size:  32  
L2 Cache Size:  2883584 bytes  

# Prerequsites:
A NVIDIA CUDA capable graphics card, reference can be found here: https://developer.nvidia.com/cuda-gpus

The NVIDIA CUDA toolkit, can be found here: https://developer.nvidia.com/cuda-toolkit


# The following python libraries:
pip install openpyxl

pip install numpy

pip install numba

pip install pypiwin32

pip install cuda-python

pip install mkl

pip install scipy

pip install pyculib


# Running with different matrix sizes:
e.g. 
python Gemm_CUDA.py 4096 4096 4096  

## Profiling (most useful if running just one kernel)
nvprof python Gemm_CUDA.py 4096 4096 4096  

# Trials
- All trials for Matrix Size will use the naive non-JIT implementation of GEMM
- All trials after 6 will not use the naive non-JIT implementation for the sake of time and sanity (4096 was taking more than 20 hours...)
- All trials will compare against naive GEMM with numba JIT and loop re-ordered GEMM with numba JIT
- Threads Per Block affects all 4 kernels, TILE_DIM affects just shared memory caching and vectorized kernels, Trials 8 to 23 will only compare the GPU kernels
- Profiling done on some interesting trials/kernels for trials 24 - 28

|Trial #	  | Matrix Size | Threads Per Block | TILE_DIM |  Comment |
|-----------|-------------|-------------------|----------|----------|
|1			    |	32		      |			16		        |		16	   |Done|
|2			    |	64		      |			16		        |		16	   |Done|
|3			    |	128		      |			16		        |		16	   |Done|
|4			    |	256		      |			16		        |		16	   |Done|
|5			    |	512		      |			16		  |		16	 |Done|
|6			    |	1024	      |			16		  |		16	 |Done|
|7			    |	4096	      |			16		  |		16	 |Done|
|8			    |	4096	     |			4		    |		4	    |Done|
|9			    |	4096	      |			4		    |		8	    |Done|
|10			    |	4096	  |			4		    |		16	 |Done|
|11			    |	4096	  |			4		    |		32	 |Done|
|12			    |	4096	  |			8		    |		4	 |Done|
|13			    |	4096	  |			8		    |		8	 |Done|
|14			    |	4096	  |			8		    |		16	 |Done|
|15			    |	4096	  |			8		    |		32	 |Done|
|x          |	4096	  |			16		  |		4	 | numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR for shared memory access kernel|
|x	        |	4096	  |			16		  |		8	 | numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR for shared memory access kernel|
|16         |	4096	  |			16		  |		16	 |Done|
|17			    |	4096	  |			16		  |		32	 |Done|
|x          |	4096	  |			32		  |		4	 |numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR for shared memory access kernel|
|x          |	4096	  |			32		  |		8	 |numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR for shared memory access kernel|
|x          |	4096	  |			32		  |		16	 |numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR for shared memory access kernel|
|18	        |	4096	  |			32		|		32	 |Done|
|19			    |	4096	  |			16	|		4	 |Works for vectorized kernel|
|20         |	4096	  |			16		|		8	 |Works for vectorized kernel|
|21 		    |	4096	  |			32	|		4	 |Works for vectorized kernel|
|22         |	4096	  |			32		|	8	 |Works for vectorized kernel|
|23         |	4096	  |			32		|   16	 |Works for vectorized kernel|
|24         |   4096      | 4               | 32     |Same as Trial 11, SM kernel profiled
|25         |   4096      | 8               | 4     |Same as Trial 12, SM kernel profiled
|26         |   4096      | 8               | 32     |Same as Trial 15, SM kernel profiled
|27         |   4096      | 8               | 4     |Same as Trial 12, Vec kernel profiled
|28         |   4096      | 32               | 32     |Same as Trial 18, Vec kernel profiled
|30			    |	32		      |			16		        |		16	   |Re-run for MKL and cuBLAS implementations|
|31		        |	64		      |			16		        |		16	   |Re-run for MKL and cuBLAS implementations|
|32			    |	128		      |			16		        |		16	   |Re-run for MKL and cuBLAS implementations|
|33			    |	256		      |			16		        |		16	   |Re-run for MKL and cuBLAS implementations|
|34			    |	512		      |			16		  |		16	 |Re-run for MKL and cuBLAS implementations|
|35			    |	1024	      |			16		  |		16	 |Re-run for MKL and cuBLAS implementations|
|36			    |	4096	      |			16		  |		16	 |Re-run for MKL and cuBLAS implementations|
--------------------------------------------------------------------------------------
# Memory Access Violation
In-depth call stack for above errors below, using improper tile dimensions for the number of threads per block causes memory access violations due to probably misaligment.
This could be due to misalignment of shared memory accesses that is more forgiving at smaller threads per block and tile dimensions.
Vectorization transpose forces alignment of shared memory and then just iterates through the working set based on tile_dim, probably why we avoid the error.

File "C:\Users\barsana\source\repos\GemmCUDA\Gemm_CUDA.py", line 49, in cuda_gemm
    C = C_global.copy_to_host()
        ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\barsana\AppData\Roaming\Python\Python311\site-packages\numba\cuda\cudadrv\devices.py", line 232, in _require_cuda_context
    return fn(*args, **kws)
           ^^^^^^^^^^^^^^^^
  File "C:\Users\barsana\AppData\Roaming\Python\Python311\site-packages\numba\cuda\cudadrv\devicearray.py", line 277, in copy_to_host
    _driver.device_to_host(hostary, self, self.alloc_size,
  File "C:\Users\barsana\AppData\Roaming\Python\Python311\site-packages\numba\cuda\cudadrv\driver.py", line 3145, in device_to_host
    fn(host_pointer(dst), device_pointer(src), size, *varargs)
  File "C:\Users\barsana\AppData\Roaming\Python\Python311\site-packages\numba\cuda\cudadrv\driver.py", line 327, in safe_cuda_api_call
    self._check_ctypes_error(fname, retcode)
  File "C:\Users\barsana\AppData\Roaming\Python\Python311\site-packages\numba\cuda\cudadrv\driver.py", line 395, in _check_ctypes_error
    raise CudaAPIError(retcode, msg)
numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR
