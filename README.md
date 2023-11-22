# GemmCUDA
# Alexandru Barsan 2023 for ECE 718
TODO: Implement more cuda profiling

# Machine Specifications
CPU: Intel i7-7700K 
GPU: Nvidia Geforce GTX 1080 Ti
RAM: DDR4 16 GB Memory size 1200 MHz DRAM Freq

# Prerequsites:
A NVIDIA CUDA capable graphics card, reference can be found here: https://developer.nvidia.com/cuda-gpus
The NVIDIA CUDA toolkit, can be found here: https://developer.nvidia.com/cuda-toolkit

# The following python libraries:
pip install openpyxl
pip install numpy
pip install numba
pip install pypiwin32
pip install cuda-python

# Running with different matrix sizes:
e.g. python Gemm_CUDA.py 4096 4096 4096

# Trials
- All trials for Matrix Size will use the naive non-JIT implementation of GEMM
- All trials after 6 will not use the naive non-JIT implementation for the sake of time and sanity (4096 was taking more than 20 hours...)
- All trials will compare against naive GEMM with numba JIT and loop re-ordered GEMM with numba JIT
- Threads Per Block affects all 4 kernels, TILE_DIM affects just shared memory caching and vectorized kernels, Trials 8 to 23 will only compare the GPU kernels

|Trial #	| Matrix Size | Threads Per Block | TILE_DIM |
|-----------|-------------|-------------------|----------|
|1			|	32		  |			16		  |		16	 |-
|2			|	64		  |			16		  |		16	 |-
|3			|	128		  |			16		  |		16	 |-
|4			|	256		  |			16		  |		16	 |-
|5			|	512		  |			16		  |		16	 |-
|6			|	1024	  |			16		  |		16	 |-
|7			|	4096	  |			16		  |		16	 |-
|8			|	4096	  |			4		  |		4	 |-
|9			|	4096	  |			4		  |		8	 |-
|10			|	4096	  |			4		  |		16	 |-
|11			|	4096	  |			4		  |		32	 |-
|12			|	4096	  |			8		  |		4	 |-
|13			|	4096	  |			8		  |		8	 |-
|14			|	4096	  |			8		  |		16	 |-
|15			|	4096	  |			8		  |		32	 |-
|16	x		|	4096	  |			16		  |		4	 | numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR
|17	-> 16x	|	4096	  |			16		  |		8	 | numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR
|18	-> 16	|	4096	  |			16		  |		16	 |-
|17			|	4096	  |			16		  |		32	 |-
|18	x		|	4096	  |			32		  |		4	 |numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR
|19	-> 18x	|	4096	  |			32		  |		8	 |numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR
|20 -> 18x	|	4096	  |			32		  |		16	 |numba.cuda.cudadrv.driver.CudaAPIError: [700] Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR
|21	-> 18	|	4096	  |			32		  |		32	 |
----------------------------------------------------------

# In-depth call stack for above errors, using improper tile dimensions for the number of threads per block causes memory access violations.
# This could be due to misalignment of shared memory accesses that is more forgiving at smaller threads per block and tile dimensions.

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