#Alexandru Barsan 2023
#This will generate Matlab-like plots that look a little nicer than excel's
#Summarized/investigated data is in Summarized Trials.xlsx
#Import used libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load the data into a dataframe for parsing
trials_path = 'trials.xlsx'
dfTrials = pd.read_excel(trials_path)

#Comparative graphs for naive python implementations and CUDA implementations
#See README.md for a breakdown of the trials
df_Fig1 = dfTrials.iloc[:7,:]
matSize = [32,64,128,256,512,1024,4096,8192,16384]
serNaive = df_Fig1.iloc[:7,1]
serNaiveJIT = df_Fig1.iloc[:7,2]
serLoopReord = df_Fig1.iloc[:7,3]
serCUDABasic = df_Fig1.iloc[:7,4]
serGMC = df_Fig1.iloc[:7,5]
serSMC = df_Fig1.iloc[:7,6]
serVec= df_Fig1.iloc[:7,7]
#print(serVec)

#Compare Naive python series to Naive CUDA
plt.figure()
plt.plot(matSize[0:7], serNaive, 'k-x', label='Naive Python')
plt.plot(matSize[0:7], serNaiveJIT, 'r-x', label='JiT-compiled Python')
plt.plot(matSize[0:7], serLoopReord, 'g-x', label='JiT-compiled Python with loop reordering')
plt.plot(matSize[0:7], serCUDABasic, 'b-x', label='Naive CUDA Kernel')

plt.title('Naive CPU & GPU Comparative')
plt.xlabel('Matrix Size')
plt.ylabel('Log Time (s)')
plt.yscale('log')
plt.xticks(range(0,4100,128))       

plt.legend()
plt.show()

#Compare CUDA Kernels
plt.figure()
plt.plot(matSize[0:7], serCUDABasic, 'k-x', label='Naive CUDA Kernel')
plt.plot(matSize[0:7], serGMC, 'r-x', label='General Memory Coalescing')
plt.plot(matSize[0:7], serSMC, 'g-x', label='Shared Memory Caching')
plt.plot(matSize[0:7], serVec, 'b-x', label='Vectorized Kernel')

plt.title('GPU Kernel Comparative')
plt.xlabel('Matrix Size')
plt.ylabel('Log Time (s)')
plt.yscale('log')
plt.xticks(range(0,4100,128))    

plt.legend()
plt.show()


#Compare Naive Python and CUDA Vectorization to MKL/CuBLAS for <1024
serMKL= dfTrials.iloc[29:39,8]
serCuBLAS= dfTrials.iloc[29:39,9]
serVec = dfTrials.iloc[29:39,7]
#print(serVec)
plt.figure()
plt.plot(matSize[0:6], serNaive[0:6], 'k-x', label='Naive Python')
plt.plot(matSize[0:6], serVec[0:6], 'r-x', label='Vectorized Kernel')
plt.plot(matSize[0:6], serCuBLAS[0:6], 'g-x', label='CuBLAS')
plt.plot(matSize[0:6], serMKL[0:6], 'b-x', label='MKL')

plt.title('Standard Library Comparative - Small Matrices')
plt.xlabel('Matrix Size')
plt.ylabel('Log Time (s)')
plt.yscale('log')
plt.xticks(range(0,1050,32))  

plt.legend()
plt.show()

#Compare CUDA Vec to MKL/CuBLAS for >1024
plt.figure()

plt.plot(matSize[5:9], serVec[5:9], 'r-x', label='Vectorized Kernel')
plt.plot(matSize[5:9], serCuBLAS[5:9], 'g-x', label='CuBLAS')
plt.plot(matSize[5:9], serMKL[5:9], 'b-x', label='MKL')

plt.title('Standard Library Comparative - Larger Matrices')
plt.xlabel('Matrix Size')
plt.ylabel('Log Time (s)')
plt.yscale('log')
plt.xticks(range(1024,16400,1024))

plt.legend()
plt.show()

