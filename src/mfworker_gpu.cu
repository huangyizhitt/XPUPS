#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "mfworker.h"

namespace MF{

void MFWorker::PrepareGPUResources()
{
	size_t size_p = m * k;
	size_t size_q = n * k;

#ifdef CAL_PORTION_RMSE
	cudaMalloc(&feature, (size_p + size_q + 1) * sizeof(float));
	cudaMalloc(&gpu_loss, workers * sizeof(float) * 32);
#else
	cudaMalloc(&feature, (size_p + size_q) * sizeof(float));
#endif
	p = feature;
	q = feature + size_p;

	cudaMalloc(&gpuR, size * sizeof(MatrixNode));
}

void MFWorker::ReleaseGPUResources()
{
	cudaFree(feature);
	cudaFree(gpuR);

#ifdef CAL_PORTION_RMSE
	cudaFree(gpu_loss);
#endif
}

void MFWorker::PullGPUData()
{
	MatrixNode *cpuR = dm.data.r_matrix.data();
	cudaMemcpy(gpuR, cpuR, sizeof(MatrixNode) * size, cudaMemcpyHostToDevice);
}


}
