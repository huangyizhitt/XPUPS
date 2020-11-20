#include <cuda_runtime.h>
#include "mfworker.h"

namespace MF{

bool MFWorker::InitGPUAffinity()
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if(deviceCount == 0) {
		printf("The system don't have any GPU!\n");
		return false;
	}

	if(xpu->gpu_dev >= deviceCount) {
		printf("The gpu_dev beyond total gpus!\n");
		return false;
	}

	cudaSetDevice(xpu->gpu_dev);
	return true;
}

void MFWorker::PrepareGPUResources()
{
	size_t size_p = m * k;
	size_t size_q = n * k;

#ifdef CAL_PORTION_RMSE
	cudaMalloc(&gpu_feature, (size_p + size_q + 1) * sizeof(float));
#else
	cudaMalloc(&gpu_feature, (size_p + size_q) * sizeof(float));
#endif
	gpu_p = gpu_feature;
	gpu_q = gpu_feature + size_p;

	cudaMalloc(&gpuR, size * sizeof(MatrixNode));
}

void MFWorker::ReleaseGPUResources()
{
	cudaFree(gpu_feature);
	cudaFree(gpuR);
}

void MFWorker::PullGPUData()
{
	MatrixNode *cpuR = dm.data.r_matrix.data();
	cudaMemcpy(gpuR, cpuR, sizeof(MatrixNode) * size, cudaMemcpyHostToDevice);
}

}
