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
	loss_size = 32 * workers;
	cudaMalloc(&feature, (size_p + size_q + 1) * sizeof(float));

	if(trans_mode != HALFQ_SHM_ACOPY) {
		cudaMallocHost(&loss, sizeof(float) * loss_size);
		cudaMalloc(&gpu_loss, sizeof(float) * loss_size);
	} else {
		cudaMallocHost(&loss, sizeof(float) * loss_size * xpu->num_streams);
                cudaMalloc(&gpu_loss, sizeof(float) * loss_size * xpu->num_streams);
	}
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
	cudaFreeHost(loss);
#endif
}

void MFWorker::PullGPUData()
{
	MatrixNode *cpuR = dm.data.r_matrix.data();
	cudaMemcpy(gpuR, cpuR, sizeof(MatrixNode) * size, cudaMemcpyHostToDevice);
}

void MFWorker::PinnedBuf(void* buf, size_t size)
{
        cudaHostRegister(buf, size, cudaHostRegisterDefault);
}

void MFWorker::UnpinnedBuf(void *buf)
{
        cuMemHostUnregister(buf);
}
}
