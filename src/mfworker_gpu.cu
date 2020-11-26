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
	cudaMalloc(&gpu_loss, size * sizeof(float));
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


void MFWorker::sgd_update_k128_gpu()
{
	curandState *rand_state;
    cudaMalloc(&rand_state, sizeof(curandState)*core_num);
	gpuErr(cudaPeekAtLastError());

	init_rand_state<<<((core_num+255)/256),256>>>(rand_state,core_num);
	gpuErr(cudaPeekAtLastError());

	int update_vector_size = 128;
	int update_count = (ceil)(1.0 * size / (core_num*update_vector_size));

#ifdef CAL_PORTION_RMSE
	sgd_k128_kernel_hogwild_warp32_loss<<<core_num/4, 128>>>(gpuR, start, size, rand_state, gpu_loss, 
						p, q, k, update_count, update_vector_size, lrate, lambda_p, lambda_q);
	cudaMemcpy(loss, gpu_loss, size * sizeof(float), cudaMemcpyDeviceToHost);
#else
	sgd_k128_kernel_hogwild_warp32<<<core_num/4, 128>>>(gpuR, start, size, rand_state,
						p, q, k, update_count, update_vector_size, lrate, lambda_p, lambda_q);
#endif

	gpuErr(cudaPeekAtLastError());
	cudaDeviceSynchronize();
	cudaFree(rand_state);
}

}
