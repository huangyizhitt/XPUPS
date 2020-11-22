#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
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

#ifdef CAL_PORTION_RMSE
__global__ void sgd_k128_kernel_hogwild_warp32_loss(
							const MatrixNode *R,
							size_t size,
							curandState *state,
							float *loss;
							float *p,
							float *q,
							int k,
							int update_count,
							int update_vector_size,
							float lrate,
							float lambda_p,
							float lambda_q
							)
{
	for(int update_ite = 0; update_ite < update_count; update_ite++)
	{

		int lane_id = threadIdx.x%32;
		int local_wid = threadIdx.x/32;
		int wid = 4*blockIdx.x + local_wid; 

		long long start_id = 0;
		if(lane_id == 0)
		{
			long long origin = (long long)(curand_uniform(&state[wid])*size);
			start_id = origin%size;
				//start_id == 0;
		}

		start_id = __shfl_sync(__activemask(), start_id, 0);
			//start_id = __shfl(start_id, 0);
			
		for(int i = 0;i < update_vector_size;i++)
		{
			int offset = (start_id + i)%size;
				
			float r = __ldg(&R[offset].r);
			int u = __ldg(&R[offset].row_index);
			int v = __ldg(&R[offset].col_index);
		
				//read the p & q into register				   
			int base_p = u*k;
			int base_q = v*k;

			float tmp_p1 = (p[base_p + lane_id]);
			float tmp_q1 = (q[base_q + lane_id]);
			
			float tmp_p2 = (p[base_p + lane_id + 32]);
			float tmp_q2 = (q[base_q + lane_id + 32]);
			
			float tmp_p3 = (p[base_p + lane_id + 64]);
			float tmp_q3 = (q[base_q + lane_id + 64]);
			
			float tmp_p4 = (p[base_p + lane_id + 96]);
			float tmp_q4 = (q[base_q + lane_id + 96]);

			float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;

				//get dot product.
			tmp_product += __shfl_down_sync(__activemask(), tmp_product, 16);
			tmp_product += __shfl_down_sync(__activemask(), tmp_product, 8);
			tmp_product += __shfl_down_sync(__activemask(), tmp_product, 4);
			tmp_product += __shfl_down_sync(__activemask(), tmp_product, 2);
			tmp_product += __shfl_down_sync(__activemask(), tmp_product, 1);

//				  tmp_product = __shfl(tmp_product,0);
			tmp_product = __shfl_sync(__activemask(), tmp_product, 0);

			float ruv = r - tmp_product;
			loss[offset] = ruv;

				//update
				//only works for k=blockDim.x=128
			p[base_p + lane_id +  0] = (tmp_p1 + lrate*(ruv*tmp_q1 - lambda_p*tmp_p1));
			q[base_q + lane_id +  0] = (tmp_q1 + lrate*(ruv*tmp_p1 - lambda_q*tmp_q1));

			p[base_p + lane_id + 32] = (tmp_p2 + lrate*(ruv*tmp_q2 - lambda_p*tmp_p2));
			q[base_q + lane_id + 32] = (tmp_q2 + lrate*(ruv*tmp_p2 - lambda_q*tmp_q2));

			p[base_p + lane_id + 64] = (tmp_p3 + lrate*(ruv*tmp_q3 - lambda_p*tmp_p3));
			q[base_q + lane_id + 64] = (tmp_q3 + lrate*(ruv*tmp_p3 - lambda_q*tmp_q3));

			p[base_p + lane_id + 96] = (tmp_p4 + lrate*(ruv*tmp_q4 - lambda_p*tmp_p4));
			q[base_q + lane_id + 96] = (tmp_q4 + lrate*(ruv*tmp_p4 - lambda_q*tmp_q4)); 
		}	 
	}
}
#else
__global__ void sgd_k128_kernel_hogwild_warp32(
							const MatrixNode *R,
							size_t size,
							curandState *state,
							float *p,
							float *q,
							int k,
							int update_count,
							int update_vector_size,
							float lrate,
							float lambda_p,
							float lambda_q
							)
{
	for(int update_ite = 0; update_ite < update_count; update_ite++)
	{

		int lane_id = threadIdx.x%32;
		int local_wid = threadIdx.x/32;
		int wid = 4*blockIdx.x + local_wid; 

		long long start_id = 0;
		if(lane_id == 0)
		{
			long long origin = (long long)(curand_uniform(&state[wid])*size);
			start_id = origin%size;
				//start_id == 0;
		}

		start_id = __shfl_sync(__activemask(), start_id, 0);
			//start_id = __shfl(start_id, 0);
			
		for(int i = 0;i < update_vector_size;i++)
		{
			int offset = (start_id + i)%size;
				
			float r = __ldg(&R[offset].r);
			int u = __ldg(&R[offset].row_index);
			int v = __ldg(&R[offset].col_index);
		
				//read the p & q into register				   
			int base_p = u*k;
			int base_q = v*k;

			float tmp_p1 = (p[base_p + lane_id]);
			float tmp_q1 = (q[base_q + lane_id]);
			
			float tmp_p2 = (p[base_p + lane_id + 32]);
			float tmp_q2 = (q[base_q + lane_id + 32]);
			
			float tmp_p3 = (p[base_p + lane_id + 64]);
			float tmp_q3 = (q[base_q + lane_id + 64]);
			
			float tmp_p4 = (p[base_p + lane_id + 96]);
			float tmp_q4 = (q[base_q + lane_id + 96]);

			float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;

				//get dot product.
			tmp_product += __shfl_down_sync(__activemask(), tmp_product, 16);
			tmp_product += __shfl_down_sync(__activemask(), tmp_product, 8);
			tmp_product += __shfl_down_sync(__activemask(), tmp_product, 4);
			tmp_product += __shfl_down_sync(__activemask(), tmp_product, 2);
			tmp_product += __shfl_down_sync(__activemask(), tmp_product, 1);

//				  tmp_product = __shfl(tmp_product,0);
			tmp_product = __shfl_sync(__activemask(), tmp_product, 0);

			float ruv = r - tmp_product;

				//update
				//only works for k=blockDim.x=128
			p[base_p + lane_id +  0] = (tmp_p1 + lrate*(ruv*tmp_q1 - lambda_p*tmp_p1));
			q[base_q + lane_id +  0] = (tmp_q1 + lrate*(ruv*tmp_p1 - lambda_q*tmp_q1));

			p[base_p + lane_id + 32] = (tmp_p2 + lrate*(ruv*tmp_q2 - lambda_p*tmp_p2));
			q[base_q + lane_id + 32] = (tmp_q2 + lrate*(ruv*tmp_p2 - lambda_q*tmp_q2));

			p[base_p + lane_id + 64] = (tmp_p3 + lrate*(ruv*tmp_q3 - lambda_p*tmp_p3));
			q[base_q + lane_id + 64] = (tmp_q3 + lrate*(ruv*tmp_p3 - lambda_q*tmp_q3));

			p[base_p + lane_id + 96] = (tmp_p4 + lrate*(ruv*tmp_q4 - lambda_p*tmp_p4));
			q[base_q + lane_id + 96] = (tmp_q4 + lrate*(ruv*tmp_p4 - lambda_q*tmp_q4)); 
		}	 
	}
}
#endif

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
