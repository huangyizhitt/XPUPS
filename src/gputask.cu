#include "task.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <numeric>

namespace global {
extern std::vector<cudaStream_t> streams;
}

namespace MF {

#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

curandState *rand_state;
static int update_vector_size = 128;
static int update_count;

__global__ void init_rand_state(curandState*state, int size)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < size)curand_init(clock() + tid,tid,0,&state[tid]);
}

void InitGPUTask(int gpu_workers, int stream)
{
	cudaMalloc(&rand_state, sizeof(curandState)*gpu_workers);
	gpuErr(cudaPeekAtLastError());
	if(stream == -1)
		init_rand_state<<<((gpu_workers+255)/256),256>>>(rand_state,gpu_workers);
	else
		init_rand_state<<<((gpu_workers+255)/256),256,0,global::streams[stream]>>>(rand_state,gpu_workers);
	gpuErr(cudaPeekAtLastError());
}

void DeInitGPUTask()
{
	cudaFree(rand_state);
}

__global__ void print_head(const MatrixNode *R, float *p, float *q)
{
	for(int i = 0; i < 5; i++) {
		printf("r[%d]: %.7f, p[%d]: %.7f, q[%d]: %.7f\n", i, R[i].r, i, p[i], i, q[i]);
	}
}

__global__ void print_tail(float *p, size_t size_p, float *q, size_t size_q)
{
	for(int i = 0; i < 5; i++) {
		int p_i = size_p * 128 - i - 1;
		int q_i = size_q * 128 - i - 1;
		printf("p[%d]: %.7f, q[%d]: %.7f\n", p_i, p[p_i], q_i, q[q_i]);
	}
}

__global__ void print_loss(float *gpu_loss, size_t size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	printf("gpu_loss[%d]: %.7f\n", i, gpu_loss[i]);
}

void g_print_tail(float *p, size_t size_p, float *q, size_t size_q)
{
	print_tail<<<1, 1>>>(p, size_p, q, size_q);
}

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

			float tmp_product = 0.0;
			float tmp_p[32], tmp_q[32];

			for(int j = 0; j < k; j+=32) {
				int index = j/32;
				tmp_p[index] = (p[base_p + lane_id + j]);
				tmp_q[index] = (q[base_q + lane_id + j]);

				tmp_product += tmp_p[index] * tmp_q[index];
			}

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
			for(int j = 0; j < k; j+=32) {
					const int index = j/32;
					p[base_p + lane_id + j] = (tmp_p[index] + lrate*(ruv*tmp_q[index] - lambda_p*tmp_p[index]));
					q[base_q + lane_id + j] = (tmp_q[index] + lrate*(ruv*tmp_p[index] - lambda_q*tmp_q[index]));
			}
		}	 
	}
}

__global__ void sgd_k128_kernel_hogwild_warp32(
							const MatrixNode *R,
							size_t size,
							curandState *state,
							float *gpu_loss,
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
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	gpu_loss[tid] = 0.0;
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

			float tmp_product = 0.0;
			float tmp_p[32], tmp_q[32];

			for(int j = 0; j < k; j+=32) {
				int index = j/32;
				tmp_p[index] = (p[base_p + lane_id + j]);
				tmp_q[index] = (q[base_q + lane_id + j]);

				tmp_product += tmp_p[index] * tmp_q[index];
			}

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
			for(int j = 0; j < k; j+=32) {
					const int index = j/32;
					p[base_p + lane_id + j] = (tmp_p[index] + lrate*(ruv*tmp_q[index] - lambda_p*tmp_p[index]));
					q[base_q + lane_id + j] = (tmp_q[index] + lrate*(ruv*tmp_p[index] - lambda_q*tmp_q[index]));
			} 

			if(lane_id == 0)
				gpu_loss[tid] += ruv * ruv;
		}	 
	}
}

__global__ void gpu_calc_rmse(
							const MatrixNode *R,
							size_t size,
							curandState *state,
							float *gpu_loss,
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
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	gpu_loss[tid] = 0.0;
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
			//only lane_id = 0 will record
			if(lane_id == 0)
				gpu_loss[tid] += ruv * ruv;
		}
	}
}
							
void *sgd_update_k128_gpu(void *args)
{
	Args *para = (Args *)args;

	MatrixNode *R = (MatrixNode *)para->data;
	int gpu_workers = para->workers;
	size_t size = para->size;
	static double start, elapse = 0;
/*	if(global::current_epoch == 1) {
		cudaMalloc(&rand_state, sizeof(curandState)*gpu_workers);
		gpuErr(cudaPeekAtLastError());
		init_rand_state<<<((gpu_workers+255)/256),256>>>(rand_state,gpu_workers);
		gpuErr(cudaPeekAtLastError());

		update_count = (ceil)(1.0 * size / (gpu_workers*update_vector_size));
	}*/
	update_count = (ceil)(1.0 * size / (gpu_workers*update_vector_size));
#ifdef CAL_PORTION_RMSE	
	float *loss = para->loss;
	float *gpu_loss = para->gpu_loss;

//	printf("current_epoch: %d, gpu_workers: %d, R: %p, size: %ld, update_count: %d, update_vector_size: %d, p: %p, q: %p, gpu_loss: %p, rand_state: %p, stream: %d\n", global::current_epoch, gpu_workers, R, size, update_count, update_vector_size, para->p, para->q, gpu_loss, rand_state, para->stream);
	if(para->stream == -1) {
		sgd_k128_kernel_hogwild_warp32<<<gpu_workers/4, 128>>>(R, size, rand_state, gpu_loss, para->p, para->q, para->k, update_count,
									update_vector_size, para->lrate, para->lambda_p, para->lambda_q);
		gpuErr(cudaPeekAtLastError());
        	cudaDeviceSynchronize();
		cudaMemcpy(loss, gpu_loss, (gpu_workers * 32) * sizeof(float), cudaMemcpyDeviceToHost);
	} else {
		cudaStream_t stream = global::streams[para->stream];
		sgd_k128_kernel_hogwild_warp32<<<gpu_workers/4, 128, 0, stream>>>(R, size, rand_state, gpu_loss, para->p, para->q, para->k, update_count,
									update_vector_size, para->lrate, para->lambda_p, para->lambda_q);
//		cudaDeviceSynchronize();
		cudaMemcpyAsync(loss, gpu_loss, (gpu_workers * 32) * sizeof(float), cudaMemcpyDeviceToHost, stream);
//		cudaMemcpy(loss, gpu_loss, (gpu_workers * 32) * sizeof(float), cudaMemcpyDeviceToHost);
//		print_loss<<<gpu_workers/4, 128>>>(gpu_loss, (gpu_workers * 32));
	}
	
//	gpu_calc_rmse<<<gpu_workers/4, 128>>>(R, size, rand_state, gpu_loss, para->p, para->q, 128, update_count,
//									update_vector_size, para->lrate, para->lambda_p, para->lambda_q);
//	gpuErr(cudaPeekAtLastError());
//	cudaDeviceSynchronize();
#else
	if(para->stream == -1) {
		sgd_k128_kernel_hogwild_warp32<<<gpu_workers/4, 128>>>(R, size, rand_state, para->p, para->q, para->k, update_count,
									update_vector_size, para->lrate, para->lambda_p, para->lambda_q);
		gpuErr(cudaPeekAtLastError());
		cudaDeviceSynchronize();
	} else {
		cudaStream_t stream = global::streams[para->stream];
		sgd_k128_kernel_hogwild_warp32<<<gpu_workers/4, 128, 0, stream>>>(R, size, rand_state, para->p, para->q, para->k, update_count,
									update_vector_size, para->lrate, para->lambda_p, para->lambda_q);
	}
#endif

/*	if(global::current_epoch == global::target_epoch) {
		cudaFree(rand_state);
	}*/
	return NULL;
}

}
