#include "task.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

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

static curandState *rand_state;
static int update_vector_size = 128;
static int update_count;

__global__ void init_rand_state(curandState*state, int size)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < size)curand_init(clock() + tid,tid,0,&state[tid]);
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

void *sgd_update_k128_gpu(void *args)
{
	Args *para = (Args *)args;

	MatrixNode *R = (MatrixNode *)para->data;
	int gpu_workers = para->workers;
	size_t size = para->size;

	if(current_epoch == 1) {
		cudaMalloc(&rand_state, sizeof(curandState)*gpu_workers);
		gpuErr(cudaPeekAtLastError());
		init_rand_state<<<((gpu_workers+255)/256),256>>>(rand_state,gpu_workers);
		gpuErr(cudaPeekAtLastError());

		update_count = (ceil)(1.0 * size / (gpu_workers*update_vector_size));
	}

#ifdef CAL_PORTION_RMSE	
	sgd_k128_kernel_hogwild_warp32_loss<<<gpu_workers/4, 128>>>(R, size, rand_state, para->gpu_loss, para->p, para->q, 128, update_count,
									update_vector_size, para->lrate, para->lambda_p, para->lambda_q);
	cudaMemcpy(para->loss, para->gpu_loss, size * sizeof(float), cudaMemcpyDeviceToHost);
#else
	sgd_k128_kernel_hogwild_warp32<<<gpu_workers/4, 128>>>(R, size, rand_state, para->p, para->q, 128, update_count,
									update_vector_size, para->lrate, para->lambda_p, para->lambda_q);
#endif

	gpuErr(cudaPeekAtLastError());
	cudaDeviceSynchronize();

	if(current_epoch == target_epoch) {
		cudaFree(rand_state);
	}
}

}