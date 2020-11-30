#include "task.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <numeric>

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

static float calc_rmse(MatrixNode *R, size_t size, float *p, float *q, int k)
{
	float loss = 0.0;
#if defined USEOMP
#pragma omp parallel for num_threads(20) schedule(static) reduction(+:loss)
#endif	
	for(size_t i = 0; i < size; i++) {
		MatrixNode &N = R[i];
		float *_p = &p[N.row_index * k];
		float *_q = &q[N.col_index * k];
		float e = N.r - std::inner_product(_p, _p+k, _q, ((float)0.0f));
		loss += e*e;
	}
	return loss;	
}

void *sgd_update_k128_gpu(void *args)
{
	Args *para = (Args *)args;

	MatrixNode *R = (MatrixNode *)para->data;
	int gpu_workers = para->workers;
	size_t size = para->size;
	debugp("current_epoch: %d, R: %p\n", global::current_epoch, R);
	if(global::current_epoch == 1) {
		cudaMalloc(&rand_state, sizeof(curandState)*gpu_workers);
		gpuErr(cudaPeekAtLastError());
		init_rand_state<<<((gpu_workers+255)/256),256>>>(rand_state,gpu_workers);
		gpuErr(cudaPeekAtLastError());

		update_count = (ceil)(1.0 * size / (gpu_workers*update_vector_size));
	}

#ifdef CAL_PORTION_RMSE	
	float *loss = para->loss;
	MatrixNode *check_data = (MatrixNode *)para->check_data;
	sgd_k128_kernel_hogwild_warp32<<<gpu_workers/4, 128>>>(R, size, rand_state, para->p, para->q, 128, update_count,
									update_vector_size, para->lrate, para->lambda_p, para->lambda_q);
	cudaDeviceSynchronize();
	cudaMemcpy(para->check_p, para->p, (para->size_p+para->size_q) * sizeof(float), cudaMemcpyDeviceToHost);
	loss[0] = calc_rmse(check_data, size, para->check_p, para->check_p+para->size_p, 128);
#else
	sgd_k128_kernel_hogwild_warp32<<<gpu_workers/4, 128>>>(R, size, rand_state, para->p, para->q, 128, update_count,
									update_vector_size, para->lrate, para->lambda_p, para->lambda_q);
#endif

	gpuErr(cudaPeekAtLastError());
	cudaDeviceSynchronize();

	if(global::current_epoch == global::target_epoch) {
		cudaFree(rand_state);
	}
	return NULL;
}

}