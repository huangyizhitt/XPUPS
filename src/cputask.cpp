#include "cputask.h"
#include <pthread.h>
#include <vector>
#include <atomic>
#include <iostream>

pthread_cond_t cpu_workers_barrier_con = PTHREAD_COND_INITIALIZER;
pthread_mutex_t cpu_workers_barrier_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t control_wake_up_con = PTHREAD_COND_INITIALIZER;
pthread_mutex_t control_wake_up_mutex = PTHREAD_MUTEX_INITIALIZER;
std::atomic<int> cpu_workers_complete(0);
std::atomic<int> epoch(0);


namespace MF{
#ifdef USE_AVX2


#elif USE_AVX512


#else
static inline void sgd_update(float *p, float *q, int k, float err, float lrate, float lambda_p, float lambda_q)
{
	for(int i = 0; i < k; i++) {
		float tmp_p = p[i];
		float tmp_q = q[i];
		p[i] = std::max(0.0f, tmp_p + lrate * (err * tmp_q - lambda_p * tmp_p));
		q[i] = std::max(0.0f, tmp_q + lrate * (err * tmp_p - lambda_q * tmp_q));
	}
}

static inline float inner_product(float *a, float *b, int k)
{
	float ret = 0;
	for(int i = 0; i < k; i++) {
		ret += a[i]*b[i];
	}
	return ret;
}

#endif

void *sgd_kernel_hogwild_cpu(void *args)
{
	CPUArgs *cpu_args = (CPUArgs *)args;
	WorkerDM *dm = cpu_args->dm;
	Grid *grid = &dm->grid;
	int k = dm->k;
	int target_epoch = cpu_args->target_epoch;
	float *p = cpu_args->p;
	float *q = cpu_args->q;
	float lrate = cpu_args->lrate;
	float lambda_p = cpu_args->lambda_p;
	float lambda_q = cpu_args->lambda_q;

	while(true) {
		printf("threads %d will block!\n", cpu_args->tid);
		pthread_mutex_lock(&cpu_workers_barrier_mutex);
		pthread_cond_wait(&cpu_workers_barrier_con, &cpu_workers_barrier_mutex);
		pthread_mutex_unlock(&cpu_workers_barrier_mutex);
		
		printf("threads %d will recover!\n", cpu_args->tid);
											
		int blockId;
		std::vector<MatrixNode *>& ptrs = grid->blocks;
		while((blockId = dm->GetFreeBlock(epoch)) >= 0) {
			for(MatrixNode *N = ptrs[blockId]; N != ptrs[blockId+1]; N++) {
				int u = N->row_index;
				int v = N->col_index;
				float r = N->r;

				int base_p = u * k;
				int base_q = v * k;
				
				float ruv = r - inner_product(p + base_p, q + base_q, k);

				sgd_update(p+base_p, q+base_q, k, ruv, lrate, lambda_p, lambda_q); 
			}
			dm->RecoverBlockFree(blockId);
		}

		pthread_mutex_lock(&control_wake_up_mutex);
		cpu_workers_complete++;
		if(cpu_workers_complete == cpu_args->workers) {
			printf("will wake up control thread!\n");
			pthread_cond_signal(&control_wake_up_con);
		}
		pthread_mutex_unlock(&control_wake_up_mutex);
		if(epoch == target_epoch) {
			printf("threads %d will stop!\n", cpu_args->tid);
			break;
		}
	}
}

}
