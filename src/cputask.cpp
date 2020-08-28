#include "cputask.h"
#include <pthread.h>
#include <vector>
#include <atomic>
#include <iostream>
#include <immintrin.h>
#include "utils.h"

pthread_cond_t cpu_workers_barrier_con = PTHREAD_COND_INITIALIZER;
pthread_mutex_t cpu_workers_barrier_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t control_wake_up_con = PTHREAD_COND_INITIALIZER;
pthread_mutex_t control_wake_up_mutex = PTHREAD_MUTEX_INITIALIZER;
std::atomic<int> cpu_workers_complete(0);


namespace MF{
#ifdef USE_AVX2
static inline float inner_product(float *p, float *q, int k)
{
	float product;
	__m256 r_vec = _mm256_setzero_ps();
	
	for(int i = 0; i < k; i += 8) {
		r_vec = _mm256_fmadd_ps(_mm256_load_ps(p+i), _mm256_load_ps(q+i), r_vec);
	}

	r_vec = _mm256_add_ps(r_vec, _mm256_permute2f128_ps(r_vec, r_vec, 0x1));
    r_vec = _mm256_hadd_ps(r_vec, r_vec);
    r_vec = _mm256_hadd_ps(r_vec, r_vec);
    _mm_store_ss(&product, _mm256_castps256_ps128(r_vec));
	return product;
}

static inline void sgd_update(float *p, float *q, int k, float err, float lrate, float lambda_p, float lambda_q)
{
	__m256 r_vec, lrate_vec, lambda_p_vec, lambda_q_vec, p_vec, q_vec, tmp_p_vec, tmp_q_vec, mid_p_vec, mid_q_vec;
	r_vec = _mm256_set1_ps(err);
    lrate_vec = _mm256_set1_ps(lrate);
    lambda_p_vec = _mm256_set1_ps(lambda_p);
    lambda_q_vec = _mm256_set1_ps(lambda_q);

	for(int pos = 0; pos < k; pos += 8) {
		tmp_p_vec = _mm256_load_ps(p + pos);
        	tmp_q_vec = _mm256_load_ps(q + pos);

        	mid_p_vec = _mm256_mul_ps(lambda_p_vec, tmp_p_vec);
       	 	mid_q_vec = _mm256_mul_ps(lambda_q_vec, tmp_q_vec);

		p_vec = _mm256_fmadd_ps(lrate_vec, _mm256_fmsub_ps(r_vec, tmp_q_vec, mid_p_vec), tmp_p_vec);	
        	q_vec = _mm256_fmadd_ps(lrate_vec, _mm256_fmsub_ps(r_vec, tmp_p_vec, mid_q_vec), tmp_q_vec);
		
		_mm256_store_ps(p+pos, p_vec);
        _mm256_store_ps(q+pos, q_vec);
	}
}
#elif USE_AVX512
static inline float inner_product(float *p, float *q, int k)
{
	__m512 r_vec = _mm512_setzero_ps();
	for(int i = 0; i < k; i+=16) {
		r_vec = _mm512_fmadd_ps(_mm512_load_ps(p+i), _mm512_load_ps(q+i), r_vec);
	}
	return _mm512_reduce_add_ps(r_vec);	
}

static inline void sgd_update(float *p, float *q, int k, float err, float lrate, float lambda_p, float lambda_q)
{
	__m512 err_vec, lrate_vec, lambda_p_vec, lambda_q_vec, tmp_p_vec, tmp_q_vec, p_vec, q_vec;
	err_vec = _mm512_set1_ps(err);
    lrate_vec = _mm512_set1_ps(lrate);
    lambda_p_vec = _mm512_set1_ps(lambda_p);
    lambda_q_vec = _mm512_set1_ps(lambda_q);

	for(int i = 0; i < k; i+=16) {
		tmp_p_vec = _mm512_load_ps(p + i);
        tmp_q_vec = _mm512_load_ps(q + i);

		p_vec = _mm512_fmadd_ps(lrate_vec, _mm512_fmsub_ps(err_vec, tmp_q_vec, _mm512_mul_ps(lambda_p_vec, tmp_p_vec)), tmp_p_vec);
         	_mm512_store_ps(p+i, p_vec);

		q_vec = _mm512_fmadd_ps(lrate_vec, _mm512_fmsub_ps(err_vec, tmp_p_vec, _mm512_mul_ps(lambda_q_vec, tmp_q_vec)), tmp_q_vec);
        	_mm512_store_ps(q+i, q_vec);
	}
}

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
	int current_epoch = cpu_args->current_epoch;
	float *p = cpu_args->p;
	float *q = cpu_args->q;
	float lrate = cpu_args->lrate;
	float lambda_p = cpu_args->lambda_p;
	float lambda_q = cpu_args->lambda_q;

	pthread_t thread = pthread_self();

	pthread_setaffinity_np(thread, sizeof(cpu_set_t), cpu_args->cpuset);
	while(true) {
		debugp("threads %d will block!\n", cpu_args->tid);
		pthread_mutex_lock(&cpu_workers_barrier_mutex);
		pthread_cond_wait(&cpu_workers_barrier_con, &cpu_workers_barrier_mutex);
		pthread_mutex_unlock(&cpu_workers_barrier_mutex);
		
		debugp("threads %d will recover!\n", cpu_args->tid);
		int blockId;
		std::vector<MatrixNode *>& ptrs = grid->blocks;
		while((blockId = dm->GetFreeBlock()) != -1) {
			if(blockId == -2) continue;
			debugp("[Thread %d] blockId %d\n", cpu_args->tid, blockId);
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
			debugp("will wake up control thread!\n");
			pthread_cond_signal(&control_wake_up_con);
		}
		pthread_mutex_unlock(&control_wake_up_mutex);
		if(current_epoch == target_epoch) {
			debugp("threads %d will stop!\n", cpu_args->tid);
			break;
		}
	}
}

}
