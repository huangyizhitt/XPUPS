#include "task.h"
#include <vector>
#include <iostream>
#include <immintrin.h>
#include "utils.h"
#include "xpu.h"

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

#if __GNUC__ < 7
static float _mm512_reduce_add_ps(__m512 a) 
{
	__m512 tmp = _mm512_add_ps(a,_mm512_shuffle_f32x4(a,a,_MM_SHUFFLE(0,0,3,2)));
    	__m128 r = _mm512_castps512_ps128(_mm512_add_ps(tmp,_mm512_shuffle_f32x4(tmp,tmp,_MM_SHUFFLE(0,0,0,1))));
    	r = _mm_hadd_ps(r,r);
    	return _mm_cvtss_f32(_mm_hadd_ps(r,r));
}
#endif

static inline float inner_product(float *p, float *q, int k)
{
	__m512 r_vec = _mm512_setzero_ps();
/*	for(int i = 0; i < k; i+=16) {
		r_vec = _mm512_fmadd_ps(_mm512_load_ps(p+i), _mm512_load_ps(q+i), r_vec);
	}*/
	r_vec = _mm512_fmadd_ps(_mm512_load_ps(p), _mm512_load_ps(q), r_vec);
	r_vec = _mm512_fmadd_ps(_mm512_load_ps(p+16), _mm512_load_ps(q+16), r_vec);
	r_vec = _mm512_fmadd_ps(_mm512_load_ps(p+32), _mm512_load_ps(q+32), r_vec);
	r_vec = _mm512_fmadd_ps(_mm512_load_ps(p+48), _mm512_load_ps(q+48), r_vec);
	r_vec = _mm512_fmadd_ps(_mm512_load_ps(p+64), _mm512_load_ps(q+64), r_vec);
	r_vec = _mm512_fmadd_ps(_mm512_load_ps(p+80), _mm512_load_ps(q+80), r_vec);
	r_vec = _mm512_fmadd_ps(_mm512_load_ps(p+96), _mm512_load_ps(q+96), r_vec);
	r_vec = _mm512_fmadd_ps(_mm512_load_ps(p+112), _mm512_load_ps(q+112), r_vec);
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
		p[i] = tmp_p + lrate * (err * tmp_q - lambda_p * tmp_p);
		q[i] = tmp_q + lrate * (err * tmp_p - lambda_q * tmp_q);
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

void *fpsgd_kernel(void *args)
{
	Args *cpu_args = (Args *)args;
	WorkerDM *dm = (WorkerDM *)cpu_args->data;
	Grid *grid = &dm->grid;
	int k = dm->k;

	float *p = cpu_args->p;
	float *q = cpu_args->q;
	float lrate = cpu_args->lrate;
	float lambda_p = cpu_args->lambda_p;
	float lambda_q = cpu_args->lambda_q;

#ifdef CAL_PORTION_RMSE		
		float *loss = cpu_args->loss;
#endif

	int blockId;
	std::vector<MatrixNode *>& ptrs = grid->blocks;

#ifdef CAL_PORTION_RMSE	
	*loss = 0.0;
#endif	
	debugp("Entry %d fpsgd_kernel!\n", cpu_args->tid);	
	while((blockId = dm->GetFreeBlock()) != -1) {
		if(blockId == -2) continue;
//		debugp("[Thread %d] blockId %d\n", cpu_args->tid, blockId);
		for(MatrixNode *N = ptrs[blockId]; N != ptrs[blockId+1]; N++) {
			int u = N->row_index;
			int v = N->col_index;
			float r = N->r;

			int base_p = u * k;
			int base_q = v * k;
//				start = cpu_second();		
			float ruv = r - inner_product(p + base_p, q + base_q, k);
//				elapse += cpu_second() - start;
#ifdef CAL_PORTION_RMSE	
			*loss += ruv * ruv;
#endif
//				start = cpu_second();
			sgd_update(p+base_p, q+base_q, k, ruv, lrate, lambda_p, lambda_q); 
//				elapse += cpu_second() - start;
		}
		dm->RecoverBlockFree(blockId);
	}
}


}
