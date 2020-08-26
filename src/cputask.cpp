#include "cputask.h"
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>

std::mutex cpu_workers_barrier_mutex;
std::condition_variable cpu_workers_barrier_con;
std::mutex control_wake_up_mutex;
std::condition_variable control_wake_up_con;
std::atomic<int> cpu_workers_complelte(0);
std::atomic<int> epoch(0);


using namespace MF{

void sgd_kernel_hogwild_cpu(CPUArgs *cpu_args)
{
	CPUArgs *cpu_args;
	WorkerDM *dm = cpu_args->dm;
	Grid *grid = &dm->grid;
	int k = dm->k;
	int target_epoch = cpu_args->epoch;
	float *p = cpu_args->p;
	float *q = cpu_args->q;
	float lrate = cpu_args->lrate;
	float lambda_p = cpu_args->lambda_p;
	float lambda_q = cpu_args->lambda_q;

	while(true) {
		{
			printf("threads %d will block!\n", cpu_args->tid);
			std::unique_lock<std::mutex> unique_lock(cpu_workers_barrier_mutex);
			cpu_workers_barrier_con.wait(unique_lock, [&](){return dm->remain_blocks > 0;})
		}
		printf("threads %d will recover!\n", cpu_args->tid);
											
		int blockId;
		std::vector<MatrixNode *>& ptrs = grid->blocks;
		while((blockId = dm->GetFreeBlock()) > 0) {
			for(MatrixNode *N = ptrs[blockId]; N != ptrs[blockId+1]; N++) {
				int u = N->row_index;
				int v = N->col_index;
				float r = N->r;

				int base_p = u * k;
				int base_q = v * k;

/*				float ruv = r - inner_product(p + base_p, q + base_q);

				sgd_update(p+base_p, q+base_q, ruv, lrate, lambda_p, lambda_q); */
			}
		}

		cpu_workers_complelte++;
		std::unique_lock<std::mutex> unique_lock(control_wake_up_mutex);
		if(cpu_workers_complelte == cpu_args->workers) {
			control_wake_up_con.notify_one();
		}

		if(epoch == target_epoch) {
			printf("threads %d will stop!\n");
			break;
		}
	}
}

}
