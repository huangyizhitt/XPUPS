#ifndef _CPU_TASK_H_
#define _CPU_TASK_H_

#include "mfdata.h"
#include <atomic>
#include <pthread.h>
#include <sched.h>


extern pthread_mutex_t cpu_workers_barrier_mutex;
extern pthread_cond_t cpu_workers_barrier_con;
extern pthread_mutex_t control_wake_up_mutex;
extern pthread_cond_t control_wake_up_con;
extern std::atomic<int> cpu_workers_complete;
extern std::atomic<int> epoch;

namespace MF{

struct CPUArgs {
	int tid;
	int workers;
	int target_epoch;
	int *current_epoch;
	float lambda_p;
	float lambda_q;
	float lrate;
	float *p;
	float *q;
	WorkerDM *dm;
	cpu_set_t *cpuset;
#ifdef CAL_PORTION_RMSE
	float *loss;
#endif
};

void *sgd_kernel_hogwild_cpu(void *args);


}
#endif



