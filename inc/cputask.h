#ifndef _CPU_TASK_H_
#define _CPU_TASK_H_

#include "mfdata.h"
#include <atomic>

extern std::mutex cpu_workers_barrier_mutex;
extern std::condition_variable cpu_workers_barrier_con;
extern std::mutex control_wake_up_mutex;
extern std::condition_variable control_wake_up_con;
extern std::atomic<int> cpu_workers_complelte;
extern std::atomic<int> epoch;

using namespace MF{

struct CPUArgs {
	int tid;
	int workers;
	int target_epoch;
	float lambda_p;
	float lambda_q;
	float lrate;
	float *p;
	float *q;
	WorkerDM *dm;
};

void sgd_kernel_hogwild_cpu(CPUArgs *cpu_args);


}
#endif



