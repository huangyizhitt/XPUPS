#ifndef _CPU_TASK_H_
#define _CPU_TASK_H_

#include "mfdata.h"
#include "xpu.h"

namespace MF{

void *fpsgd_kernel(void *args);
void *sgd_update_k128_gpu(void *args);
void InitGPUTask(int workers, int stream);
void DeInitGPUTask();
void g_print_tail(float *p, size_t size_p, float *q, size_t size_q);

}
#endif



