#ifndef _CPU_TASK_H_
#define _CPU_TASK_H_

#include "mfdata.h"
#include "xpu.h"

namespace MF{

void *fpsgd_kernel(void *args);
void *sgd_update_k128_gpu(void *args);

}
#endif



