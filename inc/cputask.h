#ifndef _CPU_TASK_H_
#define _CPU_TASK_H_

#include "mfdata.h"

namespace MF{

void *sgd_kernel_hogwild_cpu(void *args);
void *fpsgd_kernel(void *args);

}
#endif



