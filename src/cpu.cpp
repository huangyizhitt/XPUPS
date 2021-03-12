#include "xpu.h"
#include "utils.h"

namespace MF {

CPU* CPU::cur_cpu = NULL;

void* CPU::task_thread(void *args)
{
	CPUTask *task = (CPUTask *)args;
	while(true) {
		pthread_mutex_lock(&cur_cpu->barrier_mutex);
		pthread_cond_wait(&cur_cpu->barrier_con, &cur_cpu->barrier_mutex);
		pthread_mutex_unlock(&cur_cpu->barrier_mutex);
		debugp("Thread %d wake up!\n", task->tid);
		pFunc func = task->func;
		func(task->args);

		pthread_mutex_lock(&cur_cpu->wake_mutex);
		cur_cpu->complete_workers++;
		if(cur_cpu->complete_workers == cur_cpu->workers) {
			debugp("will wake up control thread!\n");
			pthread_cond_signal(&cur_cpu->wake_con);
		}
		pthread_mutex_unlock(&cur_cpu->wake_mutex);

		if(cur_cpu->current_epoch == cur_cpu->target_epoch) break;
	}
}

void CPU::Init()
{
	xpu_type = XPU_TYPE::CPU;
	XPU::Init();
	tasks.resize(workers);
	pthread_mutex_init(&barrier_mutex, NULL);
	pthread_mutex_init(&wake_mutex, NULL);
	pthread_cond_init(&barrier_con, NULL);
	pthread_cond_init(&wake_con, NULL);
	complete_workers = 0;
	SetCurCPU();
}

void CPU::CreateTasks(int task_index, pFunc func, void *args)
{
	tasks[task_index].func = func;
	tasks[task_index].args = args;
	tasks[task_index].tid = task_index;
	pthread_create(&tasks[task_index].thread, NULL, task_thread, &tasks[task_index]);
}

void CPU::RunTasks()
{
	//wake up worker threads;
	pthread_mutex_lock(&barrier_mutex);
	complete_workers = 0;
	pthread_cond_broadcast(&barrier_con);
	pthread_mutex_unlock(&barrier_mutex);

	//sleep control threads;
	if(complete_workers == 0) {
		debugp("control_thread will block!\n");
		pthread_cond_wait(&wake_con,&wake_mutex);
	}
	//wake up, the worker complete a epoch

	debugp("control_thread wake up and do something...!\n");
	pthread_mutex_unlock(&wake_mutex);
}

void CPU::JoinTasks()
{
	for(int i = 0; i < workers; i++) {
		pthread_join(tasks[i].thread, NULL);
	}
}

void CPU::DeInit()
{
	pthread_mutex_destroy(&barrier_mutex);
	pthread_mutex_destroy(&wake_mutex);
	pthread_cond_destroy(&barrier_con);
	pthread_cond_destroy(&wake_con);
}

void CPU::Transfer(void *dst, void *src, size_t size, TransferDirect direct)
{
	if(src == dst)	return;
	memcpy(dst, src, size);
}

void CPU::SetTask(pFunc func, void * args, int index)
{
	tasks[index].func = func;
	tasks[index].args = args;
}

//There is no cross-device on the CPU
int CPU::singles2halfp(void *target, const void *source, ptrdiff_t numel, int rounding_mode, int is_quiet, int nr_threads, bool cross_device)
{
	return cpu_singles2halfp(target, source, numel, rounding_mode, is_quiet, nr_threads);
}

int CPU::halfp2singles(void *target, void *source, ptrdiff_t numel, int nr_threads, bool cross_device)
{
	return cpu_halfp2singles(target, source, numel, nr_threads);
}

}
