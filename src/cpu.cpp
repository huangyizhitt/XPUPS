#include "xpu.h"
#include "utils.h"

namespace MF {

void *task_thread(void *args)
{
	CPUTaskPool *pool = (CPUTaskPool *)args;
	pthread_cond_t *barrier_con = &pool->barrier_con;
	pthread_mutex_t *barrier_mutex = &pool->barrier_mutex;
	pthread_cond_t *wake_con = &pool->wake_con;
	pthread_mutex_t *wake_mutex = &pool->wake_mutex;
	int *complete_workers = &pool->complete_workers;
	int current_epoch = pool->current_epoch;
	int target_epoch = pool->target_epoch;
	int total_works = pool->tasks.size();
	int tid = pool->tid;
	while(true) {
		pthread_mutex_lock(barrier_mutex);
		pthread_cond_wait(barrier_con, barrier_mutex);
		pthread_mutex_unlock(barrier_mutex);

		pFunc func = pool->tasks[tid].func;
		func(pool->tasks[tid].args);

		pthread_mutex_lock(wake_mutex);
		*complete_workers++;
		if(*complete_workers == total_works) {
			debugp("will wake up control thread!\n");
			pthread_cond_signal(wake_con);
		}
		pthread_mutex_unlock(wake_mutex);

		if(current_epoch == target_epoch) break;
	}
}

void CPU::Init()
{
	xpu_type = XPU_TYPE::CPU;
	XPU::Init();
	task_pool.tasks.resize(workers);
	pthread_mutex_init(&task_pool.barrier_mutex, NULL);
	pthread_mutex_init(&task_pool.wake_mutex, NULL);
	pthread_cond_init(&task_pool.barrier_con, NULL);
	pthread_cond_init(&task_pool.wake_con, NULL);
	task_pool.target_epoch = target_epoch;
	task_pool.current_epoch = 0;
	task_pool.complete_workers = 0;
}

void CPU::Bind()
{

}

void CPU::CreateTasks(int task_index, pFunc func, void *args)
{
	task_pool.tasks[task_index].func = func;
	task_pool.tasks[task_index].args = args;
	pthread_create(&task_pool.tasks[task_index].thread, NULL, task_thread, &task_pool);
}

void CPU::RunTasks()
{
	//wake up worker threads;
	pthread_mutex_lock(&task_pool.barrier_mutex);
	task_pool.complete_workers = 0;
	pthread_cond_broadcast(&task_pool.barrier_con);
	pthread_mutex_unlock(&task_pool.barrier_mutex);

	//sleep control threads;
	if(task_pool.complete_workers == 0) {
		debugp("control_thread will block!\n");
		pthread_cond_wait(&task_pool.wake_con,&task_pool.wake_mutex);
	}
	//wake up, the worker complete a epoch

	debugp("control_thread wake up and do something...!\n");
	pthread_mutex_unlock(&task_pool.wake_mutex);
}

void CPU::JoinTasks()
{
	for(int i = 0; i < workers; i++) {
		pthread_join(task_pool.tasks[i].thread, NULL);
	}
}

void CPU::DeInit()
{
	pthread_mutex_destroy(&task_pool.barrier_mutex);
	pthread_mutex_destroy(&task_pool.wake_mutex);
	pthread_cond_destroy(&task_pool.barrier_con);
	pthread_cond_destroy(&task_pool.wake_con);
}

void CPU::Transfer(void *dst, void *src, size_t size, TransferDirect direct)
{
	memcpy(dst, src, size);
}

void CPU::SetTask(pFunc func, void * args, int index)
{
	task_pool.tasks[index].func = func;
	task_pool.tasks[index].args = args;
}

int CPU::singles2halfp(void *target, const void *source, ptrdiff_t numel, int rounding_mode, int is_quiet, int nr_threads)
{
	return singles2halfp(target, source, numel, rounding_mode, is_quiet, nr_threads);
}

int CPU::halfp2singles(void *target, void *source, ptrdiff_t numel, int nr_threads)
{
	return halfp2singles(target, source, numel, nr_threads);
}

}
