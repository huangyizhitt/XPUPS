#ifndef _XPU_H_
#define _XPU_H_

#include <string.h>
#include <pthread.h>
#include <vector>

namespace MF {

enum class XPU_TYPE {
		CPU = 0,
		GPU,
		FPGA,
		TPU,
		UNKONWN_XPUTYPE,
};

enum class TransferDirect {
	C2S,
	S2C,
	S2S,
	C2C,
};

typedef void* (*pFunc)(void *);

struct Args {
	float lambda_p;
	float lambda_q;
	float lrate;
	float *p;
	float *q;
	
#ifdef CAL_PORTION_RMSE	
	float *loss;
#endif
	void *data;

#ifdef DEBUG
	int tid;
#endif
};


struct XPU {
	XPU() {}
	virtual ~XPU() {}

	//Init by env
	//if call XPU() create XPU object, must call this function
	virtual void Init();
	virtual void Bind(){}				//xpu task bind numa node and device;  
	virtual void CreateTasks(int task_index, pFunc func, void *args){}								
	virtual void RunTasks(){}							
	virtual void JoinTasks(){}
	virtual void DestroyTasks(){}
	virtual void Transfer(void *dst, void *src, size_t size, TransferDirect direct){}
	virtual int singles2halfp(void *target, const void *source, ptrdiff_t numel, int rounding_mode, int is_quiet, int nr_threads){}
	virtual int halfp2singles(void *target, void *source, ptrdiff_t numel, int nr_threads){}
		
	char xpu_name[64];
	XPU_TYPE xpu_type;
	int dev_id;								//device id
	int max_cores;
	int workers;
	int worker_ratio;
	int target_epoch;
	int current_epoch;
};

struct XPU_INFO {
	XPU_TYPE type;
	int workers;
	int work_ratio;						//work load ratio
	int start;
	size_t size;
};

struct CPUTask {
	pFunc func;
	void *args;
	pthread_t thread;
};

struct CPUTaskPool {
	pthread_cond_t barrier_con;
	pthread_mutex_t barrier_mutex;
	pthread_cond_t wake_con;
	pthread_mutex_t wake_mutex;
	int complete_workers;
	int target_epoch;
	int current_epoch;
	int tid;
	std::vector<CPUTask> tasks;
};

struct CPU : public XPU {
	CPU() {}
	virtual ~CPU() {DeInit();}

	virtual void Init();
	virtual void Bind();
	virtual void CreateTasks(int task_index, pFunc func, void *args);
	virtual void RunTasks();
	virtual void JoinTasks();
	virtual void DeInit();
	virtual void Transfer(void *dst, void *src, size_t size, TransferDirect direct);
	virtual void SetTask(pFunc func, void *args, int index);
	virtual int singles2halfp(void *target, const void *source, ptrdiff_t numel, int rounding_mode, int is_quiet, int nr_threads);
	virtual int halfp2singles(void *target, void *source, ptrdiff_t numel, int nr_threads);
	
	CPUTaskPool task_pool;
};

}

#endif

