#ifndef _MFWORKER_H_
#define _MFWORKER_H_

#include <pthread.h>
#include <sched.h>
#include "xpu.h"
#include "ps/ps.h"
#include "mfdata.h"
#include "cputask.h"

namespace MF {
class MFWorker {
public:
	MFWorker(XPU * const xpu, const int& target_epoch) : xpu(xpu), core_num(xpu->core), data_counter(0),
		target_epoch(target_epoch), current_epoch(0)
	{
		rank = ps::MyRank();
		kv_xpu = new ps::KVWorker<float>(0, 0);		
	}

	MFWorker() {}

	~MFWorker() {delete kv_xpu; free(feature); delete xpu;}

	inline void SetWorkload(const int& workload) {xpu->worker_ratio = workload;}
	//Worker init by environment
	void Init();
	
	void PrepareData();
	void PullFeature();
	void PushFeature();
	void PullCompressFeature();
	void PushCompressFeature();
	void PullAllFeature();
	void PushAllFeature();
	void PullPushFeature();
	void PushWorkerXPU();
	void PullDataFromServer();
	void PullDataInfoFromServer();
	void PullBlockAndFeature();
	void InitTestData();
	int GetWorkerID() const  {return rank;}
	void GridProblem();
	void CreateTasks();
	void StartUpTasks();
	void JoinTasks();
	void InitCPUAffinity();
	void SetCPUAffinity();
	void Test();
	int PrepareShmbuf();
	void PullCompressFeatureUseShm();
	void PushCompressFeatureUseShm();

public:
	int target_epoch;
	int current_epoch;

private:
	int rank;
	int core_num;
	int m;
	int n;
	int k = 128;
	float scale;
	float lambda_p = 0.01;
	float lambda_q = 0.01;
	float lrate = 0.005;
	float *p;
	float *q;
	float *feature;
	unsigned char *shm_buf;
#ifdef SEND_COMPRESS_Q_FEATURE
	uint16_t *halfp;
	uint16_t *halfq;
#endif
	size_t data_counter;
	XPU *xpu;
	cpu_set_t cpuset;
	WorkerDM dm;
	size_t start;
	size_t size;
	ps::KVWorker<float>* kv_xpu;
	std::vector<int> blocks;					//current hand blocks id
	std::vector<CPUArgs> args;
	std::vector<pthread_t> tids;
#ifdef CAL_PORTION_RMSE
	std::vector<float> loss;
#endif
};

}

#endif

