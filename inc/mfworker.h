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
	MFWorker(XPU * const xpu, const int& target_epoch);

	MFWorker() : data_counter(0), current_epoch(0){}

	~MFWorker(); 

	inline void SetWorkload(const int& workload) {xpu->worker_ratio = workload;}
	inline int NumaNode() const {return xpu->numa_node;}
	//Worker init by environment
	void Init();

	//Prepare the worker
	void Prepare();
	void PrepareCPUResources();
	void PrepareGPUResources();
	void PrepareResources();
	void ReleaseCPUResources();
	void ReleaseGPUResources();
	void ReleaseResources();
	void PullFeature();
	void PushFeature();
	void PullCompressFeature();
	void PushCompressFeature();
	void PullAllFeature();
	void PushAllFeature();
	void PullPushFeature();
	void PushXPUInfo();
	void PullTrainingData();
	void PullDataInfoFromServer();
	void PullBlockAndFeature();
	void InitTrainingData();
	void PullGPUData();
	int GetWorkerID() const  {return rank;}
	void GridProblem();
	void CreateCPUTasks();
	void StartUpTasks();
	void JoinTasks();
	void InitCPUAffinity();
	bool InitGPUAffinity();
	void SetCPUAffinity();
	void Test();
	int PrepareShmbuf();
	void PullCompressFeatureUseShm();
	void PushCompressFeatureUseShm();
	void PullFeatureUseShm();
	void PushFeatureUseShm();
	void sgd_update_k128_gpu();
	void Computing();

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
	MatrixNode *gpuR;
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
	float *gpu_loss;
#endif
};

}

#endif

