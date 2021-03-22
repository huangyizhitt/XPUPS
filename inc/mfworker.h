#ifndef _MFWORKER_H_
#define _MFWORKER_H_

#include <pthread.h>
#include <sched.h>
#include "xpu.h"
#include "ps/ps.h"
#include "mfdata.h"
#include "utils.h"

namespace MF {

class MFWorker {
public:
	MFWorker() {}
	~MFWorker() {PostProcess();}
	void PreProcess();
	void PostProcess();
	void Pull();
	void Push();
	void Computing();
	void CreateWorkers(pFunc func);
	void JoinWorkers();
	void Barrier();
	inline int GetCurrentEpoch() const {return xpu->current_epoch;}
	inline int GetTargetEpoch() const {return xpu->target_epoch;}
	inline int GetWorkerID() const {return rank;}
	inline TransMode GetTransMode() const {return trans_mode;}
private:
	void Init();
	void DeInit();
	void InitTrainingData();
	void PullTrainingData();
	void PrepareCPUResources();
	void PrepareGPUResources();
	void PrepareResources();
	int PrepareShmbuf();
	void GridProblemFromWorkers();
	void GridProblemFromStreams(int streams);
	
	void ReleaseCPUResources();
	void ReleaseGPUResources();
	void ReleaseResources();
	
	
	void PullAll();
	void PullAllShm();
	void PullQ();
	void PullQShm();
	void PullHalfQ();
	void PullHalfQShm();
	void PullHalfQShmEX();

	void PushAll();
	void PushAllShm();
	void PushQ();
	void PushQShm();
	void PushHalfQ();
	void PushHalfQShm();
	void PushHalfQShmEX();
	void PullHalfQShmAcopy(int stream);
	void PushHalfQShmAcopy(int stream);

	void PushXPUInfo();
	
	void PullGPUData();
	void PinnedBuf(void* buf, size_t size);
	void UnpinnedBuf(void *buf);
	int CreateShmbuf();
	int LinkPullbuf();
	void LinkShmbuf();
	void DestroyShmbuf();
/*
public:
#ifdef TEST
	double record1_start = 0;
	double record1_elapse = 0;
	double record2_start = 0;
	double record2_elapse = 0;
#endif
*/
private:
	int rank;
	int server_rank;
	int push_counts;
	int workers;
	int max_cores;
	int numa_node;
	int shm_id;
	TransMode trans_mode;
	size_t start;
	size_t size;

//info from parameter
	size_t m;
	size_t n;
	int k;
	float scale;
	float lambda_p;
	float lambda_q;
	float lrate;
	
	float *p;
	float *q;
	float *feature;
	MatrixNode *gpuR;
	unsigned char *shm_buf;
	unsigned char *pull_buf;

	XPU *xpu;
	WorkerDM dm;
	ps::KVWorker<float>* kv_xpu;
	std::vector<int> blocks;					//current hand blocks id
	std::vector<float> ps_vals;
	std::vector<Args> args;
	std::vector<int> index;
#ifdef CAL_PORTION_RMSE
	float *loss;
	int loss_size;
	float *gpu_loss;
public:
	void StartRecord();
#endif	
};

}

#endif
