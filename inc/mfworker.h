#ifndef _MFWORKER_H_
#define _MFWORKER_H_

#include <thread>
#include "xpu.h"
#include "ps/ps.h"
#include "mfdata.h"

namespace MF {
class MFWorker {
public:
	MFWorker(XPU * const xpu) : xpu(xpu), core_num(xpu->core), data_counter(0) {
		rank = ps::MyRank();
		kv_xpu = new ps::KVWorker<float>(0, 0);		
	}

	~MFWorker() {delete kv_xpu;}
	
	void PrepareData();
	int PushFeature();
	void PushWorkerXPU();
	void PullDataFromServer();
	void PullDataInfoFromServer();
	void PullBlockAndFeature();
	void InitTestData();
	int GetWorkerID() const  {return rank;}
	void Test();

public:
	int epochs = 20;

private:
	int rank;
	int core_num;
	int m;
	int n;
	int k = 128;
	int work_ratio;
	float *p;
	float *q;
	size_t data_counter;
	XPU *xpu;
	MF::Data data;
	size_t start;
	size_t size;
	ps::KVWorker<float>* kv_xpu;
	std::vector<int> blocks;					//current hand blocks id
};

}

#endif

