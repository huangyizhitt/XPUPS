#ifndef _MFWORKER_H_
#define _MFWORKER_H_

#include <thread>
#include "xpu.h"
#include "ps/ps.h"
#include "mfdata.h"

namespace MF {
class MFWorker {
public:
	MFWorker(XPU * const xpu) : xpu(xpu), core_num(xpu->core) {
		rank = ps::MyRank();
		kv_xpu = new ps::KVWorker<int>(0, 0);		
		ps::RegisterExitCallback([](){delete kv_xpu;})
	}

	~MFWorker() {delete kv_xpu;}
	
	void PrepareData();
	void PushWorkerXPU();

public:
	int epochs = 50;

private:
	int rank;
	int core_num;
	XPU *xpu;
	MF::Data data;
	ps::KVWorker<int>* kv_xpu;
};

}

#endif

