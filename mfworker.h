#ifndef _MFWORKER_H_
#define _MFWORKER_H_

#include <thread>
#include "xpu.h"
#include "ps/ps.h"

namespace MF {
class MFWorker {
public:
	MFWorker(XPU * const xpu) : xpu(xpu) {
		rank = ps::MyRank();
		kv_xpu = new ps::KVWorker<XPU>(0);
		core_num = std::thread::hardware_concurrency();
		kv_xpu->Wait(kv_xpu->Push(rank, xpu))
	}

	~MFWorker() {delete kv_xpu;}
	
	

public:
	int epochs = 50;

private:
	int rank;
	int core_num;
	XPU *xpu;
	MF::Data data;
	ps::KVWorker<XPU>* kv_xpu;
};

}

#endif

