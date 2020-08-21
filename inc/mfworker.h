#ifndef _MFWORKER_H_
#define _MFWORKER_H_

#include <thread>
#include "xpu.h"
#include "ps/ps.h"
#include "mfdata.h"

namespace MF {
class MFWorker {
public:
	MFWorker(XPU * const xpu) : xpu(xpu) {
		rank = ps::MyRank();
		std::vector<ps::Key> keys;
		std::vector<int> vals;
		keys.push_back(rank);
		vals.push_back(xpu->xpu_type);
		kv_xpu = new ps::KVWorker<int>(0, 0);
		core_num = std::thread::hardware_concurrency();
		kv_xpu->Wait(kv_xpu->Push(keys, vals));
	}

	~MFWorker() {delete kv_xpu;}
	
	

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

