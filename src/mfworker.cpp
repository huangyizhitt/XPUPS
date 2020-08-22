#include "mfworker.h"
#include "utils.h"

namespace MF {

void MFWorker::PushWorkerXPU()
{
	std::vector<Key> keys;									//XPU Info {rank, (peak_performance, mem_band)} len 2
	std::vector<float> vals;
	const std::vector<int> lens;
	CMD cmd = PUSH_INFO;
	keys.push_back(rank);
	vals.push_back(xpu->peak_performance, xpu->mem_band);
	lens.push_back(2);

	kv_xpu->Wait(kv_xpu->Push(keys, vals, lens, cmd));
}

}
