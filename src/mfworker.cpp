#include "mfworker.h"
#include "utils.h"

namespace MF {

void MFWorker::PushWorkerXPU()
{
	std::vector<ps::Key> keys;									//XPU Info {rank, (peak_performance, mem_band)} len 2
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PUSH_INFO;
	keys.push_back(rank);
	vals.push_back(xpu->peak_performance);
	vals.push_back(xpu->mem_band);
	lens.push_back(2);

	kv_xpu->Push(keys, vals, lens, cmd);
}

void MFWorker::PullDataFromServer()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_DATA;

	kv_xpu->Wait(kv_xpu->Pull(keys, vals, lens, cmd));

	Data& data = this->data;
	size_t keys_size = keys.size();
	int len = lens[0];

	for(int i = 0; i < keys_size; i++) {
		data.r_matrix[i].row_index = (int)vals[i * len + 0];
		data.r_matrix[i].col_index = (int)vals[i * len + 1];
		data.r_matrix[i].r = (float)vals[i * len + 2];
		data_counter++;
	}

	printf("recive data count: %d\n", data_counter);
}

}
