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

	kv_xpu->Wait(kv_xpu->Push(keys, vals, lens, cmd));
	
}

void MFWorker::PullDataInfoFromServer()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;

	keys.push_back(rank);
	lens.push_back(2);
	CMD cmd = PULL_DATA_INFO;

	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));
	start = (size_t)vals[0];
	size = (size_t)vals[1];
	printf("start: %ld, size: %ld\n", start, size);
}

void MFWorker::PullDataFromServer()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_DATA;

	for(size_t i = 0; i < 3; i++) {
		keys.push_back(i);
		lens.push_back(2);
	}

	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));
	
	printf("receive the data!");
	Data& data = this->data;
	size_t size = vals.size();
	int len = 2;
	for(int i = 0; i < size / 3; i++) {
		data.r_matrix[i].row_index = (int)vals[i * len + 0];
		data.r_matrix[i].col_index = (int)vals[i * len + 1];
//		data.r_matrix[i].r = (float)vals[i * len + 2];
		data_counter++;
	}

	printf("recive data count: %ld\n", data_counter);
}

}
