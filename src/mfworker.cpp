#include "mfworker.h"
#include "utils.h"
#include <cstdlib>

namespace MF {

static bool alloc_feature(false);

void MFWorker::PushWorkerXPU()
{
	std::vector<ps::Key> keys;									//XPU Info {rank, (peak_performance, mem_band)} len 2
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PUSH_INFO;
	keys.push_back(rank);
	vals.push_back(xpu->xpu_type);
	vals.push_back(xpu->workers);
	vals.push_back(xpu->worker_ratio);
	lens.push_back(3);

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

	for(size_t i = 0; i < size; i++) {
		keys.push_back(i);
		lens.push_back(3);
	}

	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));
	
/*	for(size_t i = 0; i < vals.size(); i++) {
                printf("vals[%d]: %d\n", i,(int)vals[i]);
        }*/
	Data& data = this->data;
	size_t size = keys.size();
	printf("receive the data, size: %d!\n", size);
	data.r_matrix.resize(size);	
	int len = 3;
	for(int i = 0; i < size; i++) {
		data.r_matrix[i].row_index = (int)vals[i * len + 0];
		data.r_matrix[i].col_index = (int)vals[i * len + 1];
		data.r_matrix[i].r = (float)vals[i * len + 2];
		data_counter++;
	}

	printf("recive data count: %ld\n", data_counter);
}

void MFWorker::PullWorkerAndFeature()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_FEATURE;

	for(int i = 0; i < 3; i++) {
		keys.push_back(i);
	}

	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	size_t size = keys.size();
	work_ratio = lens[0];
	blocks.resize(work_ratio);
	m = lens[1];
	n = lens[2];
	size_t size_p = m * k;
	size_t size_q = n * k;

	if(!alloc_feature) {
		p = (float *)aligned_alloc(64, size_p * sizeof(float));
		q = (float *)aligned_alloc(64, size_q * sizeof(float));
	}

	memcpy(&blocks[0], &vals[0], sizeof(int) * work_ratio);
	memcpy(p, &vals[work_ratio], sizeof(float) * size_p);
	memcpy(q, &vals[work_ratio+size_p], size_q * sizeof(float));

	printf("work_ratio: %d, m: %d, n: %d\n");
	for(int i = 0; i < work_ratio; i++) {
		printf("block id: %d\n", blocks[i]);
	}
}

//push format {keys0, feature_p} {keys1, feature_q} {lens0: m} {lens1: n}
void MFWorker::PushFeature()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PUSH_FEATURE;

	size_t size_p = m * k;
	size_t size_q = n * k; 

	vals.resize(size_p+size_q);

	keys.push_back(0);
	keys.push_back(1);
	lens.push_back(m);
	lens.push_back(n);

	memcpy(&vals[0], p, sizeof(float) * size_p);
	memcpy(&vals[size_p], q, sizeof(float) * size_q);

	for(int i = 0; i < 5; i++) {
		printf("[Worker] p[%d]: %.2f, q[%d]: %.2f\n", i, p[i], i, q[i]);
	}

	kv_xpu->Wait(kv_xpu->Push(keys, vals, lens, cmd));
}


void MFWorker::Test()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_DATA;

	for(size_t i = 0; i < 5; i++) {
		keys.push_back(i);
		lens.push_back(3);
	}

	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	for(size_t i = 0; i < vals.size(); i++) {
		printf("vals[%d]: %d\n", i,(int)vals[i]);
	}
}

}
