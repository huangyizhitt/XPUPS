#include "mfworker.h"
#include "utils.h"
#include "cputask.h"
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
	debugp("start: %ld, size: %ld\n", start, size);
}

void MFWorker::PullDataFromServer()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_DATA;

	for(size_t i = start; i < start+size; i++) {
		keys.push_back(i);
		lens.push_back(3);
	}

	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	Data& data = this->dm.data;
	size_t size = keys.size();
	debugp("receive the data, size: %d!\n", size);
	data.r_matrix.resize(size);	
	data_counter = size;
	int len = 3;
	for(int i = 0; i < size; i++) {
		data.r_matrix[i].row_index = (int)vals[i * len + 0];
		data.r_matrix[i].col_index = (int)vals[i * len + 1];
		data.r_matrix[i].r = (float)vals[i * len + 2];
	}

	printf("Recive data count: %ld\n", data_counter);
	dm.PrintHead(rank, 3);
}

//return value: 1 all epoch complete, 0 receive block and feature
void MFWorker::PullBlockAndFeature()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_FEATURE;

	for(int i = 0; i < 3; i++) {
		keys.push_back(rank+i);
	}
	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	debugp("receive block and feature!\n");
	size_t size = keys.size();

	work_ratio = lens[0];
	blocks.resize(work_ratio);
	
	size_t size_p = lens[1];
	size_t size_q = lens[2];
	m = size_p / k;
	n = size_q / k;

	if(!alloc_feature) {
		p = (float *)aligned_alloc(64, size_p * sizeof(float));
		q = (float *)aligned_alloc(64, size_q * sizeof(float));
	}

	for(int i = 0; i < work_ratio; i++) {
		blocks[i] = (int)vals[i];
	}
	memcpy(p, &vals[work_ratio], sizeof(float) * size_p);
	memcpy(q, &vals[work_ratio+size_p], size_q * sizeof(float));

	for(int i = 0; i < work_ratio; i++) {
		printf("[worker %d] block id: %d\n", rank, blocks[i]);
	}

}

//push format {keys0, feature_p} {keys1, feature_q} {lens0: m*k} {lens1: n*k}
int MFWorker::PushFeature()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PUSH_FEATURE;

	size_t size_p = m * k;
	size_t size_q = n * k; 

	vals.resize(work_ratio+size_p+size_q);

	keys.push_back(0);
	keys.push_back(1);
	keys.push_back(2);

	lens.push_back(work_ratio);
	lens.push_back(size_p);
	lens.push_back(size_q);

	for(int i = 0; i < work_ratio; i++) 
	{
		vals[i] = blocks[i];
	}
	
	memcpy(&vals[work_ratio], p, sizeof(float) * size_p);
	memcpy(&vals[work_ratio + size_p], q, sizeof(float) * size_q);

	kv_xpu->Wait(kv_xpu->Push(keys, vals, lens, cmd));
	
	if(vals[0] == STOP_WORKER) {
		printf("Worker %d will stop!\n", rank);
		return 1;
	} else {
		return 0;
	}
}

void MFWorker::InitTestData()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = INIT_DATA;

	keys.push_back(rank);
	lens.push_back(0);
	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));
	if(lens[0] != 4) {
		printf("[Worker %d] InitTestData: receive data fail!\n");
	}
	
	start = (int)vals[0];
	size = (int)vals[1];
	dm.rows = (int)vals[2];
	dm.cols = (int)vals[3];
	dm.nnz = size;
	printf("[Worker %d] start: %ld, size: %ld\n", rank, start, size);
}

void MFWorker::GridProblem()
{
	Dim2 gridDim;
		
	gridDim.x = 2 * core_num + 1;
	gridDim.y = 2 * core_num + 1;
	dm.SetGrid(gridDim);
	dm.GridData(rank);
}

void MFWorker::CreateTasks()
{
	for(int i = 0; i < core_num; i++) {
		CPUArgs arg;
		arg.tid = i;
		arg.workers = core_num;
		arg.target_epoch = epochs;
		arg.lambda_p = lambda_p;
		arg.lambda_q = lambda_q;
		arg.lrate = lrate;
		arg.p = p;
		arg.q = q;
		arg.dm = &dm;
		args.push_back(arg);
		threads.emplace_back(sgd_kernel_hogwild_cpu, &args[i]);
	}
}

void MFWorker::StartUpTasks()
{
	printf("will start up all tasks!\n");
	dm.ClearBlockFlags();
	cpu_workers_complelte = 0;
	cpu_workers_barrier_con.notify_all();
	std::unique_lock<std::mutex> unique_lock(control_wake_up_mutex);
	control_wake_up_con.wait(unique_lock, [&](){return cpu_workers_complelte == core_num;});
	epoch++;
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
