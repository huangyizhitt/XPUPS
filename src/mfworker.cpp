#include "mfworker.h"
#include "utils.h"
#include "ps/internal/env.h"
#include "dmlc/logging.h"
#include "cputask.h"
#include <cstdlib>
#include <numeric>
#include <cmath>

namespace MF {

static bool alloc_feature(false);

//Worker init by environment
void MFWorker::Init()
{
	const char* val = NULL;
	XPU *xpu = new xpu();
	xpu->Init();
	xpu->is_server = false;
	xpu->worker_ratio = 1;
	val = dmlc::CHECK_NOTNULL(ps::Environment::Get()->find("EPOCH"));
	this->xpu = xpu; 
	core_num = xpu->core;
	data_counter = 0;
	target_epoch = atoi(val);
	current_epoch = 0;
	rank = ps::MyRank();
	kv_xpu = new ps::KVWorker<float>(0, 0);	
}

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
/*
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
}*/

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

	debugp("Recive data count: %ld\n", data_counter);
//	dm.PrintHead(rank, 3);
}


//pull feature, <keys, {feature}>
void MFWorker::PullFeature()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_FEATURE;
	
	keys.push_back(0);
	keys.push_back(1);

	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	int size_p = m * k;
	int size_q = n * k;
	
	memcpy(p, &vals[0], sizeof(float) * size_p);
	memcpy(q, &vals[size_p], sizeof(float) * size_q);
//	print_feature_tail(p, q, size_p, size_q, 3, 0);
}


//push format {keys0, feature_p} {keys1, feature_q} {lens0: m*k} {lens1: n*k}
void MFWorker::PushFeature()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PUSH_FEATURE;

	size_t size_p = m * k;
	size_t size_q = n * k; 

#ifdef CAL_RMSE
	vals.resize(size_p+size_q+1);
#else
	vals.resize(size_p+size_q);
#endif

	keys.push_back(0);
	keys.push_back(1);

	lens.push_back(size_p);
	lens.push_back(size_q);

	
	memcpy(&vals[0], p, sizeof(float) * size_p);
	memcpy(&vals[size_p], q, sizeof(float) * size_q);
//	print_feature_tail(p, q, size_p, size_q, 3, 0);

#ifdef CAL_RMSE
	keys.push_back(2);
	lens.push_back(1);
	vals[size_p+size_q] =  std::accumulate(loss.begin(), loss.end(), 0.0);
//	l = std::sqrt(l / size);
//	printf("[Worker %d]epoch: %d, loss: %f\n", rank, current_epoch, l);
#endif

	kv_xpu->Wait(kv_xpu->Push(keys, vals, lens, cmd));
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
	if(lens[0] != 5) {
		printf("[Worker %d] InitTestData: receive data fail!\n");
	}
	
	start = (int)vals[0];
	size = (int)vals[1];
	m = dm.rows = (int)vals[2];
	n = dm.cols = (int)vals[3];
	scale = vals[4];

	lambda_p = lambda_p / scale;
	lambda_q = lambda_q / scale;
	dm.nnz = size;
	size_t size_p = m * k;
	size_t size_q = n * k;
	p = (float *)aligned_alloc(64, size_p * sizeof(float));
	q = (float *)aligned_alloc(64, size_q * sizeof(float));
	debugp("[Worker %d] start: %ld, size: %ld, rows: %d, cols: %d\n", rank, start, size, dm.rows, dm.cols);
}

void MFWorker::GridProblem()
{
	Dim2 gridDim;
		
	gridDim.x =  2*core_num + 1;
	gridDim.y =  2*core_num + 1;
	
	dm.SetGrid(gridDim);
	dm.GridData(rank);
}

void MFWorker::CreateTasks()
{
	tids.resize(core_num);
#ifdef CAL_RMSE	
	loss.resize(core_num);
#endif
	for(int i = 0; i < core_num; i++) {
		CPUArgs arg;
		arg.tid = i;
		arg.workers = core_num;
		arg.target_epoch = target_epoch;
		arg.current_epoch = &current_epoch;
		arg.lambda_p = lambda_p;
		arg.lambda_q = lambda_q;
		arg.lrate = lrate;
		arg.p = p;
		arg.q = q;
		arg.dm = &dm;
		arg.cpuset = &cpuset;

#ifdef CAL_RMSE	
		arg.loss = &loss[i];
#endif

		args.push_back(arg);
	}

	for(int i = 0; i < core_num; i++) {
		pthread_create(&tids[i], NULL, sgd_kernel_hogwild_cpu, &args[i]);
	}
}

void MFWorker::JoinTasks()
{
	for(int i = 0; i < core_num; i++) {
		pthread_join(tids[i], NULL);
	}
}

void MFWorker::StartUpTasks()
{
	//wake up worker threads;
	pthread_mutex_lock(&cpu_workers_barrier_mutex);
	cpu_workers_complete = 0;
	current_epoch++;
	pthread_cond_broadcast(&cpu_workers_barrier_con);
	pthread_mutex_unlock(&cpu_workers_barrier_mutex);

	//sleep control threads;
	if(cpu_workers_complete == 0) {
		debugp("control_thread will block!\n");
		pthread_cond_wait(&control_wake_up_con,&control_wake_up_mutex);
	}
	debugp("control_thread wake up and do something...!\n");
	pthread_mutex_unlock(&control_wake_up_mutex);

	//wake up control threads by worker completed
	dm.ClearBlockFlags();	
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

void MFWorker::InitCPUAffinity()
{
	CPU_ZERO(&cpuset);
	int start = rank * core_num;
	int size = core_num;
	for (int i = start; i < start + size; i++)
        CPU_SET(i, &cpuset);
}

}
