#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <numeric>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "utils.h"
#include "ps/internal/env.h"
#include "dmlc/logging.h"
#include "mfworker.h"

namespace MF {

void MFWorker::Init()
{
	const char *val = NULL;
	//bind worker to numa node, default node is node 0
	val = Environment::Get()->find("NUMA_NODE");
	if(val != NULL) {
		numa_node = std::atoi(val);
	} else {
		numa_node = 0;
	}
	BindNumaNode(numa_node);
	
	XPU *xpu;
	val = CHECK_NOTNULL(Environment::Get()->find("XPU_TYPE"));
	if(strcmp(val, "CPU") == 0) {
		xpu = new CPU;
		
	} else if(strcmp(val, "GPU") == 0) {
		
	} else if(strcmp(val, "FPGA") == 0) {
		
	} else if(strcmp(val, "TPU") == 0) {
		
	} else {
		
	}

	xpu->Init();
	xpu->Bind();
	this->xpu = xpu;

	val = Environment::Get()->find("lambda");
	if(val != NULL) {
		lambda_p = lambda_q = strtod(val, NULL);
	}

	val = Environment::Get()->find("lrate");
	if(val != NULL) {
		lrate = strtod(val, NULL);
	}

	val = Environment::Get()->find("SHM");
	if(val != NULL) {
		use_shm = std::atoi(val);
	}

	val = Environment::Get()->find("TRANSMODE");
	if(val != NULL) {
		trans_mode = std::atoi(val);
	}
	
	rank = ps::MyRank();
	workers = xpu->workers;
	max_cores = xpu->max_cores;
	kv_xpu = new ps::KVWorker<float>(0, 0);	
	
}

void MFWorker::DeInit()
{
	delete kv_xpu;
	delete xpu;
}

//Push xpu information to server
void MFWorker::PushXPUInfo()
{
	std::vector<ps::Key> keys;									
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

//Send Init Training Data CMD to Server
void MFWorker::InitTrainingData()
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
	
	start = (size_t)vals[0];
	size = (size_t)vals[1];
	m = dm.rows = (size_t)vals[2];
	n = dm.cols = (size_t)vals[3];
	scale = vals[4];

	lambda_p = lambda_p / scale;
	lambda_q = lambda_q / scale;
	dm.nnz = size;

	debugp("[Worker %d] start: %ld, size: %ld, rows: %d, cols: %d\n", rank, start, size, dm.rows, dm.cols);
}

//Pull Training Data from Server
void MFWorker::PullTrainingData()
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

	size_t recv_size = keys.size();
	if(recv_size != size) {
		debugp("receive the training data fail, size: %d, recv_size: %d!\n", size, recv_size);
		return ;
	}

	Data& data = this->dm.data;
	data.r_matrix.resize(size);	
	data_counter = size;
	int len = 3;
	for(int i = 0; i < size; i++) {
		data.r_matrix[i].row_index = (int)vals[i * len + 0];
		data.r_matrix[i].col_index = (int)vals[i * len + 1];
		data.r_matrix[i].r = (float)vals[i * len + 2];
	}

	if(xpu->xpu_type == GPU) PullGPUData();
	debugp("Recive data count: %ld\n", data_counter);
}


void MFWorker::PrepareCPUResources()
{
	size_t size_p = m * k;
	size_t size_q = n * k;

#ifdef CAL_PORTION_RMSE
	feature = (float *)aligned_alloc(64, (size_p + size_q + 1) * sizeof(float));
#else
	feature = (float *)aligned_alloc(64, (size_p + size_q) * sizeof(float));
#endif
	p = feature;
	q = feature + size_p;

	if(trans_mode == HALFQ) {
		halfp = (uint16_t *)malloc(sizeof(uint16_t) * (size_p + size_q + 2));
		halfq = halfp + size_p;
	}

}

int MFWorker::PrepareShmbuf()
{
	int key = ftok("/home", rank);
	if(key == -1) {
    	perror("ftok fail!\n");
    	return -1;
	}
	size_t shm_size = sizeof(float)*(m * k + n * k);
	int shmid = shmget(key, shm_size, IPC_CREAT | 0777);
	if(shmid == -1) {
		perror("shmget fail!\n");
		return -1;
	}

	shm_buf = (unsigned char *)shmat(shmid, NULL, 0);
	if(!shm_buf) {
		perror("shmat fail!\n");
		return -1;
	}
	return 0;
}

void MFWorker::PrepareResources()
{
	if(xpu->xpu_type == CPU) {
		PrepareCPUResources();
	} else if(xpu->xpu_type == GPU) {
		PrepareGPUResources();
	}

	ps_vals.resize(m * k + n * k + 1);
	PrepareShmbuf();
}

void MFWorker::ReleaseCPUResources()
{
	free(feature);
	if(trans_mode == HALFQ)
		free(halfp);
}

void MFWorker::ReleaseResources()
{
	if(xpu->xpu_type == CPU) {
		ReleaseCPUResources();
	} else if(xpu->xpu_type == GPU) {
		ReleaseGPUResources();
	}
}

void MFWorker::GridProblem()
{
	Dim2 gridDim;
		
	gridDim.x = 2*workers + 1;
	gridDim.y = 2*workers + 1;
	
	dm.SetGrid(gridDim);
	dm.GridData(rank);
}


void MFWorker::PreProcess()
{
	Init();
	PushXPUInfo();
	InitTrainingData();
	PrepareResources();
	PullTrainingData();
	if(xpu->xpu_type == CPU) {
		GridProblem();
	}	
}

void MFWorker::PostProcess()
{
	ReleaseResources();
	DeInit();
}

void MFWorker::PullAll()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_ALL_FEATURE;

	xpu->current_epoch++;
	keys.push_back(0);
	keys.push_back(1);

	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	size_t transfer_size = (n+m)*k*sizeof(float);

	xpu->Transfer(p, &vals[0], transfer_size, TransferDirect::S2C);
}

void MFWorker::PullAllShm()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_ALL_FEATURE_SHM;

	xpu->current_epoch++;

	//Only request the server to copy data to share memory;
	keys.push_back(rank);
	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	size_t transfer_size = (n+m)*k*sizeof(float);
	xpu->Transfer(p, shm_buf, transfer_size, TransferDirect::S2C);
}

//This function pulls P and Q in the first epoch 
//And only pulls Q in other epoch
void MFWorker::PullQ()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_FEATURE;
	
	xpu->current_epoch++;
	
	if(current_epoch == 1) {
		keys.push_back(0);
		keys.push_back(1);
	} else {
		keys.push_back(0);
	}	

	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	size_t size_p = m * k;
	size_t size_q = n * k;

	if(xpu->current_epoch == 1) {
		xpu->Transfer(p, &vals[0], (size_p+size_q) * sizeof(float), TransferDirect::S2C);	
	} else {
		xpu->Transfer(q, &vals[0], (size_q) * sizeof(float), TransferDirect::S2C);	
	}
}

//Use share memory 
void MFWorker::PullQShm()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_FEATURE_SHM;

	xpu->current_epoch++;

	//Only request the server to copy data to share memory;
	keys.push_back(rank);
	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	size_t size_p = m * k;
	size_t size_q = n * k;

	if(xpu->current_epoch == 1) {
		xpu->Transfer(p, shm_buf, (size_p+size_q)*sizeof(float), TransferDirect::S2C);
	} else {
		xpu->Transfer(q, shm_buf, (size_q)*sizeof(float), TransferDirect::S2C);
	}
}

void MFWorker::PullHalfQ()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_HALF_FEATURE;

	uint16_t *h_p, *h_q;
	double start, elapse;	
	xpu->current_epoch++;	

	if(xpu->current_epoch == 1) {
		keys.push_back(0);
		keys.push_back(1);
	} else {
		keys.push_back(0);
	}

	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	size_t size_p = m * k;
	size_t size_q = n * k;

	if(xpu->current_epoch == 1) {
		//decode
		h_p = (uint16_t *)&vals[0];
		xpu->halfp2singles(p, h_p, size_p+size_q, max_cores);
	} else {
		h_q = (uint16_t *)&vals[0];
		xpu->halfp2singles(q, h_q, size_q, max_cores);
	}
}

void MFWorker::PullHalfQShm()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PULL_HALF_FEATURE_SHM;

	uint16_t *h_p, *h_q;

	keys.push_back(rank);
	kv_xpu->Wait(kv_xpu->Pull(keys, &vals, &lens, cmd));

	size_t size_p = m * k;
	size_t size_q = n * k;

	if(xpu->current_epoch == 1) {
		h_p = (uint16_t *)shm_buf;
		xpu->halfp2singles(p, h_p, size_p+size_q, max_cores);
	} else {
		h_q = (uint16_t *)shm_buf;
		xpu->halfp2singles(q, h_q, size_q, max_cores);
	}
}

void MFWorker::PushAll()
{
	std::vector<ps::Key> keys;
	
	std::vector<int> lens;
	CMD cmd = PUSH_ALL_FEATURE;

	size_t size_p = m * k;
	size_t size_q = n * k; 
	size_t trans_size;

	keys.push_back(0);
	keys.push_back(1);

	lens.push_back(size_p);
	lens.push_back(size_q);
	trans_size = size_p+size_q;

	xpu->Transfer(&ps_vals[0], p, trans_size * sizeof(float), TransferDirect::C2S);

#ifdef CAL_PORTION_RMSE
	keys.push_back(2);
	lens.push_back(1);
	ps_vals[trans_size] =  std::accumulate(loss.begin(), loss.end(), 0.0);
	trans_size += 1;
#endif

	kv_xpu->Wait(kv_xpu->Push(keys, ps_vals, lens, cmd));
}

void MFWorker::PushAllShm()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PUSH_ALL_FEATURE_SHM;

	size_t trans_size = (n+m)*k;

	xpu->Transfer(shm_buf, p, trans_size * sizeof(float), TransferDirect::C2S);

	keys.push_back(rank);
	vals.push_back(trans_size);
	lens.push_back(1);
	
#ifdef CAL_PORTION_RMSE
	keys.push_back(rank+1);
	lens.push_back(1);
	vals.push_back(std::accumulate(loss.begin(), loss.end(), 0.0));
#endif
	kv_xpu->Wait(kv_xpu->Push(keys, vals, lens, cmd));		
}

void MFWorker::PushQ()
{
	std::vector<ps::Key> keys;
	
	std::vector<int> lens;
	CMD cmd = PUSH_FEATURE;

	size_t size_p = m * k;
	size_t size_q = n * k; 	
	size_t trans_size;
	int index;
	float *src;

	//current_epoch < target_epoch, only push q 
	if(xpu->current_epoch < xpu->target_epoch) {
		keys.push_back(rank);
		lens.push_back(size_q);
		trans_size = size_q;
		index = 1;
		src = q;
	} else {
		keys.push_back(rank);
		keys.push_back(rank+1);
		lens.push_back(size_p);
		lens.push_back(size_q);
		trans_size = size_p + size_q;
		index = 2;
		src = p;
	}

	xpu->Transfer(&ps_vals[0], src, trans_size * sizeof(float), TransferDirect::C2S);

#ifdef CAL_PORTION_RMSE
	keys.push_back(rank+index);
	lens.push_back(1);
	ps_vals[trans_size] =  std::accumulate(loss.begin(), loss.end(), 0.0);
	trans_size += 1;
#endif

	kv_xpu->Wait(kv_xpu->Push(keys, ps_vals, lens, cmd));
}

void MFWorker::PushQShm()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PUSH_FEATURE_SHM;

	size_t size_p = m * k;
	size_t size_q = n * k; 
	size_t trans_size;
	float *src;

	//current_epoch < target_epoch, only push q	
	if(xpu->current_epoch < xpu->target_epoch) {
		trans_size = size_q;
		src = q;
	} else {
		trans_size = size_p+size_q;
		src = p;
	}

	xpu->Transfer(shm_buf, src, trans_size * sizeof(float), TransferDirect::C2S);

	keys.push_back(rank);
	vals.push_back(trans_size);
	lens.push_back(1);
	
#ifdef CAL_PORTION_RMSE
	keys.push_back(rank+1);
	lens.push_back(1);
	vals.push_back(std::accumulate(loss.begin(), loss.end(), 0.0));
#endif
	kv_xpu->Wait(kv_xpu->Push(keys, vals, lens, cmd));		
}

void MFWorker::PushHalfQ()
{
	std::vector<ps::Key> keys;
	std::vector<int> lens;
	CMD cmd = PUSH_HALF_FEATURE;

	size_t size_p = m * k;
	size_t size_q = n * k; 
	size_t trans_size;
	int index;
	float *src;
	uint16_t *dst;

	if(xpu->current_epoch < xpu->target_epoch) {
		keys.push_back(rank);
		lens.push_back(size_q/2);				//compress half point
		trans_size = size_q / 2;
		index = 1;
		src = q;
		dst = halfq;
	} else {
		keys.push_back(rank);
		keys.push_back(rank+1);
		lens.push_back(size_p/2);
		lens.push_back(size_q/2);
		trans_size = (size_p+size_q)/2;
		index = 2;
		src = p;
		dst = halfp;
	}

	xpu->singles2halfp(dst, src, trans_size, FE_TONEAREST, 0, max_cores);
	xpu->Transfer(&ps_vals[0], dst, trans_size, TransferDirect::C2S);
	
#ifdef CAL_PORTION_RMSE
	keys.push_back(rank+index);
	lens.push_back(1);
	ps_vals[trans_size] =  std::accumulate(loss.begin(), loss.end(), 0.0);
#endif

	kv_xpu->Wait(kv_xpu->Push(keys, ps_vals, lens, cmd));
}

void MFWorker::PushHalfQShm()
{
	std::vector<ps::Key> keys;
	std::vector<float> vals;
	std::vector<int> lens;
	CMD cmd = PUSH_HALF_FEATURE_SHM;

	size_t size_p = m * k;
	size_t size_q = n * k; 
	size_t trans_size;
	float *src;

	if(xpu->current_epoch < xpu->target_epoch) {
		trans_size = size_q;
		src = q;
	} else {
		trans_size = size_p+size_q;
		src = p;
	}

	keys.push_back(rank);
	vals.push_back(trans_size);
	lens.push_back(1);				//compress half point
	
	xpu->singles2halfp(shm_buf, src, trans_size, FE_TONEAREST, 0, max_cores);

#ifdef CAL_PORTION_RMSE
	keys.push_back(rank+1);
	lens.push_back(1);
	vals.push_back(std::accumulate(loss.begin(), loss.end(), 0.0));
#endif
	kv_xpu->Wait(kv_xpu->Push(keys, vals, lens, cmd));	
}

void MFWorker::Pull()
{
	switch(trans_mode) {
		case ALL: 
			use_shm ? PullAllShm() : PullAll();
			break;

		case Q:
			use_shm ? PullQShm() : PullQShm();
			break;

		case HALFQ:
			use_shm ? PullHalfQShm : PullHalfQ();
			break;

		default:
			printf("Unkown trans_mode, exit!\n");
			exit(-1);
	}
}

void MFWorker::Push()
{
	switch(trans_mode) {
		case ALL: 
			use_shm ? PushAllShm() : PushAll();
			break;

		case Q:
			use_shm ? PushQShm() : PushQShm();
			break;

		case HALFQ:
			use_shm ? PushHalfQShm : PushHalfQ();
			break;

		default:
			printf("Unkown trans_mode, exit!\n");
			exit(-1);
	}
}

void MFWorker::CreateWorkers(pFunc func)
{
#ifdef CAL_PORTION_RMSE
		loss.resize(workers);
#endif

	for(int i = 0; i < workers; i++) {
		Args args;
		args.lambda_p = lambda_p;
		args.lambda_q = lambda_q;
		args.lrate = lrate;
		args.p = p;
		args.q = q;
	
#ifdef CAL_PORTION_RMSE	
		args.loss = &loss[i];
#endif
		args.data = &dm;

		xpu->CreateTasks(i, func, &args);
	}
}

void MFWorker::Computing()
{
	xpu->RunTasks();
}

void MFWorker::JoinWorkers()
{
	xpu->JoinTasks();
}

}
