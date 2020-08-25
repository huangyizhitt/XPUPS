#include "mfserver.h"
#include "utils.h"
#include "ps/base.h"
#include "ps/internal/postoffice.h"
#include <cstdio>
#include <atomic>

namespace MF{

static std::atomic<bool> data_init_stage(false);

void MFServer::GetWorkerInfo(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t vals_size = req_data.vals.size();
	size_t lens_size = req_data.lens.size();

	int worker_rank = req_data.keys[0];
	XPU_INFO xpu_info;
	

	xpu_info.type = (XPU_TYPE)req_data.vals[0];
	xpu_info.workers = (int)req_data.vals[1];
	xpu_info.work_ratio = (int)req_data.vals[2];
	printf("Worker: %d, XPU TYPE: %d, workers: %d, work_ratio: %d\n", worker_rank, xpu_info.type, xpu_info.workers, xpu_info.work_ratio);
	if(xpu_info.workers > max_workers) max_workers = xpu_info.workers;
	scale += xpu_info.work_ratio;
	worker_xpu_info.insert(std::make_pair(worker_rank, xpu_info));
	
	switch(xpu_info.type) {
		
		case CPU:
			cpus++;
			break;
		
		case GPU:
			gpus++;
			break;
		
		case FPGA:
			fpgas++;
			break;
		
		case TPU:
			tpus++;
			break;

		default:
			break;
	}
	
	ps::KVPairs<float> res;
	server->Response(req_meta, res);
}

void MFServer::PushDataInfoToWorker(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	ps::KVPairs<float> res;
	size_t n = req_data.keys.size();
	res.keys = req_data.keys;
	res.vals.push_back(0);
	res.vals.push_back(dm.nnz);
	res.lens.resize(n);

	server->Response(req_meta, res);
}

void MFServer::PrepareData()
{
	if(!data_init_stage) {

		dm.Init(nr_threads);

		Dim2 gridDim;
		
		gridDim.x = 2 * scale;
		gridDim.y = 2 * scale;
		dm.SetGrid(gridDim);
		dm.GridData(nr_threads);
		dm.InitModel();
		data_init_stage = true;
	}
}

//push free block id, format <xpu_rank, {free block id by work_ratio}>
//push feature matrix p, format<1, {model->p}>
//push feature matrix q, format<2, {model->q}>
void MFServer::PushBlockAndFeature(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	int xpu_rank = req_data.keys[0];   //keys[0]->rank, keys[1]->rank+1, keys[2]->rank+2
	int work_ratio = worker_xpu_info[xpu_rank].work_ratio;
	int blockId;

	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.lens.resize(keys_size);

	//len[0]=work_ratio, len[1]=size_p, len[2]=size_q;
	res.lens[0] = work_ratio;
	int size_p = dm.rows * dm.k;
	int size_q = dm.cols * dm.k;
	res.lens[1] = size_p;			  
	res.lens[2] = size_q;
	res.vals.resize(work_ratio+size_p+size_q);

	for(int i = 0; i < res.vals.size(); i++) {
		if(i < work_ratio) {
			while((blockId = dm.FindFreeBlock()) < 0);					//sync epoch
			res.vals[i] = blockId;
		} else if(i < work_ratio + size_p) {
			res.vals[i] = dm.model.p[i-work_ratio];
		} else {
			res.vals[i] = dm.model.q[i-work_ratio-size_p];
		}
	}

	debugp("will respose block and feature!\n");
	server->Response(req_meta, res);
}

void MFServer::PushDataToWorker(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{	
	size_t keys_size = req_data.keys.size();
	size_t vals_size = keys_size * 3;
	ps::KVPairs<float> res;

	res.keys = req_data.keys;
	res.vals.resize(vals_size);
	res.lens.resize(keys_size);
	
	size_t start = req_data.keys[0];

	for(size_t i = start; i < start + keys_size * 3; i+=3) {
		res.vals[i-start] = (dm.data.r_matrix[i/3].row_index);
		res.vals[i+1-start] = (dm.data.r_matrix[i/3].col_index);
		res.vals[i+2-start] = (dm.data.r_matrix[i/3].r);
	}

	debugp("start: %d, keys_size: %d, vals_size:%d, lens:%d\n", start, keys_size, res.vals.size(), res.lens.size());
	server->Response(req_meta, res);
}

//pull feature format {keys0, feature_p} {keys1, feature_q} {lens0: m*k} {lens1: n*k}
void MFServer::PullFeature(const ps::KVMeta& req_meta,
							const ps::KVPairs<float>& req_data,
							ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t vals_size = req_data.vals.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	int work_ratio = req_data.lens[0];

	debugp("feature size: %ld\n", vals_size);
	if(size_p != req_data.lens[1] || size_q != req_data.lens[2]) {
		printf("[Server] receive feature fail!\n");
		exit(-1);
	}
	
/*	res.keys = req_data.keys;
	res.lens.resize(keys_size);	*/

	//需要修改同步参数法则
	memcpy(&dm.model.p[0], &req_data.vals[work_ratio], sizeof(float) * size_p);
	memcpy(&dm.model.q[0], &req_data.vals[work_ratio+size_p], sizeof(float) * size_q);

	for(int i = 0; i < work_ratio; i++) {
		dm.SetBlockFree(req_data.vals[i]);
	}

	EpochStatus status = dm.EpochComplete();

	if(status == CompleteOnece) {
		//commpute loss
		printf("Epoch: %d, loss: \n", dm.current_epoch);
		dm.ClearBlockTable();
	}

	if(status == CompleteAll) {
	//	PushStopWorker(req_meta, req_data, server);
		printf("Epoch: %d, loss: \n", dm.current_epoch);
		ps::Finalize(0, true);
	} else {
		ps::KVPairs<float> res;
		server->Response(req_meta, res);
	}
}

							  
void MFServer::Test(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t vals_size = req_data.vals.size();
	ps::KVPairs<float> res;

	res.keys = req_data.keys;
	res.vals.resize(keys_size * 3);
	res.lens.resize(keys_size);

	for(int i = 0; i < keys_size * 3; i+=3) {
		res.vals[i] = i;
		res.vals[i+1] = i+1;
		res.vals[i+2] = i+2;
//		res.lens[i] = 3;
		printf("vals[%d]: %d\n", i, i);
	}
	printf("key_size %d\n", keys_size);
	server->Response(req_meta, res);
}

void MFServer::PushStopWorker(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	ps::KVPairs<float> res;

	res.keys = req_data.keys;
	res.lens.resize(keys_size);
	for(size_t i = 0; i < keys_size; i++) 
		res.vals.push_back(STOP_WORKER);	
	
	server->Response(req_meta, res);
}

void MFServer::ReceiveXPUHandle(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	CMD cmd = (CMD) req_meta.cmd;
	
	switch(cmd) {
		case PUSH_INFO:
			GetWorkerInfo(req_meta, req_data, server);
			break;
		case PULL_DATA_INFO:
			PrepareData();
			PushDataInfoToWorker(req_meta, req_data, server);
			break;

		case PULL_DATA:
			PushDataToWorker(req_meta, req_data, server);
			//Test(req_meta, req_data, server);
			break;

		case PULL_FEATURE:
			
			PushBlockAndFeature(req_meta, req_data, server);
			break;
				
			
		case PUSH_FEATURE:
			PullFeature(req_meta, req_data, server);
			break;
		
		default:
			break;
	}

/*	ps::KVPairs<float> res;
	res.keys = req_data.keys;
    res.vals.resize(keys_size);
	for(size_t i = 0; i < keys_size; i++) {
		if (req_meta.push) {
			worker_xpu_info.insert(std::make_pair(req_data.keys[i], req_data.vals[i]));
		}
	}

	server->Response(req_meta, res); */       
}


void MFServer::PrintWorkerXPU()
{
	XPU_INFO xpu_info;
	int worker_rank;
	for(std::map<int, XPU_INFO>::iterator it = worker_xpu_info.begin(); it != worker_xpu_info.end(); it++) {
		worker_rank = it->first;
		xpu_info = it->second;
		printf("Worker: %d, XPU TYPE: %d, workers: %d, work_ratio: %d\n", worker_rank, xpu_info.type, xpu_info.workers, xpu_info.work_ratio);
	}
}
							  
}
