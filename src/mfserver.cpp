#include "mfserver.h"
#include "utils.h"
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
		dm.Init();
		data_init_stage = true;
	}
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
/*		res.vals[i-start] = i;
                res.vals[i+1-start] = i+1;
                res.vals[i+2-start] = i+2; */
	}

	printf("start: %d, keys_size: %d, vals_size:%d, lens:%d\n", start, keys_size, res.vals.size(), res.lens.size());
	server->Response(req_meta, res);
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
	for(std::unordered_map<int, XPU_INFO>::iterator it = worker_xpu_info.begin(); it != worker_xpu_info.end(); it++) {
		worker_rank = it->first;
		xpu_info = it->second;
		printf("Worker: %d, XPU TYPE: %d, workers: %d, work_ratio: %d\n", worker_rank, xpu_info.type, xpu_info.workers, xpu_info.work_ratio);
	}
}

	
}
