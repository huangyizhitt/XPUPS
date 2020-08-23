#include "mfserver.h"
#include "utils.h"
#include <cstdio>

namespace MF{

void MFServer::GetWorkerInfo(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t vals_size = req_data.vals.size();
	size_t lens_size = req_data.lens.size();

	for(int i = 0; i < keys_size; i++) {
		printf("Worker: %d, peak_performance: %f, bandwidth: %f, kv pair len: %d\n", req_data.keys[i],
			req_data.vals[i * req_data.lens[i]], req_data.vals[i * req_data.lens[i] + 1], req_data.lens[i]);
	}
}

void MFServer::PrepareData()
{
	dm.LoadData();
}

void MFServer::PushDataToWorker(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	KVPairs<Val> res;
	size_t keys_size = dm.nnz;
	size_t vals_size = dm.nnz * 3;
	size_t lens = 3;

	res.keys.resize(keys_size);
	res.vals.resize(vals_size);
	res.vals.resize(dm.nnz);
	for(size_t i = 0; i < keys_size; i++) {
		res.keys.push_back(i);
		res.vals.push_back(dm.data.r_matrix[i].row_index);
		res.vals.push_back(dm.data.r_matrix[i].col_index);
		res.vals.push_back(dm.data.r_matrix[i].r);
		res.lens.push_back(lens);
	}

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

		case PULL_DATA:
			PrepareData();
			PushDataToWorker();
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
	for(std::unordered_map<int, float>::iterator it = worker_xpu_info.begin(); it != worker_xpu_info.end(); it++) {
		printf("Worker %d, XPU: %d\n", it->first, it->second);
	}
}

	
}
