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

	ps::KVPairs<float> res;
	res.keys = req_data.keys;
    res.vals.resize(keys_size);


	server->Response(req_meta, res);
}

void MFServer::ReceiveXPUHandle(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	

	CMD cmd = req_meta.cmd;

	switch(cmd) {
		case PUSH_INFO:
			
			break;

		case PULL_DATA:

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
	for(std::unordered_map<int, int>::iterator it = worker_xpu_info.begin(); it != worker_xpu_info.end(); it++) {
		printf("Worker %d, XPU: %d\n", it->first, it->second);
	}
}

	
}
