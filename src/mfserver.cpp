#include "mfserver.h"
#include <cstdio>

namespace MF{

void MFServer::ReceiveXPUHandle(const ps::KVMeta& req_meta,
                              const ps::KVPairs<int>& req_data,
                              ps::KVServer<int>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t vals_size = req_data.vals.size();

	ps::KVPairs<int> res;
	res.keys = req_data.keys;
        res.vals.resize(keys_size);
	for(size_t i = 0; i < keys_size; i++) {
		if (req_meta.push) {
			xpu_info.insert(std::make_pair(req_data.keys[i], req_data.vals[i]));
		}
	}

        server->Response(req_meta, res);        
}


void MFServer::PrintXPU()
{
	for(std::unordered_map<int, int>::iterator it = xpu_info.begin(); it != xpu_info.end(); it++) {
		printf("Worker %d, XPU: %d\n", it->first, it->second);
	}
}
	
}
