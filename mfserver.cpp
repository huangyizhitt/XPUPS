#include "mfserver.h"
#include <cstdio>

namespace MF{

void MFServer::ReceiveXPUHandle(const ps::KVMeta& req_meta,
        const ps::KVPairs<XPU>& req_data,
        ps::KVServer<XPU>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t vals_size = req_data.vals.size();

	for(size_t i = 0; i < keys_size; i++) {
		if (req_meta.push) {
			xpu_info.insert(std::make_pair(req_data.keys[i], req_data.vals[i]));
		}
	}

}

void MFServer::PrintXPU()
{
	for(std::unordered_map<int, XPU>::iterator it = xpu_info.begin(); it != xpu_info.end(); it++) {
		printf("Worker %d, XPU: %s\n", it->first, it->second.name);
	}
}
	
}