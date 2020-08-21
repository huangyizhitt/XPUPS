#ifndef _MFSERVER_H_
#define _MFSERVER_H_

#include <unordered_map>
#include "ps/ps.h"
#include "xpu.h"

namespace MF{

class MFServer {
public:
	Server() {
		server_xpu = new ps::KVServer<XPU>(0);
		server_xpu->set_request_handle(ReceiveXPUHandle);
	}

	~Server() {delete server_xpu;}

	void ReceiveXPUHandle(const ps::KVMeta& req_meta,
        const ps::KVPairs<XPU>& req_data,
        ps::KVServer<XPU>* server);

	void PrintXPU();

private:
	std::unordered_map<int, XPU> xpu_info;
	ps::KVServer<XPU>* server_xpu;
};

}

#endif
