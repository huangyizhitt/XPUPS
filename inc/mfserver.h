#ifndef _MFSERVER_H_
#define _MFSERVER_H_

#include <unordered_map>
#include "ps/ps.h"
#include "xpu.h"

namespace MF{

class MFServer {
public:
	MFServer() {
		server_xpu = new ps::KVServer<int>(0);
		server_xpu->set_request_handle(ReceiveXPUHandle);
	}

	~MFServer() {delete server_xpu;}

	static void ReceiveXPUHandle(const ps::KVMeta& req_meta,
                              const ps::KVPairs<int>& req_data,
                              ps::KVServer<int>* server);

	void PrintXPU();

private:
	static std::unordered_map<int, int> xpu_info;
	ps::KVServer<int>* server_xpu;
};

}

#endif
