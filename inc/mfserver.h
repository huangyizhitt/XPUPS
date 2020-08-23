#ifndef _MFSERVER_H_
#define _MFSERVER_H_

#include <unordered_map>
#include "ps/ps.h"
#include "xpu.h"
#include "mfdata.h"

namespace MF{

class MFServer {
public:
	MFServer(XPU * const xpu_inf) : xpu_info(xpu_inf) {
		server_xpu = new ps::KVServer<float>(0);
		server_xpu->set_request_handle(ReceiveXPUHandle);
	}

	~MFServer() {delete server_xpu;}

	static void ReceiveXPUHandle(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	static void GetWorkerInfo(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	static void PushDataToWorker(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	static void PushDataInfoToWorker(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	void PrintWorkerXPU();

	static void PrepareData();

	static void Test(const ps::KVMeta& req_meta,
                          const ps::KVPairs<float>& req_data,
                          ps::KVServer<float>* server);


private:
	static std::unordered_map<int, float> worker_xpu_info;			//<rank, performance>
	ps::KVServer<float>* server_xpu;
	XPU *xpu_info;
	static DataManager dm;
};

}

#endif
