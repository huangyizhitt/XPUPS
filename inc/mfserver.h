#ifndef _MFSERVER_H_
#define _MFSERVER_H_

#include <map>
#include "ps/ps.h"
#include "xpu.h"
#include "mfdata.h"

namespace MF{

class MFServer {
public:
	MFServer(XPU * const xpu_inf) : xpu(xpu_inf) {
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
	static size_t cpus;
	static size_t gpus;
	static size_t fpgas;
	static size_t tpus;
	static int max_workers;
	static int scale;
	ps::KVServer<float>* server_xpu;
	XPU *xpu;
	static std::map<int, XPU_INFO> worker_xpu_info;			//<XPU_TYPE, workers, work_ratio>
	static DataManager dm;
};

}

#endif
