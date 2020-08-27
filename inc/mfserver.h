#ifndef _MFSERVER_H_
#define _MFSERVER_H_

#include <unordered_map>
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

	static void PushBlockAndFeature(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	static void PullFeature(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	static void PushStopWorker(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	static void ProcessInitData(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	static void ProcessPullData(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	static void ProcessPushFeature(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	static void ProcessPullFeature(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);
	
	void PrintWorkerXPU();

	static void PrepareData();

	static void Test(const ps::KVMeta& req_meta,
                          const ps::KVPairs<float>& req_data,
                          ps::KVServer<float>* server);
	static void SetThreads(const int& t) {nr_threads = t;}

private:
	static size_t cpus;
	static size_t gpus;
	static size_t fpgas;
	static size_t tpus;
	static int max_workers;
	static int scale;
	static int nr_threads;
	ps::KVServer<float>* server_xpu;
	XPU *xpu;
	static std::unordered_map<int, XPU_INFO> worker_xpu_info;			//<XPU_TYPE, workers, work_ratio>
	static DataManager dm;
};

}

#endif
