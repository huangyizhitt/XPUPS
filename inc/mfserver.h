#ifndef _MFSERVER_H_
#define _MFSERVER_H_

#include <unordered_map>
#include <cstdint>
#include "ps/ps.h"
#include "xpu.h"
#include "mfdata.h"

#ifdef EXPLORE
#include <iostream>
#include <fstream>
#endif


namespace MF{

class MFServer {
public:
	MFServer(XPU * const xpu_inf) : xpu(xpu_inf) {
		server_xpu = new ps::KVServer<float>(0);
		server_xpu->set_request_handle(ReceiveXPUHandle);
	}

	MFServer() {
		
	}

	~MFServer() {delete server_xpu;}

	void Init();

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

	static void ProcessPullPushFeature(const ps::KVMeta& req_meta,
							const ps::KVPairs<float>& req_data,
							ps::KVServer<float>* server);

	static void ProcessPullAllFeature(const ps::KVMeta& req_meta,
							  const ps::KVPairs<float>& req_data,
							  ps::KVServer<float>* server);

	static void ProcessPushAllFeature(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);		

	static void ProcessPullCompressFeature(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server);

	static void ProcessPushCompressFeature(const ps::KVMeta& req_meta,
							const ps::KVPairs<float>& req_data,
							ps::KVServer<float>* server);
	
	void PrintWorkerXPU();

	static void PrepareData();

	static void Test(const ps::KVMeta& req_meta,
                          const ps::KVPairs<float>& req_data,
                          ps::KVServer<float>* server);
	static void SetThreads(const int& t) {nr_threads = t;}

private:
	static int cpus;					//counts cpu
	static int gpus;					//counts gpu
	static int fpgas;				//counts fpga
	static int tpus;					//counts tpu
	static int xpus;					//counts xpu
	
	static int max_workers;
	static int scale;
	static int nr_threads;

	static int target_epoch;			
	static int current_epoch;
	static int receive_times;		//receive times from worker;

#ifdef CAL_PORTION_RMSE	
	static float loss;
#endif

#ifdef EXPLORE
	static std::ofstream out;
#endif

	ps::KVServer<float>* server_xpu;
	XPU *xpu;								//xpu of server
	static std::unordered_map<int, XPU_INFO> worker_xpu_info;			//<XPU_TYPE, workers, work_ratio>
	static DataManager dm;
};

}

#endif
