#ifndef _MFSERVER_H_
#define _MFSERVER_H_

#include <unordered_map>
#include <cstdint>
#include "ps/ps.h"
#include "xpu.h"
#include "mfdata.h"
#include "utils.h"

namespace MF {

class MFServer {
public:
	MFServer() : cpus(0), gpus(0), fpgas(0), tpus(0), xpus(0){}
	~MFServer() {if(trans_mode >= ALL_SHM && trans_mode <= HALFQ_SHM_EX) DestroyShmbuf(); delete server_xpu;}
	
	void Init();
	static void ProcessHandle(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

private:
	void GetWorkerInfo(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessInitTrainingData(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	
	void ProcessPullTrainingData(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPullAll(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPullAllShm(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPullQ(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPullQShm(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPullHalfQ(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPullHalfQShm(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPushAll(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPushAllShm(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPushQ(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPushQShm(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPushHalfQ(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void ProcessPushHalfQShm(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);	

	void ProcessPullHalfQShmEX(const ps::KVMeta& req_meta,
		      const ps::KVPairs<float>& req_data,
		      ps::KVServer<float>* server);

	void ProcessPushHalfQShmEX(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server);

	void SetCurServer();
	void PrepareData();
	int CreateShmbuf();
	void DestroyShmbuf();

/*
#ifdef TEST
	double record_start;
	double record_elapse = 0;
#endif
*/

private:
	int cpus;
	int gpus;
	int fpgas;
	int tpus;
	int xpus;

	int numa_node;
	int total_work_ratio;
	int quantify_data_threads;
	int prepare_data_threads;
	int received;							//received times from workers
	int pull_counts;
	int pull_shmid;
	TransMode trans_mode;

	unsigned char *pull_buf;

#ifdef CAL_PORTION_RMSE	
	float loss;
#endif	

	static MFServer *cur_server;
	ps::KVServer<float>* server_xpu;
	XPU *xpu;
	std::unordered_map<int, XPU_INFO> worker_xpu_info;
	std::unordered_map<int, std::pair<int, unsigned char *> > shm_buf;
	DataManager dm;
};

}

#endif
