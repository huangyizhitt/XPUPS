#ifndef _MFSERVER_H_
#define _MFSERVER_H_

#include <unordered_map>
#include <cstdint>
#include <vector>
#include "ps/ps.h"
#include "xpu.h"
#include "mfdata.h"
#include "utils.h"

namespace MF {

#ifdef CAL_PORTION_RMSE
struct Record{
	float loss;
	double cur_time;

	Record() {}
	Record(const float& l, const double& t) : loss(l), cur_time(t) {}
};
#endif

class MFServer {
public:
	MFServer() : cpus(0), gpus(0), fpgas(0), tpus(0), xpus(0){}
	~MFServer() 
	{
#ifdef CAL_PORTION_RMSE	
		if(need_record) {
			RecordLoss();
		}
#endif
		if(trans_mode >= ALL_SHM && trans_mode < UNKONWN_MODE) {
		       	DestroyShmbuf();
		}	
		
		delete server_xpu;
	}
	
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

	void ProcessPullHalfQShmAcopy(const ps::KVMeta& req_meta,
						const ps::KVPairs<float>& req_data,
						ps::KVServer<float>* server);

	void ProcessPushHalfQShmAcopy(const ps::KVMeta& req_meta,
				  const ps::KVPairs<float>& req_data,
				  ps::KVServer<float>* server);
	
	void ProcessLinkShm(const ps::KVMeta& req_meta,
					  const ps::KVPairs<float>& req_data,
					  ps::KVServer<float>* server);

	void SetCurServer();
	void PrepareData();
	int CreateShmbuf();
	void DestroyShmbuf();
	int PrepareShmbuf();
	int LinkShmbuf(int worker_rank);

	void PinnedBuf(void* buf, size_t size);
	void UnpinnedBuf(void *buf);
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
	int pull_count;
	
	int pull_shmid;
	int my_rank;
	TransMode trans_mode;

	unsigned char *pull_buf;

#ifdef CAL_PORTION_RMSE	
	bool need_record = false;
	int record_counts;
	float loss;
	double start;
	double elapse;
	std::vector<Record> record;

	void RecordLoss();
	void ProcessStartRecord(const ps::KVMeta& req_meta,
                                          const ps::KVPairs<float>& req_data,
                                          ps::KVServer<float>* server);	
#endif	

	static MFServer *cur_server;
	ps::KVServer<float>* server_xpu;
	XPU *xpu;
	std::vector<int> pull_counts;
	std::vector<int> push_counts;
	std::unordered_map<int, XPU_INFO> worker_xpu_info;
	std::unordered_map<int, std::pair<int, unsigned char *> > shm_buf;
	DataManager dm;
};

}

#endif
