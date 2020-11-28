#include "mfserver.h"
#include "utils.h"
#include "ps/base.h"
#include "ps/internal/postoffice.h"
#include "ps/internal/env.h"
#include <cstdio>
#include <mutex>
#include <numeric> 
#include <cmath>
#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>

using namespace ps;

namespace MF {

#ifdef CAL_RMSE	
static double calc_rmse(std::vector<MatrixNode>& R, Model& model)
{
	double loss = 0;
	size_t nnz = R.size();
	size_t k = model.k;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+:loss)
#endif	
	for(size_t i = 0; i < nnz; i++) {
		MatrixNode &N = R[i];
		float *p = &model.p[N.row_index * k];
		float *q = &model.q[N.col_index * k];
		float e = N.r - std::inner_product(p, p+k, q, ((float)0.0f));
		loss += e*e;
	}
	return std::sqrt(loss / nnz);
}
#endif

static bool data_init_stage(false);
MFServer* MFServer::cur_server = NULL;

void MFServer::SetCurServer()
{
	cur_server = this;
}

void MFServer::Init()
{
	SetCurServer();
	//bind server to numa node, default node is node 0
	const char *val = NULL;
	val = Environment::Get()->find("NUMA_NODE");
	if(val != NULL) {
		numa_node = std::atoi(val);
	} else {
		numa_node = 0;
	}
	BindNumaNode(numa_node);

	xpu = new XPU;
	xpu->xpu_type = XPU_TYPE::CPU;
	xpu->Init();
	xpu->current_epoch = 1;

	server_xpu = new ps::KVServer<float>(0);	
	server_xpu->set_request_handle(ProcessHandle);

	total_work_ratio = 0;
	prepare_data_threads = xpu->max_cores;
	quantify_data_threads = xpu->workers;

#ifdef CAL_PORTION_RMSE	
	loss = 0.0;
#endif

	const char *file_path = "netflix_train.bin";
	val = Environment::Get()->find("DATA_PATH");
	if(val != NULL) {
		file_path = val;
	}	 

	val = Environment::Get()->find("TRANSMODE");
	if(val != NULL) {
		trans_mode = static_cast<TransMode>(std::atoi(val));
	} else {
		trans_mode = ALL;
	}

	(trans_mode == HALFQ || trans_mode == HALFQ_SHM) ? dm.Init(file_path, true) : dm.Init(file_path, false);
	printf("Server XPU TYPE: %d, data threads: %d, work threads: %d\n", static_cast<int>(xpu->xpu_type), xpu->max_cores, xpu->workers);
}

void MFServer::ProcessHandle(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server)
{
	CMD cmd = (CMD) req_meta.cmd;

	switch(cmd) {
		case PUSH_INFO:
			cur_server->GetWorkerInfo(req_meta, req_data, server);
			break;

		case INIT_DATA:
			cur_server->ProcessInitTrainingData(req_meta, req_data, server);
			break;

		case PULL_DATA:
			cur_server->ProcessPullTrainingData(req_meta, req_data, server);
			break;

		case PULL_Q_FEATURE:
			cur_server->ProcessPullQ(req_meta, req_data, server);
			break;
				
			
		case PUSH_Q_FEATURE:
			cur_server->ProcessPushQ(req_meta, req_data, server);
			break;

		case PULL_Q_FEATURE_SHM:
			cur_server->ProcessPullQShm(req_meta, req_data, server);
			break;

		case PUSH_Q_FEATURE_SHM:
			cur_server->ProcessPushQShm(req_meta, req_data, server);
			break;

		case PULL_ALL_FEATURE:
			cur_server->ProcessPullAll(req_meta, req_data, server);
			break;

		case PUSH_ALL_FEATURE:
			cur_server->ProcessPushAll(req_meta, req_data, server);
			break;

		case PULL_ALL_FEATURE_SHM:
			cur_server->ProcessPullAllShm(req_meta, req_data, server);
			break;

		case PUSH_ALL_FEATURE_SHM:
			cur_server->ProcessPushAllShm(req_meta, req_data, server);
			break;
	
		case PULL_HALF_FEATURE:
			cur_server->ProcessPullHalfQ(req_meta, req_data, server);
			break;

		case PUSH_HALF_FEATURE:
			cur_server->ProcessPushHalfQ(req_meta, req_data, server);
			break;

		case PULL_HALF_FEATURE_SHM:
			cur_server->ProcessPullHalfQShm(req_meta, req_data, server);
			break;

		case PUSH_HALF_FEATURE_SHM:
			cur_server->ProcessPushHalfQShm(req_meta, req_data, server);
			break;

		default:
			break;
	}

}

//Get Worker XPU Info
void MFServer::GetWorkerInfo(const ps::KVMeta& req_meta,
							const ps::KVPairs<float>& req_data,
							ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t vals_size = req_data.vals.size();
	size_t lens_size = req_data.lens.size();

	int worker_rank = req_data.keys[0];
	XPU_INFO xpu_info;

	xpu_info.type = (XPU_TYPE)req_data.vals[0];
	xpu_info.workers = (int)req_data.vals[1];
	xpu_info.work_ratio = (int)req_data.vals[2];
	printf("Worker: %d, XPU TYPE: %d, threads: %d, work_ratio: %d\n", worker_rank, static_cast<int>(xpu_info.type), xpu_info.workers, xpu_info.work_ratio);
	total_work_ratio += xpu_info.work_ratio;
	worker_xpu_info.insert(std::make_pair(worker_rank, xpu_info));

	switch(xpu_info.type) {
	  
		case XPU_TYPE::CPU:
		 	cpus++;
			break;
	  
		case XPU_TYPE::GPU:
		  	gpus++;
		  	break;
	  
		case XPU_TYPE::FPGA:
		  	fpgas++;
		  	break;
	  
	  	case XPU_TYPE::TPU:
		  	tpus++;
		  	break;

	  	default:
		 	break;
	}
	xpus++;
	ps::KVPairs<float> res;
	server->Response(req_meta, res);
}


void MFServer::PrepareData()
{
	if(!data_init_stage) {

		dm.PrepareTrainingData(prepare_data_threads);

		Dim2 gridDim;
		
		gridDim.x = 1;
		gridDim.y = total_work_ratio;
		dm.SetGrid(gridDim);
		dm.GridData(prepare_data_threads);
		dm.InitModel();

		data_init_stage = true;
	}
}

int MFServer::CreateShmbuf()
{
	for(const auto& n : worker_xpu_info) {
		int worker_rank = n.first;
		int key = ftok("/home", worker_rank);
		if(key == -1) {
        	perror("ftok fail!\n");
        	return -1;
		}
		size_t size = (size_t)sizeof(float)*(dm.rows * dm.k + dm.cols * dm.k);
		int shmid = shmget(key, size, IPC_CREAT | 0777);
		if(shmid == -1) {
			perror("shmget fail!\n");
			return -1;
		}

		unsigned char *buf = 
			(unsigned char *)shmat(shmid, NULL, 0);
		if(!buf) {
			perror("shmat fail!\n");
			return -1;
		}
		shm_buf[worker_rank] = std::make_pair(shmid, buf);
	}
	return 0;
}

void MFServer::ProcessInitTrainingData(const ps::KVMeta& req_meta,
					  const ps::KVPairs<float>& req_data,
					  ps::KVServer<float>* server)
{
	ps::KVPairs<float> res;
	size_t n = req_data.keys.size();
	res.keys = req_data.keys;
	res.lens.resize(n);
	
	PrepareData();

	if(trans_mode >= ALL_SHM && trans_mode <= HALFQ_SHM)
		CreateShmbuf();
	
	int rank = req_data.keys[0];
	int start = 0;
	int size = 0;

	dm.SplitData(start, size, worker_xpu_info[rank].work_ratio);
	worker_xpu_info[rank].start = start;
	worker_xpu_info[rank].size = size;
	res.vals.push_back((float)start);
	res.vals.push_back((float)size);
	res.vals.push_back((float)dm.rows);
	res.vals.push_back((float)dm.cols);
	res.vals.push_back(dm.scale);
	res.lens[0] = 5;
	server->Response(req_meta, res);	
}

void MFServer::DestroyShmbuf()
{
	for(const auto& n : shm_buf) {
		shmdt(n.second.second);
		shmctl(n.second.first, IPC_RMID, NULL);
	}
}


void MFServer::ProcessPullTrainingData(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server)	
{
	size_t keys_size = req_data.keys.size();
	size_t vals_size = keys_size * 3;
	ps::KVPairs<float> res;

	res.keys = req_data.keys;
	res.vals.resize(vals_size);
	res.lens.resize(keys_size);

	size_t start = req_data.keys[0];

	for(size_t i = start, j = 0; i < start + keys_size; i++, j+=3) {
		res.vals[j] = (dm.data.r_matrix[i].row_index);
		res.vals[j+1] = (dm.data.r_matrix[i].col_index);
		res.vals[j+2] = (dm.data.r_matrix[i].r);
		res.lens[i-start] = 3;
	}

	debugp("start: %d, keys_size: %d, vals_size:%d, lens:%d\n", start, keys_size, res.vals.size(), res.lens.size());
	//dm.PrintHead(start, 3);
	server->Response(req_meta, res);	
}

//Process PULL_Q_FEATURE cmd from workers, will send P and Q feature to workers in the first epoch
//Will send Q feature to workers in the other epoch
void MFServer::ProcessPullQ(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server)	
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size;
	float *src;
	
	ps::KVPairs<float> res;
	res.keys = req_data.keys;

	res.lens.resize(keys_size);

	if(xpu->current_epoch != 1) {
		res.lens[0] = size_q;
		vals_size = size_q;
		src = dm.model.q;
	} else {
		res.lens[0] = size_p;
		res.lens[1] = size_q;
		vals_size = size_p + size_q;
		src = dm.model.p;
	}

	//zero-copy constructor and operator=
	ps::SArray<float> vals(src, vals_size);
	res.vals = vals;

//	print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);
	server->Response(req_meta, res);	
}

void MFServer::ProcessPushQ(const ps::KVMeta& req_meta,
                      const ps::KVPairs<float>& req_data,
                      ps::KVServer<float>* server)	
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = req_data.vals.size();
	
	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.lens.resize(keys_size);
	
	//printf("current_epoch: %d\n", xpu->current_epoch);	
	if(xpu->current_epoch != xpu->target_epoch) {
		if(received == 0) {
			memcpy(&dm.model.q[0], &req_data.vals[0], sizeof(float) * size_q);	
		} else {
			for(int i = 0; i < size_q; i++) {
				dm.model.q[i] = (dm.model.q[i] + req_data.vals[i]) / 2;
			}
		}
	} else {
		int rank = req_data.keys[0];
		int start = worker_xpu_info[rank].start;
		int size = worker_xpu_info[rank].size;

		//Get feature p in the last epoch	
		int worker_start_p = dm.data.r_matrix[start].row_index * dm.k;
		int worker_size_p = (dm.data.r_matrix[start+size-1].row_index - dm.data.r_matrix[start].row_index + 1) * dm.k;
		memcpy(&dm.model.p[worker_start_p], &req_data.vals[worker_start_p], sizeof(float) * worker_size_p);
		
		if(received == 0) {
			memcpy(&dm.model.q[0], &req_data.vals[size_p], sizeof(float) * size_q);
		} else {
			for(int i = size_p; i < size_p + size_q; i++) {
				dm.model.q[i-size_p] = (dm.model.q[i-size_p] + req_data.vals[i]) / 2;
			}
		}
	}

	server->Response(req_meta, res);
#ifdef CAL_PORTION_RMSE	
	loss += req_data.vals.back();
#endif

	received++;
	if(received == xpus) {
//		current_epoch++;

#ifdef CAL_PORTION_RMSE
		printf("Epoch %d loss %.4f\n", xpu->current_epoch, std::sqrt(loss / dm.nnz));
		loss = 0;
#endif

#ifdef CAL_RMSE
		if(xpu->current_epoch < xpu->target_epoch)
			printf("Epoch %d\n", xpu->current_epoch);
		else
			printf("Epoch %d global loss %.4f\n", xpu->current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		
#endif
		xpu->current_epoch++;
		received = 0;
	}	
}

void MFServer::ProcessPullQShm(const ps::KVMeta& req_meta,
					const ps::KVPairs<float>& req_data,
					ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;

	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	int rank = req_data.keys[0];
	res.vals.resize(1);
	res.lens.resize(keys_size);
	res.lens[0] = 1;
	
	float *buf = (float *)shm_buf[rank].second;
	size_t size;
  	
	if(xpu->current_epoch != 1) {
		res.vals[0] = size = size_q * sizeof(float);
	  	memcpy(buf, &dm.model.q[0], res.vals[0]);
//		 printf("[Process Pull] dm.model.q[0]: %.3f, dm.model.q[1]: %.3f, dm.model.q[2]: %.3f\n", dm.model.q[0], dm.model.q[1], dm.model.q[2]); 
  	} else {
		res.vals[0] = size = (size_q+size_p) * sizeof(float);
		memcpy(buf, &dm.model.p[0], size);
  	}
//  print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);
	server->Response(req_meta, res);	
}

void MFServer::ProcessPushQShm(const ps::KVMeta& req_meta,
                  const ps::KVPairs<float>& req_data,
                  ps::KVServer<float>* server)	
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = req_data.vals.size();

	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.lens.resize(keys_size);

	int rank = req_data.keys[0];
	float *buf = (float *)shm_buf[rank].second;

	//printf("current_epoch: %d\n", current_epoch); 
	if(xpu->current_epoch != xpu->target_epoch) {
	  if(received == 0) {
		  memcpy(&dm.model.q[0], buf, sizeof(float) * size_q); 
	//		 printf("[Process push] dm.model.q[0]: %.3f, dm.model.q[1]: %.3f, dm.model.q[2]: %.3f\n", dm.model.q[0], dm.model.q[1], dm.model.q[2]); 
	  } else {
		  for(int i = 0; i < size_q; i++) {
			  dm.model.q[i] = (dm.model.q[i] + buf[i]) / 2;
		  }
	  }
	} else {
	  int start = worker_xpu_info[rank].start;
	  int size = worker_xpu_info[rank].size;

	  //Get feature p in the last epoch   
	  int worker_start_p = dm.data.r_matrix[start].row_index * dm.k;
	  int worker_size_p = (dm.data.r_matrix[start+size-1].row_index - dm.data.r_matrix[start].row_index + 1) * dm.k;
	  memcpy(&dm.model.p[worker_start_p], &buf[worker_start_p], sizeof(float) * worker_size_p);
	  
	  if(received == 0) {
		  memcpy(&dm.model.q[0], &buf[size_p], sizeof(float) * size_q);
	  } else {
		  for(int i = size_p; i < size_p + size_q; i++) {
			  dm.model.q[i-size_p] = (dm.model.q[i-size_p] + buf[i]) / 2;
		  }
	  }
	}

	server->Response(req_meta, res);
#ifdef CAL_PORTION_RMSE	
	loss += req_data.vals.back();
#endif

	received++;
	if(received == xpus) {
	//	  current_epoch++;

#ifdef CAL_PORTION_RMSE
	  printf("Epoch %d loss %.4f\n", xpu->current_epoch, std::sqrt(loss / dm.nnz));
	  loss = 0;
#endif

#ifdef CAL_RMSE
	  if(xpu->current_epoch < xpu->target_epoch)
		  printf("Epoch %d\n", xpu->current_epoch);
	  else
		  printf("Epoch %d global loss %.4f\n", xpu->current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		  
#endif
	  xpu->current_epoch++;
	  received = 0;
	}
}

void MFServer::ProcessPullAll(const ps::KVMeta& req_meta,
                  const ps::KVPairs<float>& req_data,
                  ps::KVServer<float>* server)	
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = size_p + size_q;

	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.lens.resize(keys_size);

	res.lens[0] = size_p;
	res.lens[1] = size_q;

	ps::SArray<float> vals(&dm.model.p[0], vals_size);
	res.vals = vals;
//	print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);
	server->Response(req_meta, res);	
}

void MFServer::ProcessPushAll(const ps::KVMeta& req_meta,
                  const ps::KVPairs<float>& req_data,
                  ps::KVServer<float>* server)	
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = req_data.vals.size();
	
	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.lens.resize(keys_size);

	if(received == 0) {
		memcpy(&dm.model.p[0], &req_data.vals[0], (size_p + size_q)*sizeof(float)); 
	} else {
		for(int i = 0; i < size_p; i++) {
			dm.model.p[i] = (dm.model.p[i] + req_data.vals[i]) / 2;
		}

		for(int i = size_p; i < size_p + size_q; i++) {
			dm.model.q[i-size_p] = (dm.model.q[i-size_p] + req_data.vals[i]) / 2;
		}
	}

	server->Response(req_meta, res);
#ifdef CAL_PORTION_RMSE	
	loss += req_data.vals.back();
#endif
	
	received++;
	if(received == xpus) {
	
#ifdef CAL_PORTION_RMSE
		printf("Epoch %d loss %.4f\n", xpu->current_epoch, std::sqrt(loss / dm.nnz));
		loss = 0;
#endif
	
#ifdef CAL_RMSE
		if(xpu->current_epoch < xpu->target_epoch) {
			printf("Epoch %d\n", current_epoch);
		} else {
			printf("Epoch %d global loss %.4f\n", xpu->current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		
		}
#endif
		xpu->current_epoch++;
		received = 0;
	}	
}

void MFServer::ProcessPullHalfQ(const ps::KVMeta& req_meta,
				const ps::KVPairs<float>& req_data,
				ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size;
	float *src;

	ps::KVPairs<float> res; 
	res.keys = req_data.keys;

	res.lens.resize(keys_size);
	if(xpu->current_epoch != 1) {
	  	//encode
	  	cpu_singles2halfp(dm.halfq, &dm.model.q[0], size_q, FE_TONEAREST, 0, quantify_data_threads);

		//prepare transmission data
		res.lens[0] = vals_size = size_q/2;					  //compress
		src = (float *)dm.halfq; 
	} else {
		//encode
		cpu_singles2halfp(dm.halfp, &dm.model.p[0], size_p+size_q, FE_TONEAREST, 0, quantify_data_threads);

		//prepare transmission data
		res.lens[0] = size_p/2;
		res.lens[1] = size_q/2;
		vals_size = (size_p+size_q)/2;
		src = (float *)dm.halfp;
	}
	
	res.vals = ps::SArray<float>(src, vals_size);
	server->Response(req_meta, res);
//  print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);	
}

void MFServer::ProcessPushHalfQ(const ps::KVMeta& req_meta,
					  const ps::KVPairs<float>& req_data,
					  ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = req_data.vals.size();
	uint16_t *h_p;
	uint16_t *h_q;
	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.lens.resize(keys_size);
  
  	//printf("current_epoch: %d\n", current_epoch); 
  	if(xpu->current_epoch != xpu->target_epoch) {
		h_q = (uint16_t *)&req_data.vals[0];
	  	if(received == 0) {
//			  memcpy(&dm.model.q[0], &req_data.vals[0], sizeof(float) * size_q);  
			cpu_halfp2singles(&dm.model.q[0], h_q, size_q, quantify_data_threads);
	  	} else {
#if defined(USE_AVX2) || defined(USE_AVX512)
			halfp2singles_madd(&dm.model.q[0], h_q, size_q, quantify_data_threads, 0.5);
#else
		  	for(int i = 0; i < size_q; i++) {
				float tmp;
				cpu_halfp2singles(&tmp, h_q+i, 1, quantify_data_threads);
			  	dm.model.q[i] = (dm.model.q[i] + tmp) / 2;
			}
#endif
	  	}
  	} else {
	  	int rank = req_data.keys[0];
	  	int start = worker_xpu_info[rank].start;
	  	int size = worker_xpu_info[rank].size;

	  	//Get feature p in the last epoch   
	  	int worker_start_p = dm.data.r_matrix[start].row_index * dm.k;
	  	int worker_size_p = (dm.data.r_matrix[start+size-1].row_index - dm.data.r_matrix[start].row_index + 1) * dm.k;
//	  	printf("start: %d, start_p: %d, size_p: %d\n", start, worker_start_p, worker_size_p);
//	  	memcpy(&dm.model.p[worker_start_p], &req_data.vals[worker_start_p], sizeof(float) * worker_size_p);
		h_p = (uint16_t *)&req_data.vals[0];
		h_q = h_p + size_p;
		cpu_halfp2singles(&dm.model.p[worker_start_p], &h_p[worker_start_p], worker_size_p, quantify_data_threads);

	  	if(received == 0) {
//		  	memcpy(&dm.model.q[0], &req_data.vals[size_p], sizeof(float) * size_q);
			cpu_halfp2singles(&dm.model.q[0], h_q, size_q, quantify_data_threads);
	  	} else {
#if defined(USE_AVX2) || defined(USE_AVX512)
			halfp2singles_madd(&dm.model.q[0], h_q, size_q, quantify_data_threads, 0.5);
#else
		  	for(int i = 0; i < size_q; i++) {
//			  	dm.model.q[i-size_p] = (dm.model.q[i-size_p] + req_data.vals[i]) / 2;
				float tmp;
				cpu_halfp2singles(&tmp, h_q+i, 1, quantify_data_threads);
				dm.model.q[i] = (dm.model.q[i] + tmp) / 2;
		  	}
#endif
	  	}
  	}

  	server->Response(req_meta, res);
#ifdef CAL_PORTION_RMSE	
  	loss += req_data.vals.back();
#endif

  	received++;
  	if(received == xpus) {
//	  current_epoch++;

#ifdef CAL_PORTION_RMSE
	  	printf("Epoch %d loss %.4f\n", xpu->current_epoch, std::sqrt(loss / dm.nnz));
	  	loss = 0;
#endif

#ifdef CAL_RMSE
	  	if(xpu->current_epoch < xpu->target_epoch)
		  	printf("Epoch %d\n", xpu->current_epoch);
	  	else
		  	printf("Epoch %d global loss %.4f\n", xpu->current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		  
#endif
	  	xpu->current_epoch++;
	  	received = 0;
  	}	
}

void MFServer::ProcessPullHalfQShm(const ps::KVMeta& req_meta,
				const ps::KVPairs<float>& req_data,
				ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;

	ps::KVPairs<float> res; 
	res.keys = req_data.keys;
	int rank = req_data.keys[0];
	res.lens.resize(keys_size);
	res.vals.resize(1);

	if(xpu->current_epoch != 1) {
		//encode
		cpu_singles2halfp(dm.halfq, &dm.model.q[0], size_q, FE_TONEAREST, 0, quantify_data_threads);

		//prepare transmission data
		res.vals[0] = size_q * sizeof(uint16_t);					//compress
		res.lens[0] = 1;
		uint16_t *_buf = (uint16_t *)shm_buf[rank].second;
		
		memcpy(_buf, dm.halfq, res.vals[0]);
		  
		server->Response(req_meta, res);
	} else {
		//encode
		cpu_singles2halfp(dm.halfp, &dm.model.p[0], size_p+size_q, FE_TONEAREST, 0, quantify_data_threads);
		//prepare transmission data
		res.vals[0] = (size_p+size_q) * sizeof(uint16_t);
		res.lens[0] = 1;
		uint16_t *_buf = (uint16_t *)shm_buf[rank].second;
		memcpy(_buf, dm.halfp, res.vals[0]);
		server->Response(req_meta, res);
	}
	//  print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);

}

void MFServer::ProcessPushHalfQShm(const ps::KVMeta& req_meta,
				  const ps::KVPairs<float>& req_data,
				  ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = req_data.vals.size();
	uint16_t *h_p;
	uint16_t *h_q;
	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.lens.resize(keys_size);

	int rank = req_data.keys[0];
	//printf("current_epoch: %d\n", current_epoch); 
	if(xpu->current_epoch != xpu->target_epoch) {
		h_q = (uint16_t *)shm_buf[rank].second;
		if(received == 0) {
//			  memcpy(&dm.model.q[0], &req_data.vals[0], sizeof(float) * size_q);  
			cpu_halfp2singles(&dm.model.q[0], h_q, size_q, quantify_data_threads);
		} else {
#if defined(USE_AVX2) || defined(USE_AVX512)
			halfp2singles_madd(&dm.model.q[0], h_q, size_q, quantify_data_threads, 0.5);
#else
			for(int i = 0; i < size_q; i++) {
				float tmp;
				cpu_halfp2singles(&tmp, h_q+i, 1, quantify_data_threads);
				dm.model.q[i] = (dm.model.q[i] + tmp) / 2;
			}
#endif
		}
	} else {
		int start = worker_xpu_info[rank].start;
		int size = worker_xpu_info[rank].size;

		//Get feature p in the last epoch	
		int worker_start_p = dm.data.r_matrix[start].row_index * dm.k;
		int worker_size_p = (dm.data.r_matrix[start+size-1].row_index - dm.data.r_matrix[start].row_index + 1) * dm.k;
//		printf("start: %d, start_p: %d, size_p: %d\n", start, worker_start_p, worker_size_p);
//		memcpy(&dm.model.p[worker_start_p], &req_data.vals[worker_start_p], sizeof(float) * worker_size_p);
		h_p = (uint16_t *)shm_buf[rank].second;
		h_q = h_p + size_p;
		cpu_halfp2singles(&dm.model.p[worker_start_p], &h_p[worker_start_p], worker_size_p, quantify_data_threads);
		if(received == 0) {
//			memcpy(&dm.model.q[0], &req_data.vals[size_p], sizeof(float) * size_q);
			cpu_halfp2singles(&dm.model.q[0], h_q, size_q, quantify_data_threads);
		} else {
#if defined(USE_AVX2) || defined(USE_AVX512)
			halfp2singles_madd(&dm.model.q[0], h_q, size_q, quantify_data_threads, 0.5);
#else
			for(int i = 0; i < size_q; i++) {
//				dm.model.q[i-size_p] = (dm.model.q[i-size_p] + req_data.vals[i]) / 2;
				float tmp;
				cpu_halfp2singles(&tmp, h_q+i, 1, quantify_data_threads);
				dm.model.q[i] = (dm.model.q[i] + tmp) / 2;
			}
#endif
		}
	}

	server->Response(req_meta, res);
#ifdef CAL_PORTION_RMSE	
	loss += req_data.vals.back();
#endif

	received++;
	if(received == xpus) {
//	  current_epoch++;

#ifdef CAL_PORTION_RMSE
		printf("Epoch %d loss %.4f\n", xpu->current_epoch, std::sqrt(loss / dm.nnz));
		loss = 0;
#endif

#ifdef CAL_RMSE
		if(xpu->current_epoch < xpu->target_epoch)
			printf("Epoch %d\n",  xpu->current_epoch);
		else
			printf("Epoch %d global loss %.4f\n",  xpu->current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		  
#endif
		xpu->current_epoch++;
		received = 0;
	}	
}

void MFServer::ProcessPullAllShm(const ps::KVMeta& req_meta,
					const ps::KVPairs<float>& req_data,
					ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = size_p + size_q;

	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	int rank = req_data.keys[0];
	res.vals.resize(1);
	res.vals[0] = vals_size;
	res.lens.resize(keys_size);
	res.lens[0] = 1;

	float *buf = (float *)shm_buf[rank].second;
	memcpy(buf, &dm.model.p[0], vals_size * sizeof(float));
//	print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);
	server->Response(req_meta, res);	
}

void MFServer::ProcessPushAllShm(const ps::KVMeta& req_meta,
                  const ps::KVPairs<float>& req_data,
                  ps::KVServer<float>* server)	
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = req_data.vals.size();

	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.lens.resize(keys_size);

	int rank = req_data.keys[0];
	float *buf = (float *)shm_buf[rank].second;

	//printf("current_epoch: %d\n", current_epoch); 

	if(received == 0) {
		memcpy(&dm.model.p[0], buf, (size_p+size_q)*sizeof(float));
	} else {
			for(int i = 0; i < size_p; i++) {
				dm.model.p[i] = (dm.model.p[i] + buf[i]) / 2;
			}

			for(int i = size_p; i < size_p + size_q; i++) {
				dm.model.q[i-size_p] = (dm.model.q[i-size_p] + buf[i]) / 2;
			}
	}

	server->Response(req_meta, res);
#ifdef CAL_PORTION_RMSE	
	loss += req_data.vals.back();
#endif

	received++;
	if(received == xpus) {
	//	  current_epoch++;

#ifdef CAL_PORTION_RMSE
	  printf("Epoch %d loss %.4f\n", xpu->current_epoch, std::sqrt(loss / dm.nnz));
	  loss = 0;
#endif

#ifdef CAL_RMSE
	  if(xpu->current_epoch < xpu->target_epoch)
		  printf("Epoch %d\n", xpu->current_epoch);
	  else
		  printf("Epoch %d global loss %.4f\n", xpu->current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		  
#endif
	  xpu->current_epoch++;
	  received = 0;
	}	
}
				  
}
