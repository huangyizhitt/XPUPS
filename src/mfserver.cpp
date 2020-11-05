#include "mfserver.h"
#include "utils.h"
#include "ps/base.h"
#include "ps/internal/postoffice.h"
#include <cstdio>
#include <mutex>
#include <numeric> 
#include <cmath>
#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>

namespace MF{

//static int merge_feature = 0;

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
//static std::mutex mtx;

void MFServer::Init()
{
	const char* val = NULL;
	XPU *xpu = new XPU;
	xpu->Init();
	xpu->is_server = true;
	xpu->worker_ratio = 0;
	this->xpu = xpu;
	
	val = CHECK_NOTNULL(ps::Environment::Get()->find("EPOCH"));
	target_epoch = atoi(val);

	server_xpu = new ps::KVServer<float>(0);
	server_xpu->set_request_handle(ReceiveXPUHandle);
	SetDataThreads(xpu->max_core);
	SetThreads(xpu->workers);
	printf("Server XPU TYPE: %d, data threads: %d, work threads: %d\n", xpu->xpu_type, xpu->max_core, xpu->workers);
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
	printf("Worker: %d, XPU TYPE: %d, threads: %d, work_ratio: %d\n", worker_rank, xpu_info.type, xpu_info.workers, xpu_info.work_ratio);
	if(xpu_info.workers > max_workers) max_workers = xpu_info.workers;
	scale += xpu_info.work_ratio;
	worker_xpu_info.insert(std::make_pair(worker_rank, xpu_info));
	
	switch(xpu_info.type) {
		
		case CPU:
			cpus++;
			break;
		
		case GPU:
			gpus++;
			break;
		
		case FPGA:
			fpgas++;
			break;
		
		case TPU:
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

		dm.Init(data_nr_threads);

		Dim2 gridDim;
		
		gridDim.x = 1;
		gridDim.y = scale;
		dm.SetGrid(gridDim);
		dm.GridData(data_nr_threads);
		dm.InitModel();

#ifdef EXPLORE
		size_t size_q = dm.cols * dm.k;
		for(size_t i = 0; i < size_q; i++) {
			out << dm.model.q[i] << ",";
		}	
		out << std::endl;
		printf("%.7f\n", dm.model.q[10]);
#endif

		data_init_stage = true;
	}
}

//Prepare training data and send the data info to workers
void MFServer::ProcessInitData(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	ps::KVPairs<float> res;
	size_t n = req_data.keys.size();
	res.keys = req_data.keys;
	res.lens.resize(n);
	
	PrepareData();
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

//Process PULLDATA cmd from Workers, will send Data to Workers
//Data format{keys[0], MatrixNode[0]}
void MFServer::ProcessPullData(const ps::KVMeta& req_meta,
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

//Process PULL_FEATURE cmd from workers, will send feature to workers
//Data format{keys[0], p, keys[1], q}
void MFServer::ProcessPullFeature(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = size_p + size_q;
	
	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.vals.resize(vals_size);
	res.lens.resize(keys_size);

	if(current_epoch != 1) {
		res.lens[0] = size_q;
		memcpy(&res.vals[0], &dm.model.q[0], size_q * sizeof(float));
	} else {
		res.lens[0] = size_p;
		res.lens[1] = size_q;
		memcpy(&res.vals[0], &dm.model.p[0], size_p * sizeof(float));
		memcpy(&res.vals[size_p], &dm.model.q[0], size_q * sizeof(float));
	}
	
//	print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);
	server->Response(req_meta, res);
}
													  
//Process PUSH_FEATURE CMD from workers, will get feature from workers
//Data format{keys[0], p, keys[1], q}
//only in the last epoch need get feature p
void MFServer::ProcessPushFeature(const ps::KVMeta& req_meta,
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
	
	//printf("current_epoch: %d\n", current_epoch);	
	if(current_epoch != target_epoch) {
		if(receive_times == 0) {
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
		
		if(receive_times == 0) {
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

	receive_times++;
	if(receive_times == xpus) {
//		current_epoch++;

#ifdef CAL_PORTION_RMSE
		printf("Epoch %d loss %.4f\n", current_epoch, std::sqrt(loss / dm.nnz));
		loss = 0;
#endif

#ifdef CAL_RMSE
		if(current_epoch < target_epoch)
			printf("Epoch %d\n", current_epoch);
		else
			printf("Epoch %d global loss %.4f\n", current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		
#endif
		current_epoch++;
		receive_times = 0;
	}
	
}


#ifdef SEND_COMPRESS_Q_FEATURE
void MFServer::ProcessPullCompressFeature(const ps::KVMeta& req_meta,
							const ps::KVPairs<float>& req_data,
							ps::KVServer<float>* server)
{
  	size_t keys_size = req_data.keys.size();
  	size_t size_p = dm.rows * dm.k;
  	size_t size_q = dm.cols * dm.k;
  	size_t vals_size = (size_p + size_q)/2;
  
  	ps::KVPairs<float> res; 
  	res.keys = req_data.keys;

  	res.lens.resize(keys_size);

  	if(current_epoch != 1) {
		  //encode
		  singles2halfp(dm.halfq, &dm.model.q[0], size_q, FE_TONEAREST, 0, nr_threads);

		  //prepare transmission data
		  res.lens[0] = size_q/2;					  //compress
		  float *_q = (float *)dm.halfq;
		  res.vals.CopyFrom(_q, res.lens[0]);
		  
		  server->Response(req_meta, res);
	 } else {
		  //encode
		  singles2halfp(dm.halfp, &dm.model.p[0], size_p, FE_TONEAREST, 0, nr_threads);
		  singles2halfp(dm.halfq, &dm.model.q[0], size_q, FE_TONEAREST, 0, nr_threads);

		  //prepare transmission data
		  res.lens[0] = size_p/2;
		  res.lens[1] = size_q/2;
		  float *_p = (float *)dm.halfp;
		  res.vals.CopyFrom(_p, (size_p+size_q)/2);

		  server->Response(req_meta, res);
	 }
  
//  print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);
}

void MFServer::ProcessPullCompressFeatureUseShm(const ps::KVMeta& req_meta,
							const ps::KVPairs<float>& req_data,
							ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = (size_p + size_q)/2;
  
	ps::KVPairs<float> res; 
	res.keys = req_data.keys;
	int rank = req_data.keys[0];
	res.lens.resize(keys_size);
	res.vals.resize(1);
	
	if(current_epoch != 1) {
		  //encode
		  singles2halfp(dm.halfq, &dm.model.q[0], size_q, FE_TONEAREST, 0, nr_threads);

		  //prepare transmission data
		  res.vals[0] = size_q * sizeof(uint16_t);					  //compress
		  res.lens[0] = 1;
		  uint16_t *_buf = (uint16_t *)shm_buf[rank].second;
		  
		  memcpy(_buf, dm.halfq, res.vals[0]);
		 	
		  server->Response(req_meta, res);
	 } else {
		  //encode
		  singles2halfp(dm.halfp, &dm.model.p[0], size_p, FE_TONEAREST, 0, nr_threads);
		  singles2halfp(dm.halfq, &dm.model.q[0], size_q, FE_TONEAREST, 0, nr_threads);

		  //prepare transmission data
		  res.vals[0] = (size_p+size_q) * sizeof(uint16_t);
		  res.lens[0] = 1;
		  uint16_t *_buf = (uint16_t *)shm_buf[rank].second;
		  memcpy(_buf, dm.halfp, res.vals[0]);
		  server->Response(req_meta, res);
	 }
  
//	print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);
}


void MFServer::ProcessPushCompressFeature(const ps::KVMeta& req_meta,
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
  	if(current_epoch != target_epoch) {
		h_q = (uint16_t *)&req_data.vals[0];
	  	if(receive_times == 0) {
//			  memcpy(&dm.model.q[0], &req_data.vals[0], sizeof(float) * size_q);  
			halfp2singles(&dm.model.q[0], h_q, size_q, nr_threads);
	  	} else {
		  	for(int i = 0; i < size_q; i++) {
				float tmp;
				halfp2singles(&tmp, h_q+i, 1, nr_threads);
			  	dm.model.q[i] = (dm.model.q[i] + tmp) / 2;
			}
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
		halfp2singles(&dm.model.p[worker_start_p], &h_p[worker_start_p], worker_size_p, nr_threads);

	  	if(receive_times == 0) {
//		  	memcpy(&dm.model.q[0], &req_data.vals[size_p], sizeof(float) * size_q);
			halfp2singles(&dm.model.q[0], h_q, size_q, nr_threads);
	  	} else {
		  	for(int i = 0; i < size_q; i++) {
//			  	dm.model.q[i-size_p] = (dm.model.q[i-size_p] + req_data.vals[i]) / 2;
				float tmp;
				halfp2singles(&tmp, h_q+i, 1, nr_threads);
				dm.model.q[i] = (dm.model.q[i] + tmp) / 2;
		  	}
	  	}
  	}

  	server->Response(req_meta, res);
#ifdef CAL_PORTION_RMSE	
  	loss += req_data.vals.back();
#endif

  	receive_times++;
  	if(receive_times == xpus) {
//	  current_epoch++;

#ifdef CAL_PORTION_RMSE
	  	printf("Epoch %d loss %.4f\n", current_epoch, std::sqrt(loss / dm.nnz));
	  	loss = 0;
#endif

#ifdef CAL_RMSE
	  	if(current_epoch < target_epoch)
		  	printf("Epoch %d\n", current_epoch);
	  	else
		  	printf("Epoch %d global loss %.4f\n", current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		  
#endif
	  	current_epoch++;
	  	receive_times = 0;
  	}
  
}

void MFServer::ProcessPushCompressFeatureUseShm(const ps::KVMeta& req_meta,
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
	if(current_epoch != target_epoch) {
		h_q = (uint16_t *)shm_buf[rank].second;
		if(receive_times == 0) {
//			  memcpy(&dm.model.q[0], &req_data.vals[0], sizeof(float) * size_q);  
			halfp2singles(&dm.model.q[0], h_q, size_q, nr_threads);
		} else {
			for(int i = 0; i < size_q; i++) {
				float tmp;
				halfp2singles(&tmp, h_q+i, 1, nr_threads);
				dm.model.q[i] = (dm.model.q[i] + tmp) / 2;
			}
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
		halfp2singles(&dm.model.p[worker_start_p], &h_p[worker_start_p], worker_size_p, nr_threads);

		if(receive_times == 0) {
//			memcpy(&dm.model.q[0], &req_data.vals[size_p], sizeof(float) * size_q);
			halfp2singles(&dm.model.q[0], h_q, size_q, nr_threads);
		} else {
			for(int i = 0; i < size_q; i++) {
//				dm.model.q[i-size_p] = (dm.model.q[i-size_p] + req_data.vals[i]) / 2;
				float tmp;
				halfp2singles(&tmp, h_q+i, 1, nr_threads);
				dm.model.q[i] = (dm.model.q[i] + tmp) / 2;
			}
		}
	}

	server->Response(req_meta, res);
#ifdef CAL_PORTION_RMSE	
	loss += req_data.vals.back();
#endif

	receive_times++;
	if(receive_times == xpus) {
//	  current_epoch++;

#ifdef CAL_PORTION_RMSE
		printf("Epoch %d loss %.4f\n", current_epoch, std::sqrt(loss / dm.nnz));
		loss = 0;
#endif

#ifdef CAL_RMSE
		if(current_epoch < target_epoch)
			printf("Epoch %d\n", current_epoch);
		else
			printf("Epoch %d global loss %.4f\n", current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		  
#endif
		current_epoch++;
		receive_times = 0;
	}
  
}

#endif

void MFServer::ProcessPullPushFeature(const ps::KVMeta& req_meta,
							const ps::KVPairs<float>& req_data,
							ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = size_p + size_q;

	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.vals.resize(vals_size);
	res.lens.resize(keys_size);
		
	if(receive_times == 0) {
		memcpy(&dm.model.p[0], &req_data.vals[0], sizeof(float) * size_p);
		memcpy(&dm.model.q[0], &req_data.vals[size_p], sizeof(float) * size_q);
	} else {
		for(int i = 0; i < size_p; i++) {
			dm.model.p[i] = (dm.model.p[i] + req_data.vals[i]) / 2;	
		}

		for(int i = size_p; i < size_p + size_q; i++) {
			dm.model.q[i-size_p] = (dm.model.q[i-size_p] + req_data.vals[i]) / 2;				
		}
	}

#ifdef CAL_PORTION_RMSE	
	loss += req_data.vals.back();
#endif
	
	receive_times++;
	if(receive_times == xpus) {
	//		current_epoch++;
		memcpy(&res.vals[0], &dm.model.p[0], size_p * sizeof(float));
		memcpy(&res.vals[size_p], &dm.model.q[0], size_q * sizeof(float));
		print_feature_head(&res.vals[0], &res.vals[size_p], 3, 1);
		res.lens[0] = size_p;
		res.lens[1] = size_q;
		server->Response(req_meta, res);
#ifdef CAL_PORTION_RMSE
		printf("Epoch %d loss %.4f\n", current_epoch, std::sqrt(loss / dm.nnz));
		loss = 0;
#endif
	
#ifdef CAL_RMSE
		if(current_epoch < target_epoch)
			printf("Epoch %d\n", current_epoch);
		else
			printf("Epoch %d loss %.4f\n", current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		
#endif
		current_epoch++;
		receive_times = 0;	
	}
}

//Process PULL_FEATURE cmd from workers, will send feature to workers
//Data format{keys[0], p, keys[1], q}
void MFServer::ProcessPullAllFeature(const ps::KVMeta& req_meta,
							  const ps::KVPairs<float>& req_data,
							  ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t size_p = dm.rows * dm.k;
	size_t size_q = dm.cols * dm.k;
	size_t vals_size = size_p + size_q;

	ps::KVPairs<float> res;
	res.keys = req_data.keys;
	res.vals.resize(vals_size);
	res.lens.resize(keys_size);

	res.lens[0] = size_p;
	res.lens[1] = size_q;

	memcpy(&res.vals[0], &dm.model.p[0], size_p * sizeof(float));
	memcpy(&res.vals[size_p], &dm.model.q[0], size_q * sizeof(float));
//	print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);
	server->Response(req_meta, res);
}

//Process PUSH_FEATURE CMD from workers, will get feature from workers
//Data format{keys[0], p, keys[1], q}
void MFServer::ProcessPushAllFeature(const ps::KVMeta& req_meta,
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

	if(receive_times == 0) {
		for(int i = 0; i < size_p; i++) {
			dm.model.p[i] = req_data.vals[i];

		}

		for(int i = size_p; i < size_p + size_q; i++) {
			dm.model.q[i-size_p] = req_data.vals[i];
		} 
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
	
	receive_times++;
	if(receive_times == xpus) {
	
#ifdef CAL_PORTION_RMSE
		printf("Epoch %d loss %.4f\n", current_epoch, std::sqrt(loss / dm.nnz));
		loss = 0;
#endif
	
#ifdef CAL_RMSE
		if(current_epoch < target_epoch) {
			printf("Epoch %d\n", current_epoch);
		} else {
			printf("Epoch %d global loss %.4f\n", current_epoch, calc_rmse(dm.data.r_matrix, dm.model));		
		}
#endif
		current_epoch++;
		receive_times = 0;
	}
}


							  
void MFServer::Test(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	size_t keys_size = req_data.keys.size();
	size_t vals_size = req_data.vals.size();
	ps::KVPairs<float> res;

	res.keys = req_data.keys;
	res.vals.resize(keys_size * 3);
	res.lens.resize(keys_size);

	for(int i = 0; i < keys_size * 3; i+=3) {
		res.vals[i] = i;
		res.vals[i+1] = i+1;
		res.vals[i+2] = i+2;
//		res.lens[i] = 3;
		printf("vals[%d]: %d\n", i, i);
	}
	printf("key_size %d\n", keys_size);
	server->Response(req_meta, res);
}

void MFServer::ReceiveXPUHandle(const ps::KVMeta& req_meta,
                              const ps::KVPairs<float>& req_data,
                              ps::KVServer<float>* server)
{
	CMD cmd = (CMD) req_meta.cmd;
	
	switch(cmd) {
		case PUSH_INFO:
			GetWorkerInfo(req_meta, req_data, server);
			break;

		case INIT_DATA:
			ProcessInitData(req_meta, req_data, server);
			break;

		case PULL_DATA:
			ProcessPullData(req_meta, req_data, server);
			break;

		case PULL_FEATURE:
			ProcessPullFeature(req_meta, req_data, server);
			break;
				
			
		case PUSH_FEATURE:
			ProcessPushFeature(req_meta, req_data, server);
			break;

		case PULL_PUSH_FEATURE:
			ProcessPullPushFeature(req_meta, req_data, server);
			break;

		case PULL_ALL_FEATURE:
			ProcessPullAllFeature(req_meta, req_data, server);
			break;

		case PUSH_ALL_FEATURE:
			ProcessPushAllFeature(req_meta, req_data, server);
			break;
#ifdef SEND_COMPRESS_Q_FEATURE	
		case PULL_HALF_FEATURE:
			ProcessPullCompressFeature(req_meta, req_data, server);
			break;

		case PUSH_HALF_FEATURE:
	//		printf("entry push half feature\n");
			ProcessPushCompressFeature(req_meta, req_data, server);
			break;

		case PULL_HALF_FEATURE_SHM:
			ProcessPullCompressFeatureUseShm(req_meta, req_data, server);
			break;

		case PUSH_HALF_FEATURE_SHM:
			ProcessPushCompressFeatureUseShm(req_meta, req_data, server);
			break;
#endif

		default:
			break;
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

void MFServer::DestroyShmbuf()
{
	for(const auto& n : shm_buf) {
		shmdt(n.second.second);
		shmctl(n.second.first, IPC_RMID, NULL);
	}
}


void MFServer::PrintWorkerXPU()
{
	XPU_INFO xpu_info;
	int worker_rank;
	for(std::unordered_map<int, XPU_INFO>::iterator it = worker_xpu_info.begin(); it != worker_xpu_info.end(); it++) {
		worker_rank = it->first;
		xpu_info = it->second;
		printf("Worker: %d, XPU TYPE: %d, workers: %d, work_ratio: %d\n", worker_rank, xpu_info.type, xpu_info.workers, xpu_info.work_ratio);
	}
}
							  
}
