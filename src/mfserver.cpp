#include "mfserver.h"
#include "utils.h"
#include "ps/base.h"
#include "ps/internal/postoffice.h"
#include <cstdio>
#include <mutex>
#include <numeric> 
#include <cmath>

namespace MF{

static int merge_feature = 0;

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

		dm.Init(nr_threads);

		Dim2 gridDim;
		
		gridDim.x = 1;
		gridDim.y = scale;
		dm.SetGrid(gridDim);
		dm.GridData(nr_threads);
		dm.InitModel();
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
	
	int rank = req_data.keys[0];
	int start = 0;
	int size = 0;

	dm.SplitData(start, size, worker_xpu_info[rank].work_ratio);

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
	
	merge_feature = 0;

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

	if(merge_feature == 0) {
		for(int i = 0; i < size_p; i++) {
			dm.model.p[i] = req_data.vals[i];
//			dm.model.p[i] = req_data.vals[i];
		}

		for(int i = size_p; i < size_p + size_q; i++) {
//			dm.model.q[i-size_p] = (dm.model.q[i-size_p] + req_data.vals[i]) / 2;
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
	
	merge_feature++;
//	print_feature_tail(&dm.model.p[0], &dm.model.q[0], size_p, size_q, 3, 1);
#ifdef CAL_RMSE	
	receive_times++;
	loss += req_data.vals.back();
	if(receive_times == xpus) {
		epoch++;
#ifdef CAL_ALLLOSS
		printf("Epoch %d loss %.4f\n", epoch, calc_rmse(dm.data.r_matrix, dm.model));
#else
		printf("Epoch %d loss %.4f\n", epoch, std::sqrt(loss / dm.nnz));
#endif
		receive_times = 0;
		loss = 0;
	}
#endif
	server->Response(req_meta, res);
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
		
		default:
			break;
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
