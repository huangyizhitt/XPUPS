#include "mfdata.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <unordered_set>
#include <numeric>
#include <random>

namespace MF {

struct sort_node_by_p
{
    bool operator() (MatrixNode const &lhs, MatrixNode const &rhs)
    {
        return std::tie(lhs.row_index, lhs.col_index) < std::tie(rhs.row_index, rhs.col_index);
    }
};

struct sort_node_by_q
{
    bool operator() (MatrixNode const &lhs, MatrixNode const &rhs)
    {
        return std::tie(lhs.col_index, lhs.row_index) < std::tie(rhs.col_index, rhs.row_index);
    }
};

void DataManager::Init(const char * file, const bool& use_half)
{
	train_file_path = file;
	this->use_half = use_half;
	k = 128;
}

bool DataManager::LoadData()
{
	double start, elapse;
	start = cpu_second();
	
	if(train_file_path)
			fp = fopen(train_file_path, "rb");

	if(fp == nullptr) {
		printf("Fail: cannot open %s!\n", train_file_path);
		return false;
	}
	
	fseek(fp, 0L, SEEK_END);
    nnz = ftell(fp) / 12;
	data.r_matrix.resize(nnz);
	rewind(fp);
	
	size_t idx = 0;
	while(true) {
		int flag = 0;
		int row_index, col_index;
		float r;

		flag += fread(&row_index, sizeof(int), 1, fp);
		flag += fread(&col_index, sizeof(int), 1, fp);
		flag += fread(&r, sizeof(float), 1, fp);

		if(flag != 3) break;

		if(row_index + 1 > rows) rows = row_index + 1;
		if(col_index + 1 > cols) cols = col_index + 1;

		data.r_matrix[idx].row_index = row_index;
		data.r_matrix[idx].col_index = col_index;
		data.r_matrix[idx].r = r;
		idx++;
	}

	fclose(fp);
	
	elapse = cpu_second() - start;
	printf("rows:%d, cols:%d, nnz:%ld, idx:%ld, cost time: %f\n",rows, cols, nnz, idx, elapse);
	return true;
}

void DataManager::CollectDataInfo(int nr_threads)
{
	float ex = 0, ex2 = 0;
	
#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:ex,ex2)
#endif

	for(size_t i = 0; i < nnz; i++)
    {
        MatrixNode &N = data.r_matrix[i];
        ex += (double)N.r;
        ex2 += (double)N.r*N.r;
    }

    ex /= (float)nnz;
    ex2 /= (float)nnz;
    means = (float)ex;
    stddev = (float)sqrt(ex2-ex*ex);
}

int* DataManager::GenerateRandomMap(size_t size)
{
	std::srand(time(NULL));
    std::vector<int> map(size, 0);
    for(size_t i = 0; i < size; i++)
        map[i] = i;

    random_shuffle(map.begin(), map.end());
    
    int *map_ptr = (int *)malloc(size * sizeof(int));
    for(size_t i = 0; i < size; i++)map_ptr[i] = map[i];
    
    return map_ptr;
}

int* DataManager::GenerateInvMap(int *map, size_t size)
{
	int*inv_map = (int *)malloc(size * sizeof(int));
    for(int i = 0;i < size;i++)inv_map[map[i]] = i;
    return inv_map;
}

void DataManager::DestroyMap(int *map)
{
	if(map) {
		free(map);
		map = nullptr;
	}
}

void DataManager::ScaleData(float mf_scale, int nr_threads)
{
	if(mf_scale == 1.0)
		return;

#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(static)
#endif
	for(size_t i = 0; i < nnz; i++)
		data.r_matrix[i].r *= mf_scale;	
}

void DataManager::ShuffleData()
{
	for(size_t i = 0; i < nnz; i++)
    {
        MatrixNode &N = data.r_matrix[i];
        N.row_index = p_map[N.row_index];
        N.col_index = q_map[N.col_index];
    }
}

void DataManager::PrepareTrainingData(int nr_threads)
{
	double start, elapse;
	scale = 1.0f;
	
	if(!LoadData()) {
		printf("Load data fail, abort!\n");
		exit(1);
	}
	CollectDataInfo(nr_threads);
	scale = std::max((float)1e-4, stddev);

	printf("shuffle problem ...\n");
	start = cpu_second();
	p_map = GenerateRandomMap(rows);
	q_map = GenerateRandomMap(cols);
	inv_p_map = GenerateInvMap(p_map, rows);
	inv_q_map = GenerateInvMap(q_map, cols);
	ShuffleData();
	ScaleData(1.0/scale, nr_threads);
	printf("time elapsed:%.8fs\n",(cpu_second() - start));

	counts_p.resize(rows, 0);
	counts_q.resize(cols, 0);
}

void DataManager::SetGrid(const Dim2& grid_dim)
{
	grid.gridDim = grid_dim;
	int nr_bins_x = grid.gridDim.x;
	int nr_bins_y = grid.gridDim.y;
	
	block_size = nr_bins_x * nr_bins_y;
	remain_blocks = block_size;
	complete_blocks = 0;
	counts_epoch.resize(block_size, 1);

	grid.blockDim.x = (int)ceil((double)cols / nr_bins_x);
	grid.blockDim.y = (int)ceil((double)rows / nr_bins_y);

	busy_x.resize(nr_bins_x, false);
	busy_y.resize(nr_bins_y, false);
}

void DataManager::GridData(int nr_threads)
{
	printf("Grid Problem to all XPU...\n");
	double start, elapse;
	start = cpu_second();
	counts.resize(block_size, 0);
	for(size_t i = 0; i < nnz; i++) {
		MatrixNode N = data.r_matrix[i];
		int blockIdx = GetBlockId(grid, N);
		counts[blockIdx] += 1;
		counts_p[N.row_index] += 1;
		counts_q[N.col_index] += 1;
	}

	std::vector<MatrixNode *>& ptrs = grid.blocks;
	ptrs.resize(block_size + 1);
	ptrs[0] = &data.r_matrix[0];

	for(int block = 0; block < block_size; block++) {
		ptrs[block+1] = ptrs[block] + counts[block];
	}

	std::vector<MatrixNode*> pivots(ptrs.begin(), ptrs.end()-1);
    	for(int block = 0; block < block_size; ++block)
    	{
        	for(MatrixNode* pivot = pivots[block]; pivot != ptrs[block+1];)
        	{
            		int curr_block = GetBlockId(grid, *pivot);
			if(curr_block == block)
            		{
               			++pivot;
                		continue;
            		}

            		MatrixNode *next = pivots[curr_block];
			std::swap(*pivot, *next);
            		pivots[curr_block] += 1;
        	}
    	}
	

#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(dynamic)
#endif
    for(int block = 0; block < block_size; ++block)
    {
        if(rows > cols)
            std::sort(ptrs[block], ptrs[block+1], sort_node_by_p());
        else
            std::sort(ptrs[block], ptrs[block+1], sort_node_by_q());
    }	
    elapse = cpu_second() - start;
    printf("Grid Problem to all XPU complete, cost: %.8f\n", elapse);
}

int DataManager::GetBlockId(Grid& grid, MatrixNode& n)
{
	return (n.row_index/grid.blockDim.y) * grid.gridDim.x + n.col_index/grid.blockDim.x;
}

int DataManager::GetBlockId(Grid& grid, int row, int col)
{
	return row * grid.gridDim.x + col; 
}

void DataManager::CountFeature()
{
	for(size_t i = 0; i < nnz; i++) {
		MatrixNode N = data.r_matrix[i];
		int blockIdx = GetBlockId(grid, N);
		counts_p[N.row_index] += 1;
		counts_q[N.col_index] += 1;
	}
}

void DataManager::InitModel()
{
	
	model.m = rows;
	model.n = cols;
	model.k = k;
	model.scale = means / scale;

	std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
	float s = (float)sqrt(1.0/k);
	
	model.feature.resize((rows+cols) * k, 0);
	model.p = &model.feature[0];
	model.q = &model.feature[rows*k];

	auto init1 = [&](float *feature, size_t size, std::vector<int>& counts)
	{
		for(size_t i = 0; i < size; ++i)
        	{
            		if(counts[i] > 0) {
				for(size_t j = 0; j < k; j++)
                    			feature[i * k + j] = (float)(distribution(generator)*s);
			} else {
                		for(size_t j = 0; j < k; j++)
                    		//	feature[i * k + j] = std::numeric_limits<float>::quiet_NaN();
					feature[i * k + j] = 0;
            		}
        	}
	};

	init1(model.p, rows, counts_p);
	init1(model.q, cols, counts_q);

	if(use_half) {
		halfp = (short *)malloc(sizeof(short) * (rows + cols) * k);
		halfq = halfp + (rows * k);
	}

	printf("init model success!\n");
}

void DataManager::InitModelShm(void *shm_buf)
{
	float *tmp = (float *)shm_buf; 
	model.m = rows;
	model.n = cols;
	model.k = k;
	model.scale = means / scale;

	std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
	float s = (float)sqrt(1.0/k);

	model.p = tmp;
	model.q = tmp + rows*k;

	auto init1 = [&](float *feature, size_t size, std::vector<int>& counts)
	{
		for(size_t i = 0; i < size; ++i)
        	{
            		if(counts[i] > 0) {
				for(size_t j = 0; j < k; j++)
                    			feature[i * k + j] = (float)(distribution(generator)*s);
			} else {
                		for(size_t j = 0; j < k; j++)
                    		//	feature[i * k + j] = std::numeric_limits<float>::quiet_NaN();
					feature[i * k + j] = 0;
            		}
        	}
	};

	init1(model.p, rows, counts_p);
	init1(model.q, cols, counts_q);

	if(use_half) {
		halfp = (short *)malloc(sizeof(short) * (rows + cols) * k);
		halfq = halfp + (rows * k);
	}

	printf("init model success!\n");
}

//Find the free orthogonal block
int DataManager::FindFreeBlock()
{
	std::lock_guard<std::mutex> lock(mtx);
	if(remain_blocks == 0) return -1;
	int blockid = GetBlockId(grid, block_y, block_x);
	
	while(busy_x[block_x] || busy_y[block_y] || counts_epoch[blockid] != current_epoch) {
		block_x++;
		block_y++;
	
		if(block_y >= grid.gridDim.y) {
			move++;
			block_x = move;
			block_y = 0;
		} else if(block_x == grid.gridDim.x) {
			block_x = 0;
		}

		blockid = GetBlockId(grid, block_y, block_x);
	}

	busy_x[block_x] = true;
	busy_y[block_y] = true;	
	remain_blocks--;
	return blockid;
}

void DataManager::ClearBlockTable()
{
	std::lock_guard<std::mutex> lock(mtx);
	current_epoch++;
	//Recovery schedule
	remain_blocks = block_size;
	complete_blocks = 0;
	block_x = 0;
	block_y = 0;
	move = 0;
}

void DataManager::SetBlockFree(int blockId)
{
	int y = blockId / grid.gridDim.x;
	int x = blockId % grid.gridDim.x;
	std::lock_guard<std::mutex> lock(mtx);
	busy_x[x] = false;
	busy_y[y] = false;
	counts_epoch[blockId]++;
	complete_blocks++;
}

//注意，以后是否要加入轮转
void DataManager::SplitData(int& start, int& size, int work_ratio)
{
	std::lock_guard<std::mutex> lock(mtx);
	start = 0, size = 0;
	for(int i = 0; i < block_x; i++) {							//find start block;
		start += counts[i];
	}

	for(int i = block_x; i < block_x + work_ratio; i++) {		//find block size;
		size += counts[i];
	}
	block_x+=work_ratio;
}

//公共，可以提到基类
void DataManager::PrintHead(int start, int head)
{
	for(int i = start; i < start+head; i++) {
		printf("[Server]u: %d, v: %d, r: %.2f\n", data.r_matrix[i].row_index, data.r_matrix[i].col_index, data.r_matrix[i].r);
	}	
}


int WorkerDM::GetBlockId(Grid& grid, MatrixNode& n)
{
	return ((n.row_index-start_rows)/grid.blockDim.y) * grid.gridDim.x + (n.col_index-0)/grid.blockDim.x;
}

int WorkerDM::GetBlockId(Grid& grid, int row, int col)
{
	return row * grid.gridDim.x + col; 
}


void WorkerDM::SetGrid(const Dim2& grid_dim)
{
	grid.gridDim = grid_dim;
	int nr_bins_x = grid.gridDim.x;
	int nr_bins_y = grid.gridDim.y;
	
	block_size = nr_bins_x * nr_bins_y;
	grid.blockDim.x = (int)ceil((double)cols / nr_bins_x);
	grid.blockDim.y = (int)ceil((double)(end_rows - start_rows + 1) / nr_bins_y);
}

void WorkerDM::InitBlockScheduler()
{
	int nr_bins_x = grid.gridDim.x;
	int nr_bins_y = grid.gridDim.y;

		//init the block scheduler
	busy_x.resize(nr_bins_x, false);
	busy_y.resize(nr_bins_y, false);

/*	for(int y = 0; y < nr_bins_y; y++) {
		for(int x = 0; x < nr_bins_x; x++) {
			ready_queue.push_back(Block(x, y, y * nr_bins_x + x));
		}
	}*/

	for(int x = 0; x < nr_bins_x; x++) {
		for(int i = 0; i < nr_bins_y; i++) {
			int _x = (x+i)%nr_bins_x;
			ready_queue.push_back(Block(_x, i, i * nr_bins_x + _x ));
		}
	}

	pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE);
}

void WorkerDM::GridData(int rank, int nr_threads)
{
	printf("[Work %d]Grid Problem...\n", rank);
	double start, elapse;
	start = cpu_second();
	counts.resize(block_size, 0);
	for(size_t i = 0; i < nnz; i++) {
		MatrixNode N = data.r_matrix[i];
		int blockIdx = GetBlockId(grid, N);
		counts[blockIdx] += 1;
	}

	std::vector<MatrixNode *>& ptrs = grid.blocks;
	ptrs.resize(block_size + 1);
	ptrs[0] = &data.r_matrix[0];

	for(int block = 0; block < block_size; block++) {
		ptrs[block+1] = ptrs[block] + counts[block];
	}

	std::vector<MatrixNode*> pivots(ptrs.begin(), ptrs.end()-1);
    	for(int block = 0; block < block_size; ++block)
    	{
        	for(MatrixNode* pivot = pivots[block]; pivot != ptrs[block+1];)
        	{
            		int curr_block = GetBlockId(grid, *pivot);
					if(curr_block == block)
            		{
               			++pivot;
                		continue;
            		}

            		MatrixNode *next = pivots[curr_block];
					std::swap(*pivot, *next);
            		pivots[curr_block] += 1;
        	}
    	}


#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(dynamic)
#endif
    for(int block = 0; block < block_size; ++block)
    {
        if(rows > cols)
            std::sort(ptrs[block], ptrs[block+1], sort_node_by_p());
        else
            std::sort(ptrs[block], ptrs[block+1], sort_node_by_q());
    }
	
    elapse = cpu_second() - start;
    printf("[Work %d]Grid Problem complete, cost: %.8f\n", rank, elapse);
}

void WorkerDM::GridQ(int rank, int nr_threads)
{
		printf("[Work %d]GridQ Problem...\n", rank);
		double start, elapse;
		start = cpu_second();
		counts.resize(block_size, 0);
		infos.resize(block_size);
		
		for(size_t i = 0; i < nnz; i++) {
			MatrixNode N = data.r_matrix[i];
			int blockIdx = GetBlockId(grid, N);
			counts[blockIdx] += 1;
		}
	
		std::vector<MatrixNode *>& ptrs = grid.blocks;
		ptrs.resize(block_size + 1);
		ptrs[0] = &data.r_matrix[0];

		int start_r = 0;
		int start_q = 0;
		int count_q = 0;
		for(int block = 0; block < block_size; block++) {
			ptrs[block+1] = ptrs[block] + counts[block];
			infos[block].start_r = start_r;
			start_r += counts[block];
			infos[block].size_r = counts[block];
			infos[block].start_q = start_q;
			infos[block].size_q = std::min(grid.blockDim.x, cols-count_q);
                        start_q += grid.blockDim.x;
			count_q += grid.blockDim.x;

//			 printf("block: %d, start_r: %lld, size_r: %lld, start_q: %d, size_q: %d\n", block, infos[block].start_r, infos[block].size_r, infos[block].start_q, infos[block].size_q);
		}
	
		std::vector<MatrixNode*> pivots(ptrs.begin(), ptrs.end()-1);
			for(int block = 0; block < block_size; ++block)
			{
				for(MatrixNode* pivot = pivots[block]; pivot != ptrs[block+1];)
				{
						int curr_block = GetBlockId(grid, *pivot);
//						printf("block: %d, curr_block: %d, row: %d, col: %d\n", block, curr_block, pivot->row_index, pivot->col_index);
						if(curr_block == block)
						{
							++pivot;
							continue;
						}
	
						MatrixNode *next = pivots[curr_block];
						std::swap(*pivot, *next);
						pivots[curr_block] += 1;
				}
			}
	
//	     printf("block_size: %d\n", block_size);	
/*#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(dynamic)
#endif
		for(int block = 0; block < block_size; ++block)
		{
			std::sort(ptrs[block], ptrs[block+1], sort_node_by_q());	
		}        
		
		for(int block = 0; block < block_size; ++block) {
			if(block > 0)
				infos[block].start_q = std::min(infos[block-1].start_q+infos[block-1].size_q , ptrs[block][0].col_index);
			else
				infos[block].start_q = 0;
			infos[block].size_q = (ptrs[block][counts[block]-1].col_index - infos[block].start_q);
                        printf("block: %d, start_r: %lld, size_r: %lld, start_q: %d, size_q: %d\n", block, infos[block].start_r, infos[block].size_r, infos[block].start_q, infos[block].size_q);
		}
*/	
			
		elapse = cpu_second() - start;
		printf("[Work %d]Grid Problem complete, cost: %.8f\n", rank, elapse);

}

int WorkerDM::GetFreeBlock()
{
	pthread_spin_lock(&lock);
//	std::lock_guard<std::mutex> lock(mtx);
	if(ready_queue.empty()) {
		pthread_spin_unlock(&lock); 
		return -1;
	}
	//find free block;
	Block block = ready_queue.front();
	ready_queue.pop_front();

	// following code will cause dead lock
	/*	while(busy_x[block.x] || busy_y[block.y]) {
		ready_queue.push_back(block);
		block = ready_queue.front();
		ready_queue.pop_front();
	}*/


	if(busy_x[block.x] || busy_y[block.y]) {
		ready_queue.push_back(block);
		pthread_spin_unlock(&lock);
                return -2;		
	}

	//store free block into using queue
	using_queue.push_back(block);
	busy_x[block.x] = true;
	busy_y[block.y] = true;
	pthread_spin_unlock(&lock);
	return block.id;
}

void WorkerDM::PrintHead(int rank, int head)
{
	for(int i = 0; i < head; i++) {
		printf("[Worker %d]u: %d, v: %d, r: %.2f\n", rank, data.r_matrix[i].row_index, data.r_matrix[i].col_index, data.r_matrix[i].r);
	}	
}

void WorkerDM::ClearBlockFlags()
{
	ready_queue.swap(using_queue);
}

void WorkerDM::RecoverBlockFree(int blockId)
{
//	std::lock_guard<std::mutex> lock(mtx);
	pthread_spin_lock(&lock); 
	busy_x[blockId % grid.gridDim.x] = false;
	busy_y[blockId / grid.gridDim.x] = false;
	pthread_spin_unlock(&lock);
}
}
