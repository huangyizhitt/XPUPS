#include "mfdata.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <unordered_set>
#include <numeric>


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
	printf("rows:%lu, cols:%lu, nnz:%lu, idx:%lu, cost time: %f\n",rows, cols, nnz, idx, elapse);
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

void DataManager::Init(int nr_threads)
{
	double start, elapse;
	scale = 1.0f;
	
	LoadData();
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

	grid.blockDim.x = (int)ceil((double)cols / nr_bins_x);
	grid.blockDim.y = (int)ceil((double)rows / nr_bins_y);
}

void DataManager::GridData(int nr_threads)
{
	printf("Grid Problem to all XPU...\n");
	double start, elapse;
	start = cpu_second();
	std::vector<int> counts(block_size, 0);
	for(size_t i = 0; i < nnz; i++) {
		MatrixNode N = data.r_matrix[i];
		int blockIdx = GetBlockId(grid, N);
		counts[blockIdx] += 1;
		counts_p[N.row_index] += 1;
		counts_p[N.col_index] += 1;
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


void DataManager::InitFeatureP()
{
	
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
	
	model.p.resize(rows * k, 0);
	model.q.resize(k * cols, 0);

	auto init1 = [&](std::vector<float>& feature, size_t size, std::vector<int>& counts)
	{
		for(size_t i = 0; i < size; ++i)
        {
            if(counts[i] > 0) {
				for(size_t j = 0; j < k; j++)
                    feature[i * k + j] = (float)(distribution(generator)*s);
			} else {
                for(size_t j = 0; d < k; j++)
                    feature[i * k + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
	}

	init1(model.p, rows, counts_p);
	init1(model.q, rows, counts_q);

	printf("init model success!\n");
}

}
