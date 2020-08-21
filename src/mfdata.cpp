#include "mfdata.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <ctime>

const int nr_threads = 20;

namespace MF {

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

void DataManager::CollectDataInfo()
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

void DataManager::ScaleData(float mf_scale)
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

}
