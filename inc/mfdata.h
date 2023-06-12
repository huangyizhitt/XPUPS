#ifndef _MFDATA_H_
#define _MFDATA_H_

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <deque>

namespace MF{

struct Dim2 {
	int x;
	int y;
};

struct MatrixNode {
	int row_index;					// row index
	int col_index;					// col index
	float r;						// the element of the R[row][col]
};

struct Data{
	std::vector<MatrixNode> r_matrix;
};

struct Model{
	int m;
	int n;
	int k;
	float scale;
	float *p;           //m*k
	float *q;			//k*n
	std::vector<float> feature;		
};

//2D Grid
struct Grid
{
	Dim2 gridDim;							//num of block in grid			
	Dim2 blockDim;              			//num of element in block
	Dim2 blockStart;						//block start coordinate in grid
	std::vector<MatrixNode *> blocks;		//blocks[i] point the head of the ith block
};

//Acopy info
struct AcopyInfo {
	int id;
	int start_q;
	int size_q;
	long long start_r;
	long long size_r;
};

enum EpochStatus {
	CompleteOnece,
	CompleteAll,
	UnComplete,
};

class DataManager {
public:
	DataManager()	
	{
		
	}
	
	~DataManager() 
	{
		if(use_half)
			free(halfp);
	}
	
	void Init(const char * train_file, const char *test_file, const char *accuracy_file, const bool& use_half, int k);
	void DeInit();
	void SetGrid(const Dim2& gridDim);
	void GridData(int nr_threads);
	void PrepareTrainingData(int nr_threads);
	
//private:
	bool LoadData();
	void CollectDataInfo(int nr_threads);
	void ScaleData(float mf_scale, int nr_threads);
	void ShuffleData();
	int *GenerateRandomMap(size_t size);
	int *GenerateInvMap(int *map, size_t size);
	void InitModel();
	void InitModelShm(void *shm_buf);
//	void GridProblem();
	void DestroyMap(int *map);
	void ScaleModel();
	void ShuffleModel();
	int GetBlockId(Grid& grid, MatrixNode& r);					//by matrix node's row and col index;
	int GetBlockId(Grid& grid, int row, int col);					//by block's row and col index; 
	int FindFreeBlock();
	void SetBlockFree(int blockId);
	EpochStatus EpochComplete();
	void ClearBlockTable();
	void CountFeature();
	void SplitData(int& start, int& size, int work_ratio);
	void PrintHead(int start, int head = 5);

	void Testing();

	FILE *fp;
	int *p_map;
	int *q_map;
	int *inv_p_map;
	int *inv_q_map;
	size_t nnz = 0;								//total element size
	int rows = 0;								//row size
	int cols = 0;								//col size
	int k;										//rows * cols -> rows * k and k * cols
	int block_size;
	int epoch;
	int current_epoch = 1;
	int remain_blocks;							//remain blocks in current epoch
	int complete_blocks;
	int block_x = 0;
	int block_y = 0;
	int move = 0;
	float means;
	float stddev;
	float scale;
	bool use_half;
	short *halfp;
	short *halfq;
	struct Grid grid;
	Data data;
	Model model;
	std::mutex mtx;
	std::vector<int> counts;					//counts every block size;
	std::vector<int> counts_p;
	std::vector<int> counts_q;
	std::vector<int> counts_epoch;				//counts the block epoch;
	std::vector<bool> busy_x;
	std::vector<bool> busy_y;

	char train_file_path[128];
	char test_file_path[128];
	char accuracy_file_path[128];
};

struct Block {
	Block(const int x = 0, const int y = 0, const int id = 0) : x(x), y(y), id(id) {}
	int x;
	int y;
	int id;
};

class WorkerDM {
public:
	void PrintHead(int rank, int head = 5);
	void SetGrid(const Dim2& grid_dim);
	void GridData(int rank, int nr_thread);
	void GridQ(int rank, int nr_threads);						//used to acopy
	int GetBlockId(Grid& grid, MatrixNode& r);					//by matrix node's row and col index;
	int GetBlockId(Grid& grid, int row, int col);					//by block's row and col index; 
	int GetFreeBlock(int epoch);
	int GetFreeBlock();
	void RecoverBlockFree(int blockId);
	void InitBlockScheduler();
	void ClearBlockFlags();

	size_t nnz = 0;
	int rows = 0;
	int cols = 0;
	int start_rows;
	int end_rows;
	int k = 128;
	int block_size;
	
	struct Grid grid;
	Data data;
	Model model;
	pthread_spinlock_t lock;
	std::vector<int> counts;
	std::vector<AcopyInfo> infos;
	std::vector<bool> busy_x;
	std::vector<bool> busy_y;
	std::deque<Block> ready_queue;
	std::deque<Block> using_queue;
};

}

#endif

