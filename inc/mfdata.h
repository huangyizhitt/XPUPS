#ifndef _MFDATA_H_
#define _MFDATA_H_

#include <vector>
#include <cstdio>

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
	std::vector<float> p;           //m*k
	std::vector<float> q;			//k*n
};

//2D Grid
struct Grid
{
	Dim2 gridDim;							//num of block in grid			
	Dim2 blockDim;              			//num of element in block
	Dim2 blockStart;						//block start coordinate in grid
	std::vector<MatrixNode *> blocks;		//blocks[i] point the head of the ith block
};


class DataManager {
public:
	DataManager(const char *file) : train_file_path(file)							//build manager and open file	
	{
		
	}
	
//	~Datamanager() {}
	
	void Init(int nr_threads);
	void DeInit();
	void SetGrid(const Dim2& gridDim);
	void GridData(int nr_threads);
	void ChoseBlock();
	
//private:
	bool LoadData();
	void CollectDataInfo(int nr_threads);
	void ScaleData(float mf_scale, int nr_threads);
	void ShuffleData();
	int *GenerateRandomMap(size_t size);
	int *GenerateInvMap(int *map, size_t size);
	void InitModel();
//	void GridProblem();
	void DestroyMap(int *map);
	void ScaleModel();
	void ShuffleModel();
	int GetBlockId(Grid& grid, MatrixNode& r);					//by matrix node's row and col index;
	int GetBlockId(Grid& grid, int row, int col);					//by block's row and col index; 

	FILE *fp;
	const char *train_file_path;
	int *p_map;
	int *q_map;
	int *inv_p_map;
	int *inv_q_map;
	size_t nnz = 0;								//total element size
	size_t rows = 0;							//row size
	size_t cols = 0;							//col size
	int k;										//rows * cols -> rows * k and k * cols
	int block_size;
	float means;
	float stddev;
	float scale;
	struct Grid grid;
	Data data;
	Model model;
	std::vector<int> counts_p;
	std::vector<int> counts_q;
};

}

#endif

