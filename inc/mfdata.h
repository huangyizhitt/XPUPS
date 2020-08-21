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
	float *p;
	float *q;
};

class DataManager {
public:
	DataManager(const char *file) : train_file_path(file)							//build manager and open file	
	{
		
	}
	
//	~Datamanager() {}
	
	void Init();
	void DeInit();
	
private:
	bool LoadData();
	void CollectDataInfo();
	void ScaleData(float mf_scale);
	void ShuffleData();
	int *GenerateRandomMap(size_t size);
	int *GenerateInvMap(int *map, size_t size);
	void InitModel();
	void GridProblem();
	void DestroyMap(int *map);
	void ScaleModel();
	void ShuffleModel();
	
private:
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
	float means;
	float stddev;
	Data data;
	std::vector<int> count_p;
	std::vector<int> count_q;
};

}

#endif

