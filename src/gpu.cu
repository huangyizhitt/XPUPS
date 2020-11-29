#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "xpu.h"

namespace global {

int current_epoch = 0;
int target_epoch = 0;

}

namespace MF {

void GPU::Init()
{
	xpu_type = XPU_TYPE::GPU;
	XPU::Init();
	global::target_epoch = XPU::target_epoch;
}

bool GPU::Bind()
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if(deviceCount == 0) {
		printf("The system don't have any GPU!\n");
		return false;
	}

	if(dev_id >= deviceCount) {
		printf("The gpu %d beyond total gpu devices!\n", dev_id);
		return false;
	}

	cudaSetDevice(dev_id);
	return true;
}

void GPU::CreateTasks(int task_index, pFunc func, void * args)
{
	task.func = func;
	task.args = args;
}

void GPU::RunTasks()
{
	global::current_epoch = XPU::current_epoch;
	task.func(task.args);
}

void GPU::JoinTasks()
{
	cudaDeviceSynchronize();
}

void GPU::Transfer(void *dst, void *src, size_t size, TransferDirect direct)
{
	cudaMemcpyKind copy_type = static_cast<cudaMemcpyKind>(direct);
	cudaMemcpy(dst, src, size, copy_type);
}

__global__ void singles2half_kernel(half *target, const float *source, size_t nx, size_t ny)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if(tx < nx && tx < ny) {
		target[ty * nx + tx] = __float2half(source[ty * nx + tx]);
	}
}

int GPU::singles2halfp(void *target, const void *source, ptrdiff_t numel, int rounding_mode, int is_quiet, int nr_threads, bool cross_device)
{
	half *mid;
	float *src = (float *)source;
	
	size_t bytes = sizeof(half)*numel;
	//cross device: target in cpu, source in gpu
	if(cross_device) {
		cudaMalloc(&mid, bytes);
	} else {
		mid = (half *)target;
	}

	size_t nx = 1024;
	size_t ny = (numel + nx - 1) / nx;

	dim3 blockSize(32, 32, 1);
	dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y, 1);
	singles2half_kernel<<<gridSize, blockSize>>>(mid, src, nx, ny);

	cudaDeviceSynchronize();
	if(cross_device) {
		Transfer(target, mid, bytes, TransferDirect::C2S);
		cudaFree(mid);
	}
	return 0;
}

__global__ void halfp2singles_kernel(float *target, half *source, size_t nx, size_t ny)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if(tx < nx && tx < ny) {
		target[ty * nx + tx] = __half2float(source[ty * nx + tx]);
	}	
}

int GPU::halfp2singles(void *target, void *source, ptrdiff_t numel, int nr_threads, bool cross_device)
{
	half *mid;
	size_t bytes = sizeof(half)*numel;
	float *dst = (float *)target;
	//cross device: source in cpu, target in gpu
	if(cross_device) {
		cudaMalloc(&mid, bytes);
		Transfer(mid, source, bytes, TransferDirect::S2C);
	} else {
		mid = (half *)source;
	}

	size_t nx = 1024;
	size_t ny = (numel + nx - 1) / nx;

	dim3 blockSize(32, 32, 1);
	dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y, 1);


	halfp2singles_kernel<<<gridSize, blockSize>>>(dst, mid, nx, ny);

	cudaDeviceSynchronize();

	if(cross_device) {
		cudaFree(mid);
	}
	return 0;
}

}
