#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <vector>
#include "xpu.h"
#include "utils.h"
#include "ps/internal/env.h"

namespace global {

int current_epoch = 0;
int target_epoch = 0;
std::vector<cudaStream_t> streams;

}

namespace MF {
GPU::~GPU()
{
	cudaFree(transfer_buf);
}

void GPU::Init()
{
	xpu_type = XPU_TYPE::GPU;
	XPU::Init();
	global::target_epoch = XPU::target_epoch;
}

void GPU::InitAcopy()
{
	const char *val = NULL;
	val = ps::Environment::Get()->find("EPOCH");
	if(val != NULL)
		XPU::num_streams = std::atoi(val);
	else
		XPU::num_streams = 10;

	global::streams.resize(XPU::num_streams);
	for(int stream = 0; stream < XPU::num_streams; stream++) {
		cudaStreamCreate(&global::streams[stream]);
	}
}

void GPU::DeInitAcopy()
{
	for (uint64_t stream = 0; stream < XPU::num_streams; stream++)
        // Destroy streams.
        cudaStreamDestroy(global::streams[stream]); 
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

void GPU::PrepareTransferBuf(size_t size)
{
	cudaMalloc(&transfer_buf, size * sizeof(half));
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

void GPU::Transfer(void *dst, void *src, size_t size, TransferDirect direct, int stream)
{
	cudaMemcpyKind copy_type = static_cast<cudaMemcpyKind>(direct);
	cudaMemcpyAsync(dst, src, size, copy_type, global::streams[stream]);
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
		mid = (half *)transfer_buf;
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
	}
	return 0;
}

int GPU::singles2halfp(void *target, const void *source, ptrdiff_t numel, int stream, int rounding_mode, int is_quiet, int nr_threads, bool cross_device)
{
	half *mid;
	float *src = (float *)source;
	
	size_t bytes = sizeof(half)*numel;
	//cross device: target in cpu, source in gpu
	if(cross_device) {
		mid = (half *)transfer_buf;
	} else {
		mid = (half *)target;
	}

	size_t nx = 1024;
	size_t ny = (numel + nx - 1) / nx;

	dim3 blockSize(32, 32, 1);
	dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y, 1);
	singles2half_kernel<<<gridSize, blockSize, 0, global::streams[stream]>>>(mid, src, nx, ny);

	if(cross_device) {
		Transfer(target, mid, bytes, TransferDirect::C2S, stream);
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

//	long long start, elapse;

	//cross device: source in cpu, target in gpu
	if(cross_device) {
	//	cudaMalloc(&mid, bytes);
//		start = cpu_microsecond();
		mid = (half *)transfer_buf;
		Transfer(mid, source, bytes, TransferDirect::S2C);
//		elapse = cpu_microsecond() - start;
//		printf("GPU PCIE transfer time: %.7fs, bytes: %ld, bandwidth: %.7fGB/s, numel: %ld\n", (float)elapse / 1000000, bytes, (float)bytes / (elapse * 1000), numel);
	} else {
		mid = (half *)source;
	}

	size_t nx = 1024;
	size_t ny = (numel + nx - 1) / nx;

	dim3 blockSize(32, 32, 1);
	dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y, 1);


	halfp2singles_kernel<<<gridSize, blockSize>>>(dst, mid, nx, ny);

	cudaDeviceSynchronize();

	return 0;
}

int GPU::halfp2singles(void *target, void *source, ptrdiff_t numel, int stream, int nr_threads, bool cross_device)
{
	half *mid;
	size_t bytes = sizeof(half)*numel;
	float *dst = (float *)target;

//	long long start, elapse;

	//cross device: source in cpu, target in gpu
	if(cross_device) {
	//	cudaMalloc(&mid, bytes);
//		start = cpu_microsecond();
		mid = (half *)transfer_buf;
		Transfer(mid, source, bytes, TransferDirect::S2C, stream);
//		elapse = cpu_microsecond() - start;
//		printf("GPU PCIE transfer time: %.7fs, bytes: %ld, bandwidth: %.7fGB/s, numel: %ld\n", (float)elapse / 1000000, bytes, (float)bytes / (elapse * 1000), numel);
	} else {
		mid = (half *)source;
	}

	size_t nx = 1024;
	size_t ny = (numel + nx - 1) / nx;

	dim3 blockSize(32, 32, 1);
	dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y, 1);


	halfp2singles_kernel<<<gridSize, blockSize, 0, global::streams[stream]>>>(dst, mid, nx, ny);

	return 0;
}

void GPU::AcopySync(int stream)
{
	cudaStreamSynchronize(global::streams[stream]);   
}

}
