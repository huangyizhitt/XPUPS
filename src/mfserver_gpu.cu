#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "mfserver.h"

namespace MF{

void MFServer::PinnedBuf(void* buf, size_t size)
{
	cudaHostRegister(buf, size, cudaHostRegisterDefault);
}

void MFServer::UnpinnedBuf(void *buf)
{
	cuMemHostUnregister(buf);
}
}
