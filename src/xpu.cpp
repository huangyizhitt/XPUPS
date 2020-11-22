#include <string.h>
#include <cstdlib>
#include <string>
#include <numa.h>
#include "ps/internal/env.h"
#include "dmlc/logging.h"
#include "xpu.h"
#include "ps/ps.h"

using namespace ps;
using namespace dmlc;

void XPU::Init()
{
	const char *val = NULL;
	std::string name, type, max_core, threads;

	if(ps::IsWorker()) {
		name = "WORKER_XPU_NAME";
		type = "WORKER_XPU_TYPE";
		max_core = "WORKER_XPU_MAX_CORE";
		threads = "WORKER_XPU_THREADS";
	} else {
		name = "SERVER_XPU_NAME";
		type = "SERVER_XPU_TYPE";
		max_core = "SERVER_XPU_MAX_CORE";
		threads = "SERVER_XPU_THREADS";
	}
	val = CHECK_NOTNULL(Environment::Get()->find(name.c_str()));
	strcpy(xpu_name, val);
	val = CHECK_NOTNULL(Environment::Get()->find(type.c_str()));
	if(strcmp(val, "CPU") == 0) {
		xpu_type = CPU;
	} else if(strcmp(val, "GPU") == 0) {
		xpu_type = GPU;
	} else if(strcmp(val, "FPGA") == 0) {
		xpu_type = FPGA;
	} else if(strcmp(val, "TPU") == 0) {
		xpu_type = TPU;
	} else {
		xpu_type = UNKONWN_XPUTYPE;
	}

	val = CHECK_NOTNULL(Environment::Get()->find(max_core.c_str()));
	this->max_core = std::atoi(val);
	val = CHECK_NOTNULL(Environment::Get()->find(threads.c_str()));
	core = workers = std::atoi(val);

	//bind xpu to numa node, default node is node 0
	val = Environment::Get()->find("NUMA_NODE");
	if(val != NULL) {
		numa_node = std::atoi(val);
	} else {
		numa_node = 0;
	}

	if(xpu_type == GPU) {
		val = Environment::Get()->find("GPU_DEVICE");
		if(val != NULL) {
			gpu_dev = std::atoi(val);
		} else {
			gpu_dev = 0;
		}
	} 
}

void XPU::NumaBindNode()
{
	if(numa_available() < 0) {
		printf("System can not support numa arch!\n");
		return;
	}	

	numa_set_localalloc();
        int max_node = numa_max_node();
        if(numa_node <= max_node && numa_run_on_node(numa_node) == 0) {
                printf("Bind task to node %d success!\n", numa_node);
        } else {
                printf("Bind task to node %d fail!\n", numa_node);
        }
}


