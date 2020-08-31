#ifndef _XPU_H_
#define _XPU_H_

#include <string.h>

enum XPU_TYPE {
	CPU = 0,
	GPU,
	FPGA,
	TPU,
	UNKONWN_XPUTYPE,
};

struct XPU {

	XPU() {}
	XPU(const char *name, XPU_TYPE xpu_type, int core, int max_core,
		int workers, int worker_ratio, bool is_server) : 
		xpu_type(xpu_type), core(core), max_core(max_core), workers(workers), 
		worker_ratio(worker_ratio), is_server(is_server) {strcpy(xpu_name, name);}

	//Init by env
	//if call XPU() create XPU object, must call this function
	void Init();
		
	char xpu_name[64];
	XPU_TYPE xpu_type;
	int core;
	int max_core;
	int workers;
	int id;							//global id in system;
	int worker_ratio;
	bool is_server;
	bool is_virtual;				//virtual cpu;
};

struct XPU_INFO {
	XPU_TYPE type;
	int workers;
	int work_ratio;						//work load ratio
};

#endif

