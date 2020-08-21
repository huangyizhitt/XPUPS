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
		float peak_performance, float mem_band, float mem_size, bool is_server) : 
		xpu_type(xpu_type), core(core), peak_performance(peak_performance), mem_band(mem_band),
		mem_size(mem_size), is_server(is_server) {strcpy(xpu_name, name);}
	
	char xpu_name[64];
	XPU_TYPE xpu_type;
	int max_core;
	int core;
	int workers;
	int id;							//global id in system;
	float peak_performance;
	float mem_band;
	float mem_size;	
	bool is_server;
	bool is_virtual;				//virtual cpu;
};

#endif

