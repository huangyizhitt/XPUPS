#include <string.h>
#include <cstdlib>
#include "ps/internal/env.h"
#include "dmlc/logging.h"
#include "xpu.h"

using namespace ps;
using namespace dmlc

void XPU::Init()
{
	const char *val = NULL;
	val = CHECK_NOTNULL(Environment::Get()->find("XPU_NAME"));
	strcpy(xpu_name, val);
	val = CHECK_NOTNULL(Environment::Get()->find("XPU_TYPE"));
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

	val = CHECK_NOTNULL(Environment::Get()->find("XPU_MAX_CORE"));
	max_core = std::atoi(val);
	val = CHECK_NOTNULL(Environment::Get()->find("XPU_THREADS"));
	core = workers = std::atoi(val);	
}


