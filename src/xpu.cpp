#include <string.h>
#include <cstdlib>
#include <string>
#include "ps/internal/env.h"
#include "dmlc/logging.h"
#include "xpu.h"
#include "ps/ps.h"

using namespace ps;
using namespace dmlc;

namespace MF {
void XPU::Init()
{
	const char *val = NULL;

	val = Environment::Get()->find("XPU_NAME");
	if(val != NULL)
		strcpy(xpu_name, val);

	val = Environment::Get()->find("XPU_MAX_CORES");
	if(val != NULL)
		this->max_cores = std::atoi(val);

	val = Environment::Get()->find("XPU_WORKERS");
	if(val != NULL)
		workers = std::atoi(val);

	val = ps::Environment::Get()->find("WORK_LOAD");
	if(val != NULL)
		worker_ratio = atoi(val);

	if(xpu_type == XPU_TYPE::CPU) {
		dev_id = 0;
	} else {
		val = Environment::Get()->find("DEVICE_ID");
		if(val != NULL) {
			dev_id = std::atoi(val);
		} else {
			dev_id = 0;
		}
	} 

	val = Environment::Get()->find("EPOCH");
	if(val != NULL)
		target_epoch = std::atoi(val);
	else
		target_epoch = 20;

}

}
