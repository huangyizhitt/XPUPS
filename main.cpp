#include <iostream>
#include "ps/ps.h"
#include "mfserver.h"
#include "mfworker.h"

std::unordered_map<int, float> MF::MFServer::worker_xpu_info;
MF::DataManager MF::MFServer::dm("netflix_train.bin");

int main(int argc, char **argv)
{
	XPU *xpu;
	MF::MFServer* server;
	MF::MFWorker* worker;

	ps::Start(0);
	if (ps::IsScheduler()) {
    	std::cout << "start scheduler" << std::endl;
  	}
	if (ps::IsServer()) {
    		std::cout << "start server" << std::endl;
		xpu = new XPU("W-2155", CPU, 2, 20, 2000, 60, 128, true);
		server = new MF::MFServer(xpu);
		ps::RegisterExitCallback([server, xpu](){ delete server; delete xpu;});
  	}

	if (ps::IsWorker()) {
		std::cout << "start worker" << std::endl;
		xpu = new XPU("W-2155", CPU, 9, 20, 2000, 60, 128, false);
		worker = new MF::MFWorker(xpu);
		worker->PushWorkerXPU();
		worker->PullDataInfoFromServer();
		worker->PullDataFromServer();
//		worker->Test();
		ps::RegisterExitCallback([worker, xpu](){ delete worker; delete xpu;});
	}

	ps::Finalize(0, true);
}
