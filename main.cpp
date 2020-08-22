#include <iostream>
#include "ps/ps.h"
#include "mfserver.h"
#include "mfworker.h"

std::unordered_map<int, int> MF::MFServer::xpu_info;

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
		server = new MF::MFServer();
  	}

	if (ps::IsWorker()) {
		std::cout << "start worker" << std::endl;
		xpu = new XPU("W-2155", CPU, 9, 20, 2000, 60, 128, false);
		worker = new MF::MFWorker(xpu);
	}

	ps::Finalize(0, true);
	
	if(ps::IsServer()) {
		delete server;
	}

	if(ps::IsWorker()) {
		delete xpu;
		delete worker;
	}
}
