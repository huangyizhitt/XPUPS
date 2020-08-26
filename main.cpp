#include <iostream>
#include <unistd.h>
#include "ps/ps.h"
#include "ps/base.h"
#include "ps/internal/postoffice.h"
#include "mfserver.h"
#include "mfworker.h"

std::unordered_map<int, XPU_INFO> MF::MFServer::worker_xpu_info;
MF::DataManager MF::MFServer::dm("netflix_train.bin", 128, 3);
size_t MF::MFServer::cpus(0);
size_t MF::MFServer::gpus(0);
size_t MF::MFServer::fpgas(0);
size_t MF::MFServer::tpus(0);
int MF::MFServer::max_workers(0);
int MF::MFServer::scale(0);
int MF::MFServer::nr_threads(0);


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
		xpu = new XPU("W-2155", CPU, 20, 20, 20, 0, true);
		server = new MF::MFServer(xpu);
		server->SetThreads(xpu->workers);
		ps::RegisterExitCallback([server, xpu](){ delete server; delete xpu;});
  	}

	if (ps::IsWorker()) {
		int epoch = 20;
		std::cout << "start worker" << std::endl;
		xpu = new XPU("W-2155", CPU, 9, 20, 9, 1, false);
		worker = new MF::MFWorker(xpu);
		worker->PushWorkerXPU();
		worker->InitTestData();
		worker->PullDataFromServer();
		worker->GridProblem();
		worker->CreateTasks();

		for(int i = 0; i < epoch; i++) {
			sleep(0.5);//pull feature
			worker->StartUpTasks();			//start up tasks to compute 
			//push feature
		}

		ps::RegisterExitCallback([worker, xpu](){ delete worker; delete xpu;});
	}

	ps::Finalize(0, true);
}
