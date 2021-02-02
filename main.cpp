#include <iostream>
#include <unistd.h>
#include "ps/ps.h"
#include "ps/base.h"
#include "ps/internal/postoffice.h"
#include "mfserver.h"
#include "mfworker.h"
#include "utils.h"

int main(int argc, char **argv)
{
	MF::MFServer* server;
	MF::MFWorker* worker;

	ps::Start(0);
	if (ps::IsScheduler()) {
    	std::cout << "start scheduler, pid: " << getpid() << std::endl;
  	}
	if (ps::IsServer()) {
    	std::cout << "start server, pid" << getpid() << std::endl;
//		xpu = new XPU("W-2155", CPU, 20, 20, 20, 0, true);
		server = new MF::MFServer();
		server->Init();
//		server->SetThreads(xpu->workers);
		ps::RegisterExitCallback([server](){ delete server;});
  	}

	if (ps::IsWorker()) {
		std::cout << "start worker, pid: " << getpid() << std::endl;
		worker = new MF::MFWorker();
		worker->PreProcess();

		double start, elapse = 0, pull_start, pull_elapse = 0, push_start, push_elapse = 0, compute_start, compute_elapse = 0;
		start = cpu_second();
		while(true) {
//			printf("Begin epoch\n");
			pull_start = cpu_second();
			worker->Pull();
//			printf("pull success!\n");
			pull_elapse += cpu_second() - pull_start;
//                        printf("Pull cost time: %.3f\n", elapse);

			compute_start = cpu_second();
			worker->Computing();
//			printf("Computing success!\n");
			compute_elapse += cpu_second() - compute_start;
//                        printf("Compute cost time: %.3f\n", elapse);

			push_start = cpu_second();
			worker->Push();
//			printf("Push success!\n");
                        push_elapse += cpu_second() - push_start;
//                        printf("Push cost time: %.3f\n", elapse);
			worker->Barrier();
//			elapse = cpu_second() - start;
  //                      printf("Push cost time: %.3f\n", elapse);
			if(worker->GetCurrentEpoch() == worker->GetTargetEpoch()) break;
		}
		elapse = cpu_second() - start;
//		printf("[Worker %d] 20 epoch cost time: %.3f\n", worker->GetWorkerID(), elapse);
//		printf("20 epoch pull cost time: %.3f\n", elapse);
//		printf("20 epoch push cost time: %.3f\n", elapse);
		printf("[Worker %d] 20 epoch total cost time: %.3f, pull cost time: %.3f, compute cost time: %.3f, push cost time: %.3f\n", worker->GetWorkerID(), elapse, pull_elapse, compute_elapse, push_elapse);
//		printf("20 epoch communication cost time %.3f\n", elapse);
		worker->JoinWorkers();
		ps::RegisterExitCallback([worker](){ delete worker;});
	}

	ps::Finalize(0, true);
}
