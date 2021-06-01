#include <iostream>
#include <unistd.h>
#include "ps/ps.h"
#include "ps/base.h"
#include "ps/internal/postoffice.h"
#include "mfserver.h"
#include "mfworker.h"
#include "utils.h"

#ifdef TEST
#include <fstream>
#endif


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
		printf("[Work %d] Complete Init!\n", worker->GetWorkerID());
		worker->Pull();
		int target_epoch=worker->GetTargetEpoch();
		int current_epoch=0;

#ifdef CAL_PORTION_RMSE
		TransMode trans_mode = worker->GetTransMode();
		if(trans_mode >= ALL_SHM && trans_mode <= trans_mode) {
			worker->StartRecord();
		}
#endif

#ifdef TEST
		bool record = true;
		if(argc != 2) {
			printf("parameter error, cannot record test result!\n");
			record = false;
		}
		double start, elapse = 0, pull_start, pull_elapse = 0, push_start, push_elapse = 0, compute_start, compute_elapse = 0;
		start = cpu_second();
#endif
		while(current_epoch < target_epoch) {
//			printf("[Work %d]Begin epoch\n", worker->GetWorkerID());

//			printf("[Work %d]Pull success!\n", worker->GetWorkerID());
#ifdef TEST
			compute_start = cpu_second();
#endif
			worker->Computing();
//			printf("[Work %d]Compute success!\n", worker->GetWorkerID());
#ifdef TEST
			compute_elapse += cpu_second() - compute_start;

			push_start = cpu_second();
#endif

			worker->Push();
//			printf("[Work %d]Push success!\n", worker->GetWorkerID());
#ifdef TEST                        
			push_elapse += cpu_second() - push_start;
#endif
			//                        printf("Push cost time: %.3f\n", elapse);
			worker->Barrier();
//			elapse = cpu_second() - start;
  //                      printf("Push cost time: %.3f\n", elapse);
			if(++current_epoch == target_epoch) break;

#ifdef TEST
                        pull_start = cpu_second();
#endif

                        worker->Pull();
//                      printf("[Work %d]Pull success!\n", worker->GetWorkerID());
#ifdef TEST
                        pull_elapse += cpu_second() - pull_start;
#endif
		}
#ifdef TEST
		elapse = cpu_second() - start;
		if(record) {
			std::ofstream out(argv[1], std::ios_base::out | std::ios_base::app);
			out << worker->GetWorkerName() << "," << pull_elapse << "," << compute_elapse << "," << push_elapse << "," << elapse << std::endl;
			out.close();
		}
//		printf("[Worker %d] 20 epoch cost time: %.3f\n", worker->GetWorkerID(), elapse);
//		printf("20 epoch pull cost time: %.3f\n", elapse);
//		printf("20 epoch push cost time: %.3f\n", elapse);
		printf("[Worker %d] 20 epoch total cost time: %.3f, pull cost time: %.3f, compute cost time: %.3f, push cost time: %.3f\n", worker->GetWorkerID(), elapse, pull_elapse, compute_elapse, push_elapse);
//		printf("20 epoch communication cost time %.3f\n", elapse);
//		printf("pull: transmission:%.3f + half2single:%.3f\n", worker->record1_elapse, worker->record2_elapse);
#endif
		worker->JoinWorkers();
		ps::RegisterExitCallback([worker](){ delete worker;});
	}

	ps::Finalize(0, true);
	return 0;
}
