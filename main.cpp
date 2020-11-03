#include <iostream>
#include <unistd.h>
#include "ps/ps.h"
#include "ps/base.h"
#include "ps/internal/postoffice.h"
#include "mfserver.h"
#include "mfworker.h"
#include "utils.h"

std::unordered_map<int, XPU_INFO> MF::MFServer::worker_xpu_info;
std::unordered_map<int, std::pair<int, unsigned char *> > MF::MFServer::shm_buf;
MF::DataManager MF::MFServer::dm("netflix_train.bin", 128, 3);
int MF::MFServer::cpus(0);
int MF::MFServer::gpus(0);
int MF::MFServer::fpgas(0);
int MF::MFServer::tpus(0);
int MF::MFServer::xpus(0);
int MF::MFServer::max_workers(0);
int MF::MFServer::scale(0);
int MF::MFServer::nr_threads(0);

#ifdef EXPLORE
std::ofstream  MF::MFServer::out("feature.csv", std::ios::out);
#endif


int MF::MFServer::receive_times(0);
int MF::MFServer::current_epoch(1);
int MF::MFServer::target_epoch(20);
#ifdef CAL_PORTION_RMSE
float MF::MFServer::loss(0.0);
#endif


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
//		xpu = new XPU("W-2155", CPU, 20, 20, 20, 0, true);
		server = new MF::MFServer();
		server->Init();
//		server->SetThreads(xpu->workers);
		ps::RegisterExitCallback([server](){ delete server;});
  	}

	if (ps::IsWorker()) {
		std::cout << "start worker" << std::endl;
//		xpu = new XPU("W-2155", CPU, 8, 20, 8, 1, false);
		worker = new MF::MFWorker();
		worker->Init();
		worker->InitCPUAffinity();
		worker->PushWorkerXPU();
		worker->InitTestData();
		worker->PullDataFromServer();
		worker->GridProblem();
		worker->CreateTasks();

		double start, elapse = 0;
		start = cpu_second();
		while(true) {
#ifdef SEND_ALL_FEATURE
//			printf("Begin epoch\n");
//			start = cpu_second();
			worker->PullAllFeature();
//			elapse = cpu_second() - start;
//                        printf("Pull cost time: %.3f\n", elapse);

//			start = cpu_second();
			worker->StartUpTasks();
//			elapse = cpu_second() - start;
 //                       printf("Compute cost time: %.3f\n", elapse);

//			start = cpu_second();
			worker->PushAllFeature();
//                        elapse = cpu_second() - start;
//                        printf("Push cost time: %.3f\n", elapse);
#elif SEND_Q_FEATURE
//			start = cpu_second();
			worker->PullFeature();
//			elapse = cpu_second() - start;
   //                     printf("Pull cost time: %.3f\n", elapse);

//			start = cpu_second();
			worker->StartUpTasks();
//			elapse = cpu_second() - start;
//                        printf("Compute cost time: %.3f\n", elapse);

//			start = cpu_second();
			worker->PushFeature();
//			elapse = cpu_second() - start;
//                        printf("Push cost time: %.3f\n", elapse);
#elif SEND_COMPRESS_Q_FEATURE
//			start = cpu_second();
			worker->PullCompressFeature();
//			elapse += cpu_second() - start;
//			printf("Pull cost time: %.3f\n", elapse);

//			start = cpu_second();
			worker->StartUpTasks();
//			elapse += cpu_second() - start;
//			printf("Compute cost time: %.3f\n", elapse);

//			start = cpu_second();
			worker->PushCompressFeature();
//			elapse += cpu_second() - start;
//			printf("Push cost time: %.3f\n", elapse);
#endif
			ps::Postoffice::Get()->Barrier(0, ps::kWorkerGroup);
//			elapse = cpu_second() - start;
  //                      printf("Push cost time: %.3f\n", elapse);
			if(worker->current_epoch == worker->target_epoch) break;
		}
		elapse = cpu_second() - start;
		printf("20 epoch cost time: %.3f\n", elapse);
//		printf("20 epoch compute cost time: %.3f\n", elapse);
//		printf("20 epoch communication cost time %.3f\n", elapse);
		worker->JoinTasks();
		ps::RegisterExitCallback([worker](){ delete worker;});
	}

	ps::Finalize(0, true);
}
