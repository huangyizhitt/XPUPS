#include "utils.h"
#include <sys/time.h>

long long cpu_microsecond(void)
{
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, nullptr);
    ust = ((long long)tv.tv_sec)*1000000;
    ust += tv.tv_usec;
    return ust;	
}


double cpu_second(void)
{
	struct timeval tv;
    double t;

    gettimeofday(&tv, nullptr);
    t = tv.tv_sec + ((double)tv.tv_usec)/1000000;
    return t;
}

void print_feature_head(float *p, float *q, int head = 5, bool is_server = false)
{
	const char *name = (is_server ? "Server" : "Worker");

	for(int i = 0; i < head; i++) {
		printf("[%s] p[%d] %.2f, q[%d] %.2f\n", name, i, p[i], i, q[i]);
	}
}


