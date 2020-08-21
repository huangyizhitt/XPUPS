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

