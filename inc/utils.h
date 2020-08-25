#ifndef _UTILS_H_
#define _UTILS_H_

#ifdef DEBUG
#define debugp(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define debugp(fmt, ...)
#endif 

enum CMD {
	PUSH_INFO,
	PULL_DATA_INFO,
	PULL_DATA,
	PULL_FEATURE,
	PUSH_FEATURE,
	STOP_WORKER,
};

long long cpu_microsecond(void);
double cpu_second(void);
void print_feature_head(float *p, float *q, int head = 5, bool is_server = false);

#endif

