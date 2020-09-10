#ifndef _UTILS_H_
#define _UTILS_H_

#ifdef DEBUG
#define debugp(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define debugp(fmt, ...)
#endif 

#include <cstdio>

enum CMD {
	PUSH_INFO,
	INIT_DATA,
	PULL_DATA,
	PULL_FEATURE,
	PUSH_FEATURE,
	PULL_PUSH_FEATURE,
	PULL_ALL_FEATURE,
	PUSH_ALL_FEATURE,
	STOP_WORKER = 987654,
};

long long cpu_microsecond(void);
double cpu_second(void);
void print_feature_head(const float *p, const float *q, int head = 5, bool is_server = false);
void print_feature_tail(const float *p, const float *q, size_t size_p, size_t size_q, int tail = 3, bool is_server = false);

#endif

