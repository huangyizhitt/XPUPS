#ifndef _UTILS_H_
#define _UTILS_H_

#ifdef DEBUG
#define debugp(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define debugp(fmt, ...)
#endif 

enum CMD {
	PUSH_INFO,
	INIT_DATA,
	PULL_DATA,
	PULL_FEATURE,
	PUSH_FEATURE,
	STOP_WORKER = 987654,
};

long long cpu_microsecond(void);
double cpu_second(void);
void print_feature_head(const float *p, const float *q, int head = 5, bool is_server = false);
void print_feature_tail(const std::vector<float>& p, const std::vecotr<float>& q, int tail = 3, bool is_server = false);

#endif

