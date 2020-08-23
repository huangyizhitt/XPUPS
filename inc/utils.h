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
};

long long cpu_microsecond(void);
double cpu_second(void);


#endif

