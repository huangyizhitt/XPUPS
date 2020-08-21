#ifndef _UTILS_H_
#define _UTILS_H_

#ifdef DEBUG
#define debugp(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define debugp(fmt, ...)
#endif 

long long cpu_microsecond(void);
double cpu_second(void);


#endif

