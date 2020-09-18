#ifndef _UTILS_H_
#define _UTILS_H_

#ifdef DEBUG
#define debugp(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define debugp(fmt, ...)
#endif 

#include <cstdio>
#include <cstddef>

#define  INT16_TYPE          short
#define UINT16_TYPE unsigned short
#define  INT32_TYPE          int
#define UINT32_TYPE unsigned int

#define HALFP_PINF ((UINT16_TYPE) 0x7C00u)  // +inf
#define HALFP_MINF ((UINT16_TYPE) 0xFC00u)  // -inf
#define HALFP_PNAN ((UINT16_TYPE) 0x7E00u)  // +nan (only is_quite bit set, no payload)
#define HALFP_MNAN ((UINT16_TYPE) 0xFE00u)  // -nan (only is_quite bit set, no payload)

/* Define our own values for rounding_mode if they aren't already defined */
#ifndef FE_TONEAREST
    #define FE_TONEAREST    0x0000
    #define FE_UPWARD       0x0100
    #define FE_DOWNWARD     0x0200
    #define FE_TOWARDZERO   0x0300
#endif
#define     FE_TONEARESTINF 0xFFFF  /* Round to nearest, ties away from zero (apparently no standard C name for this) */


enum CMD {
	PUSH_INFO,
	INIT_DATA,
	PULL_DATA,
	PULL_FEATURE,
	PUSH_FEATURE,
	PULL_PUSH_FEATURE,
	PULL_ALL_FEATURE,
	PUSH_ALL_FEATURE,
	PULL_HALF_FEATURE,
	PUSH_HALF_FEATURE,
	STOP_WORKER = 987654,
};

long long cpu_microsecond(void);
double cpu_second(void);
void print_feature_head(const float *p, const float *q, int head = 5, bool is_server = false);
void print_feature_tail(const float *p, const float *q, size_t size_p, size_t size_q, int tail = 3, bool is_server = false);
int singles2halfp(void *target, const void *source, ptrdiff_t numel, int rounding_mode, int is_quiet, int nr_threads);
int halfp2singles(void *target, void *source, ptrdiff_t numel, int nr_threads);

#endif

