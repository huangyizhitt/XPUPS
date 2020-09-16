#include "utils.h"
#include <sys/time.h>
#include <cstdio>
#include <fenv.h>
#include <omp.h>

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

void print_feature_head(const float *p, const float *q, int head, bool is_server)
{
	const char *name = (is_server ? "Server" : "Worker");

	for(int i = 0; i < head; i++) {
		printf("[%s] p[%d] %.2f, q[%d] %.2f\n", name, i, p[i], i, q[i]);
	}
}

void print_feature_tail(const float *p, const float *q, size_t size_p, size_t size_q, int tail, bool is_server)
{
	const char *name = (is_server ? "Server" : "Worker");
	for(int i = 1; i <= tail; i++) {
		printf("[%s] p[%d] %.2f, q[%d] %.2f\n", name, i, p[size_p-i], i, q[size_q-i]);
	}
}

// Macros ---------------------------------------------------------------------

#define  INT16_TYPE          short
#define UINT16_TYPE unsigned short
#define  INT32_TYPE          int
#define UINT32_TYPE unsigned int

int singles2halfp(void *target, const void *source, ptrdiff_t numel, int rounding_mode, int is_quiet)
{
    UINT16_TYPE *hp = (UINT16_TYPE *) target; // Type pun output as an unsigned 16-bit int
    const UINT32_TYPE *xp = (const UINT32_TYPE *) source; // Type pun input as an unsigned 32-bit int
    UINT16_TYPE    hs, he, hm, hr, hq;
    UINT32_TYPE x, xs, xe, xm, xt, zm, zt, z1;
    ptrdiff_t i;
    int hes, N;
    static int next;  // Little Endian adjustment
    static int checkieee = 1;  // Flag to check for IEEE754, Endian, and word size
    double one = 1.0; // Used for checking IEEE754 floating point format
    UINT32_TYPE *ip; // Used for checking IEEE754 floating point format

    if( checkieee ) { // 1st call, so check for IEEE754, Endian, and word size
        ip = (UINT32_TYPE *) &one;
        if( *ip ) { // If Big Endian, then no adjustment
            next = 0;
        } else { // If Little Endian, then adjustment will be necessary
            next = 1;
            ip++;
        }
        if( *ip != 0x3FF00000u ) { // Check for exact IEEE 754 bit pattern of 1.0
            return 1;  // Floating point bit pattern is not IEEE 754
        }
        if( sizeof(INT16_TYPE) != 2 || sizeof(INT32_TYPE) != 4 ) {
            return 1;  // short is not 16-bits, or long is not 32-bits.
        }
        checkieee = 0; // Everything checks out OK
    }

    if( source == NULL || target == NULL ) { // Nothing to convert (e.g., imag part of pure real)
        return 0;
    }

    // Make sure rounding mode is valid
    if( rounding_mode < 0 )
        rounding_mode = fegetround( );
    if( rounding_mode != FE_TONEAREST &&
        rounding_mode != FE_UPWARD &&
        rounding_mode != FE_DOWNWARD &&
        rounding_mode != FE_TOWARDZERO &&
        rounding_mode != FE_TONEARESTINF ) {
        rounding_mode = FE_TONEAREST;
    }
    hq = is_quiet ? (UINT16_TYPE) 0x0200u: (UINT16_TYPE) 0x0000u;  // Force NaN results to be quiet?

    // Loop through the elements
#pragma omp parallel for private(x,xs,xe,xm,xt,zm,zt,z1,hs,he,hm,hr,hes,N)
    for( i=0; i<numel; i++ ) {
        x = xp[i];
        if( (x & 0x7FFFFFFFu) == 0 ) {  // Signed zero
            hp[i] = (UINT16_TYPE) (x >> 16);  // Return the signed zero
        } else { // Not zero
            xs = x & 0x80000000u;  // Pick off sign bit
            xe = x & 0x7F800000u;  // Pick off exponent bits
            xm = x & 0x007FFFFFu;  // Pick off mantissa bits
            xt = x & 0x00001FFFu;  // Pick off trailing 13 mantissa bits beyond the shift (used for rounding normalized determination)
            if( xe == 0 ) {  // Denormal will underflow, return a signed zero or smallest denormal depending on rounding_mode
                if( (rounding_mode == FE_UPWARD   && xm && !xs) ||  // Mantissa bits are non-zero and sign bit is 0
                    (rounding_mode == FE_DOWNWARD && xm &&  xs) ) { // Mantissa bits are non-zero and sigh bit is 1
                    hp[i] = (UINT16_TYPE) (xs >> 16) | (UINT16_TYPE) 1u;  // Signed smallest denormal
                } else {
                    hp[i] = (UINT16_TYPE) (xs >> 16);  // Signed zero
                }
            } else if( xe == 0x7F800000u ) {  // Inf or NaN (all the exponent bits are set)
                if( xm == 0 ) { // If mantissa is zero ...
                    hp[i] = (UINT16_TYPE) ((xs >> 16) | 0x7C00u); // Signed Inf
                } else {
                    hm = (UINT16_TYPE) (xm >> 13); // Shift mantissa over
                    if( hm ) { // If we still have some non-zero bits (payload) after the shift ...
                        hp[i] = (UINT16_TYPE) ((xs >> 16) | 0x7C00u | hq | hm); // Signed NaN, shifted mantissa bits set
                    } else {
                        hp[i] = (UINT16_TYPE) ((xs >> 16) | 0x7E00u); // Signed NaN, only 1st mantissa bit set (quiet)
                    }
                }
            } else { // Normalized number
                hs = (UINT16_TYPE) (xs >> 16); // Sign bit
                hes = ((int)(xe >> 23)) - 127 + 15; // Exponent unbias the single, then bias the halfp
                if( hes >= 0x1F ) {  // Overflow
                    hp[i] = (UINT16_TYPE) ((xs >> 16) | 0x7C00u); // Signed Inf
                } else if( hes <= 0 ) {  // Underflow exponent, so halfp will be denormal
                    xm |= 0x00800000u;  // Add the hidden leading bit
                    N = (14 - hes);  // Number of bits to shift mantissa to get it into halfp word
                    hm = (N < 32) ? (UINT16_TYPE) (xm >> N) : (UINT16_TYPE) 0u; // Halfp mantissa
                    hr = (UINT16_TYPE) 0u; // Rounding bit, default to 0 for now (this will catch FE_TOWARDZERO and other cases)
                    if( N <= 24 ) {  // Mantissa bits have not shifted away from the end
                        zm = (0x00FFFFFFu >> N) << N;  // Halfp denormal mantissa bit mask
                        zt = 0x00FFFFFFu & ~zm;  // Halfp denormal trailing manntissa bits mask
                        z1 = (zt >> (N-1)) << (N-1);  // First bit of trailing bit mask
                        xt = xm & zt;  // Trailing mantissa bits
                        if( rounding_mode == FE_TONEAREST ) {
                            if( xt > z1 || xt == z1 && (hm & 1u) ) { // Trailing bits are more than tie, or tie and mantissa is currently odd
                                hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
                            }
                        } else if( rounding_mode == FE_TONEARESTINF ) {
                            if( xt >= z1  ) { // Trailing bits are more than or equal to tie
                                hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
                            }
                        } else if( (rounding_mode == FE_UPWARD   && xt && !xs) ||  // Trailing bits are non-zero and sign bit is 0
                                   (rounding_mode == FE_DOWNWARD && xt &&  xs) ) { // Trailing bits are non-zero and sign bit is 1
                            hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
                        }
                    } else {  // Mantissa bits have shifted at least one bit beyond the end (ties not possible)
                        if( (rounding_mode == FE_UPWARD   && xm && !xs) ||  // Trailing bits are non-zero and sign bit is 0
                            (rounding_mode == FE_DOWNWARD && xm &&  xs) ) { // Trailing bits are non-zero and sign bit is 1
                            hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
                        }
                    }
                    hp[i] = (hs | hm) + hr; // Combine sign bit and mantissa bits and rounding bit, biased exponent is zero
                } else {
                    he = (UINT16_TYPE) (hes << 10); // Exponent
                    hm = (UINT16_TYPE) (xm >> 13); // Mantissa
                    hr = (UINT16_TYPE) 0u; // Rounding bit, default to 0 for now
                    if( rounding_mode == FE_TONEAREST ) {
                        if( xt > 0x00001000u || xt == 0x00001000u && (hm & 1u) ) { // Trailing bits are more than tie, or tie and mantissa is currently odd
                            hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
                        }
                    } else if( rounding_mode == FE_TONEARESTINF ) {
                        if( xt >= 0x00001000u  ) { // Trailing bits are more than or equal to tie
                            hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
                        }
                    } else if( (rounding_mode == FE_UPWARD   && xt && !xs) ||  // Trailing bits are non-zero and sign bit is 0
                               (rounding_mode == FE_DOWNWARD && xt &&  xs) ) { // Trailing bits are non-zero and sign bit is 1
                        hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
                    }
                    hp[i] = (hs | he | hm) + hr;  // Adding rounding bit might overflow into exp bits, but that is OK
                }
            }
        }
    }
    return 0;
}

int halfp2singles(void *target, void *source, ptrdiff_t numel)
{
    UINT16_TYPE *hp = (UINT16_TYPE *) source; // Type pun input as an unsigned 16-bit int
    UINT32_TYPE *xp = (UINT32_TYPE *) target; // Type pun output as an unsigned 32-bit int
    UINT16_TYPE h, hs, he, hm;
    UINT32_TYPE xs, xe, xm;
    INT32_TYPE xes;
    ptrdiff_t i;
    int e;
    static int next;  // Little Endian adjustment
    static int checkieee = 1;  // Flag to check for IEEE754, Endian, and word size
    double one = 1.0; // Used for checking IEEE754 floating point format
    UINT32_TYPE *ip; // Used for checking IEEE754 floating point format
    
    if( checkieee ) { // 1st call, so check for IEEE754, Endian, and word size
        ip = (UINT32_TYPE *) &one;
        if( *ip ) { // If Big Endian, then no adjustment
            next = 0;
        } else { // If Little Endian, then adjustment will be necessary
            next = 1;
            ip++;
        }
        if( *ip != 0x3FF00000u ) { // Check for exact IEEE 754 bit pattern of 1.0
            return 1;  // Floating point bit pattern is not IEEE 754
        }
        if( sizeof(INT16_TYPE) != 2 || sizeof(INT32_TYPE) != 4 ) {
            return 1;  // short is not 16-bits, or long is not 32-bits.
        }
        checkieee = 0; // Everything checks out OK
    }
    
    if( source == NULL || target == NULL ) // Nothing to convert (e.g., imag part of pure real)
        return 0;
    
#pragma omp parallel for private(xs,xe,xm,h,hs,he,hm,xes,e)
    for( i=0; i<numel; i++ ) {
        h = hp[i];
        if( (h & 0x7FFFu) == 0 ) {  // Signed zero
            xp[i] = ((UINT32_TYPE) h) << 16;  // Return the signed zero
        } else { // Not zero
            hs = h & 0x8000u;  // Pick off sign bit
            he = h & 0x7C00u;  // Pick off exponent bits
            hm = h & 0x03FFu;  // Pick off mantissa bits
            if( he == 0 ) {  // Denormal will convert to normalized
                e = -1; // The following loop figures out how much extra to adjust the exponent
                do {
                    e++;
                    hm <<= 1;
                } while( (hm & 0x0400u) == 0 ); // Shift until leading bit overflows into exponent bit
                xs = ((UINT32_TYPE) hs) << 16; // Sign bit
                xes = ((INT32_TYPE) (he >> 10)) - 15 + 127 - e; // Exponent unbias the halfp, then bias the single
                xe = (UINT32_TYPE) (xes << 23); // Exponent
                xm = ((UINT32_TYPE) (hm & 0x03FFu)) << 13; // Mantissa
                xp[i] = (xs | xe | xm); // Combine sign bit, exponent bits, and mantissa bits
            } else if( he == 0x7C00u ) {  // Inf or NaN (all the exponent bits are set)
                xp[i] = (((UINT32_TYPE) hs) << 16) | ((UINT32_TYPE) 0x7F800000u) | (((UINT32_TYPE) hm) << 13); // Signed Inf or NaN
            } else { // Normalized number
                xs = ((UINT32_TYPE) hs) << 16; // Sign bit
                xes = ((INT32_TYPE) (he >> 10)) - 15 + 127; // Exponent unbias the halfp, then bias the single
                xe = (UINT32_TYPE) (xes << 23); // Exponent
                xm = ((UINT32_TYPE) hm) << 13; // Mantissa
                xp[i] = (xs | xe | xm); // Combine sign bit, exponent bits, and mantissa bits
            }
        }
    }
    return 0;
}

