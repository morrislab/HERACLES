/* Scalar stand-ins for the handful of SSE2 intrinsics RapidNJ uses in its
 * alignment-to-distance code (bitDistanceGap, bitDistanceProtein, the sequence
 * loaders), so the source builds on non-x86 hosts such as Apple Silicon.
 *
 * On x86 the real <emmintrin.h> is used instead (see stdinclude.h).  HERACLES
 * only ever runs RapidNJ on a precomputed distance matrix (-i pd), so this path
 * is never executed there; it exists to make the program link everywhere.
 */
#ifndef RAPIDNJ_SSE_COMPAT_H
#define RAPIDNJ_SSE_COMPAT_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t u[4]; } __m128i;

static inline __m128i _mm_set_epi32(int e3, int e2, int e1, int e0) {
  __m128i r;
  r.u[0] = (uint32_t)e0; r.u[1] = (uint32_t)e1; r.u[2] = (uint32_t)e2; r.u[3] = (uint32_t)e3;
  return r;
}
static inline __m128i _mm_setzero_si128(void) { __m128i r; memset(&r, 0, sizeof r); return r; }
static inline __m128i _mm_add_epi32(__m128i a, __m128i b) {
  __m128i r; for (int i = 0; i < 4; ++i) r.u[i] = a.u[i] + b.u[i]; return r;
}
static inline __m128i _mm_and_si128(__m128i a, __m128i b) {
  __m128i r; for (int i = 0; i < 4; ++i) r.u[i] = a.u[i] & b.u[i]; return r;
}
static inline __m128i _mm_andnot_si128(__m128i a, __m128i b) {
  __m128i r; for (int i = 0; i < 4; ++i) r.u[i] = (~a.u[i]) & b.u[i]; return r;
}
static inline __m128i _mm_or_si128(__m128i a, __m128i b) {
  __m128i r; for (int i = 0; i < 4; ++i) r.u[i] = a.u[i] | b.u[i]; return r;
}
static inline __m128i _mm_xor_si128(__m128i a, __m128i b) {
  __m128i r; for (int i = 0; i < 4; ++i) r.u[i] = a.u[i] ^ b.u[i]; return r;
}
static inline __m128i _mm_srli_epi32(__m128i a, int n) {
  __m128i r; for (int i = 0; i < 4; ++i) r.u[i] = (n > 31) ? 0u : (a.u[i] >> n); return r;
}
static inline __m128i _mm_cmpeq_epi8(__m128i a, __m128i b) {
  __m128i r; const unsigned char *pa = (const unsigned char *)&a, *pb = (const unsigned char *)&b;
  unsigned char *pr = (unsigned char *)&r;
  for (int i = 0; i < 16; ++i) pr[i] = (pa[i] == pb[i]) ? 0xFFu : 0x00u;
  return r;
}
static inline void *_mm_malloc(size_t size, size_t align) {
  void *p = NULL;
  if (align < sizeof(void *)) align = sizeof(void *);
  if (posix_memalign(&p, align, size ? size : 1) != 0) return NULL;
  return p;
}
static inline void _mm_free(void *p) { free(p); }

#endif /* RAPIDNJ_SSE_COMPAT_H */
