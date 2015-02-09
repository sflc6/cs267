#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <sys/time.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>

#define malloc32(N) \
  (((uintptr_t) (malloc(N + 31)) + 31) & ~(uintptr_t) 0x0FF)

__attribute__((always_inline))
inline double dotSSE(double* a, double* b, int n) {
  double dot[2];
  __m128d vdot = _mm_set1_pd(0);

  for (int i = 0; i < (n / 2) * 2; i += 2) {
    __m128d v_a = _mm_load_pd(&a[i]);
    __m128d v_b = _mm_load_pd(&b[i]);

    v_a = _mm_mul_pd(v_a, v_b);
    vdot = _mm_add_pd(vdot, v_a);
  }

  _mm_store_pd(dot, vdot);

  return (n % 2 == 0 ? dot[0] + dot[1] :
          dot[0] + dot[1] + a[n - 1] * b[n - 1]);
}

float reduceSSE(const float* a, int n) {
  float sum[4];
  __m128 vsum = _mm_set1_ps(0);

  // Ensure that n is a multiple of 4
  assert((n & 3) == 0);

  // Ensure that a is 16-byte aligned
  assert(((uintptr_t) a & 15) == 0);

  // Execute the adds
  for (int i = 0; i < n; i += 4) {
    __m128 value = _mm_load_ps(&a[i]);
    vsum = _mm_add_ps(vsum, value);
  }

  // Store results back in sum
  _mm_store_ps(sum, vsum);

  // Return
  return sum[0] + sum[1] + sum[2] + sum[3];
}

float reduceAVX(const float* a, int n) {
  float sum[8];
  __m256 vsum = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);

  // Ensure that n is a multiple of 8
  assert((n & 7) == 0);

  // Ensure that a is 16-byte aligned
  assert(((uintptr_t) a & 31) == 0);

  // Execute the adds
  for (int i = 0; i < n; i += 8) {
    __m256 value = _mm256_load_ps(&a[i]);
    vsum = _mm256_add_ps(vsum, value);
  }

  // Store results back in sum
  _mm256_store_ps(sum, vsum);

  // Return
  float res = sum[0];
  res += sum[1];
  res += sum[2];
  res += sum[3];
  res += sum[4];
  res += sum[5];
  res += sum[6];
  res += sum[7];
  return res;
}

void benchmark_reduce() {
  int N = 8 * 100000;

  float* ptr16 = (float*) malloc32(sizeof(float) * N);
  for (int j = 0; j < N; ++j) ptr16[j] = (j * 13) % 10;

  float* ptr32 = (float*) malloc32(sizeof(float) * N);
  for (int i = 0; i < N; ++i) ptr32[i] = (i * 13) % 10;

  // [TIMING BLOCK] sse
  struct timeval timer_sse;
  gettimeofday(&timer_sse, NULL);
  double tic_sse =
      (double) timer_sse.tv_sec +
      (double) (1e-6 * timer_sse.tv_usec);
  printf("[Timing] sse ...\n");
  ///
  printf("SSE sum: %f\n", reduceSSE(ptr16, N));
  ///
  gettimeofday(&timer_sse, NULL);
  double toc_sse =
      (double) timer_sse.tv_sec +
      (double) (1e-6 * timer_sse.tv_usec);
  double sse =
      toc_sse -
      tic_sse;
  printf("[Timing] sse: ");
  printf("%f\n", sse);

  // [TIMING BLOCK] avx
  struct timeval timer_avx;
  gettimeofday(&timer_avx, NULL);
  double tic_avx =
      (double) timer_avx.tv_sec +
      (double) (1e-6 * timer_avx.tv_usec);
  printf("[Timing] avx ...\n");
  ///
  printf("AVX sum: %f\n", reduceAVX(ptr32, N));
  ///
  gettimeofday(&timer_avx, NULL);
  double toc_avx =
      (double) timer_avx.tv_sec +
      (double) (1e-6 * timer_avx.tv_usec);
  double avx =
      toc_avx -
      tic_avx;
  printf("[Timing] avx: ");
  printf("%f\n", avx);

  // [TIMING BLOCK] naive
  struct timeval timer_naive;
  gettimeofday(&timer_naive, NULL);
  double tic_naive =
      (double) timer_naive.tv_sec +
      (double) (1e-6 * timer_naive.tv_usec);
  printf("[Timing] naive ...\n");
  ///
  float sum = 0;
  for (int i = 0; i < N; ++i) sum += ptr32[i];
  ///
  gettimeofday(&timer_naive, NULL);
  double toc_naive =
      (double) timer_naive.tv_sec +
      (double) (1e-6 * timer_naive.tv_usec);
  double naive =
      toc_naive -
      tic_naive;
  printf("[Timing] naive: ");
  printf("%f\n", naive);

  printf("True sum: %f\n", sum);
}

int main() {
  int N = 31;
  double* ptr = (double*) malloc32(sizeof(double) * N);
  for (int i = 0; i < N; ++i) ptr[i] = i + 1;

  printf("%f\n", dotSSE(ptr, ptr, N));

  double ret = 0;
  for (int i = 0; i < N; ++i) {
    ret += ptr[i] * ptr[i];
  }
  printf("%f\n", ret);

  return 0;
}

