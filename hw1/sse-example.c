#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <emmintrin.h>
#include <pmmintrin.h>

#define malloc16(N) \
  (((uintptr_t) (malloc(N + 15)) + 15) & ~(uintptr_t) 0x0F)

float reduce(const float* a, int n) {
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

int main() {
  int N = 20;
  // void* mem = malloc(sizeof(float) * N + 15);
  // float* ptr = (float*) (((uintptr_t) mem + 15) & ~(uintptr_t) 0x0F);

  float* ptr = (float*) malloc16(sizeof(float) * N);

  for (int i = 0; i < N; ++i) ptr[i] = i;

  printf("sum: %f\n", reduce(ptr, N));
  printf("sum: %d\n", N * (N - 1) / 2);

  return 0;
}

