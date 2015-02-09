#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

// 16 byte aligned malloc
#define malloc16(N) \
  (((uintptr_t) (malloc(N + 15)) + 15) & ~(uintptr_t) 0x0F)
#define get_align16(ptr) ((int) (((uintptr_t) (ptr)) & 15))
#define check_aligned16(ptr) assert(get_align16(ptr) == 0);

// 32 byte aligned malloc
#define malloc32(N) \
  (((uintptr_t) (malloc(N + 31)) + 31) & ~(uintptr_t) 0xFF)
#define get_align32(ptr) ((int) (((uintptr_t) (ptr)) & 31))
#define check_aligned32(ptr) assert(get_align32(ptr) == 0);

__attribute__((always_inline))
inline double dotAVX(double* a, double* b, int n) {
  assert(get_align32(a) == 0);
  assert(get_align32(b) == 0);

  double dot[4];
  __m256d vdot = _mm256_set_pd(0, 0, 0, 0);

  for (int i = 0; i < n; i += 4) {
    __m256d v_a = _mm256_load_pd(&a[i]);
    __m256d v_b = _mm256_load_pd(&b[i]);

    v_a = _mm256_mul_pd(v_a, v_b);
    vdot = _mm256_add_pd(vdot, v_a);
  }

  _mm256_store_pd(dot, vdot);

  return dot[0] + dot[1] + dot[2] + dot[3];
}

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

  return dot[0] + dot[1];
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double* A_t, double* B,
                     double* C) {
  /* For each row i of A */
  for (int j = 0; j < N; ++j)
  {
    /* For each column j of B */
    for (int i = 0; i < M; ++i)
    {
      /* Compute C(i,j) */
      // double cij = C[i+j*lda];
      // for (int k = 0; k < K; ++k)
      // {
      //   cij += A_t[k+i*lda] * B[k+j*lda];
      // }
      // C[i+j*lda] = cij;

      C[i+j*lda] += dotSSE(&A_t[i*lda], &B[j*lda], K);
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major
 * format. On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
  // Re-allocate space for A, B, and C. This time, ensure that they're
  // all 16-byte aligned. A needs to be transposed. Also, if lda is
  // odd, then we need to add an extra column and row of zeros.
  int new_lda = lda + lda % 2;
  double* al_A_t = (double*) malloc16(sizeof(double) * new_lda * new_lda);
  double* al_B = (double*) malloc16(sizeof(double) * new_lda * new_lda);

  // Ensure that the pointers are in fact aligned
  check_aligned16(al_A_t);
  check_aligned16(al_B);

  // If we have an even matrix, then work directly with the provided
  // output buffer. Otherwise, create a temporary buffer.
  double* destination;
  if (lda % 2 == 0) {
    destination = C;
  } else {
    destination = (double*) malloc16(sizeof(double) * new_lda * new_lda);
  }

  // Move A over -- this requires a transpose
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < lda; ++j) {
      al_A_t[j+i*new_lda] = A[i+j*lda];
    }
  }

  // Move B over -- we can copy this over verbatim
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < lda; ++j) {
      al_B[i+j*new_lda] = B[i+j*lda];
    }
  }

  // If we have odd matrices, then add padding
  if (lda % 2 == 1) {
    for (int i = 0; i < new_lda; ++i) {
      al_A_t[(new_lda-1)+i*new_lda] = 0;
      al_A_t[i+(new_lda-1)*new_lda] = 0;

      al_B[(new_lda-1)+i*new_lda] = 0;
      al_B[i+(new_lda-1)*new_lda] = 0;
    }
  }

  /* For each block-row of A */
  for (int i = 0; i < new_lda; i += BLOCK_SIZE) {
    /* For each block-column of B */
    for (int j = 0; j < new_lda; j += BLOCK_SIZE) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < new_lda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the
         * matrix */
        int M = min(BLOCK_SIZE, new_lda-i);
        int N = min(BLOCK_SIZE, new_lda-j);
        int K = min(BLOCK_SIZE, new_lda-k);

        /* Perform individual block dgemm */
        check_aligned16(al_A_t + k + i * new_lda);

        do_block(new_lda, M, N, K, al_A_t + k + i * new_lda,
                 al_B + k + j * new_lda, destination + i + j * new_lda);
      }
    }
  }

  // Copy back to C
  if (lda % 2 == 1) {
    for (int i = 0; i < lda; ++i) {
      for (int j = 0; j < lda; ++j) {
        C[i+j*lda] = destination[i+j*new_lda];
      }
    }
  }
}

