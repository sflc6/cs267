#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

// 16 byte aligned malloc
#define malloc16(N) \
  (((uintptr_t) (malloc(N + 15)) + 15) & ~(uintptr_t) 0x0F)

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
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k)
      {
        cij += A_t[k+i*lda] * B[k+j*lda];
      }
      C[i+j*lda] = cij;
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major
 * format. On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
  // Re-allocate space for A, B, and C. This time, ensure that they're
  // all 16-byte aligned. Also, A needs to be transposed.
  double* al_A_t = (double*) malloc16(sizeof(double) * lda * lda);
  double* al_B = (double*) malloc16(sizeof(double) * lda * lda);
  double* al_C = (double*) malloc16(sizeof(double) * lda * lda);

  // Ensure that the pointers are in fact aligned
  assert(((uintptr_t) al_A_t & 15) == 0);
  assert(((uintptr_t) al_B & 15) == 0);
  assert(((uintptr_t) al_C & 15) == 0);

  // Move A over -- this requires a transpose
  memcpy(al_A_t, A, lda * lda * sizeof(double));
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < lda; ++j) {
      al_A_t[j+i*lda] = A[i+j*lda];
    }
  }

  // Move B over -- we can copy this over verbatim
  memcpy(al_B, B, lda * lda * sizeof(double));

  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE) {
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the
         * matrix */
        int M = min (BLOCK_SIZE, lda-i);
        int N = min (BLOCK_SIZE, lda-j);
        int K = min (BLOCK_SIZE, lda-k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, al_A_t + k + i*lda, al_B + k + j*lda,
                 C + i + j*lda);
      }
    }
  }
}

