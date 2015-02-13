#include <emmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

const char* dgemm_desc = "Loop-tiled dgemm; A is re-ordered where A's blocks are placed in column-major order, and each block is in column-major order. Blocks of B are transposed and cached. Block multiplies are executed via SSE.";

#define min(a,b) (((a)<(b))?(a):(b))

// Do not change this
#define UNROLL_FACTOR 4

#define REARRANGE_A(FUNC, BLCK) \
__attribute__((always_inline)) \
inline double* FUNC(int lda, int new_lda, double* A,\
                    double* rearranged_A) {\
  double* p_A;\
  double* p_rearranged_A;\
  int new_width, new_height, height;\
  int i, j, k, l;\
  for (k = 0; k < lda; k += BLCK) {\
    new_width = min(BLCK, new_lda - k);\
    for (i = 0; i < new_lda; i += BLCK) {\
      /* Useful pointers/values */\
      p_A = A + i + k * lda;\
      p_rearranged_A = rearranged_A + new_width * i + new_lda * k;\
      new_height = min(BLCK, new_lda - i);\
      height = min(BLCK, lda - i);\
      \
      /* Fill in the current block, whose top left is at (i, k) in the\
       * original matrix. */\
      for (j = 0; j < new_width; j++) {\
        /* Fill in the actual values */\
        for (l = 0; l < height; l++) {\
          p_rearranged_A[l + j * new_height] = p_A[l + j * lda];\
        }\
        /* Pad with zeros */\
        for (l = height; l < new_height; l++) {\
          p_rearranged_A[l + j * new_height] = 0.0;\
        }\
      }\
    }\
  }\
  return rearranged_A;\
}\

#define SQUARE_DGEMM(FUNC, REARR_FUNC, BLCK) \
void FUNC(int lda, double* A, double* B, double* __restrict__ C) { \
  /* Create a padded C to work with aligned loads */\
  int new_lda = lda; \
  int i, j, k, l, m; \
  /* Round up to the nearest multiple of UNROLL_FACTOR */ \
  new_lda = (lda + UNROLL_FACTOR - 1) / UNROLL_FACTOR * UNROLL_FACTOR; \
  double padded_C[new_lda * new_lda] __attribute__((aligned(16))); \
  /* Fill in the padded version of C -- don't bother filling in the \
   * padded region, since we won't really be using the entries there. */\
  for (j = 0; j < lda; ++j) { \
    for (i = 0; i < lda; ++i) { \
      padded_C[i + j * new_lda] = C[i + j * lda]; \
    } \
  } \
  /* Re-arrange A */\
  double rearranged_A[new_lda * new_lda] __attribute__((aligned(16))); \
  REARR_FUNC(lda, new_lda, A, rearranged_A); \
  /* BLCK should be fairly small, so store cur_block_B on the \
   * stack rather than on the heap. */\
  double cur_block_B[BLCK* BLCK]; \
  double* p_B; \
  for (j = 0; j < new_lda; j += BLCK) { \
    for (k = 0; k < new_lda; k += BLCK) { \
      int K_new = min(new_lda - k, BLCK); \
      int K = min(lda - k, BLCK); \
      int N = min(new_lda - j, BLCK); \
      /* Store away the current block from B that we're looking at. If \
       * we're lucky, this should get stored in the L1 cache. */ \
      p_B = B + k + j * lda; \
      for (l = 0; l < N; ++l) { \
        for (m = 0; m < K; ++m) { \
          cur_block_B[l + m * N] = p_B[m + l * lda]; \
        } \
      } \
      /* Execute the loop-tiled matrix multiply */\
      for (i = 0; i < new_lda; i += BLCK) { \
        int M = min(BLCK, new_lda - i); \
        do_block(new_lda, M, N, K, \
                 rearranged_A + i * K_new + k * new_lda, \
                 cur_block_B, padded_C + i + j * new_lda); \
      } \
    } \
  } \
  /* If we're currently using padding, then unpad */\
  for (j = 0; j < lda; j++) { \
    for (i = 0; i < lda; i++) { \
      C[i + j * lda] = padded_C[i + j * new_lda]; \
    } \
  } \
} \

#define HAMMERTIME(FUNC_REARRANGE, FUNC_SQUARE_DGEMM, BLCK) \
  REARRANGE_A(FUNC_REARRANGE, BLCK) \
  SQUARE_DGEMM(FUNC_SQUARE_DGEMM, FUNC_REARRANGE, BLCK)

void do_block(int new_lda, int M, int N, int K,
              double* __restrict__ rearranged_A,
              double* __restrict__ cur_block_B,
              double* __restrict__ padded_C) {
  // A0 holds two rows in one column of A. A1 holds the next two rows
  // in the same column. So, A0 and A1 jointly hold four rows.
  register __m128d A0;
  register __m128d A1;

  // B0 - B3 hold four rows in one column of B. Values are duplicated
  // in each register to broadcast the multiplies made.
  register __m128d B0;
  register __m128d B1;
  register __m128d B2;
  register __m128d B3;

  register __m128d C0;
  register __m128d C1;
  register __m128d C2;
  register __m128d C3;
  register __m128d C4;
  register __m128d C5;
  register __m128d C6;
  register __m128d C7;

  // Ordering j before i yields a ~3% boost.
  for (int j = 0; j < N; j += UNROLL_FACTOR) {
    for (int i = 0; i < M; i += UNROLL_FACTOR) {
      // Column 1
      padded_C += i + j * new_lda;
      C0 = _mm_load_pd(padded_C);
      C1 = _mm_load_pd(padded_C + 2);

      // Column 2
      C2 = _mm_load_pd(padded_C + new_lda);
      C3 = _mm_load_pd(padded_C + new_lda + 2);

      // Column 3
      C4 = _mm_load_pd(padded_C + 2 * new_lda);
      C5 = _mm_load_pd(padded_C + 2 * new_lda + 2);

      // Column 4
      C6 = _mm_load_pd(padded_C + 3 * new_lda);
      C7 = _mm_load_pd(padded_C + 3 * new_lda + 2);

      // Iterate over the common dimension
      for (int k = 0; k < K; k++) {
        // Load into A registers
        A0 = _mm_load_pd(rearranged_A + i + k * M);
        A1 = _mm_load_pd(rearranged_A + i + k * M + 2);

        // Load into B registers
        B0 = _mm_load1_pd(cur_block_B + j + k * N);
        B1 = _mm_load1_pd(cur_block_B + j + k * N + 1);
        B2 = _mm_load1_pd(cur_block_B + j + k * N + 2);
        B3 = _mm_load1_pd(cur_block_B + j + k * N + 3);

        // Load into C registers
        C0 = _mm_add_pd(C0, _mm_mul_pd(A0, B0));
        C1 = _mm_add_pd(C1, _mm_mul_pd(A1, B0));
        C2 = _mm_add_pd(C2, _mm_mul_pd(A0, B1));
        C3 = _mm_add_pd(C3, _mm_mul_pd(A1, B1));
        C4 = _mm_add_pd(C4, _mm_mul_pd(A0, B2));
        C5 = _mm_add_pd(C5, _mm_mul_pd(A1, B2));
        C6 = _mm_add_pd(C6, _mm_mul_pd(A0, B3));
        C7 = _mm_add_pd(C7, _mm_mul_pd(A1, B3));
      }

      // Store results back in padded_C
      _mm_store_pd(padded_C, C0);
      _mm_store_pd(padded_C + 2, C1);

      _mm_store_pd(padded_C + new_lda, C2);
      _mm_store_pd(padded_C + new_lda + 2, C3);

      _mm_store_pd(padded_C + 2 * new_lda, C4);
      _mm_store_pd(padded_C + 2 * new_lda + 2, C5);

      _mm_store_pd(padded_C + 3 * new_lda, C6);
      _mm_store_pd(padded_C + 3 * new_lda + 2, C7);

      padded_C -= i + j * new_lda;
    }
  }
}

// Generate code for the various block sizes
HAMMERTIME(rearrange_a_32, square_dgemm_32, 32)
HAMMERTIME(rearrange_a_52, square_dgemm_52, 52)
HAMMERTIME(rearrange_a_68, square_dgemm_68, 68)
HAMMERTIME(rearrange_a_80, square_dgemm_80, 80)
HAMMERTIME(rearrange_a_88, square_dgemm_88, 88)

void square_dgemm(int lda, double* A, double* B, double* __restrict__ C) {
  if (lda <= 32) {
    square_dgemm_32(lda, A, B, C);
  } else if (lda <= 97) {
    square_dgemm_52(lda, A, B, C);
  } else if (lda <= 192) {
    square_dgemm_68(lda, A, B, C);
  } else if (lda <= 639) {
    square_dgemm_88(lda, A, B, C);
  } else {
    square_dgemm_80(lda, A, B, C);
  }
}

