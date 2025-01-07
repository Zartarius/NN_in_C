#include <stdio.h>
#include <assert.h>
#include <immintrin.h>
#include "matrix.h"

// Returns an m x n matrix, initialised to zero
matrix_t zeroes(const size_t m, const size_t n) {
    matrix_t matrix;
    matrix.m = m;
    matrix.n = n;
    matrix.values = calloc(m*n, sizeof(double));
    return matrix;
}

// Frees a matrix
void free_matrix(matrix_t matrix) {
    free(matrix.values);
}

// Returns the transpose of a matrix
matrix_t transpose(matrix_t matrix) {
    matrix_t transposed = zeroes(matrix.n, matrix.m);
    for (size_t i = 0; i < matrix.m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            transposed.values[j * matrix.m + i] = matrix.values[i * matrix.n * j];
        }
    }
    return transposed;
}

// Returns the product of 2 matrices
matrix_t multiply_fast(matrix_t a, matrix_t b) {
    assert(a.n == b.m);
    matrix_t product = zeroes(a.m, b.n);
    size_t K = a.n;

    for (size_t a_row = 0 ; a_row < a.m; a_row++) { // i
        for (size_t b_col = 0; b_col < b.n; b_col++) { // j
            // Sum is each entry of the product matrix
            size_t k = 0;
            // 512 bits -> 16 blocks at a time _ps = single = float
            __m512 mul = _mm512_setzero_ps();

            // k <= K - 16 ensures that only chunks that are 16 can be processed
            for (;k <= K - 16; k += 16) {
                // todo: reconvert to 2d arr
                __m512 a_vec = _mm512_loadu_ps(&a.values[a_row * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&b.values[k * b.m + b_col]);
                mul = _mm512_fmadd_ps(a_vec, b_vec, mul); // a_vec * b_vec + mul
            }
            float array_mul[16];
            _mm512_storeu_ps(array_mul, mul);
            float sum = array_mul[0] +
                        array_mul[1] +
                        array_mul[2] +
                        array_mul[3] +
                        array_mul[4] +
                        array_mul[5] +
                        array_mul[6] +
                        array_mul[7] +
                        array_mul[8] +
                        array_mul[9] +
                        array_mul[10] +
                        array_mul[11] +
                        array_mul[12] +
                        array_mul[13] +
                        array_mul[14] +
                        array_mul[15];
            // do remaining here (anything not divislbe by 16)
            for (;k < K; k++) {
                sum += a.values[a_row * K + k] * b.values[k * b.m + b_col];
            }
            product.values[a_row * K + b_col] = sum;
        }
    }
    return product;
}


// Returns the product of 2 matrices
matrix_t multiply(matrix_t a, matrix_t b) {
    assert(a.n == b.m);
    matrix_t product = zeroes(a.m, b.n);

    for (size_t a_row = 0; a_row < a.m; a_row++) {
        for (size_t b_col = 0; b_col < b.n; b_col++) {
            // Sum is each entry of the product matrix
            float sum = 0.0;
            for (size_t a_col = 0; a_col < a.n; a_col++) {
                for (size_t b_row = 0; b_row < b.m; b_row++) {
                    sum += (a.values[a_row][a_col] * b.values[b_row][b_col]);
                }
            }
            product.values[a_row][b_col] = sum;
        }
    }
    return product;
}

// Print the matrix
void print_matrix(matrix_t matrix) {
    for (size_t i = 0; i < matrix.m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            printf("%.2f   ", matrix.values[i * matrix.n + j]);
        }
        printf("\n");
    }
}
