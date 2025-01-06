#include <stdio.h>
#include <assert.h>
#include "matrix.h"

// Returns an m x n matrix, initialised to zero
matrix_t zeroes(size_t m, size_t n) {
    matrix_t matrix;
    matrix.m = m;
    matrix.n = n;
    matrix.values = (float**) calloc(m, sizeof(float*));

    for (size_t i = 0; i < m; i++) {
        matrix.values[i] = (float*) calloc(n, sizeof(float));
    }
    return matrix;
}

// Frees a matrix
void free_matrix(matrix_t matrix) {
    for (size_t i = 0; i < matrix.m; i++) {
        free(matrix.values[i]);
    }
    free(matrix.values);
}

// Returns the transpose of a matrix
matrix_t transpose(matrix_t matrix) {
    matrix_t transposed = zeroes(matrix.n, matrix.m);
    for (size_t i = 0; i < matrix.m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            transposed.values[j][i] = matrix.values[i][j];
        }
    }
    return transposed;
}

// Returns the product of 2 matrices
matrix_t multiply(matrix_t a, matrix_t b) {
    assert(a.n == b.m);
    matrix_t product = zeroes(a.m, b.n);

    for (size_t a_row = 0 ; a_row < a.m; a_row++) {
        for (size_t b_col = 0; b_col < b.n; b_col++) {
            // Sum is each entry of the product matrix
            float sum = 0.0;
            for (size_t a_col = 0; a_col < a.n; a_col++) {
                for (size_t b_row; b_row < b.m; b_row++) {
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
            printf("%.2f   ", matrix.values[i][j]);
        }
        printf("\n");
    }
}
