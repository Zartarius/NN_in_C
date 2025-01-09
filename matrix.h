#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>

typedef struct {
    float** values;
    size_t m; // Number of rows
    size_t n; // Number of columns
} matrix_t;

matrix_t zeroes(const size_t m, const size_t n);
matrix_t random_matrix(const size_t m, const size_t n);
static inline float float_abs(float num);
void normalise(matrix_t matrix);
void matrix_add_vector(matrix_t matrix, matrix_t vector);
void free_matrix(matrix_t matrix);
matrix_t transpose(matrix_t matrix);
static void* multiply_row(void* arg);
// matrix_t multiply(matrix_t a, matrix_t b);
matrix_t* multiply(matrix_t* A, matrix_t* B);
void print_matrix(matrix_t matrix);

#endif
