#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>

typedef struct {
    float* values; // Using a 1D array to simulate a matrix
    size_t m; // Number of rows
    size_t n; // Number of columns
} matrix_t;

matrix_t zeroes(const size_t m, const size_t n);
matrix_t random_matrix(const size_t m, const size_t n);
void normalise(matrix_t matrix);
matrix_t matrix_add_vector(matrix_t matrix, matrix_t vector);
matrix_t transpose(matrix_t matrix);
matrix_t matrix_tile_multiply(matrix_t a, matrix_t b);
float add(float x, float y);
float multiply(float x, float y);
matrix_t matrix_apply(matrix_t* a, matrix_t* b, const float scalar, float (*function)(float, float));
void print_matrix(matrix_t matrix);
void determine_cache(void);
#endif
