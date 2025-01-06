#include <stdlib.h>

typedef struct {
    float** values;
    size_t m; // Number of rows
    size_t n; // Number of columns
} matrix_t;

matrix_t zeroes(const size_t m, const size_t n);
void free_matrix(matrix_t matrix);
matrix_t transpose(matrix_t matrix);
matrix_t multiply(matrix_t a, matrix_t b);
void print_matrix(matrix_t matrix);
