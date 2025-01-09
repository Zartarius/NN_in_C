#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <immintrin.h>
#include <pthread.h>
#include "matrix.h"

typedef struct {
    matrix_t* A;
    matrix_t* B;
    matrix_t* C;
    size_t row; // Row index for the result matrix
} thread_data_t;

// Returns an m x n matrix, initialised to zero
matrix_t zeroes(const size_t m, const size_t n) {
    matrix_t matrix;
    matrix.m = m;
    matrix.n = n;
    matrix.values = (float**) calloc(m, sizeof(float*));
    assert(matrix.values != NULL);

    for (size_t i = 0; i < m; i++) {
        matrix.values[i] = (float*) calloc(n, sizeof(float));
        assert(matrix.values[i] != NULL);
    }
    return matrix;
}

// Returns an m x n matrix initialised to random values between -1 and 1
matrix_t random_matrix(const size_t m, const size_t n) {
    srand(time(NULL));

    matrix_t matrix;
    matrix.m = m;
    matrix.n = n;
    matrix.values = (float**) malloc(m * sizeof(float*));
    assert(matrix.values != NULL);

    for (size_t i = 0; i < m; i++) {
        matrix.values[i] = (float*) malloc(n * sizeof(float));
        assert(matrix.values[i] != NULL);
    }
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            matrix.values[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }
    return matrix;
}

// Private helper function
static inline float float_abs(float num) {
    return (num < 0) ? -num : num;
}

// Normalises a matrix to have values between -1 and 1
void normalise(matrix_t matrix) {
    float max = float_abs(matrix.values[0][0]);

    for (size_t i = 0; i < matrix.m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            float num = float_abs(matrix.values[i][j]);
            max = (num > max) ? num : max;
        }
    }
    if (max == 0.0) {
        return;
    }
    for (size_t i = 0; i < matrix.m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            matrix.values[i][j] /= max;
        }
    }
}

// Add a vector row-wise to a matrix
void matrix_add_vector(matrix_t matrix, matrix_t vector) {
    assert((matrix.n == vector.n) && vector.m == 1);

    for (size_t i = 0; i < matrix.m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            matrix.values[i][j] += vector.values[0][j];
        }
    }
}

// Frees a matrix
void free_matrix(matrix_t matrix) {
    for (size_t i = 0; i < matrix.m; i++) {
        free(matrix.values[i]);
    }
    free(matrix.values);
}

// Returns the transpose of a matrix
matrix_t transpose(matrix_t original) {
    matrix_t transposed = zeroes(original.n, original.m);

    for (int i = 0; i < transposed.m; i++) {
        for (int j = 0; j < transposed.n; j++) {
            transposed.values[i][j] = original.values[j][i];
        }
    }
    return transposed;
}

static void* multiply_row(void* arg) {
    thread_data_t* data = (thread_data_t*) arg;
    matrix_t* A = data->A;
    matrix_t* B = data->B;
    matrix_t* C = data->C;
    size_t row = data->row;

    for (size_t j = 0; j < B->n; j++) {
        C->values[row][j] = 0;
        for (size_t k = 0; k < A->n; k++) {
            C->values[row][j] += A->values[row][k] * B->values[k][j];
        }
    }
    return NULL;
}

matrix_t* multiply(matrix_t* A, matrix_t* B) {
    if (A->n != B->m) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return NULL;
    }

    matrix_t* C = (matrix_t*)malloc(sizeof(matrix_t));
    C->m = A->m;
    C->n = B->n;
    C->values = (float**)malloc(C->m * sizeof(float*));
    for (size_t i = 0; i < C->m; i++) {
        C->values[i] = (float*)malloc(C->n * sizeof(float));
    }

    pthread_t* threads = (pthread_t*)malloc(C->m * sizeof(pthread_t));
    thread_data_t* thread_data = (thread_data_t*)malloc(C->m * sizeof(thread_data_t));

    for (size_t i = 0; i < C->m; i++) {
        thread_data[i].A = A;
        thread_data[i].B = B;
        thread_data[i].C = C;
        thread_data[i].row = i;
        pthread_create(&threads[i], NULL, multiply_row, (void*)&thread_data[i]);
    }

    for (size_t i = 0; i < C->m; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(thread_data);
    return C;
}

/*
// Returns the product of 2 matrices
matrix_t multiply(matrix_t a, matrix_t b) {
    assert(a.n == b.m);
    matrix_t product = zeroes(a.m, b.n);
    size_t K = a.n;

    for (size_t a_row = 0; a_row < a.m; a_row++) {      // i
        for (size_t b_col = 0; b_col < b.n; b_col++) {  // j
            // Sum is each entry of the product matrix
            size_t k = 0;
            // 128 bits -> 4 blocks at a time _ps = single = float
            __m128 mul = _mm_setzero_ps();

            // k <= K - 4 ensures that only chunks that are 16 can be processed
            for (; k <= K - 4; k += 4) {
                // todo: reconvert to 2d arr
                __m128 a_vec = _mm_loadu_ps(&a.values[a_row][k]);
                // because its not contigious
                __m128 b_vec = _mm_set_ps(
                    b.values[k + 3][b_col], 
                    b.values[k + 2][b_col], 
                    b.values[k + 1][b_col], 
                    b.values[k][b_col]
                );
                mul = _mm_add_ps(_mm_mul_ps(a_vec, b_vec),
                                 mul);  // a_vec * b_vec + mul
            }
            float array_mul[16];
            _mm_storeu_ps(array_mul, mul);
            float sum =
                array_mul[0] + array_mul[1] + array_mul[2] + array_mul[3];
            // do remaining here (anything not divislbe by 16)
            for (; k < K; k++) {
                sum += a.values[a_row][k] * b.values[k][b_col];
            }
            product.values[a_row][b_col] = sum;
        }
    }
    return product;
}
*/

/*
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
}*/

void print_matrix(matrix_t matrix) {
    printf("\n");
    for (size_t i = 0; i < matrix.m; i++) {
        printf("| ");
        for (size_t j = 0; j < matrix.n; j++) {
            if (j < matrix.n - 1) {
                printf("%.2f   ", matrix.values[i][j]);
            } else {
                printf("%.2f ", matrix.values[i][j]);
            }
        }
        printf("|\n");
    }
    printf("\n");
}
