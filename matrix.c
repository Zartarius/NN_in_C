#include "matrix.h"

#include <assert.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    matrix_t* A;
    matrix_t* B;
    matrix_t* C;
    size_t row;  // Row index for the result matrix
} thread_data_t;

// Returns an m x n matrix, initialised to zero
matrix_t zeroes(const size_t m, const size_t n) {
    matrix_t matrix;
    matrix.m = m;
    matrix.n = n;
    matrix.values = (float**)calloc(m, sizeof(float*));
    assert(matrix.values != NULL);

    for (size_t i = 0; i < m; i++) {
        matrix.values[i] = (float*)calloc(n, sizeof(float));
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
    matrix.values = (float**)malloc(m * sizeof(float*));
    assert(matrix.values != NULL);

    for (size_t i = 0; i < m; i++) {
        matrix.values[i] = (float*)malloc(n * sizeof(float));
        assert(matrix.values[i] != NULL);
    }
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            matrix.values[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
    return matrix;
}

// Private helper function
static inline float float_abs(float num) { return (num < 0) ? -num : num; }

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

    for (size_t i = 0; i < transposed.m; i++) {
        for (size_t j = 0; j < transposed.n; j++) {
            transposed.values[i][j] = original.values[j][i];
        }
    }
    return transposed;
}

static void* multiply_row(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    matrix_t* a = data->A;
    matrix_t* b = data->B;
    matrix_t* c = data->C;
    size_t row = data->row;

    for (size_t j = 0; j < B->n; j++) {
        __m256 mul = _mm256_setzero_ps();
        size_t k = 0;
        for (; k <= a->n - 8; k += 8) {
            __m256 a_vec = _mm256_load_ps(&a->values[row][k]);
            __m256 b_vec = _mm256_set_ps(
                b->values[k + 7][j], b->values[k + 6][j], b->values[k + 5][j],
                b->values[k + 4][j], b->values[k + 3][j], b->values[k + 2][j],
                b->values[k + 1][j], b->values[k][j]);
            mul = _mm256_add_ps(_mm256_mul_ps(a_vec, b_vec), mul);
        }
        float array_mul[8];
        _mm256_storeu_ps(array_mul, mul);
        float sum = array_mul[0] + array_mul[1] + array_mul[2] + array_mul[3] +
                    array_mul[4] + array_mul[5] + array_mul[6] + array_mul[7];
        // for remaining
        for (; k < a->n; k++) {
            sum += a->values[row][k] * b->values[k][j];
        }
        c->values[row][j] = sum;
    }
    return NULL;
}

matrix_t multiply(matrix_t a, matrix_t a) {
    assert(a.n == b.m);

    matrix_t c = zeroes(a.m, b.n);

    pthread_t* threads = malloc(C.m * sizeof(pthread_t));
    thread_data_t* thread_data = malloc(C.m * sizeof(thread_data_t));

    for (size_t i = 0; i < C.m; i++) {
        thread_data[i].A = &a;
        thread_data[i].B = &b;
        thread_data[i].C = &c;
        thread_data[i].row = i;
        pthread_create(&threads[i], NULL, multiply_row, (void*)&thread_data[i]);
    }

    for (size_t i = 0; i < C.m; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(thread_data);
    return c;
}

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
