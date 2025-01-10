#include "matrix.h"

#include <assert.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    matrix_t* a;
    matrix_t* b;
    matrix_t* c;
    size_t row;  // Row index for the result matrix
} thread_data_t;

// Returns an m x n matrix, initialised to zero
matrix_t zeroes(const size_t m, const size_t n) {
    matrix_t matrix;
    matrix.m = m;
    matrix.n = n;
    matrix.values = (float*) calloc(m * n, sizeof(float));
    assert(matrix.values != NULL);

    return matrix;
}

// Returns an m x n matrix initialised to random values between -1 and 1
matrix_t random_matrix(const size_t m, const size_t n) {
    srand(time(NULL));

    matrix_t matrix;
    matrix.m = m;
    matrix.n = n;
    matrix.values = (float*) malloc(m * n * sizeof(float));
    assert(matrix.values != NULL);

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            matrix.values[i * n + j] = (float) rand() / (float) RAND_MAX;
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
    float max = float_abs(matrix.values[0]);

    for (size_t i = 0; i < matrix.m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            float num = float_abs(matrix.values[i * matrix.n + j]);
            max = (num > max) ? num : max;
        }
    }
    if (max == 0.0) {
        return;
    }
    for (size_t i = 0; i < matrix.m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            matrix.values[i * matrix.n + j] /= max;
        }
    }
}

// Add a vector row-wise to a matrix
void matrix_add_vector(matrix_t matrix, matrix_t vector) {
    assert((matrix.n == vector.n) && vector.m == 1);

    for (size_t i = 0; i < matrix.m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            matrix.values[i * matrix.n + j] += vector.values[j];
        }
    }
}

// Returns the transpose of a matrix
matrix_t transpose(matrix_t original) {
    matrix_t transposed = zeroes(original.n, original.m);

    for (size_t i = 0; i < original.m; i++) {
        for (size_t j = 0; j < original.n; j++) {
            transposed.values[j * transposed.n + i] = original.values[i * original.n + j];
        }
    }
    return transposed;
}

static void* multiply_row(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    matrix_t* a = data->a;
    matrix_t* b = data->b;
    matrix_t* c = data->c;
    size_t row = data->row;

    for (size_t j = 0; j < b->n; j++) {
        __m256 mul = _mm256_setzero_ps();
        float sum = 0;
        size_t k = 0;
        if (a->n > 8) {
            for (; k <= a->n - 8; k += 8) {
                __m256 a_vec = _mm256_load_ps(&a->values[row * a->n + k]);
                __m256 b_vec =
                    _mm256_set_ps(b->values[(k + 7) * b->n + j], b->values[(k + 6) * b->n + j],
                                  b->values[(k + 5) * b->n + j], b->values[(k + 4) * b->n + j],
                                  b->values[(k + 3) * b->n + j], b->values[(k + 2) * b->n + j],
                                  b->values[(k + 1) * b->n + j], b->values[k * b->n + j]);
                mul = _mm256_add_ps(_mm256_mul_ps(a_vec, b_vec), mul);
            }
            float array_mul[8];
            _mm256_storeu_ps(array_mul, mul);
            sum += array_mul[0] + array_mul[1] + array_mul[2] + array_mul[3] +
                   array_mul[4] + array_mul[5] + array_mul[6] + array_mul[7];
            // for remaining
        }
        for (; k < a->n; k++) {
            sum += a->values[row * a->n + k] * b->values[k * b->n + j];
        }
        c->values[row * c->n + j] = sum;
    }
    return NULL;
}

matrix_t multiply(matrix_t a, matrix_t b) {
    assert(a.n == b.m);

    matrix_t c = zeroes(a.m, b.n);

    pthread_t* threads = malloc(c.m * sizeof(pthread_t));
    thread_data_t* thread_data = malloc(c.m * sizeof(thread_data_t));

    for (size_t i = 0; i < c.m; i++) {
        thread_data[i].a = &a;
        thread_data[i].b = &b;
        thread_data[i].c = &c;
        thread_data[i].row = i;
        pthread_create(&threads[i], NULL, multiply_row, (void*) &thread_data[i]);
    }
    for (size_t i = 0; i < c.m; i++) {
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
                printf("%.2f   ", matrix.values[i * matrix.n + j]);
            } else {
                printf("%.2f ", matrix.values[i * matrix.n + j]);
            }
        }
        printf("|\n");
    }
    printf("\n");
}
