#include "matrix.h"

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>

#define TILE_SIZE 330

static size_t tile_size = TILE_SIZE;

// Structure to pass arguments to the thread function
typedef struct {
    matrix_t *a;
    matrix_t *b;
    matrix_t *c;
    size_t start_row;
    size_t start_col;
} thread_args_t;

/*
typedef struct {
    matrix_t* a;
    matrix_t* b;
    matrix_t* c;
    size_t row;  // Row index for the result matrix
} thread_data_t;
*/

// Returns an m x n matrix, initialised to zero
matrix_t zeroes(const size_t m, const size_t n) {
    matrix_t matrix;
    matrix.m = m;
    matrix.n = n;
    matrix.values = (float *)calloc(m * n, sizeof(float));
    assert(matrix.values != NULL);

    return matrix;
}

// Returns an m x n matrix initialised to random values between -1 and 1
matrix_t random_matrix(const size_t m, const size_t n) {
    srand(time(NULL));

    matrix_t matrix;
    matrix.m = m;
    matrix.n = n;
    matrix.values = (float *)malloc(m * n * sizeof(float));
    assert(matrix.values != NULL);

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            matrix.values[i * n + j] = (float)rand() / (float)RAND_MAX;
        }
    }
    return matrix;
}

inline void free_matrix(matrix_t matrix) { free(matrix.values); }
// Private helper function
static inline float float_abs(float num) { return (num < 0) ? -num : num; }

void determine_cache(void) {
    size_t cache_size = 0;
    FILE *fp = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
    if (fp) {
        char buffer[16];
        if (fgets(buffer, sizeof(buffer), fp)) {
            cache_size = strtoul(buffer, NULL, 10) * 1024;
        }
        fclose(fp);
    } else {
        perror("fopen");
        exit(1);
    }
    tile_size = (int)sqrt((cache_size / sizeof(float)) / 3);
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
            transposed.values[j * transposed.n + i] =
                original.values[i * original.n + j];
        }
    }
    return transposed;
}

// Function to compute the product of a tile using AVX
static void *compute_tile(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    matrix_t *c = args->c;
    size_t start_row = args->start_row;
    size_t start_col = args->start_col;

    // Compute the product of the tile
    for (size_t i = start_row; i < start_row + tile_size && i < a->m; i++) {
        for (size_t j = start_col; j < start_col + tile_size && j < b->n; j++) {
            __m256 sum_vec =
                _mm256_setzero_ps();  // Initialize AVX vector for accumulation
            float sum = 0.0f;

            // Process 8 elements at a time using AVX
            size_t k = 0;
            for (; k <= a->n - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(
                    &a->values[i * a->n + k]);  // Load 8 elements from matrix A
                __m256 b_vec = _mm256_set_ps(
                    b->values[(k + 7) * b->n + j],
                    b->values[(k + 6) * b->n + j],
                    b->values[(k + 5) * b->n + j],
                    b->values[(k + 4) * b->n + j],
                    b->values[(k + 3) * b->n + j],
                    b->values[(k + 2) * b->n + j],
                    b->values[(k + 1) * b->n + j],
                    b->values[k * b->n + j]);  // Load 8 elements from matrix B
                sum_vec = _mm256_add_ps(
                    sum_vec,
                    _mm256_mul_ps(
                        a_vec, b_vec));  //_mm256_fmadd_ps(a_vec, b_vec,
                                         // sum_vec); // Multiply and accumulate
            }

            // Sum the elements of the AVX vector
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            for (size_t t = 0; t < 8; t++) {
                sum += temp[t];
            }

            // Process remaining elements
            for (; k < a->n; k++) {
                sum += a->values[i * a->n + k] * b->values[k * b->n + j];
            }

            c->values[i * c->n + j] = sum;  // Store the result
        }
    }

    return NULL;
}

// Function to multiply two matrices using tiles, threads, and AVX
matrix_t matrix_tile_multiply(matrix_t a, matrix_t b) {
    assert(a.n == b.m);

    // Create the result matrix
    matrix_t c;
    c.m = a.m;
    c.n = b.n;
    c.values = (float *)malloc(c.m * c.n * sizeof(float));
    assert(c.values != NULL);

    // Calculate the number of tiles
    size_t num_tiles_row = (a.m + tile_size - 1) / tile_size;
    size_t num_tiles_col = (b.n + tile_size - 1) / tile_size;

    // Create threads for each tile
    pthread_t threads[num_tiles_row * num_tiles_col];
    thread_args_t args[num_tiles_row * num_tiles_col];

    // Initialize thread arguments and create threads
    for (size_t i = 0; i < num_tiles_row; i++) {
        for (size_t j = 0; j < num_tiles_col; j++) {
            // Create the thread arguments
            args[i * num_tiles_col + j].a = &a;
            args[i * num_tiles_col + j].b = &b;
            args[i * num_tiles_col + j].c = &c;
            args[i * num_tiles_col + j].start_row = i * tile_size;
            args[i * num_tiles_col + j].start_col = j * tile_size;

            // Create the thread
            if (pthread_create(&threads[i * num_tiles_col + j], NULL,
                               compute_tile,
                               &args[i * num_tiles_col + j]) != 0) {
                printf("Error: Failed to create thread\n");
                exit(1);
            }
        }
    }

    // Wait for all threads to finish
    for (size_t i = 0; i < num_tiles_row * num_tiles_col; i++) {
        assert(pthread_join(threads[i], NULL) == 0);
    }

    return c;
}

/*
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
                    _mm256_set_ps(b->values[(k + 7) * b->n + j], b->values[(k +
6) * b->n + j], b->values[(k + 5) * b->n + j], b->values[(k + 4) * b->n + j],
                                  b->values[(k + 3) * b->n + j], b->values[(k +
2) * b->n + j], b->values[(k + 1) * b->n + j], b->values[k * b->n + j]); mul =
_mm256_add_ps(_mm256_mul_ps(a_vec, b_vec), mul);
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
        pthread_create(&threads[i], NULL, multiply_row, (void*)
&thread_data[i]);
    }
    for (size_t i = 0; i < c.m; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(thread_data);
    return c;
}
*/

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
