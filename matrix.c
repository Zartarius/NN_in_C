#include "matrix.h"

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include "include/threads.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>

extern size_t tile_size;

// Structure to pass arguments to the thread function.
// Struct originally designed for use in 1 function,
// but now we are using it in multiple, with some struct
// fields being redundant in some functions.
typedef struct {
    matrix_t *a;
    matrix_t *b;
    matrix_t *c;
    size_t start_row;
    size_t start_col;
} thread_args_t;

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

static THREAD_ENTRY parallel_row_adder(thread_func_param_t arg) {
    thread_args_t args = *(thread_args_t *)arg;
    float* matrix_values = args.a->values;
    float* vector_values = args.b->values;
    float* result_values = args.c->values;
    size_t n = args.a->n;                      // Equivalent to args.b->n
    size_t starting_cell = args.start_row * n; // The row to apply the addition on
    
    /*
    for (size_t i = 0; i < n; i++) {
        result_values[starting_cell + i] = matrix_values[starting_cell + i] + vector_values[i];
    }
    */
    // size_t end_cell = (args.start_row + 1) * n;
    // size_t i = args.start_row * n;
    size_t i = 0;
    for (; i < n - 7; i += 8) {
        size_t offset = starting_cell + i;
        result_values[offset] = matrix_values[offset] + vector_values[i];
        result_values[offset + 1] = matrix_values[offset + 1] + vector_values[i + 1];
        result_values[offset + 2] = matrix_values[offset + 2] + vector_values[i + 2];
        result_values[offset + 3] = matrix_values[offset + 3] + vector_values[i + 3];
        result_values[offset + 4] = matrix_values[offset + 4] + vector_values[i + 4];
        result_values[offset + 5] = matrix_values[offset + 5] + vector_values[i + 5];
        result_values[offset + 6] = matrix_values[offset + 6] + vector_values[i + 6];
        result_values[offset + 7] = matrix_values[offset + 7] + vector_values[i + 7];
    }

    for (; i < n; i++) {
        result_values[starting_cell + i] = matrix_values[starting_cell + i] + vector_values[i];
    }
    
    return (thread_func_return_t)(uintptr_t)NULL;
}

// Add a vector row-wise to a matrix, to each row
matrix_t matrix_add_vector(matrix_t matrix, matrix_t vector) {
    assert((matrix.n == vector.n) && vector.m == 1);

    #ifdef _WIN32
    thread_t *threads =  malloc(matrix.m *sizeof(thread_t));
    thread_args_t *args =  malloc(matrix.m * sizeof(thread_args_t));
    #else
    thread_t threads[matrix.m];
    thread_args_t args[matrix.m];
    #endif

    matrix_t result = zeroes(matrix.m, matrix.n);
    for (size_t i = 0; i < matrix.m; i++) {
        args[i].a = &matrix;
        args[i].b = &vector;
        args[i].c = &result; 
        args[i].start_row = i;
        args[i].start_col = -1; // We don't need this
        
        THREAD_CREATE(threads[i], parallel_row_adder, &args[i]);
    }

    THREAD_JOIN_AND_CLOSE(threads, matrix.m);

    printf("Broke out\n");

    #ifdef _WIN32
    free(threads);
    free(args);
    #endif

    return result;
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
static THREAD_ENTRY compute_tile(thread_func_param_t arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    matrix_t *c = args->c;
    size_t start_row = args->start_row;
    size_t start_col = args->start_col;

    // Compute the product of the tile
    for (size_t i = start_row; i < start_row + tile_size && i < a->m; i++) {
        for (size_t j = start_col; j < start_col + tile_size && j < b->n; j++) {
            __m256 sum_vec = _mm256_setzero_ps(); // Initialize AVX vector for accumulation
            float sum = 0.0f;

            // Process 8 elements at a time using AVX
            size_t k = 0;
            for (; k <= a->n - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(
                    &a->values[i * a->n + k]); // Load 8 elements from matrix A
                __m256 b_vec = _mm256_set_ps(
                    b->values[(k + 7) * b->n + j],
                    b->values[(k + 6) * b->n + j],
                    b->values[(k + 5) * b->n + j],
                    b->values[(k + 4) * b->n + j],
                    b->values[(k + 3) * b->n + j],
                    b->values[(k + 2) * b->n + j],
                    b->values[(k + 1) * b->n + j],
                    b->values[k * b->n + j]); // Load 8 elements from matrix B
                sum_vec = _mm256_fmadd_ps(
                    a_vec, b_vec,
                    sum_vec); //_mm256_fmadd_ps(a_vec, b_vec,
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

            c->values[i * c->n + j] = sum; // Store the result
        }
    }

    return (thread_func_return_t)(uintptr_t)NULL;
}

// Function to multiply two matrices using tiles, threads, and AVX
matrix_t matrix_tile_multiply(matrix_t a, matrix_t b) {
    // printf("%zu %zu |  %zu %zu\n", a.n, a.m, b.n, b.m);
    printf("%zu %zu |  %zu %zu\n", a.n, a.m, b.n, b.m);
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
    #ifdef _WIN32
    thread_t *threads = malloc(num_tiles_row * num_tiles_col *sizeof(thread_t));
    thread_args_t *args = malloc(num_tiles_row * num_tiles_col * sizeof(thread_args_t));
    #else
    thread_t threads[num_tiles_row * num_tiles_col];
    thread_args_t args[num_tiles_row * num_tiles_col];
    #endif

    // Initialize thread arguments and create threads
    for (size_t i = 0; i < num_tiles_row; i++) {
        for (size_t j = 0; j < num_tiles_col; j++) {
            // Create the thread arguments
            args[i * num_tiles_col + j].a = &a;
            args[i * num_tiles_col + j].b = &b;
            args[i * num_tiles_col + j].c = &c;
            args[i * num_tiles_col + j].start_row = i * tile_size;
            args[i * num_tiles_col + j].start_col = j * tile_size;

            THREAD_CREATE(threads[i * num_tiles_col + j],  compute_tile, &args[i * num_tiles_col + j]);
        }
    }

    THREAD_JOIN_AND_CLOSE(threads, num_tiles_row * num_tiles_col);

    #ifdef _WIN32
    free(threads);
    free(args);
    #endif

    return c;
}

float add(float x, float y) {
    return x + y;
}

float subtract(float x, float y) {
    return x - y;
}

float multiply(float x, float y) {
    return x * y;
}

// Takes in either 1 or 2 matrices (pass in b as NULL if only passing in 1 matrix).
// Scalar ignored if 2 matrices passed in
matrix_t matrix_apply(matrix_t* a, matrix_t* b, const float scalar, float (*function)(float, float)) {
    assert(a != NULL);
    matrix_t result = zeroes(a->m, a->n);

    // We want to do a scalar multiplication, or addition of a constant to the whole matrix
    if (b == NULL) {
        for (size_t i = 0; i < a->m; i++) {
            for (size_t j = 0; j < a->n; j++) {
                size_t index = i * a->n + j;
                result.values[index] = function(a->values[index], scalar);
            }
        }
        return result;
    }

    assert((a-> m == b->m) && (a->n == b->n));
    for (size_t i = 0; i < a->m; i++) {
        for (size_t j = 0; j < a->n; j++) {
            size_t index = i * a->n + j;
            result.values[index] = scalar * function(a->values[index], b->values[index]);
        }
    }
    return result;
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

matrix_t scalar_to_one_hot_encoding(size_t index, size_t num_classes) {
    matrix_t result = zeroes(1, num_classes);
    result.values[index] = 1;
    return result;
}

matrix_t extract_vector(matrix_t matrix, size_t index) {
    matrix_t result = zeroes(1, matrix.n);
    for (size_t i = 0; i < matrix.n; i++) {
        result.values[i] = matrix.values[index * matrix.n + i];
    }
    return result;
}
