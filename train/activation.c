#include "activation.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>

#include "../include/threads.h"

extern size_t tile_size;

typedef struct {
    matrix_t *a;
    matrix_t *b;
    size_t start_row;
    size_t start_col;
} thread_args_t;

static thread_func_return_t WINAPI matrix_activation_sigmoid(thread_func_param_t arg);
static thread_func_return_t WINAPI matrix_d_activation_sigmoid(thread_func_param_t  arg);
static thread_func_return_t WINAPI matrix_activation_softsign(thread_func_param_t  arg);
static thread_func_return_t WINAPI matrix_d_activation_softsign(thread_func_param_t  arg);
static thread_func_return_t WINAPI matrix_activation_relu(thread_func_param_t  arg);
static thread_func_return_t WINAPI matrix_d_activation_relu(thread_func_param_t  arg);
static thread_func_return_t WINAPI matrix_activation_tanh(thread_func_param_t  arg);
static thread_func_return_t WINAPI matrix_d_activation_tanh(thread_func_param_t  arg);
static thread_func_return_t WINAPI matrix_activation_leaky_relu(thread_func_param_t  arg);
static thread_func_return_t WINAPI matrix_d_activation_leaky_relu(thread_func_param_t  arg);

static thread_func_return_t WINAPI matrix_activation_sigmoid(thread_func_param_t  arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            b->values[i * a->n + j] =
                1.0 / (1.0 + exp(-a->values[i * a->n + j]));
        }
    }
    return (thread_func_return_t)(uintptr_t)NULL;
}

static thread_func_return_t WINAPI matrix_d_activation_sigmoid(thread_func_param_t  arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float value = exp(-a->values[i * a->n + j]);
            b->values[i * a->n + j] = value / ((1.0 + value) * (1.0 + value));
        }
    }
    return (thread_func_return_t)(uintptr_t)NULL;
}

static thread_func_return_t WINAPI matrix_activation_softsign(thread_func_param_t  arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float val = a->values[i * a->n + j];
            b->values[i * a->n + j] = val / (1.0 + fabs(val));
        }
    }
    return (thread_func_return_t)(uintptr_t)NULL;
}

static thread_func_return_t WINAPI matrix_d_activation_softsign(thread_func_param_t  arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float val = a->values[i * a->n + j];
            b->values[i * a->n + j] = 1 / ((1 + fabs(val)) * (1 + fabs(val)));
        }
    }
    return (thread_func_return_t)(uintptr_t)NULL;
}

static thread_func_return_t WINAPI matrix_activation_relu(thread_func_param_t  arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            b->values[i * a->n + j] = fmax(0.0, a->values[i * a->n + j]);
        }
    }
    return (thread_func_return_t)(uintptr_t)NULL;
}

static thread_func_return_t WINAPI matrix_d_activation_relu(thread_func_param_t  arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            b->values[i * a->n + j] = a->values[i * a->n + j] > 0;
        }
    }
    return (thread_func_return_t)(uintptr_t)NULL;
}

static thread_func_return_t WINAPI matrix_activation_tanh(thread_func_param_t  arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            b->values[i * a->n + j] = tanh(a->values[i * a->n + j]);
        }
    }
    return (thread_func_return_t)(uintptr_t)NULL;
}

static thread_func_return_t WINAPI matrix_d_activation_tanh(thread_func_param_t  arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float val = tanh(a->values[i * a->n + j]);
            b->values[i * a->n + j] = 1 - val * val;
        }
    }
    return (thread_func_return_t)(uintptr_t)NULL;
}

#define LEAKY_RELU_ALPHA 0.01

static thread_func_return_t WINAPI matrix_activation_leaky_relu(thread_func_param_t  arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float val = a->values[i * a->n + j];
            b->values[i * a->n + j] = (val > 0) ? val : LEAKY_RELU_ALPHA * val;
        }
    }
    return (thread_func_return_t)(uintptr_t)NULL;
}

static thread_func_return_t WINAPI matrix_d_activation_leaky_relu(thread_func_param_t  arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float val = a->values[i * a->n + j];
            b->values[i * a->n + j] = val > 0 ? 1 : LEAKY_RELU_ALPHA;
        }
    }
    return (thread_func_return_t)(uintptr_t)NULL;
}

matrix_t matrix_activation(matrix_t a, activation_func_t activation,
                           bool derivative) {
    thread_func_return_t(*activation_function)(thread_func_param_t) = NULL;
    switch (activation) {
        case SIGMOID:
            activation_function = derivative ? matrix_d_activation_sigmoid
                                             : matrix_activation_sigmoid;
            break;
        case SOFTSIGN:
            activation_function = derivative ? matrix_d_activation_softsign
                                             : matrix_activation_softsign;
            break;
        case RELU:
            activation_function =
                derivative ? matrix_d_activation_relu : matrix_activation_relu;
            break;
        case TANH:
            activation_function =
                derivative ? matrix_d_activation_tanh : matrix_activation_tanh;
            break;
        case LEAKY_RELU:
            activation_function = derivative ? matrix_d_activation_leaky_relu
                                             : matrix_activation_leaky_relu;
            break;
    }

    matrix_t b = zeroes(a.m, a.n);

    size_t num_tiles_row = (a.m + tile_size - 1) / tile_size;
    size_t num_tiles_col = (a.n + tile_size - 1) / tile_size;

    size_t num_tiles_row_col = num_tiles_row * num_tiles_col;

    #ifdef _WIN32
    thread_t *threads = malloc(num_tiles_row_col *sizeof(thread_t));
    thread_args_t *args = malloc(num_tiles_row_col * sizeof(thread_args_t));
    #else
    thread_t threads[matrix.m];
    thread_args_t args[matrix.m];
    #endif

    for (size_t i = 0; i < num_tiles_row; i++) {
        for (size_t j = 0; j < num_tiles_col; j++) {
            args[i * num_tiles_col + j].a = &a;
            args[i * num_tiles_col + j].b = &b;
            args[i * num_tiles_col + j].start_row = i * tile_size;
            args[i * num_tiles_col + j].start_col = j * tile_size;
            THREAD_CREATE(threads[i * num_tiles_col + j], activation_function, &args[i * num_tiles_col + j]);
        }
    }

    THREAD_JOIN_AND_CLOSE(threads, num_tiles_row_col);

    #ifdef _WIN32
    free(threads);
    free(args);
    #endif

    return b;
}
