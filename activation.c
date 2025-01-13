#include "activation.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <assert.h>

extern size_t tile_size;

typedef struct {
    matrix_t *a;
    matrix_t *b;
    size_t start_row;
    size_t start_col;
} thread_args_t;

static void *matrix_activation_sigmoid(void *arg);
static void *matrix_activation_softsign(void *arg);
static void *matrix_activation_step(void *arg);
static void *matrix_activation_relu(void *arg);
static void *matrix_activation_tanh(void *arg);
static void *matrix_activation_leaky_relu(void *arg);

static void *matrix_activation_sigmoid(void *arg) {
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
    return NULL;
}

static void *matrix_activation_softsign(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            double val = a->values[i * a->n + j];
            b->values[i * a->n + j] = val / (1.0 + fabs(val));
        }
    }
    return NULL;
}

static void *matrix_activation_step(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            b->values[i * a->n + j] =
                (a->values[i * a->n + j] >= 0) ? 1.0 : 0.0;
        }
    }
    return NULL;
}

static void *matrix_activation_relu(void *arg) {
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
    return NULL;
}

static void *matrix_activation_tanh(void *arg) {
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
    return NULL;
}

#define LEAKY_RELU_ALPHA 0.01

static void *matrix_activation_leaky_relu(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            double val = a->values[i * a->n + j];
            b->values[i * a->n + j] = (val > 0) ? val : LEAKY_RELU_ALPHA * val;
        }
    }
    return NULL;
}

matrix_t matrix_activation(matrix_t a, activation_func_t activation) {
    void *(*activation_function)(void *) = NULL;
    switch (activation) {
        case STEP:
            activation_function = matrix_activation_step;
            break;
        case SIGMOID:
            activation_function = matrix_activation_sigmoid;
            break;
        case SOFTSIGN:
            activation_function = matrix_activation_softsign;
            break;
        case RELU:
            activation_function = matrix_activation_relu;
            break;
        case TANH:
            activation_function = matrix_activation_tanh;
            break;
        case LEAKY_RELU:
            activation_function = matrix_activation_leaky_relu;
            break;
    }

    matrix_t b = zeroes(a.m, a.n);

    size_t num_tiles_row = (a.m + tile_size - 1) / tile_size;
    size_t num_tiles_col = (a.n + tile_size - 1) / tile_size;

    pthread_t threads[num_tiles_row * num_tiles_col];
    thread_args_t args[num_tiles_row * num_tiles_col];

    for (size_t i = 0; i < num_tiles_row; i++) {
        for (size_t j = 0; j < num_tiles_col; j++) {
            args[i * num_tiles_col + j].a = &a;
            args[i * num_tiles_col + j].b = &b;
            args[i * num_tiles_col + j].start_row = i * tile_size;
            args[i * num_tiles_col + j].start_col = j * tile_size;
            if (pthread_create(&threads[i * num_tiles_col + j], NULL,
                               activation_function,
                               &args[i * num_tiles_col + j]) != 0) {
                perror("pthread_create");
                exit(1);
            }
        }
    }

    for (size_t i = 0; i < num_tiles_row * num_tiles_col; i++) {
        assert(pthread_join(threads[i], NULL) == 0);
    }

    return b;
}