#include "loss.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

extern size_t tile_size;

typedef struct {
    matrix_t *a;
    matrix_t *b;
    matrix_t *c;
    size_t start_row;
    size_t start_col;
} thread_args_t;

static void *matrix_loss_mse(void *arg);
static void *matrix_loss_mae(void *arg);
static void *matrix_loss_hubler(void *arg);
static void *matrix_loss_log(void *arg);
static void *matrix_loss_categorical(void *arg);

static void *matrix_loss_mse(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    float *sum = calloc(1, sizeof(float));
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            float error = (actual - predict) * (actual - predict);
            *sum += error;
        }
    }
    return sum;
}

static void *matrix_d_loss_mse(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    matrix_t *c = args->c;
    size_t n = a->m * a->n;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            c->values[i * c->n + j] = (2.0 / n) * (predict - actual);
        }
    }
    return NULL;
}

static void *matrix_loss_mae(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    float *sum = calloc(1, sizeof(float));
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            *sum += fabs(actual - predict);
        }
    }
    return sum;
}

static void *matrix_d_loss_mae(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    matrix_t *c = args->c;
    size_t n = a->m * a->n;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            c->values[i * c->n + j] = ((actual - predict) > 0   ? 1
                                       : (actual - predict) < 0 ? -1
                                                                : 0) /
                                      (float)n;
        }
    }
    return NULL;
}

#define HUBLER_THRESHOLD 0.01

static void *matrix_loss_hubler(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    float *sum = calloc(1, sizeof(float));
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            *sum += (fabs(actual - predict) > HUBLER_THRESHOLD)
                        ? (actual - predict) * (actual - predict) / 2.0
                        : HUBLER_THRESHOLD * fabs(actual - predict) -
                              HUBLER_THRESHOLD * HUBLER_THRESHOLD / 2.0;
        }
    }
    return sum;
}

static void *matrix_d_loss_hubler(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    matrix_t *c = args->c;
    size_t n = a->m * a->n;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            c->values[i * c->n + j] =
                ((fabs(actual - predict) > HUBLER_THRESHOLD)
                     ? -(actual - predict)
                     : -HUBLER_THRESHOLD * ((actual - predict) > 0   ? 1
                                            : (actual - predict) < 0 ? -1
                                                                     : 0)) /
                (float)n;
        }
    }
    return NULL;
}

static void *matrix_loss_log(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    float *sum = calloc(1, sizeof(float));
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            *sum += actual * log(predict) + (1 - actual) * log(1 - predict);
        }
    }
    return sum;
}

static void *matrix_d_loss_log(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    matrix_t *c = args->c;
    size_t n = a->m * a->n;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            c->values[i * c->n + j] =
                ((predict - actual) / (predict * (1 - predict))) / (float)n;
        }
    }
    return NULL;
}

static void *matrix_loss_categorical(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    float *sum = calloc(1, sizeof(float));
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            *sum += actual * log(predict);
        }
    }
    return sum;
}

static void *matrix_d_loss_categorical(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    matrix_t *c = args->c;
    size_t n = a->m * a->n;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            c->values[i * c->n + j] = (actual / predict) / n;
        }
    }
    return NULL;
}

static void *matrix_d_loss_categorical_softmax(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    matrix_t *a = args->a;
    matrix_t *b = args->b;
    matrix_t *c = args->c;
    size_t n = a->m * a->n;
    for (size_t i = args->start_row;
         i < args->start_row + tile_size && i < a->m; i++) {
        for (size_t j = args->start_col;
             j < args->start_col + tile_size && j < a->n; j++) {
            float actual = a->values[i * a->n + j];
            float predict = b->values[i * b->n + j];
            c->values[i * c->n + j] = (predict - actual) / n;
        }
    }
    return NULL;
}

float matrix_loss(matrix_t Y, matrix_t actual, loss_func_t loss) {
    void *(*loss_function)(void *) = NULL;
    switch (loss) {
        case MSE:
            loss_function = matrix_loss_mse;
            break;
        case MAE:
            loss_function = matrix_loss_mae;
            break;
        case HUBLER:
            loss_function = matrix_loss_hubler;
            break;
        case LOG:
            loss_function = matrix_loss_log;
            break;
        case CATEGORICAL:
            loss_function = matrix_loss_categorical;
            break;
    }

    assert(Y.m == actual.m && Y.n == actual.n);

    size_t num_tiles_row = (Y.m + tile_size - 1) / tile_size;
    size_t num_tiles_col = (Y.n + tile_size - 1) / tile_size;

    size_t num_tiles_row_col = num_tiles_row * num_tiles_col;

    pthread_t threads[num_tiles_row_col];
    thread_args_t args[num_tiles_row_col];

    for (size_t i = 0; i < num_tiles_row; i++) {
        for (size_t j = 0; j < num_tiles_col; j++) {
            args[i * num_tiles_col + j].a = &Y;
            args[i * num_tiles_col + j].b = &actual;
            args[i * num_tiles_col + j].c = NULL;
            args[i * num_tiles_col + j].start_row = i * tile_size;
            args[i * num_tiles_col + j].start_col = j * tile_size;
            if (pthread_create(&threads[i * num_tiles_col + j], NULL,
                               loss_function,
                               &args[i * num_tiles_col + j]) != 0) {
                perror("pthread_create");
                exit(1);
            }
        }
    }
    int *result;
    float sum = 0;

    for (size_t i = 0; i < num_tiles_row_col; i++) {
        assert(pthread_join(threads[i], (void **)&result) == 0);
        sum += *result;
        free(result);
    }

    return sum / (Y.m * Y.n);
}

matrix_t matrix_d_loss(matrix_t Y, matrix_t actual, loss_func_t loss,
                       bool uses_softmax) {
    void *(*loss_d_function)(void *) = NULL;
    switch (loss) {
        case MSE:
            loss_d_function = matrix_d_loss_mae;
            break;
        case MAE:
            loss_d_function = matrix_d_loss_mse;
            break;
        case HUBLER:
            loss_d_function = matrix_d_loss_hubler;
            break;
        case LOG:
            loss_d_function = matrix_d_loss_log;
            break;
        case CATEGORICAL:
            loss_d_function = uses_softmax ? matrix_d_loss_categorical_softmax
                                           : matrix_d_loss_categorical;
            break;
    }

    assert(Y.m == actual.m && Y.n == actual.n);

    matrix_t c = zeroes(Y.m, Y.n);

    size_t num_tiles_row = (Y.m + tile_size - 1) / tile_size;
    size_t num_tiles_col = (Y.n + tile_size - 1) / tile_size;

    size_t num_tiles_row_col = num_tiles_row * num_tiles_col;

    pthread_t threads[num_tiles_row_col];
    thread_args_t args[num_tiles_row_col];

    for (size_t i = 0; i < num_tiles_row; i++) {
        for (size_t j = 0; j < num_tiles_col; j++) {
            args[i * num_tiles_col + j].a = &Y;
            args[i * num_tiles_col + j].b = &actual;
            args[i * num_tiles_col + j].c = &c;
            args[i * num_tiles_col + j].start_row = i * tile_size;
            args[i * num_tiles_col + j].start_col = j * tile_size;
            if (pthread_create(&threads[i * num_tiles_col + j], NULL,
                               loss_d_function,
                               &args[i * num_tiles_col + j]) != 0) {
                perror("pthread_create");
                exit(1);
            }
        }
    }
    return c;
}