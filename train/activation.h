#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../matrix.h"

typedef enum {
    STEP,
    SIGMOID,
    SOFTSIGN,
    RELU,
    TANH,
    LEAKY_RELU
} activation_func_t;

matrix_t matrix_activation(matrix_t a, activation_func_t activation);

#endif
