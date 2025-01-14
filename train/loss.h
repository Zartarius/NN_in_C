#ifndef LOSS_H
#define LOSS_H

#include <stdbool.h>

#include "../matrix.h"

typedef enum {
    MSE,
    MAE,
    HUBLER,
    LOG,          // Binary Cross Entropy
    CATEGORICAL,  // Categorical Cross Entropy
} loss_func_t;

float matrix_loss(matrix_t Y, matrix_t actual, loss_func_t loss);

matrix_t matrix_d_loss(matrix_t Y, matrix_t actual, loss_func_t loss,
                       bool uses_softmax);

#endif