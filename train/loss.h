#ifndef LOSS_H
#define LOSS_H

#include "../matrix.h"

typedef enum {
    MSE,
    MAE,
    HUBLER,
    LOG, // Binary Cross Entropy
    CATEGORICAL, // Categorical Cross Entropy
} loss_func_t;

#endif