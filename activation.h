#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"
#include "neural_network.h"

matrix_t matrix_activation(matrix_t a, activation_func_t activation);

#endif