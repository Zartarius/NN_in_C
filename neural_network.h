#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "matrix.h"

typedef struct {
    matrix_t distribution;
    size_t *prediction;
} result_t;

void create_network(size_t* layer_info, const size_t size_layer_info);
result_t predict(matrix_t X);
#endif
