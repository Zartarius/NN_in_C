#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "matrix.h"

typedef enum {
    STEP,
    SIGMOID,
    SOFTSIGN,
    RELU,
    TANH,
    LEAKY_RELU
} activation_func_t;

typedef struct {
    float* distribution;
    size_t prediction;
} result_t;

void init_layers(size_t* layer_info, const size_t size_layer_info);
// result_t predict()

#endif