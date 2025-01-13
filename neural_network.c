#include "neural_network.h"
#include "activation.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 8

// Private struct for representing a single NN layer
typedef struct {
    matrix_t weights;  // .n == number of neurons in layer
    matrix_t biases;
} layer_t;

size_t tile_size = TILE_SIZE;

static layer_t* layers;
static size_t num_layers = 0;
static activation_func_t activation = 0;

void create_network(size_t* layer_info, const size_t size_layer_info) {
    assert(size_layer_info >=
           2);  // Ensure there are at least input and output layers

    num_layers = size_layer_info;
    layers = (layer_t*)malloc(num_layers *
                              sizeof(layer_t));  // Allocate memory for layers

    for (size_t i = 1; i < num_layers; i++) {
        // Initialize biases to zero

        size_t prev_layer_neurons = layer_info[i - 1];
        // Create weights matrix
        layers[i].weights = zeroes(layer_info[i - 1], layer_info[i]);
        // Create Biases - 1 column
        layers[i].biases = zeroes(1, layer_info[i]);

        double stddev = sqrt(
            2.0 /
            prev_layer_neurons);  // Standard deviation for the initialization
        for (size_t j = 0; j < prev_layer_neurons; j++) {
            for (size_t k = 0; k < layer_info[i]; k++) {
                layers[i].weights.values[j * layer_info[i] + k] =
                    ((float)rand() / RAND_MAX) * 2 * stddev - stddev;
                layers[i].biases.values[k] =
                    ((float)rand() / RAND_MAX) * 2 * stddev - stddev;
            }
        }
    }
}

result_t predict(matrix_t X) {
    matrix_t input = X;

    for (size_t i = 1; i < num_layers; i++) {
        matrix_t output = matrix_tile_multiply(input, layers[i].weights);
        matrix_t bias = layers[i].biases;
        for (size_t j = 0; j < output.n; j++) {
            output.values[j] =
                activation_func(output.values[i] + bias.values[0]);
        }
    }
}

void determine_cache(void) {
    size_t cache_size = 0;
    FILE* fp = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
    if (fp != NULL) {
        char buffer[16];
        if (fgets(buffer, sizeof(buffer), fp)) {
            cache_size = strtoul(buffer, NULL, 10) * 1024;
        }
        fclose(fp);
    } else {
        perror("fopen");
        exit(1);
    }
    tile_size = (int)sqrt((cache_size / sizeof(float)) / 3);
}
