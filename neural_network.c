#include "neural_network.h"
#include <assert.h>

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Private struct for representing a single NN layer
typedef struct {
    matrix_t weights; // .n == number of neurons in layer
    matrix_t biases;
} layer_t;

static layer_t* layers;
static size_t num_layers = 0;

#define TILE_SIZE 8

size_t tile_size = TILE_SIZE;

void init_layers(size_t* layer_info, const size_t size_layer_info) {
    assert(size_layer_info >= 2); // Ensure there are at least input and output layers

    num_layers = size_layer_info;
    layers = (layer_t*) malloc(num_layers * sizeof(layer_t)); // Allocate memory for layers

    for (size_t i = 0; i < num_layers; i++) {
        // Initialize biases to zero
        layers[i].biases = zeroes(1, layer_info[i]); // Create a 1 x neurons_in_layer matrix for biases
        

        if (i == 0) { // Skip the input layer
            continue;
        }

        // size_t prev_layer_neurons = layer_info[i - 1]; 
        layers[i].weights = random_matrix(layer_info[i - 1], layer_info[i]); // Create weights matrix

        /*
        double stddev = sqrt(2.0 / prev_layer_neurons); // Standard deviation for He initialization
        for (size_t j = 0; j < prev_layer_neurons; j++) {
            for (size_t k = 0; k < layer_info[i]; k++) {
                layers[i].weights->data[j][k] = ((double)rand() / RAND_MAX) * 2 * stddev - stddev; // Random weights
            }
        }
        */
    }
}

void determine_cache(void) {
    size_t cache_size = 0;
    FILE *fp = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
    if (fp) {
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
