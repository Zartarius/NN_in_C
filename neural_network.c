#include "neural_network.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "train/activation.h"

#define TILE_SIZE 8

// Private struct for representing a single NN layer
typedef struct {
    matrix_t weights;
    matrix_t biases;
} layer_t;

size_t tile_size = TILE_SIZE;

static layer_t* layers;
static size_t num_layers = 0; // The number of layers, excluding input layer
static activation_func_t activation = LEAKY_RELU;

void create_network(size_t* layer_info, const size_t size_layer_info) {
    assert(size_layer_info >=
           2);  // Ensure there are at least input and output layers

    num_layers = size_layer_info - 1;
    // Subtract one because we don't need to store the input layer
    layers = (layer_t*) malloc(num_layers *
                              sizeof(layer_t));  // Allocate memory for layers
    assert(layers != NULL);

    for (size_t i = 0; i < num_layers; i++) {
        // Create Biases - 1 column
        layers[i].biases =
            zeroes(1, layer_info[i + 1]);  // layer_info[0] is the input layer
        assert(layers[i].biases.values != NULL);

        // Create weights matrix
        layers[i].weights = zeroes(layer_info[i], layer_info[i + 1]);
        assert(layers[i].weights.values != NULL);

        float stddev = sqrt(
            2.0 / layer_info[i]);  // Standard deviation for the initialization
        for (size_t j = 0; j < layer_info[i]; j++) {
            for (size_t k = 0; k < layer_info[i]; k++) {
                layers[i].weights.values[j * layer_info[i + 1] + k] =
                    ((float)rand() / RAND_MAX) * 2.0 * stddev - stddev;
                // layers[i].biases.values[k] = ((float) rand() / RAND_MAX) * 2
                // * stddev - stddev;
            }
        }
    }
}

static size_t argmax(float* distribution, size_t num_classes) {
    size_t result = 0;
    float max_prob = distribution[0];
    printf("num_classes: %zu\n", num_classes);
    for (size_t i = 0; i < num_classes; i++) {
        if (distribution[i] > max_prob) {
            max_prob = distribution[i];
            result = i;
        }
    }
    return result;
}

static float* softmax_regression(matrix_t input, size_t row_number) {
    float* distribution = malloc(input.n * sizeof(float));
    float sum = 0.0;
    for (size_t i = 0; i < input.n; i++) {
        distribution[i] = exp(input.values[input.n * row_number + i]);
        sum += distribution[i];  // Calculate the sum of the distribution,
                                 // simultaneously
    }
    for (size_t i = 0; i < input.n; i++) {
        distribution[i] /= sum;  // Normalize the distribution
    }
    return distribution;
}

result_t* predict(matrix_t X) {
    matrix_t input = X;
    for (size_t i = 0; i < num_layers; i++) {
        matrix_t output = matrix_tile_multiply(input, layers[i].weights);
        matrix_add_vector(output, layers[i].biases);

        if (i == num_layers - 1) {
            input = output;  // In this case, 'input' is the raw values from the
                             // final layer
        } else {
            input = matrix_activation(output, activation, false);
            free(output.values);
        }
    }
    result_t* predictions = (result_t*)malloc(input.m * sizeof(result_t));

    for (size_t i = 0; i < input.m; i++) {
        predictions[i].distribution = softmax_regression(input, i);
        predictions[i].prediction =
            argmax(predictions[i].distribution, input.n);
    }
    return predictions;
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
