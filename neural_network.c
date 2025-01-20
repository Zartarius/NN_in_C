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

static layer_t *layers;
static size_t num_layers = 0; // The number of layers, excluding input layer
static activation_func_t activation = LEAKY_RELU;

void create_network(size_t *layer_info, const size_t size_layer_info) {
    assert(size_layer_info >= 2); // Ensure there are at least input and output layers

    num_layers = size_layer_info - 1;
    // Subtract one because we don't need to store the input layer
    layers = (layer_t *)malloc(num_layers * sizeof(layer_t)); // Allocate memory for layers
    
    assert(layers != NULL);

    for (size_t i = 0; i < num_layers; i++) {
        // Create Biases - 1 column
        layers[i].biases = zeroes(1, layer_info[i + 1]);  
        assert(layers[i].biases.values != NULL);

        // Create weights matrix
        layers[i].weights = zeroes(layer_info[i], layer_info[i + 1]);
        assert(layers[i].weights.values != NULL);
 
        float stddev = sqrt(2.0 / layer_info[i]);  // Standard deviation for the initialization
        size_t num_rows = layers[i].weights.m;
        size_t num_cols = layers[i].weights.n;

        for (size_t j = 0; j < num_rows; j++) {
            for (size_t k = 0; k < num_cols; k++) {
                layers[i].weights.values[j * num_cols + k] = 
                    ((float)rand() / RAND_MAX) * 2.0 * stddev - stddev;
                // layers[i].biases.values[k] = ((float) rand() / RAND_MAX) * 2
                // * stddev - stddev;
            }
        }
        /*
        for (size_t j = 0; j < layer_info[i]; j++) {
            for (size_t k = 0; k < layer_info[i + 1]; k++) {
                layers[i].weights.values[j * layer_info[i + 1] + k] =
                    ((float)rand() / RAND_MAX) * 2.0 * stddev - stddev;
                // layers[i].biases.values[k] = ((float) rand() / RAND_MAX) * 2
                // * stddev - stddev;
            }
        }
        */
    }
}

static size_t argmax(float *distribution, size_t num_classes) {
    size_t result = 0;
    float max_prob = distribution[0];

    for (size_t i = 0; i < num_classes; i++) {
        if (distribution[i] > max_prob) {
            max_prob = distribution[i];
            result = i;
        }
    }
    return result;
}

static float *softmax_regression(matrix_t input, size_t row_number) {
    float *distribution = malloc(input.n * sizeof(float));
    float sum = 0.0;
    for (size_t i = 0; i < input.n; i++) {
        distribution[i] = exp(input.values[input.n * row_number + i]);
        sum += distribution[i]; // Calculate the sum of the distribution,
                                // simultaneously
    }
    for (size_t i = 0; i < input.n; i++) {
        distribution[i] /= sum; // Normalize the distribution
    }
    return distribution;
}

result_t *predict(matrix_t X) {
    matrix_t input = X;
    for (size_t i = 0; i < num_layers; i++) {
        printf("Layer: %zu\n", i);
        matrix_t output = matrix_tile_multiply(input, layers[i].weights);
        matrix_add_vector(output, layers[i].biases);
        printf("Success on %zu\n", i);

        if (i == num_layers - 1) {
            input = output; // In this case, 'input' is the raw values from the
                            // final layer
        } else {
            input = matrix_activation(output, activation, false);
            free(output.values);
        }
        printf("Success on activation %zu\n", i);
    }
    result_t *predictions = (result_t *)malloc(input.m * sizeof(result_t));

    for (size_t i = 0; i < input.m; i++) {
        predictions[i].distribution = softmax_regression(input, i);
        predictions[i].prediction = argmax(predictions[i].distribution, input.n);
    }
    return predictions;
}

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__) 
#include <sys/sysctl.h>
#endif

void determine_cache(void) {
    size_t cache_size = 0;

    #ifdef _WIN32
    DWORD bufferSize = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;

    // Get the required buffer size
    if (GetLogicalProcessorInformation(NULL, &bufferSize) != FALSE || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        perror("GetLogicalProcessorInformation");
        exit(1);
    }

    // Allocate memory for the processor information
    buffer = malloc(bufferSize);
    if (buffer == NULL) {
        perror("malloc");
        exit(1);
    }

    // Retrieve the processor information
    if (GetLogicalProcessorInformation(buffer, &bufferSize) == FALSE) {
        perror("GetLogicalProcessorInformation");
        free(buffer);
        exit(1);
    }

    // Parse the processor information to find the L2 cache size
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
    for (DWORD i = 0; i < bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); i++) {
        if (ptr->Relationship == RelationCache) {
            CACHE_DESCRIPTOR cache = ptr->Cache;
            if (cache.Level == 2) { // L2 Cache
                cache_size = cache.Size;
                break;
            }
        }
        ptr++;
    }

    free(buffer);

    if (cache_size == 0) {
        fprintf(stderr, "Failed to retrieve L2 cache size.\n");
        exit(1);
    }

    // Check if the system is macOS
    #elif __APPLE__
        // Use sysctl to get the cache size on macOS
        size_t len = sizeof(cache_size);
        if (sysctlbyname("hw.l3cachesize", &cache_size, &len, NULL, 0) != 0) {
            perror("sysctlbyname");
            exit(1);
        }

    // Check if the system is Linux
    #elif __linux__
        FILE *fp = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
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

    // Unsupported system
    #else
        fprintf(stderr, "Cache size determination is not supported on this operating system.\n");
        return;
    #endif

    // Calculate tile size based on cache size
    tile_size = (int)sqrt((cache_size / sizeof(float)) / 3);
}
