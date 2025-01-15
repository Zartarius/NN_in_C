#include "train/activation.h"
#include "matrix.h"
#include "neural_network.h"
#include "parse_csv.h"

int main() {
    // determine_cache(); // COMMENT THIS OUT IF YOU ARE A MAC USER
    size_t num_parameters = 784;
    size_t num_samples = 20;
    size_t num_classes = 10;
    
    size_t layer_info[] = {num_parameters, 256, 128, num_classes};
    size_t num_layers = sizeof(layer_info) / sizeof(size_t);
    
    create_network(layer_info, num_layers);
    matrix_t inputs = random_matrix(num_samples, num_parameters);
    result_t* predictions = predict(inputs);

    for (size_t i = 0; i < num_samples; i++) {
        printf("%zu\n", predictions[i].prediction);
        for (size_t j = 0; j < num_classes; j++) {
            printf("%f    ", predictions[i].distribution[j]);
        }
        printf("\n");
    }

    return 0;
}
