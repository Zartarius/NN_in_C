#include "matrix.h"
#include "neural_network.h"
#include "parse_csv.h"
#include "train/activation.h"

int main() {
    determine_cache(); 
    size_t num_parameters = 784;
    size_t num_samples = 10000;
    size_t num_classes = 10;
    
    size_t layer_info[] = {num_parameters, 256, 128, num_classes};
    size_t num_layers = sizeof(layer_info) / sizeof(size_t);


    create_network(layer_info, num_layers);

    printf("Created network\n");

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
