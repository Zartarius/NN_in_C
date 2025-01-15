#include "matrix.h"
#include "neural_network.h"
#include "parse_csv.h"
#include "train/activation.h"

int main() {
    // determine_cache(); // COMMENT THIS OUT IF YOU ARE A MAC USER
    size_t layer_info[] = {784, 128, 128, 10};
    size_t num_layers = sizeof(layer_info) / sizeof(size_t);

    create_network(layer_info, num_layers);
    matrix_t inputs = random_matrix(1, 784);
    result_t *predictions = predict(inputs);

    for (size_t i = 0; i < 20; i++) {
        printf("%zu\n", predictions[i].prediction);
        for (size_t j = 0; j < 10; j++) {
            printf("%f    ", predictions[i].distribution[j]);
        }
        printf("\n");
    }
}
