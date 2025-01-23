#include "matrix.h"
#include "neural_network.h"
#include "parse_csv.h"
#include "train/activation.h"

int main() {
    /*
    determine_cache();
    size_t num_parameters = 784;
    size_t num_samples = 10;
    size_t num_classes = 10;

    size_t layer_info[] = {num_parameters, 256, 128, num_classes};
    size_t num_layers = sizeof(layer_info) / sizeof(size_t);

    create_network(layer_info, num_layers);

    printf("Created network\n");

    matrix_t inputs = random_matrix(num_samples, num_parameters);
    result_t results = predict(inputs);

    for (size_t i = 0; i < num_samples; i++) {
        printf("%zu\n", results.prediction[i]);
        for (size_t j = 0; j < num_classes; j++) {
            printf("%f    ", results.distribution.values[i * results.distribution.n + j]);
        }
        printf("\n");
    } */

    printf("Trying to train\n");

    matrix_t *csv_info = read_csv("./data/mnist_train_data.csv", ',', 0, true);
    matrix_t csv_inputs = csv_info[0];
    matrix_t csv_outputs = csv_info[1];
    
    // !bug: < 8 causes segfaults
    size_t layer_info[] = {
        csv_inputs.n,
        10,
        20,
        10
    };

    create_network(layer_info, sizeof(layer_info) / sizeof(size_t));

    matrix_t inputs = extract_vector(csv_inputs, 0);
    matrix_t outputs = scalar_to_one_hot_encoding(csv_outputs.values[0], 10);

    printf("DIMS: %zu %zu\n", inputs.m, inputs.n);

    train(inputs, outputs);

    result_t results = predict(inputs);

    for (size_t i = 0; i < inputs.m; i++) {
        printf("%zu\n", results.prediction[i]);
        for (size_t j = 0; j < 10; j++) {
            printf("%f    ", results.distribution.values[i * results.distribution.n + j]);
        }
        printf("\n");
    }

    free(csv_info);

    return 0;
}
