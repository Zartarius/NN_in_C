#include "activation.h"
#include "matrix.h"
#include "neural_network.h"
#include "parse_csv.h"
// #include <stdio.h>

int main() {
    // print_matrix(data[1]);
    determine_cache();
    size_t layer_info[] = {16, 250, 150, 2};
    size_t num_layer = sizeof(layer_info) / sizeof(size_t);
    create_network(layer_info, num_layers);
    matrix_t inputs = random_matrix(1, 16);
    predict(inputs);
}
