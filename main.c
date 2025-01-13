#include "matrix.h"
#include "neural_network.h"
#include "activation.h"
#include "parse_csv.h"
// #include <stdio.h>

int main() {
    // matrix_t* data = read_csv("test.csv", 0, true);
    // print_matrix(data[0]);
    // print_matrix(data[1]);
    determine_cache();
    matrix_t a = random_matrix(1000, 1000);
    print_matrix(a);

    matrix_t b = random_matrix(1000, 1000);
    print_matrix(b);
    matrix_t c = matrix_tile_multiply(a, b);
    print_matrix(c);
    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    printf("test\n");
}
