#include "matrix.h"

int main() {
    matrix_t a = random_matrix(1000, 1500);
    matrix_t b = transpose(a);
    print_matrix(a);
    print_matrix(b);
    print_matrix(multiply(a, b));
}