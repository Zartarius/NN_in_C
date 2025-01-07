#include "matrix.h"

int main() {
    matrix_t a = random_matrix(1000, 1500);
    matrix_t b = transpose(a);
    multiply(a, b);
}