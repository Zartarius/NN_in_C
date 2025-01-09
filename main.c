#include "matrix.h"
#include "parse_csv.h"
// #include <stdio.h>

int main() {
    // matrix_t* data = read_csv("test.csv", 0, true);
    // print_matrix(data[0]);
    // print_matrix(data[1]);
    matrix_t a = zeroes(10, 10);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            a.values[i][j] = (float) i == j;
        }
    }
    print_matrix(a);

    matrix_t b = random_matrix(10, 10);
    print_matrix(b);
    print_matrix(multiply(a, b));
}
