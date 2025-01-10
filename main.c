#include "matrix.h"
#include "parse_csv.h"
// #include <stdio.h>

int main() {
    // matrix_t* data = read_csv("test.csv", 0, true);
    // print_matrix(data[0]);
    // print_matrix(data[1]);
    matrix_t a = zeroes(5, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            a.values[i][j] = (float)i == j;
        }
    }
    print_matrix(a);

    matrix_t b = random_matrix(5, 5);
    print_matrix(b);
    matrix_t c = multiply(a, b);
    print_matrix(c);
    free_matrix(c);
    free_matrix(b);
    free_matrix(a);
    printf("test\n");
}
