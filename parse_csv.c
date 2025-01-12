#include "parse_csv.h"

#define BUFFER_SIZE 16384
#define FLOAT_LENGTH 16

static size_t count_cols(FILE* data, const char delimiter);
static size_t count_rows(FILE* data);

// Converts csv file into useable matrices
// Returns an array of 2 matrix_t structs, X and y
// Parameters:
// filename: the name of the file to read
// output_column: the index of the column containing the dependent variable
// is_header: true if the file contains a header row, false otherwise
matrix_t* read_csv(char* const filename, const char delimiter, size_t output_column, bool is_header) {
    FILE* data = fopen(filename, "a+");
    assert(data != NULL);
    // File must end in newline for function to work correctly
    assert(fputc('\n', data) != EOF);
    rewind(data);
    size_t num_cols = count_cols(data, delimiter);
    size_t num_rows = count_rows(data) - (size_t) is_header;

    float* dataframe = malloc(num_rows * num_cols * sizeof(float));

    // Skip to the second line if there is a header
    char buffer[BUFFER_SIZE];
    if (is_header) {
        fgets(buffer, BUFFER_SIZE, data);
    }

    for (size_t m = 0; m < num_rows; m++) {
        size_t n = 0;
        fgets(buffer, BUFFER_SIZE, data);
        char ascii_float[FLOAT_LENGTH];
        size_t index = 0;
        for (char* letter = (char*) buffer; *letter != '\0'; letter++) {
            if (*letter == delimiter || *letter == '\n') {
                ascii_float[index] = '\0';
                dataframe[m * num_cols + n] = atof(ascii_float);
                n++;
                index = 0;
                continue;
            }
            ascii_float[index++] = *letter;
        }
    }
    fclose(data);

    matrix_t X = zeroes(num_rows, num_cols - 1);
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            if (j != output_column) {
                X.values[i * (num_cols - 1) + (j - (size_t) (j > output_column))] = dataframe[i * num_cols + j];
            }
        }
    }

    matrix_t y = zeroes(num_rows, 1);
    for (size_t i = 0; i < num_rows; i++) {
        y.values[i] = dataframe[i * num_cols + output_column];
    }
    free(dataframe);

    matrix_t* output = malloc(2 * sizeof(matrix_t));
    output[0] = X;
    output[1] = y;
    return output;
}

// Returns the number of columns in the csv file
static size_t count_cols(FILE* data, const char delimiter) {
    size_t count = 1;
    char buffer[BUFFER_SIZE];
    fgets(buffer, BUFFER_SIZE, data);

    for (char* letter = (char*) buffer; *letter != '\0'; letter++) {
        count += (size_t) (*letter == delimiter);
    }
    rewind(data);
    return count;
}

// Returns the number of rows in the csv file
static size_t count_rows(FILE* data) {
    size_t count = 0;
    char buffer[BUFFER_SIZE];

    while (fgets(buffer, BUFFER_SIZE, data) != NULL) {
        if (buffer[0] == '\n') {
            rewind(data);
            return count;
        }
        count++;
    }
    rewind(data);
    return count;
}
