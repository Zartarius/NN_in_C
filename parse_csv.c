#include "parse_csv.h"

#define BUFFER_SIZE 16384
#define FLOAT_LENGTH 16

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

    float** dataframe = temporary_dataframe(num_rows, num_cols);

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
                dataframe[m][n++] = atof(ascii_float);
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
                X.values[i][j - (size_t) (j > output_column)] = dataframe[i][j];
            }
        }
    }

    matrix_t y = zeroes(num_rows, 1);
    for (size_t i = 0; i < num_rows; i++) {
        y.values[i][0] = dataframe[i][output_column];
    }

    matrix_t* output = malloc(2 * sizeof(matrix_t));
    output[0] = X;
    output[1] = y;
    free_dataframe(dataframe, num_rows);
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

// Creates a temporary dataframe from the csv file
static float** temporary_dataframe(const size_t m, const size_t n) {
    float** dataframe = malloc(m * sizeof(float*));
    assert(dataframe != NULL);

    for (size_t i = 0; i < m; i++) {
        dataframe[i] = malloc(n * sizeof(float));
        assert(dataframe[i] != NULL);
    }
    return dataframe;
}

// Frees the temporary dataframe
static void free_dataframe(float** dataframe, const size_t m) {
    for (size_t i = 0; i < m; i++) {
        free(dataframe[i]);
    }
    free(dataframe);
}
