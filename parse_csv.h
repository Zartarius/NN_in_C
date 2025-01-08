#ifndef PARSE_CSV_H
#define PARSE_CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include "matrix.h"

matrix_t* read_csv(char* const filename, size_t output_column, bool is_header);
static size_t count_cols(FILE* data);
static size_t count_rows(FILE* data);
static float** temporary_dataframe(const size_t m, const size_t n);
static void free_dataframe(float** dataframe, const size_t m);

#endif