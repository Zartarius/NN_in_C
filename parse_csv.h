#ifndef PARSE_CSV_H
#define PARSE_CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include "matrix.h"

matrix_t* read_csv(char* const filename, const char delimiter, size_t output_column, bool is_header);

#endif
