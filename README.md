# Overview
Implementation of a basic, but optimised Neural Network in C from scratch. The goal is to accurately classify handwritten digits from the MNIST dataset.

(Project currently in progress)

# Files
`main.c` - Main file for testing, and implementing the neural network later on.

`matrix.h` - A C library meant to substitute as a simpler numpy library. Utilises multithreading and SIMD, non-portable implementation. 

`parse_csv.h` - A C library used to convert a csv data file into a useable `matrix_t` format, similar to pandas dataframes in python.

`data/mnist_train|test_data.csv` - Csv files for training and testing the mnist dataset. These datasets are smaller than the full mnist dataset, due to github file size restrictions.

`neural_network.h` - Provides the actual interface for the neural netowrk, allowing the user to pass in the testing and training data, and customising the number of layers, neurons, activation function etc. 
