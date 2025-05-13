/**
 * tools.h
 * Utility functions for activation and initialization used in the neural network
 */

#ifndef TOOLS_H
#define TOOLS_H

#include <cmath>
#include <cstdlib>

// Sigmoid activation function
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of sigmoid, using output value
inline double sigmoidDerivative(double y) {
    return y * (1.0 - y); // y = sigmoid(x)
}

// Generates a random weight in the range [-1, 1]
inline double randomWeight() {
    return ((double)rand() / RAND_MAX) * 2 - 1; // range [-1, 1]
}

#endif