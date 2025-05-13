/**
 * neuralNetwork.h
 * Header file for a basic feedforward neural network class
 * Supports one hidden layer and backpropagation
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>

class neuralNetwork {
private:
    int inputSize, hiddenSize, outputSize;
    double learningRate;
    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> hiddenLayer;
    std::vector<double> outputLayer;

public:
    neuralNetwork(int input, int hidden, int output, double lr);                      // Constructor to initialize network sizes and weights
    std::vector<double> feedForward(const std::vector<double>& input);                // Performs a forward pass through the network
    void train(const std::vector<double>& input, const std::vector<double>& target);  // Trains the network using backpropagation
};

#endif