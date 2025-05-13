/**
 * neuralNetwork.cpp
 * Implementation of a feedforward neural network with backpropagation
 */

#include "neuralNetwork.h"
#include "tools.h"
#include <cstdlib>
#include <ctime>
#include <cmath>

// Constructor to initialize weights and structure
neuralNetwork::neuralNetwork(int input, int hidden, int output, double lr)
    : inputSize(input), hiddenSize(hidden), outputSize(output), learningRate(lr) {
    srand((unsigned)time(nullptr));
    weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
    weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
    for (auto& row : weightsInputHidden)
        for (auto& w : row)
            w = randomWeight();
    for (auto& row : weightsHiddenOutput)
        for (auto& w : row)
            w = randomWeight();
    hiddenLayer.resize(hiddenSize);
    outputLayer.resize(outputSize);
}

// Forward pass
std::vector<double> neuralNetwork::feedForward(const std::vector<double>& input) {
    // Hidden layer activation
    for (int h = 0; h < hiddenSize; ++h) {
        double sum = 0.0;
        for (int i = 0; i < inputSize; ++i)
            sum += input[i] * weightsInputHidden[i][h];
        hiddenLayer[h] = sigmoid(sum);
    }
    // Output layer activation
    for (int o = 0; o < outputSize; ++o) {
        double sum = 0.0;
        for (int h = 0; h < hiddenSize; ++h)
            sum += hiddenLayer[h] * weightsHiddenOutput[h][o];
        outputLayer[o] = sigmoid(sum);
    }
    return outputLayer;
}

// Backpropagation training
void neuralNetwork::train(const std::vector<double>& input, const std::vector<double>& target) {
    feedForward(input);

    // Output layer error & delta
    std::vector<double> outputErrors(outputSize);
    std::vector<double> outputDeltas(outputSize);
    for (int o = 0; o < outputSize; ++o) {
        outputErrors[o] = target[o] - outputLayer[o];
        outputDeltas[o] = outputErrors[o] * sigmoidDerivative(outputLayer[o]);
    }
    // Hidden layer error & delta
    std::vector<double> hiddenErrors(hiddenSize);
    std::vector<double> hiddenDeltas(hiddenSize);
    for (int h = 0; h < hiddenSize; ++h) {
        double error = 0.0;
        for (int o = 0; o < outputSize; ++o)
            error += outputDeltas[o] * weightsHiddenOutput[h][o];
        hiddenErrors[h] = error;
        hiddenDeltas[h] = hiddenErrors[h] * sigmoidDerivative(hiddenLayer[h]);
    }
    // Update weights hidden → output
    for (int h = 0; h < hiddenSize; ++h)
        for (int o = 0; o < outputSize; ++o)
            weightsHiddenOutput[h][o] += learningRate * outputDeltas[o] * hiddenLayer[h];
    // Update weights input → hidden
    for (int i = 0; i < inputSize; ++i)
        for (int h = 0; h < hiddenSize; ++h)
            weightsInputHidden[i][h] += learningRate * hiddenDeltas[h] * input[i];
}