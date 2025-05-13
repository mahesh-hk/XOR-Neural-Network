/**
 * main.cpp
 * Demonstrates a simple feedforward neural network trained to solve XOR
 * The network is trained using backpropagation and uses only STL structures
 */

#include "neuralNetwork.h"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "=== NeuralNet ===\n";
    std::cout << "Simple Neural Network for XOR\n\n";

    // Initialize network: 2 inputs, 4 hidden neurons, 1 output, learning rate 0.1
    neuralNetwork net(2, 4, 1, 0.1);

    // XOR input/output pairs
    std::vector<std::vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    std::vector<std::vector<double>> outputs = {
        {0},
        {1},
        {1},
        {0}
    };

    std::cout << "Training XOR...\n\n";
    for (int epoch = 0; epoch < 20000; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i)
            net.train(inputs[i], outputs[i]);
        // Display progress every 5000 epochs (25% of training)
        if ((epoch + 1) % 5000 == 0)
            std::cout << "Epoch " << (epoch + 1) << " complete\n";
    }

    // Display final results
    std::cout << "\nResults after training:\n";
    std::cout << std::fixed << std::setprecision(4);
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input = inputs[i];
        auto result = net.feedForward(input);
        double prediction = result[0];
        int binary = prediction > 0.5 ? 1 : 0;
        int expected = static_cast<int>(outputs[i][0]);
        std::cout << input[0] << " XOR " << input[1]
                  << " = " << prediction
                  << " (Expected: " << expected << ", Predicted: " << binary << ")\n";
        if (binary == expected)
            correct++;
    }
    std::cout << "\nAccuracy: " << (correct / 4.0) * 100 << "%\n";
    std::cout << "\n";
    system("pause");
    return 0;
}