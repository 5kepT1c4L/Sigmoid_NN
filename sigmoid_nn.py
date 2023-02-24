import numpy as np
from random import random

class MLP:
    
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.inputs] + num_hidden + [num_outputs]

        self.weights = [] # List containing all the weights in the 4 layer neural network (3)
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

        self.activations = [] # List containing all the activations (4)
        for i in range(len(layers)): 
            a = np.zeros(layers[i])
            self.activations.append(a)

        self.derivatives = [] # List containing all the derivatives of next layer's error with respect to the corresponding weight matrix
        for i in range(len(layers)-1):
            a = np.zeros((layers[i], layers[i+1]))
            self.derivatives.append(a)

    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):

            net_inputs = np.dot(activations, w) # Matrix multiplcation 

            activations = self._sigmoid(net_inputs) # Each new layer undergoes a sigmoid linearization, even hidden layers
        
            self.activations[i+1] = activations
        
        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def back_propagate(self, error, verbose=False):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]

            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i] # Represents the a_1 (Current Activation)
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped) 
            
            error = np.dot(delta, self.weights[i].T)

            if verbose: 
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error
        
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            print("Original W{} {}".format(i, weights))
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
            print("Updated W{} {}".format(i, weights))

    def train(self, inputs, targets, epochs, learning_rate):
        
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):

                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output)**2)


if __name__ == "__main__":
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])
    mlp = MLP(2, [5], 1)

    mlp.train(inputs, targets, 50, 0.1)


