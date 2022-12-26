import numpy as np

class MLP:
    
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.inputs] + num_hidden + [num_outputs]

        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

    def forward_propagate(self, inputs):

        activations = inputs

        for w in self.weights:

            net_inputs = np.dot(activations, w) # Matrix multiplcation 

            activations = self._sigmoid(net_inputs) # Each new layer undergoes a sigmoid linearization, even hidden layers
        
        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    mlp = MLP()

    inputs = np.random.rand(mlp.inputs)

    outputs = mlp.forward_propagate(inputs)

    print("The network input is: {}".format(inputs))
    print("The network output is: {}".format(outputs))
