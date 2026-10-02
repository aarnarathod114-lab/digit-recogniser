"""A neural network written from scratch with NumPy."""

import numpy as np

from dataset import load_data


def relu(z):
    """Keep positive numbers, turn negative numbers into 0."""
    return np.maximum(0, z)


def softmax(z):
    """Turn each row of scores into probabilities that add up to 1."""
    exponentials = np.exp(z - z.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class NeuralNetwork:
    """Layers of neurons, where every neuron connects to every neuron in the next layer."""

    def __init__(self, layer_sizes=(784, 128, 10), seed=0):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for inputs, outputs in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Start with small random weights (He initialisation) and zero biases.
            scale = np.sqrt(2 / inputs)
            self.weights.append(rng.normal(0, scale, (inputs, outputs)).astype(np.float32))
            self.biases.append(np.zeros(outputs, dtype=np.float32))

    def forward(self, x):
        """Pass images through every layer and return the output of each layer."""
        activations = [x]
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            activations.append(softmax(z) if i == last else relu(z))
        return activations

    def predict(self, x):
        """Return the digit the network thinks each image shows."""
        return self.forward(x)[-1].argmax(axis=1)

    def accuracy(self, x, labels):
        """Return the fraction of images the network gets right."""
        return float((self.predict(x) == labels).mean())


def main():
    x_train, y_train, x_test, y_test = load_data()
    network = NeuralNetwork()
    print("Weight shapes:", [w.shape for w in network.weights])
    print("Output for the first image:")
    print(network.forward(x_test[:1])[-1][0].round(3))
    print(f"Accuracy before any training: {network.accuracy(x_test, y_test):.2%}")


if __name__ == "__main__":
    main()