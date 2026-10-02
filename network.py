"""A neural network written from scratch with NumPy."""

import numpy as np

from dataset import load_data, one_hot


def relu(z):
    """Keep positive numbers, turn negative numbers into 0."""
    return np.maximum(0, z)


def softmax(z):
    """Turn each row of scores into probabilities that add up to 1."""
    exponentials = np.exp(z - z.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def cross_entropy(probabilities, targets):
    """Measure how wrong the predictions are. Lower is better, 0 is perfect."""
    correct_class = (probabilities * targets).sum(axis=1)
    return float(-np.log(correct_class + 1e-12).mean())


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

    def backward(self, activations, targets):
        """Backpropagation: find how much each weight and bias contributed to the error."""
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        # Error at the output layer: predicted probabilities minus the correct answers.
        delta = (activations[-1] - targets) / len(targets)
        for i in reversed(range(len(self.weights))):
            weight_gradients[i] = activations[i].T @ delta
            bias_gradients[i] = delta.sum(axis=0)
            if i > 0:
                # Pass the error back one layer. ReLU blocks it where the neuron was off.
                delta = (delta @ self.weights[i].T) * (activations[i] > 0)
        return weight_gradients, bias_gradients

    def train_step(self, x, targets, learning_rate):
        """Learn from one batch of images, and return the loss before the update."""
        activations = self.forward(x)
        weight_gradients, bias_gradients = self.backward(activations, targets)
        for i in range(len(self.weights)):
            # Gradient descent: move each number a small step in the direction that lowers the loss.
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]
        return cross_entropy(activations[-1], targets)

    def predict(self, x):
        """Return the digit the network thinks each image shows."""
        return self.forward(x)[-1].argmax(axis=1)

    def accuracy(self, x, labels):
        """Return the fraction of images the network gets right."""
        return float((self.predict(x) == labels).mean())

    def save(self, path):
        """Save the learned weights and biases to a file."""
        arrays = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        """Rebuild a network from a file made by save()."""
        with np.load(path) as data:
            count = len(data.files) // 2
            weights = [data[f"w{i}"] for i in range(count)]
            biases = [data[f"b{i}"] for i in range(count)]
        network = cls([weights[0].shape[0]] + [w.shape[1] for w in weights])
        network.weights = weights
        network.biases = biases
        return network


def main():
    """Quick check that learning works: train on only 64 images, over and over."""
    x_train, y_train, x_test, y_test = load_data()
    x, labels = x_train[:64], y_train[:64]
    targets = one_hot(labels)
    network = NeuralNetwork()
    for step in range(101):
        loss = network.train_step(x, targets, learning_rate=0.1)
        if step % 20 == 0:
            print(f"Step {step:3d}   loss {loss:.4f}   accuracy on these 64 images {network.accuracy(x, labels):.2%}")


if __name__ == "__main__":
    main()