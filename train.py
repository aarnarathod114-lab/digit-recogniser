"""Train the neural network on MNIST and save the result."""

import json
import time

import numpy as np

from dataset import load_data, one_hot
from network import NeuralNetwork

VALIDATION_SIZE = 5000
BATCH_SIZE = 64
MODEL_PATH = "model.npz"
HISTORY_PATH = "history.json"

# The best settings found by experiments.py.
BEST_SETTINGS = {
    "hidden_layers": (256, 128),
    "epochs": 30,
    "learning_rate": 0.05,
    "momentum": 0.9,
    "decay": 0.95,
    "max_shift": 2,
}


def load_splits():
    """Load MNIST and hold back the last 5,000 training images as a validation set.

    The validation set is for comparing settings, so the test set stays unseen until the very end.
    """
    x_train, y_train, x_test, y_test = load_data()
    training = (x_train[:-VALIDATION_SIZE], y_train[:-VALIDATION_SIZE])
    validation = (x_train[-VALIDATION_SIZE:], y_train[-VALIDATION_SIZE:])
    return training, validation, (x_test, y_test)


def shift_images(images, rng, max_shift):
    """Slide a batch of images a few pixels in a random direction.

    A digit moved slightly is still the same digit, so this gives the network
    extra examples for free and stops it from memorising exact pixel positions.
    """
    down, right = rng.integers(-max_shift, max_shift + 1, size=2)
    grid = images.reshape(-1, 28, 28)
    return np.roll(grid, (down, right), axis=(1, 2)).reshape(len(images), 784)


def train(training, validation, hidden_layers=(128,), epochs=10, learning_rate=0.1,
          momentum=0.0, decay=1.0, max_shift=0, verbose=True):
    """Train a new network and return it, along with its results after every epoch."""
    x_train, y_train = training
    x_val, y_val = validation
    targets = one_hot(y_train)

    network = NeuralNetwork((784, *hidden_layers, 10))
    rng = np.random.default_rng(0)
    history = []

    for epoch in range(1, epochs + 1):
        # One epoch = one pass over all training images, in a new random order each time.
        order = rng.permutation(len(x_train))
        losses = []
        for i in range(0, len(order), BATCH_SIZE):
            batch = order[i:i + BATCH_SIZE]
            images = x_train[batch]
            if max_shift > 0:
                images = shift_images(images, rng, max_shift)
            losses.append(network.train_step(images, targets[batch], learning_rate, momentum))

        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_accuracy": network.accuracy(x_train, y_train),
            "val_accuracy": network.accuracy(x_val, y_val),
        })
        if verbose:
            latest = history[-1]
            print(f"Epoch {epoch:2d}/{epochs}   loss {latest['loss']:.4f}   "
                  f"training accuracy {latest['train_accuracy']:.2%}   "
                  f"validation accuracy {latest['val_accuracy']:.2%}")

        # Take smaller steps as training goes on, to settle into a good answer.
        learning_rate *= decay

    return network, history


def main():
    training, validation, (x_test, y_test) = load_splits()

    start = time.time()
    network, history = train(training, validation, **BEST_SETTINGS)
    print(f"Training took {time.time() - start:.0f} seconds")
    print(f"Final accuracy on the 10,000 test images: {network.accuracy(x_test, y_test):.2%}")

    network.save(MODEL_PATH)
    with open(HISTORY_PATH, "w") as file:
        json.dump(history, file, indent=2)
    print(f"Saved the trained network to {MODEL_PATH}")


if __name__ == "__main__":
    main()