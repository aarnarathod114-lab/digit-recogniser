"""Train the neural network on MNIST and save the result."""

import json
import time

import numpy as np

from dataset import load_data, one_hot
from network import NeuralNetwork

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.1
VALIDATION_SIZE = 5000
MODEL_PATH = "model.npz"
HISTORY_PATH = "history.json"


def main():
    x_train, y_train, x_test, y_test = load_data()

    # Hold back some training images to check progress on,
    # so the test set stays completely unseen until the very end.
    x_val, y_val = x_train[-VALIDATION_SIZE:], y_train[-VALIDATION_SIZE:]
    x_train, y_train = x_train[:-VALIDATION_SIZE], y_train[:-VALIDATION_SIZE]
    targets = one_hot(y_train)

    network = NeuralNetwork()
    rng = np.random.default_rng(0)
    history = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # One epoch = one pass over all training images, in a new random order each time.
        order = rng.permutation(len(x_train))
        losses = []
        for i in range(0, len(order), BATCH_SIZE):
            batch = order[i:i + BATCH_SIZE]
            losses.append(network.train_step(x_train[batch], targets[batch], LEARNING_RATE))

        loss = float(np.mean(losses))
        train_accuracy = network.accuracy(x_train, y_train)
        val_accuracy = network.accuracy(x_val, y_val)
        history.append({
            "epoch": epoch,
            "loss": loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })
        print(f"Epoch {epoch:2d}/{EPOCHS}   loss {loss:.4f}   "
              f"training accuracy {train_accuracy:.2%}   validation accuracy {val_accuracy:.2%}")

    print(f"Training took {time.time() - start:.0f} seconds")
    print(f"Final accuracy on the 10,000 test images: {network.accuracy(x_test, y_test):.2%}")

    network.save(MODEL_PATH)
    with open(HISTORY_PATH, "w") as file:
        json.dump(history, file, indent=2)
    print(f"Saved the trained network to {MODEL_PATH}")


if __name__ == "__main__":
    main()