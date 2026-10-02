"""Load MNIST and prepare it for the neural network."""

import numpy as np

from download_data import DATA_PATH


def prepare_images(images):
    """Flatten each 28x28 image into 784 numbers and scale pixels from 0-255 to 0-1."""
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def one_hot(labels, num_classes=10):
    """Turn each label into ten numbers: 1 at the digit's position, 0 everywhere else."""
    encoded = np.zeros((labels.size, num_classes), dtype=np.float32)
    encoded[np.arange(labels.size), labels] = 1
    return encoded


def load_data():
    """Return the training and test sets, ready for the network."""
    with np.load(DATA_PATH) as data:
        x_train = prepare_images(data["x_train"])
        y_train = data["y_train"]
        x_test = prepare_images(data["x_test"])
        y_test = data["y_test"]
    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = load_data()
    print("Training images:", x_train.shape, x_train.dtype)
    print("Test images:    ", x_test.shape, x_test.dtype)
    print("Pixel range:    ", x_train.min(), "to", x_train.max())
    print("First label:    ", y_train[0])
    print("As one-hot:     ", one_hot(y_train[:1])[0])


if __name__ == "__main__":
    main()