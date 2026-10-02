"""Tests that check the neural network and the data preparation are correct.

Run them with:  python -m pytest
"""

import numpy as np

from dataset import one_hot, prepare_images
from network import NeuralNetwork, cross_entropy, softmax


def make_batch(samples=5, inputs=20, classes=4, seed=0):
    """Make a small batch of random inputs and one-hot targets."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(samples, inputs))
    targets = one_hot(rng.integers(0, classes, samples), classes).astype(np.float64)
    return x, targets


def make_float64_network(layer_sizes, seed=1):
    """Make a network that uses high-precision numbers, for exact maths checks."""
    network = NeuralNetwork(layer_sizes, seed=seed)
    rng = np.random.default_rng(seed)
    network.weights = [w.astype(np.float64) for w in network.weights]
    network.biases = [rng.normal(0, 0.1, b.shape) for b in network.biases]
    return network


def test_softmax_rows_add_up_to_one():
    scores = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
    assert np.allclose(softmax(scores).sum(axis=1), 1)


def test_softmax_copes_with_huge_numbers():
    probabilities = softmax(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.all(np.isfinite(probabilities))


def test_one_hot_puts_a_one_at_the_label():
    encoded = one_hot(np.array([0, 3, 9]))
    assert encoded.shape == (3, 10)
    assert encoded[1].tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


def test_prepare_images_flattens_and_scales():
    images = np.full((2, 28, 28), 255, dtype=np.uint8)
    prepared = prepare_images(images)
    assert prepared.shape == (2, 784)
    assert prepared.max() == 1.0


def test_forward_gives_one_probability_per_class():
    network = NeuralNetwork((784, 128, 10))
    output = network.forward(np.zeros((3, 784), dtype=np.float32))[-1]
    assert output.shape == (3, 10)
    assert np.allclose(output.sum(axis=1), 1)


def test_backpropagation_matches_numerical_gradients():
    """The most important test: check every gradient from backward() against a slow but simple estimate.

    For each weight, nudge it up and down by a tiny amount and measure how the loss changes.
    If backpropagation is correct, the two answers agree.
    """
    network = make_float64_network((20, 7, 6, 4))
    x, targets = make_batch()
    weight_gradients, bias_gradients = network.backward(network.forward(x), targets)

    nudge = 1e-6
    pairs = list(zip(network.weights, weight_gradients)) + list(zip(network.biases, bias_gradients))
    for values, gradients in pairs:
        for index in np.ndindex(values.shape):
            original = values[index]
            values[index] = original + nudge
            loss_up = cross_entropy(network.forward(x)[-1], targets)
            values[index] = original - nudge
            loss_down = cross_entropy(network.forward(x)[-1], targets)
            values[index] = original
            estimate = (loss_up - loss_down) / (2 * nudge)
            assert abs(estimate - gradients[index]) < 1e-6


def test_training_makes_the_loss_smaller():
    network = make_float64_network((20, 16, 4))
    x, targets = make_batch(samples=32)
    first_loss = network.train_step(x, targets, learning_rate=0.1)
    for _ in range(200):
        last_loss = network.train_step(x, targets, learning_rate=0.1)
    assert last_loss < first_loss / 10


def test_save_then_load_gives_the_same_network(tmp_path):
    network = NeuralNetwork((784, 32, 10), seed=3)
    path = tmp_path / "model.npz"
    network.save(path)
    loaded = NeuralNetwork.load(path)

    x = np.random.default_rng(0).random((4, 784), dtype=np.float32)
    assert np.array_equal(network.predict(x), loaded.predict(x))
    assert np.allclose(network.forward(x)[-1], loaded.forward(x)[-1])