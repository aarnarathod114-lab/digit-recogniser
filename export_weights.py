"""Export the trained network to one small file that the web page in docs/ can load."""

from pathlib import Path

import numpy as np

from network import NeuralNetwork
from train import MODEL_PATH

OUTPUT_PATH = Path("docs") / "weights.bin"


def main():
    network = NeuralNetwork.load(MODEL_PATH)
    layer_sizes = [network.weights[0].shape[0]] + [w.shape[1] for w in network.weights]

    # The file is one long list of numbers: how many layer sizes there are, the sizes
    # themselves, then the weights and biases of each layer in order.
    parts = [np.array([len(layer_sizes), *layer_sizes])]
    for weights, biases in zip(network.weights, network.biases):
        parts.append(weights.ravel())
        parts.append(biases)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    np.concatenate(parts).astype("<f4").tofile(OUTPUT_PATH)
    print(f"Layer sizes: {layer_sizes}")
    print(f"Saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()