"""Analyse the trained network: how accurate it is, and where it goes wrong."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Draw charts straight to files, without opening a window.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dataset import load_data
from network import NeuralNetwork
from train import HISTORY_PATH, MODEL_PATH

IMAGES = Path("images")

# One set of colours for every chart, so they look like they belong together.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUES = LinearSegmentedColormap.from_list("blues", [SURFACE, "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def confusion_matrix(labels, predictions):
    """Count every (true digit, predicted digit) pair. Row = true digit, column = prediction."""
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def print_report(matrix):
    """Print overall accuracy, accuracy per digit, and the most common mistakes."""
    print(f"Test accuracy: {matrix.trace() / matrix.sum():.2%} "
          f"({matrix.sum() - matrix.trace()} mistakes out of {matrix.sum()})")

    print("\nAccuracy for each digit:")
    for digit in range(10):
        print(f"  {digit}: {matrix[digit, digit] / matrix[digit].sum():.2%}")

    mistakes = matrix.copy()
    np.fill_diagonal(mistakes, 0)
    print("\nMost common mistakes:")
    for index in np.argsort(mistakes, axis=None)[::-1][:5]:
        true_digit, predicted = divmod(int(index), 10)
        print(f"  a {true_digit} mistaken for a {predicted}: {mistakes[true_digit, predicted]} times")


def plot_training_curves(history):
    """Draw how the loss and the accuracy changed during training."""
    epochs = [entry["epoch"] for entry in history]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    left.plot(epochs, [entry["loss"] for entry in history], color=BLUE, linewidth=2)
    left.set_title("Training loss")
    left.set_ylim(bottom=0)

    series = [("train_accuracy", "Training", BLUE), ("val_accuracy", "Validation", ORANGE)]
    for key, name, colour in series:
        values = [entry[key] * 100 for entry in history]
        right.plot(epochs, values, color=colour, linewidth=2, label=f"{name} (ends at {values[-1]:.1f}%)")
    right.set_title("Accuracy (%)")
    right.legend(frameon=False, loc="lower right")

    for axes in (left, right):
        axes.set_xlabel("Epoch")
        axes.set_xticks([epoch for epoch in epochs if epoch == 1 or epoch % 5 == 0 or len(epochs) <= 10])
        axes.grid(axis="y", color=GRID, linewidth=1)
        axes.set_axisbelow(True)
        axes.spines[["top", "right", "left"]].set_visible(False)
        axes.tick_params(length=0)

    figure.tight_layout()
    figure.savefig(IMAGES / "training_curves.png", dpi=150)
    plt.close(figure)


def plot_confusion_matrix(matrix):
    """Draw the confusion matrix, colouring only the mistakes so they stand out."""
    mistakes = matrix.copy()
    np.fill_diagonal(mistakes, 0)

    figure, axes = plt.subplots(figsize=(7, 7))
    axes.imshow(mistakes, cmap=BLUES, vmin=0)
    for true_digit in range(10):
        for predicted in range(10):
            count = matrix[true_digit, predicted]
            if true_digit == predicted:
                axes.text(predicted, true_digit, count, ha="center", va="center", color=MUTED, fontsize=9)
            elif count > 0:
                colour = "white" if count > mistakes.max() * 0.55 else INK
                axes.text(predicted, true_digit, count, ha="center", va="center", color=colour)

    axes.set_title("Where the network goes wrong")
    axes.set_xlabel("Digit the network predicted")
    axes.set_ylabel("True digit")
    axes.set_xticks(range(10))
    axes.set_yticks(range(10))
    axes.xaxis.tick_top()
    axes.xaxis.set_label_position("top")
    # Thin lines between the cells.
    axes.set_xticks(np.arange(-0.5, 10), minor=True)
    axes.set_yticks(np.arange(-0.5, 10), minor=True)
    axes.grid(which="minor", color=SURFACE, linewidth=2)
    axes.tick_params(which="both", length=0)
    axes.spines[:].set_visible(False)

    figure.text(0.5, 0.02, "Grey numbers on the diagonal are correct answers. Blue cells are mistakes.",
                ha="center", color=MUTED, fontsize=10)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(IMAGES / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def plot_mistakes(x_test, labels, probabilities):
    """Draw the 12 mistakes the network was most confident about."""
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    wrong = np.flatnonzero(predictions != labels)
    worst = wrong[np.argsort(confidence[wrong])[::-1][:12]]

    figure, grid = plt.subplots(2, 6, figsize=(11, 4.8), layout="constrained")
    for axes, index in zip(grid.flat, worst):
        axes.imshow(x_test[index].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        axes.set_title(f"True {labels[index]}, predicted {predictions[index]}",
                       fontsize=10, fontweight="normal", loc="center")
        axes.set_xlabel(f"{confidence[index]:.0%} sure", fontsize=9)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines[:].set_color(GRID)
    figure.suptitle("The network's most confident mistakes", x=0.01, ha="left", fontweight="bold")
    figure.savefig(IMAGES / "mistakes.png", dpi=150)
    plt.close(figure)


def main():
    x_train, y_train, x_test, y_test = load_data()
    network = NeuralNetwork.load(MODEL_PATH)
    probabilities = network.forward(x_test)[-1]
    matrix = confusion_matrix(y_test, probabilities.argmax(axis=1))
    print_report(matrix)

    IMAGES.mkdir(exist_ok=True)
    with open(HISTORY_PATH) as file:
        plot_training_curves(json.load(file))
    plot_confusion_matrix(matrix)
    plot_mistakes(x_test, y_test, probabilities)
    print(f"\nSaved 3 charts to the {IMAGES} folder")


if __name__ == "__main__":
    main()