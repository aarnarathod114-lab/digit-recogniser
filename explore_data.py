"""Print one MNIST digit in the terminal to see what the data looks like."""

import sys

import numpy as np

from download_data import DATA_PATH

SHADES = " .:-=+*#%@"


def show_digit(image):
    """Draw a 28x28 image with text: the brighter the pixel, the denser the character."""
    for row in image:
        print("".join(SHADES[int(pixel) * len(SHADES) // 256] * 2 for pixel in row))


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    with np.load(DATA_PATH) as data:
        image = data["x_train"][index]
        label = data["y_train"][index]
    show_digit(image)
    print(f"Image number {index} is labelled: {label}")


if __name__ == "__main__":
    main()