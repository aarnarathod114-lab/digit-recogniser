"""A web page where you draw a digit and the neural network recognises it."""

import gradio as gr
import numpy as np
from PIL import Image

from network import NeuralNetwork
from train import MODEL_PATH

network = NeuralNetwork.load(MODEL_PATH)


def to_ink(drawing):
    """Turn the sketchpad's picture into a grid of numbers: 0 = blank paper, 1 = ink."""
    pixels = np.asarray(drawing, dtype=np.float32) / 255
    if pixels.ndim == 2:
        return 1 - pixels
    darkness = 1 - pixels[..., :3].mean(axis=2)
    if pixels.shape[2] == 4:
        # The fourth channel is transparency: a see-through pixel has no ink.
        return darkness * pixels[..., 3]
    return darkness


def preprocess(ink):
    """Make a drawing look like an MNIST image: 28x28, with the digit sized and centred the same way."""
    rows = np.flatnonzero(ink.max(axis=1) > 0.1)
    columns = np.flatnonzero(ink.max(axis=0) > 0.1)
    if rows.size == 0:
        return None

    # Cut out just the digit, then shrink it so its longer side is 20 pixels.
    digit = ink[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
    scale = 20 / max(digit.shape)
    height = max(1, round(digit.shape[0] * scale))
    width = max(1, round(digit.shape[1] * scale))
    picture = Image.fromarray((digit * 255).astype(np.uint8))
    small = np.asarray(picture.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32) / 255

    # Place it on a 28x28 grid so that its centre of mass sits in the middle.
    ys, xs = np.indices(small.shape)
    top = round(14 - float((ys * small).sum() / small.sum()))
    left = round(14 - float((xs * small).sum() / small.sum()))
    top = min(max(top, 0), 28 - height)
    left = min(max(left, 0), 28 - width)
    image = np.zeros((28, 28), dtype=np.float32)
    image[top:top + height, left:left + width] = small
    return image


def predict(drawing):
    """Return the network's top guesses, and the 28x28 image it actually looked at."""
    if drawing is None or drawing["composite"] is None:
        return None, None
    image = preprocess(to_ink(drawing["composite"]))
    if image is None:
        return None, None

    probabilities = network.forward(image.reshape(1, 784))[-1][0]
    guesses = {str(digit): float(probabilities[digit]) for digit in range(10)}
    # Enlarge the 28x28 image ten times so it is easy to see (dark ink on white).
    seen = np.kron(255 - image * 255, np.ones((10, 10))).astype(np.uint8)
    return guesses, seen


demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(
        label="Draw a digit from 0 to 9",
        canvas_size=(400, 400),
        brush=gr.Brush(default_size=15, colors=["#000000"], color_mode="fixed"),
        layers=False,
        transforms=(),
    ),
    outputs=[
        gr.Label(num_top_classes=3, label="The network's guess"),
        gr.Image(label="What the network sees (28 x 28 pixels)", height=280, width=280),
    ],
    live=True,
    flagging_mode="never",
    title="Handwritten Digit Recogniser",
    description="A neural network written from scratch with NumPy, with no machine learning libraries. "
                "It was trained on 55,000 handwritten digits and gets 99.03% of unseen test digits right.",
)

if __name__ == "__main__":
    demo.launch()