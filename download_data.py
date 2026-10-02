"""Download the MNIST handwritten-digit dataset into the data/ folder."""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
DATA_PATH = Path("data") / "mnist.npz"


def download():
    """Download the dataset once, and check the file is not corrupted."""
    if DATA_PATH.exists():
        print(f"Already downloaded: {DATA_PATH}")
    else:
        DATA_PATH.parent.mkdir(exist_ok=True)
        print("Downloading MNIST (about 11 MB)...")
        urllib.request.urlretrieve(URL, DATA_PATH)
        print(f"Saved to {DATA_PATH}")

    file_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if file_hash != SHA256:
        raise RuntimeError("The file is corrupted. Delete the data folder and run again.")


def main():
    download()
    with np.load(DATA_PATH) as data:
        print("Training images:", data["x_train"].shape)
        print("Training labels:", data["y_train"].shape)
        print("Test images:    ", data["x_test"].shape)
        print("Test labels:    ", data["y_test"].shape)


if __name__ == "__main__":
    main()