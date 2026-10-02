"""Compare different training settings, adding one improvement at a time.

Every experiment is judged on the validation set. The test set is never used here,
so the final test accuracy reported by train.py stays an honest measurement.
"""

import json
import time

from train import load_splits, train

EPOCHS = 30
RESULTS_PATH = "experiments.json"

# Each experiment keeps the settings of the one before it and changes one thing.
EXPERIMENTS = [
    ("Baseline: one hidden layer of 128",
     {"hidden_layers": (128,), "learning_rate": 0.1}),
    ("+ momentum",
     {"hidden_layers": (128,), "learning_rate": 0.05, "momentum": 0.9}),
    ("+ learning-rate decay",
     {"hidden_layers": (128,), "learning_rate": 0.05, "momentum": 0.9, "decay": 0.95}),
    ("+ larger network (256, 128)",
     {"hidden_layers": (256, 128), "learning_rate": 0.05, "momentum": 0.9, "decay": 0.95}),
    ("+ shifted images",
     {"hidden_layers": (256, 128), "learning_rate": 0.05, "momentum": 0.9, "decay": 0.95, "max_shift": 2}),
    ("Even larger network (512, 256)",
     {"hidden_layers": (512, 256), "learning_rate": 0.05, "momentum": 0.9, "decay": 0.95, "max_shift": 2}),
]


def main():
    training, validation, _ = load_splits()
    results = []

    print(f"Running {len(EXPERIMENTS)} experiments of {EPOCHS} epochs each. This takes several minutes.\n")
    print(f"{'Experiment':<36}{'Training':>10}{'Validation':>12}{'Seconds':>9}")
    for name, settings in EXPERIMENTS:
        start = time.time()
        network, history = train(training, validation, epochs=EPOCHS, verbose=False, **settings)
        seconds = time.time() - start
        final = history[-1]
        results.append({
            "name": name,
            "settings": settings,
            "train_accuracy": final["train_accuracy"],
            "val_accuracy": final["val_accuracy"],
            "seconds": round(seconds),
        })
        print(f"{name:<36}{final['train_accuracy']:>10.2%}{final['val_accuracy']:>12.2%}{seconds:>9.0f}")

    with open(RESULTS_PATH, "w") as file:
        json.dump(results, file, indent=2)
    print(f"\nSaved the results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()