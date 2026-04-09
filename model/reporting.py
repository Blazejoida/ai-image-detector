import pandas as pd
import matplotlib.pyplot as plt
import os
from config import RESULTS_CSV, PLOT_PATH


def save_results(acc_baseline, f1_baseline, acc_resnet, f1_resnet):
    df = pd.DataFrame({
        "Model":    ["Baseline CNN", "ResNet50 (fine-tuned)"],
        "Accuracy": [round(acc_baseline * 100, 2), round(acc_resnet * 100, 2)],
        "F1-score": [round(f1_baseline, 4),        round(f1_resnet, 4)],
    })
    df.to_csv(RESULTS_CSV, index=False)
    print(df.to_string(index=False))
    return df


def save_plots(history_baseline, history_resnet):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history_baseline["epoch"], history_baseline["loss"],
                 label="Baseline CNN", marker="o")
    axes[0].plot(history_resnet["epoch"],   history_resnet["loss"],
                 label="ResNet50",     marker="s")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history_baseline["epoch"],
                 [a * 100 for a in history_baseline["accuracy"]],
                 label="Baseline CNN", marker="o")
    axes[1].plot(history_resnet["epoch"],
                 [a * 100 for a in history_resnet["accuracy"]],
                 label="ResNet50",     marker="s")
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()
    print(f"Saved plot to {PLOT_PATH}")
