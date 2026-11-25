"""
Builds all the tables and figures used in the report, starting from the files
saved by train_nn.py and feature_extraction.py (no training here).

Outputs in results/:
  metrics.csv, results_table.tex     - main results table
  class_counts.csv                   - number of images per breed in each split
  top_confusions.csv                 - most frequent confusions for each approach
  figures/confusion_<approach>.png   - confusion matrices with class names
  figures/training_curves.png        - loss / accuracy curves of the two CNN approaches
  figures/examples_full_finetune.png - correct and wrong test predictions
  figures/confusion_report.png       - two confusion matrices side by side (for the PDF)
  per_class_f1.csv                   - F1-score of every breed for each approach
  significance.csv                   - McNemar test between pairs of approaches

Run:  python make_report.py
"""

import os
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score

import common as C

APPROACHES = [
    ("full_finetune", "Full fine-tuning"),
    ("head_only", "Classifier head only"),
    ("features_logreg", "Features + logistic regression"),
    ("features_linear_svm", "Features + linear SVM"),
]
FIG_DIR = os.path.join(C.RESULTS_DIR, "figures")
IMG_DIR = os.path.join(C.DATA_DIR, "oxford-iiit-pet", "images")


def read_predictions(name):
    with open(os.path.join(C.RESULTS_DIR, f"{name}_test_predictions.csv")) as f:
        rows = list(csv.DictReader(f))
    names = [r["image"] for r in rows]
    y_true = np.array([int(r["true"]) for r in rows])
    y_pred = np.array([int(r["pred"]) for r in rows])
    return names, y_true, y_pred


def results_table(summaries):
    rows = []
    for name, label in APPROACHES:
        s = summaries[name]
        rows.append([label, s["trainable_params"], s["train_time_s"], s["train_acc"],
                     s["val_acc"], s["test_acc"], s["test_macro_f1"]])

    with open(os.path.join(C.RESULTS_DIR, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["approach", "trainable_params", "train_time_s", "train_acc",
                    "val_acc", "test_acc", "test_macro_f1"])
        for r in rows:
            w.writerow([r[0], r[1], r[2]] + [round(v, 4) for v in r[3:]])

    # LaTeX version, ready to paste in the report
    with open(os.path.join(C.RESULTS_DIR, "results_table.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrcccc}\n\\toprule\n")
        f.write("Approach & Trainable params & Time (s) & Train acc & Val acc & Test acc & Test macro F1 \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r[0]} & {r[1]:,} & {r[2]:.0f} & {r[3]:.3f} & {r[4]:.3f} & {r[5]:.3f} & {r[6]:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("saved metrics.csv and results_table.tex")


def class_counts(classes):
    counts = {k: np.zeros(C.NUM_CLASSES, dtype=int) for k in ["train", "val", "test"]}
    for split in ["train", "val"]:
        with open(os.path.join(C.SPLIT_DIR, f"{split}.csv")) as f:
            for r in csv.DictReader(f):
                counts[split][int(r["label"])] += 1
    test_labels = C.get_labels(C.load_pets("test", None))
    for l in test_labels:
        counts["test"][l] += 1

    with open(os.path.join(C.RESULTS_DIR, "class_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["breed", "train", "val", "test"])
        for i, c in enumerate(classes):
            w.writerow([c, counts["train"][i], counts["val"][i], counts["test"][i]])
        w.writerow(["TOTAL", counts["train"].sum(), counts["val"].sum(), counts["test"].sum()])
    for k, v in counts.items():
        print(f"{k}: {v.sum()} images, min {v.min()} / max {v.max()} per breed")


def plot_confusion(name, label, classes, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(C.NUM_CLASSES))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)    # rows = true class, sum to 1
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C.NUM_CLASSES))
    ax.set_yticks(range(C.NUM_CLASSES))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted breed")
    ax.set_ylabel("True breed")
    ax.set_title(f"{label} - confusion matrix on the test set (row-normalized)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"confusion_{name}.png"), dpi=150)
    plt.close(fig)
    return cm


def top_confusions(all_cms, classes, k=10):
    with open(os.path.join(C.RESULTS_DIR, "top_confusions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["approach", "true", "predicted", "count", "share_of_true_class"])
        for name, cm in all_cms.items():
            off = cm.copy()
            np.fill_diagonal(off, 0)
            order = np.argsort(off, axis=None)[::-1][:k]
            for flat in order:
                i, j = np.unravel_index(flat, off.shape)
                w.writerow([name, classes[i], classes[j], off[i, j],
                            round(off[i, j] / cm[i].sum(), 3)])


def plot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (name, label), color in zip(APPROACHES[:2], ["tab:blue", "tab:red"]):
        path = os.path.join(C.RESULTS_DIR, f"{name}_history.csv")
        with open(path) as f:
            h = list(csv.DictReader(f))
        ep = [int(r["epoch"]) for r in h]
        axes[0].plot(ep, [float(r["train_loss"]) for r in h], "--", color=color, label=f"{label} train")
        axes[0].plot(ep, [float(r["val_loss"]) for r in h], "-", color=color, label=f"{label} val")
        axes[1].plot(ep, [float(r["train_acc_aug"]) for r in h], "--", color=color, label=f"{label} train")
        axes[1].plot(ep, [float(r["val_acc"]) for r in h], "-", color=color, label=f"{label} val")
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].set_ylabel("accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)


def plot_examples(name, classes, n=6):
    """First n correct and first n wrong test predictions (random but fixed order)."""
    names, y_true, y_pred = read_predictions(name)
    rng = np.random.default_rng(C.SEED)
    order = rng.permutation(len(names))
    correct = [i for i in order if y_true[i] == y_pred[i]][:n]
    wrong = [i for i in order if y_true[i] != y_pred[i]][:n]

    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 6))
    for row, ids, title in [(0, correct, "correct"), (1, wrong, "wrong")]:
        for col, i in enumerate(ids):
            ax = axes[row, col]
            img = Image.open(os.path.join(IMG_DIR, names[i])).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            color = "green" if row == 0 else "red"
            ax.set_title(f"true: {classes[y_true[i]]}\npred: {classes[y_pred[i]]}",
                         fontsize=8, color=color)
        axes[row, 0].set_ylabel(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"examples_{name}.png"), dpi=120)
    plt.close(fig)


def plot_confusion_pair(names, classes):
    """Two row-normalized confusion matrices next to each other, for the report."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.3))
    for ax, (name, label) in zip(axes, names):
        _, y_true, y_pred = read_predictions(name)
        cm = confusion_matrix(y_true, y_pred, labels=range(C.NUM_CLASSES))
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(C.NUM_CLASSES))
        ax.set_yticks(range(C.NUM_CLASSES))
        ax.set_xticklabels(classes, rotation=90, fontsize=6.5)
        ax.set_yticklabels(classes, fontsize=6.5)
        ax.set_xlabel("Predicted breed")
        ax.set_ylabel("True breed")
        ax.set_title(label)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_report.png"), dpi=200)
    plt.close(fig)


def per_class_f1(classes):
    scores = {}
    for name, _ in APPROACHES:
        _, y_true, y_pred = read_predictions(name)
        scores[name] = f1_score(y_true, y_pred, average=None, labels=range(C.NUM_CLASSES))
    with open(os.path.join(C.RESULTS_DIR, "per_class_f1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["breed"] + [n for n, _ in APPROACHES])
        for i, c in enumerate(classes):
            w.writerow([c] + [round(scores[n][i], 3) for n, _ in APPROACHES])


def mcnemar_tests():
    """
    Exact McNemar test on the test set: are two approaches really different,
    or is the difference in accuracy compatible with chance?
    n01 = images only the first approach gets right, n10 = only the second.
    """
    preds = {name: read_predictions(name) for name, _ in APPROACHES}
    with open(os.path.join(C.RESULTS_DIR, "significance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["approach_a", "approach_b", "only_a_correct", "only_b_correct", "p_value"])
        for i in range(len(APPROACHES)):
            for j in range(i + 1, len(APPROACHES)):
                a, b = APPROACHES[i][0], APPROACHES[j][0]
                _, t, pa = preds[a]
                _, _, pb = preds[b]
                n01 = int(((pa == t) & (pb != t)).sum())
                n10 = int(((pa != t) & (pb == t)).sum())
                p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
                w.writerow([a, b, n01, n10, f"{p:.2g}"])


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    classes = C.load_pets("test", None).classes

    summaries = {}
    for name, _ in APPROACHES:
        with open(os.path.join(C.RESULTS_DIR, f"{name}_summary.json")) as f:
            summaries[name] = json.load(f)

    results_table(summaries)
    class_counts(classes)

    all_cms = {}
    for name, label in APPROACHES:
        _, y_true, y_pred = read_predictions(name)
        all_cms[name] = plot_confusion(name, label, classes, y_true, y_pred)
    top_confusions(all_cms, classes)
    plot_curves()
    plot_examples("full_finetune", classes)
    plot_confusion_pair([APPROACHES[1], APPROACHES[2]], classes)
    per_class_f1(classes)
    mcnemar_tests()
    print("figures saved in", FIG_DIR)


if __name__ == "__main__":
    main()