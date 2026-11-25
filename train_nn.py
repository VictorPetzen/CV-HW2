"""
Approaches 1 and 2: training MobileNetV3-Small with PyTorch.

  --mode full : full fine-tuning, the whole network is trained
  --mode head : classifier-head training, model.features is frozen and only the
                two Linear layers of model.classifier are trained

Model selection: the checkpoint (epoch) with the best validation accuracy is kept.
The test set is used only once, at the end, on the selected checkpoint.

Run:  python train_nn.py --mode full
      python train_nn.py --mode head
"""

import os
import csv
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

import common as C

# default hyperparameters (fixed before running, not tuned on the test set)
DEFAULT_LR = {"full": 3e-4, "head": 1e-3}
EPOCHS = 15
WEIGHT_DECAY = 1e-4


def train_one_epoch(model, loader, optimizer, loss_fn, mode):
    model.train()
    if mode == "head":
        # the frozen part stays in eval mode, so its BatchNorm statistics
        # are not changed by the new dataset
        model.features.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(C.DEVICE), y.to(C.DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "head"], required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.mode]
    name = "full_finetune" if args.mode == "full" else "head_only"

    C.set_seed()
    sets, classes = C.get_datasets()
    train_loader = C.get_loader(sets["train"], shuffle=True)
    val_loader = C.get_loader(sets["val"], shuffle=False)

    model = C.build_model()
    if args.mode == "head":
        for p in model.features.parameters():
            p.requires_grad = False
    model = model.to(C.DEVICE)
    n_trainable = C.count_trainable(model)
    print(f"{name}: {n_trainable:,} trainable parameters, device {C.DEVICE}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(C.CHECKPOINT_DIR, f"{name}.pt")
    history = []
    best_val_acc, best_epoch = -1.0, -1

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, args.mode)
        y_val, p_val, val_loss = C.predict(model, val_loader)
        val_acc = accuracy_score(y_val, p_val)
        scheduler.step()
        history.append([epoch, tr_loss, tr_acc, val_loss, val_acc])
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {val_loss:.3f} acc {val_acc:.3f}")
        # model selection on the validation set only
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt_path)
    train_time = time.time() - start
    print(f"training time {train_time:.1f} s, best epoch {best_epoch} (val acc {best_val_acc:.3f})")

    with open(os.path.join(C.RESULTS_DIR, f"{name}_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc_aug", "val_loss", "val_acc"])
        w.writerows(history)

    # ---- final evaluation of the selected checkpoint ----
    model.load_state_dict(torch.load(ckpt_path, map_location=C.DEVICE))
    y_tr, p_tr, _ = C.predict(model, C.get_loader(sets["train_eval"], shuffle=False))
    y_te, p_te, _ = C.predict(model, C.get_loader(sets["test"], shuffle=False))  # used once
    C.save_predictions(os.path.join(C.RESULTS_DIR, f"{name}_test_predictions.csv"),
                       sets["test"], y_te, p_te)

    summary = {
        "approach": name,
        "trainable_params": n_trainable,
        "train_time_s": round(train_time, 1),
        "hyperparameters": {"loss": "cross-entropy", "optimizer": "AdamW", "lr": lr,
                            "weight_decay": WEIGHT_DECAY, "scheduler": "cosine",
                            "batch_size": C.BATCH_SIZE, "epochs": args.epochs},
        "selected_epoch": best_epoch,
        "checkpoint": ckpt_path,
        "train_acc": accuracy_score(y_tr, p_tr),
        "val_acc": best_val_acc,
        "test_acc": accuracy_score(y_te, p_te),
        "test_macro_f1": f1_score(y_te, p_te, average="macro"),
        "environment": C.environment_info(),
    }
    C.save_json(summary, os.path.join(C.RESULTS_DIR, f"{name}_summary.json"))
    print(f"TEST: accuracy {summary['test_acc']:.4f}, macro F1 {summary['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
