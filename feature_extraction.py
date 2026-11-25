"""
Approach 3: frozen MobileNetV3-Small as feature extractor + scikit-learn classifiers.

Features = output of model.features followed by model.avgpool (global average
pooling), flattened -> one 576-dimensional vector per image. The classifier part
of the network is not used. Features are extracted without augmentation.

Two traditional classifiers are compared:
  - logistic regression (multinomial)
  - linear SVM
Each one is a Pipeline(StandardScaler, classifier): the scaler is fitted on the
training features only. The regularization C is chosen on the validation set,
the selected model is then evaluated once on the test set.

Run:  python feature_extraction.py
"""

import os
import time

import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

import common as C

C_VALUES = {
    "logreg": [0.01, 0.1, 1.0, 10.0],
    "linear_svm": [0.001, 0.01, 0.1, 1.0],
}


def make_classifier(kind, c):
    if kind == "logreg":
        clf = LogisticRegression(C=c, max_iter=3000, random_state=C.SEED)
    else:
        clf = LinearSVC(C=c, max_iter=10000, random_state=C.SEED)
    return make_pipeline(StandardScaler(), clf)


@torch.no_grad()
def extract(model, ds):
    model.eval()
    feats, labels = [], []
    for x, y in C.get_loader(ds, shuffle=False):
        f = model.features(x.to(C.DEVICE))
        f = model.avgpool(f).flatten(1)         # (batch, 576)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main():
    C.set_seed()
    sets, classes = C.get_datasets()

    model = C.build_model().to(C.DEVICE)
    for p in model.parameters():
        p.requires_grad = False                 # completely frozen

    # ---- feature extraction (all splits, no augmentation) ----
    start = time.time()
    X_tr, y_tr = extract(model, sets["train_eval"])
    X_val, y_val = extract(model, sets["val"])
    extract_time_trainval = time.time() - start
    X_te, y_te = extract(model, sets["test"])
    print(f"features: train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")
    print(f"extraction time (train+val) {extract_time_trainval:.1f} s")

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(C.CHECKPOINT_DIR, "features.npz"),
                        X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val, X_te=X_te, y_te=y_te)

    for kind, c_list in C_VALUES.items():
        # ---- choose C on the validation set ----
        start = time.time()
        best_c, best_val, best_clf, val_scores = None, -1.0, None, {}
        for c in c_list:
            clf = make_classifier(kind, c).fit(X_tr, y_tr)
            acc = accuracy_score(y_val, clf.predict(X_val))
            val_scores[str(c)] = acc
            print(f"{kind} C={c}: val acc {acc:.4f}")
            if acc > best_val:
                best_c, best_val, best_clf = c, acc, clf
        fit_time = time.time() - start

        # ---- final evaluation (test used once) ----
        p_tr = best_clf.predict(X_tr)
        p_te = best_clf.predict(X_te)
        name = f"features_{kind}"
        C.save_predictions(os.path.join(C.RESULTS_DIR, f"{name}_test_predictions.csv"),
                           sets["test"], y_te, p_te)

        clf = best_clf[-1]
        n_params = clf.coef_.size + clf.intercept_.size
        summary = {
            "approach": name,
            "trainable_params": int(n_params),
            # time = feature extraction of train+val + fitting all the C values
            "train_time_s": round(extract_time_trainval + fit_time, 1),
            "feature_extraction_time_s": round(extract_time_trainval, 1),
            "classifier_fit_time_s": round(fit_time, 1),
            "hyperparameters": {"classifier": kind, "C_tried": c_list, "C_selected": best_c,
                                "feature_dim": int(X_tr.shape[1]), "scaling": "StandardScaler (fit on train)"},
            "val_scores": val_scores,
            "train_acc": accuracy_score(y_tr, p_tr),
            "val_acc": best_val,
            "test_acc": accuracy_score(y_te, p_te),
            "test_macro_f1": f1_score(y_te, p_te, average="macro"),
            "environment": C.environment_info(),
        }
        C.save_json(summary, os.path.join(C.RESULTS_DIR, f"{name}_summary.json"))
        print(f"{name}: C={best_c}, TEST accuracy {summary['test_acc']:.4f}, "
              f"macro F1 {summary['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
