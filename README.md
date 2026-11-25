# Homework 2 - Image Classification

Transfer learning with PyTorch on the Oxford-IIIT Pet dataset (37 breeds), Computer Vision course, Fall 2026.
Three ways of adapting the pre-trained MobileNetV3-Small network are compared:

1. **Full fine-tuning** - the whole network is trained (`train_nn.py --mode full`)
2. **Classifier-head training** - `model.features` frozen, only the classifier layers are trained (`train_nn.py --mode head`)
3. **Feature extraction + scikit-learn** - frozen network, 576-d pooled features, logistic regression and linear SVM (`feature_extraction.py`)

## Environment

- Python 3.12
- PyTorch / TorchVision / scikit-learn: the exact versions used for the reported results are saved in every `results/*_summary.json` (field `environment`)
- Other dependencies: numpy, scipy, matplotlib, pillow (see `requirements.txt`)
- GPU used: NVIDIA RTX 5070 (CUDA 12.8). The code also runs on CPU, only slower.

Install (GPU, CUDA 12.8 build of PyTorch):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Dataset

Oxford-IIIT Pet Dataset, O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs", CVPR 2012.
Source: https://www.robots.ox.ac.uk/~vgg/data/pets/ - License: CC BY-SA 4.0 (the copyright of the images remains with their original owners).

The dataset is **not** included in this repository. It is downloaded automatically into `data/` by
`torchvision.datasets.OxfordIIITPet(download=True)` the first time any script is run.

Split protocol:
- official `test` partition: used only once, for the final evaluation of each selected model
- official `trainval` partition: split 80% train / 20% validation, stratified by breed, `random_state=42`
- the split is saved in `splits/train.csv` and `splits/val.csv` (index, file name, label) and reloaded by every script, so all approaches use exactly the same images

## How to run

From this folder:

```
python train_nn.py --mode full
python train_nn.py --mode head
python feature_extraction.py
python make_report.py
```

The first three commands train/select each model and evaluate it once on the test set; `make_report.py` only reads the saved results and builds tables and figures.

## Settings

| | Full fine-tuning | Head only | Features + sklearn |
|---|---|---|---|
| Trainable layers | all | `classifier[0]` Linear(576,1024) and `classifier[3]` Linear(1024,37) | sklearn classifier on 576-d features |
| Loss | cross-entropy | cross-entropy | log-loss (logreg) / hinge (SVM) |
| Optimizer | AdamW, weight decay 1e-4, cosine schedule | AdamW, weight decay 1e-4, cosine schedule | - |
| Learning rate | 3e-4 | 1e-3 | - |
| Batch size / epochs | 32 / 15 | 32 / 15 | - |
| Model selection | epoch with best val accuracy | epoch with best val accuracy | C with best val accuracy |

- Preprocessing: the transforms of `MobileNet_V3_Small_Weights.DEFAULT` (resize 256, center crop 224, ImageNet mean/std).
- Augmentation (training set of approaches 1 and 2 only): `RandomResizedCrop(224, scale=(0.6, 1.0))`, `RandomHorizontalFlip`.
- Feature scaling for approach 3: `StandardScaler` fitted on the training features only. No dimensionality reduction.
- No class balancing (the classes are almost balanced, see `results/class_counts.csv`).

## Reproducibility

- Seed 42 for Python `random`, NumPy and PyTorch (CPU and CUDA), and for the data loader workers and shuffling.
- `torch.backends.cudnn.deterministic = True`, `cudnn.benchmark = False`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Remaining sources of variation: a few CUDA operations have no deterministic implementation (PyTorch prints a warning for them), and results can change slightly with a different GPU, driver or library version. Training times depend on the hardware.
- Selected checkpoints: `checkpoints/full_finetune.pt`, `checkpoints/head_only.pt`; selected epoch / C and all hyperparameters are in `results/*_summary.json`.

## File structure

```
.
├── common.py              # settings, seeds, dataset + split, model, helpers
├── train_nn.py            # approaches 1 and 2 (PyTorch training)
├── feature_extraction.py  # approach 3 (frozen features + scikit-learn)
├── make_report.py         # tables and figures from the saved results
├── requirements.txt
├── splits/                # train.csv, val.csv (stratified split of trainval)
├── checkpoints/           # selected model weights (features.npz is not in the repo)
└── results/
    ├── metrics.csv, results_table.tex     # main results
    ├── class_counts.csv                   # images per breed in each split
    ├── top_confusions.csv                 # most frequent confusions
    ├── per_class_f1.csv                   # F1-score of every breed
    ├── significance.csv                   # McNemar tests between approaches
    ├── *_summary.json                     # hyperparameters, selected checkpoint, metrics, versions
    ├── *_history.csv                      # loss/accuracy per epoch (approaches 1, 2)
    ├── *_test_predictions.csv             # prediction for every test image
    └── figures/                           # confusion matrices, training curves, examples
```