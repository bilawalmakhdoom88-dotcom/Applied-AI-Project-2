# Applied-AI-Project-2
# Project 2: Scene Classification with Transfer Learning

**Course:** Introduction to Applied AI | BS 8th Semester, Department of Electronics
**Author:** [MAKHDOOM BILAWAL]
**Week 5 lab:** Neural networks, convolutions, and transfer learning

A ResNet-18 pretrained on ImageNet, adapted to classify natural scenes into six classes. The model is trained in two phases (feature extraction, then fine-tuning) and evaluated once on a held-out test set.

## Results at a glance

| Stage | Trainable parameters | Validation accuracy |
|---|---|---|
| Feature extraction (new `fc` head only) | 3,078 of 11,179,590 (0.03%) | 0.901 |
| Fine-tuning (`layer4` + head) | 8,396,806 | 0.931 |

**Test accuracy (used once): 0.935** on 3,000 images (macro F1 0.936).

## Dataset

- [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (Kaggle), 150 × 150 RGB images, 6 classes: buildings, forest, glacier, mountain, sea, street.
- Split: **11,929 train / 2,105 val / 3,000 test**. The validation set is 15% of the original training folder; the test set is the separate `seg_test` folder.
- Image tensors: shape (150, 150, 3), dtype uint8, values 0 to 255.

## Method

1. **Preprocessing:** images resized to 224 × 224 and normalized with ImageNet mean and std.
2. **Augmentation (training set only):** RandomResizedCrop (scale 0.7 to 1.0), horizontal flip, and colour jitter. Validation and test images are only resized and normalized.
3. **Model:** ResNet-18 with ImageNet weights; the final layer is replaced by a new 512 → 6 linear layer (512 × 6 weights + 6 biases = 3,078 parameters).
4. **Phase 1, feature extraction (3 epochs):** all pretrained layers frozen, Adam with lr = 1e-3 on the new head.
5. **Phase 2, fine-tuning (3 epochs):** `layer4` unfrozen, Adam with two learning rates: 1e-4 for the pretrained `layer4` and 1e-3 for the new head.
6. **Model selection:** the weights from the epoch with the best validation accuracy are kept. The test set plays no part in any choice.

## Learning curves

| Epoch | Phase | Train loss | Val loss | Train acc | Val acc |
|---|---|---|---|---|---|
| 1 | Feature extraction | 0.588 | 0.327 | 0.809 | 0.894 |
| 2 | Feature extraction | 0.326 | 0.290 | 0.890 | 0.899 |
| 3 | Feature extraction | 0.299 | 0.276 | 0.895 | 0.901 |
| 4 | Fine-tuning | 0.262 | 0.227 | 0.906 | 0.927 |
| 5 | Fine-tuning | 0.170 | 0.204 | 0.938 | 0.931 |
| 6 | Fine-tuning | 0.124 | 0.219 | 0.955 | 0.929 |

Fine-tuning improved validation accuracy by 3.0 percentage points (0.901 → 0.931). In epoch 6 training loss kept falling while validation loss rose, an early sign of overfitting, so the epoch-5 weights were kept. Training accuracy is below validation accuracy in the early epochs because augmented training images are harder than clean validation images.

![Learning curves](images/learning_curves.png)

## Test results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| buildings | 0.952 | 0.906 | 0.928 | 437 |
| forest | 0.994 | 0.985 | 0.989 | 474 |
| glacier | 0.921 | 0.868 | 0.894 | 553 |
| mountain | 0.883 | 0.918 | 0.900 | 525 |
| sea | 0.945 | 0.982 | 0.963 | 510 |
| street | 0.926 | 0.956 | 0.941 | 501 |

- **Weakest class:** glacier (recall 0.868), most often confused with **mountain** (57 glacier images predicted as mountain, 31 the other way round).
- **Second most confused pair:** buildings ↔ street (35 + 18 errors).
- **Strongest class:** forest (recall 0.985).

![Confusion matrix](images/confusion_matrix.png)

## Error analysis

![Error gallery](images/error_gallery.png)

**Most instructive error:** [one sentence about a misclassified image: what the model predicted, and whether the model, the label, or the image itself was at fault.]

## Repository contents

```
.
├── Week5_Transfer_Learning.ipynb   # full notebook with all parts and written analysis
├── resnet18_intel.pt               # trained weights (about 45 MB)
├── images/                         # learning curves, confusion matrix, error gallery
└── README.md
```

## How to reproduce

1. Open the notebook on Kaggle, add the Intel Image Classification dataset, and set Accelerator to GPU T4 and Internet to On.
2. Run all cells in order. Training takes about 40 seconds per epoch on a T4.
3. To reuse the trained model:

```python
import torch, torch.nn as nn
from torchvision import models

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 6)   # replace the head before loading
model.load_state_dict(torch.load("resnet18_intel.pt", map_location="cpu"))
model.eval()
```

Class order (alphabetical): buildings, forest, glacier, mountain, sea, street.

## Next: Week 6 proposal

[One paragraph: the problem you want to solve, who benefits, the dataset link, and the number of classes and images. For example the domain (crop disease, medical images, manufacturing defects, documents, satellite images, or your own idea), a Kaggle dataset with at least a few hundred labeled images per class, and its license.]

## Tools

Python, PyTorch, torchvision, scikit-learn, NumPy, Matplotlib, Kaggle Notebooks (GPU T4).
# week 6 Brain Tumour MRI Screening with ResNet-18

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Use-Educational%20only-orange)

Binary classification of brain MRI slices (**tumour** vs. **no tumour**) using transfer learning with an ImageNet-pretrained ResNet-18. The project puts as much weight on **data quality** (duplicate removal and leakage auditing) and **honest evaluation** as on the model itself.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis or clinical decisions.

---

## Table of Contents

- [Highlights](#highlights)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Results](#results)
- [Getting Started](#getting-started)
- [Using the Trained Model](#using-the-trained-model)
- [Repository Contents](#repository-contents)
- [Limitations](#limitations)
- [References](#references)
- [Author and License](#author-and-license)

---

## Highlights

- **Leak-aware data preparation:** 187 exact duplicates removed, followed by a near-duplicate audit that removed 260 validation and 230 test images having a twin in the training set.
- **Imbalance handling:** class-weighted cross-entropy for a 3 : 1 tumour-heavy dataset.
- **Two-phase transfer learning:** head-only training, then fine-tuning of the last residual block.
- **Decision-threshold tuning** on the validation set for a target tumour sensitivity.
- **Explainability:** Grad-CAM heat-maps for correct predictions and errors.
- Runs end to end in about **two minutes of GPU time** on Kaggle.

## Dataset

| Item | Details |
|---|---|
| Source | Kaggle dataset [`bilawalmakhdoom/brain-tumor`](https://www.kaggle.com/datasets/bilawalmakhdoom/brain-tumor) |
| Size | 7,200 MRI slices |
| Folders | `glioma`, `meningioma`, `pituitary`, `notumor` (1,800 images each) |
| Task framing | The three tumour folders are merged into **tumor** (5,400); `notumor` becomes **no_tumor** (1,800) |
| Image properties | Widths from 173 to 1,338 px (median 512); mixed colour modes (RGB, grey, RGBA) |

File names carry no patient identifier, which shapes the splitting strategy and is discussed under [Limitations](#limitations).

## Pipeline

1. **Index** all images and merge the four folders into two classes.
2. **Inspect** image sizes, colour modes and sample slices.
3. **Remove exact duplicates** using MD5 hashes (7,200 to 7,013 images).
4. **Split** into train / validation / test with `StratifiedGroupKFold` (4,870 / 1,169 / 974 images).
5. **Audit near-duplicates** with 32 x 32 standardised fingerprints and drop validation / test images that match a training image. Final sizes: **4,870 / 909 / 744**.
6. **Pre-process** to 224 x 224, grey-scale replicated to 3 channels, ImageNet normalisation; augmentation (crop, flips, rotation, brightness / contrast) on training data only.
7. **Train** in two phases (head only for 2 epochs, then `layer4` + head for 3 epochs) using Adam and class weights (no_tumor 2.087, tumor 0.658). The best epoch by validation macro-F1 is kept.
8. **Evaluate** on the untouched test set, then **tune the decision threshold** on validation.
9. **Explain** predictions with Grad-CAM and **save** the checkpoint.

## Results

Held-out test set: **744 images** (57 no_tumor, 687 tumor).

### Default decision rule (argmax)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| no_tumor | 0.962 | 0.895 | 0.927 | 57 |
| tumor | 0.991 | 0.997 | 0.994 | 687 |

**Accuracy 0.989 | Macro-F1 0.961.** Only 8 of 744 images are misclassified (6 healthy scans flagged as tumour, 2 tumours missed).

### Tuned threshold (t = 0.986, target recall 0.95)

| TP | FN | TN | FP | Sensitivity | Specificity | PPV | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 665 | 22 | 55 | 2 | 0.968 | 0.965 | 0.997 | 0.994 |

### Trade-off to be aware of

| Operating point | Missed tumours | False alarms |
|---|---|---|
| Default (about 0.5) | **2** | 6 |
| Tuned (t = 0.986) | 22 | **2** |

The threshold search returns the *strictest* cut-off that still meets the recall target, so it reduces false alarms at the cost of more missed tumours. If missing a tumour is the worse outcome, raise `TARGET_RECALL` (for example to 0.99) or use a lower threshold.

## Getting Started

### Run on Kaggle (recommended)

1. Create a new Kaggle notebook and import `brain-tumor-mri-professional-report.ipynb` (**File, then Import Notebook**).
2. Attach the dataset via **Add Input** and search for `bilawalmakhdoom/brain-tumor`.
3. In the first code cell, make sure `DATA_ROOT` matches the real dataset path shown in the Kaggle sidebar.
4. In **Settings**, choose a **GPU** accelerator and switch **Internet on** (needed to download the ResNet-18 weights unless you add them as a Kaggle model input).
5. Select **Run, then Restart & Run All**.

The trained checkpoint is written to `/kaggle/working/week6_model.pt`.

### Configuration

All settings live in the first code cell:

| Parameter | Default | Meaning |
|---|---|---|
| `DATA_ROOT` / `TRAIN_DIR` | Kaggle dataset path | Folder containing the class sub-folders |
| `TEST_DIR` | `None` | `None` means the test set is carved from the data |
| `GRAYSCALE` | `True` | Convert images to grey-scale (3-channel) |
| `IMG_SIZE` | `224` | Network input size |
| `BACKBONE` | `resnet18` | Also supports `resnet50` and `efficientnet_b0` |
| `BINARY` | `True` | Merge tumour types into one class |
| `NORMAL_KEYWORD` | `notumor` | Folder-name keyword that marks the healthy class |
| `TARGET_RECALL` | `0.95` | Minimum tumour sensitivity for threshold tuning |

### Requirements

Kaggle already provides everything. For a local run you need Python 3 and:

```
torch  torchvision  numpy  pandas  scikit-learn  matplotlib  pillow
```

## Using the Trained Model

The checkpoint stores the weights, class order, backbone name, image size and the grey-scale flag. The tuned threshold (**0.986**) is *not* stored and must be applied by you.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

ckpt = torch.load('week6_model.pt', map_location='cpu')

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(ckpt['classes']))
model.load_state_dict(ckpt['state_dict'])
model.eval()

tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((ckpt['img_size'], ckpt['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = Image.open('scan.jpg').convert('L')
x = tf(img).unsqueeze(0)

with torch.no_grad():
    probs = F.softmax(model(x), dim=1)[0]

p_tumor = probs[ckpt['classes'].index('tumor')].item()
THRESHOLD = 0.986
label = 'tumor' if p_tumor >= THRESHOLD else 'no_tumor'
print(f'{label} (p_tumor = {p_tumor:.3f})')
```

## Repository Contents

| File | Description |
|---|---|
| `brain-tumor-mri-professional-report.ipynb` | Full annotated notebook with code, outputs and written analysis |
| `README.md` | This file |
| `week6_model.pt` | Trained checkpoint (created when the notebook is run) |

## Limitations

- **No patient identifiers.** Slices from one patient may fall into different splits despite the near-duplicate audit, which could make results optimistic.
- **Small healthy-class evaluation set** (69 validation, 57 test images) after leakage removal. Healthy-class metrics are uncertain, and one image shifts recall by about 1.8 points.
- **Single split and seed.** No cross-validation or confidence intervals.
- **Binary framing** hides how well each tumour type (glioma, meningioma, pituitary) is detected.
- **No external validation** on data from another hospital or scanner.
- Grad-CAM heat-maps are a plausibility check, not proof of correct reasoning
