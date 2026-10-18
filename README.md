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
