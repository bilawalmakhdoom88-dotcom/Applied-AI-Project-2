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

# Week 7: Text and Multimodal AI (Project 2)

> Giving the Week 6 brain-tumour MRI image model the ability to *read*: classical text methods, measured end to end, then fused with image evidence into one transparent, privacy-safe case record.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?logo=pytorch&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.8.16-09A3D5?logo=spacy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LogReg-F7931E?logo=scikitlearn&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle&logoColor=white)

| | |
|---|---|
| **Author** | Bilawal Makhdoom |
| **Date** | 2 November 2026 |
| **Course** | Introduction to Applied AI |
| **Domain** | Brain-tumour MRI (Week 6 model) |
| **Notebook** | `week7-text-and-multimodel-ai.ipynb` |
| **Environment** | Kaggle notebook, Python 3.13, spaCy 3.8.16, PyTorch 2.11.0 (CUDA 12.8 build) |

---

## Table of Contents

1. [Overview](#overview)
2. [Results at a Glance](#results-at-a-glance)
3. [System Architecture](#system-architecture)
4. [Getting Started](#getting-started)
5. [Detailed Findings](#detailed-findings)
   - [Part 1: TF-IDF by hand](#part-1-tf-idf-by-hand)
   - [Part 2: SMS spam classification](#part-2-sms-spam-classification)
   - [Part 3: Named entity recognition](#part-3-named-entity-recognition)
   - [Part 4: Regex fields with negative tests](#part-4-regex-fields-with-negative-tests)
   - [Part 5: Character error rate and OCR repair](#part-5-character-error-rate-and-ocr-repair)
   - [Part 6: Image + text case record](#part-6-image--text-case-record)
6. [Key Formulas](#key-formulas)
7. [Privacy and Safety Design](#privacy-and-safety-design)
8. [Limitations](#limitations)
9. [Next Steps](#next-steps)
10. [Notebook Structure](#notebook-structure)

---

## Overview

Project 2 builds the text side of an applied-AI system and then combines it with the image model from Week 6. Every component is **measured**, not just demonstrated, and each written result can be checked against the printed output of its cell.

**What is built**

- A TF-IDF implementation reproduced by hand and verified against scikit-learn
- An SMS spam classifier comparing word and character n-grams, with full error analysis
- A named-entity recognition (NER) pipeline scored against gold annotations, improved with an entity ruler, and evaluated on a **held-out** set from the MRI domain
- A regex field extractor with positive and **negative** tests
- A character-error-rate (CER) metric with context-aware OCR repair
- A late-fusion **case record** that combines image probabilities with clinical-note evidence, masks personal identifiers first, and explains every decision

**Notebook layout.** Each section has a description *before* the cell (what it does and why), the cell with its real output, and a **Calculations** or **Reading the output** block *after* it that re-derives the printed numbers by hand.

---

## Results at a Glance

| Part | Topic | Key idea | Measured result (this run) |
|---|---|---|---|
| 1 | TF-IDF by hand | smooth IDF, L2 norm, cosine | hand = scikit-learn; cosine(d2, d3) = **0.544** |
| 2 | SMS spam: word vs character n-grams | logistic regression, macro-F1 | macro-F1 **0.974** (word) vs **0.982** (char) |
| 3 | NER against a gold set, then entity ruler | entity-level P, R, F1 | provided set **0.516 → 0.875**; held-out domain set **0.556 → 0.811** |
| 4 | Regex fields with negative tests | pattern tests, word boundaries | **5/5** lab tests, **8/8** domain tests (5 negative) |
| 5 | Character error rate and OCR repair | CER = (S + D + I) / N | lab pairs 0.062, 0.059, 0.143 → **0.000**; invoice 0.000 |
| 6 | Image + text case record with privacy | late fusion, masking first | **REFER**, P(tumor) = 0.996, no PII in the JSON |
| 7 | Check Your Understanding | | six answers with working |

---

## System Architecture

The case record in Part 6 is a **late-fusion** pipeline: image and text evidence are produced separately and combined with simple rules a clinician can read.

```
                         ┌──────────────────────┐
   MRI image ──────────► │  Week 6 model        │──► P(tumor)  ─────────┐
                         │  (ResNet-18)         │                       │
                         └──────────────────────┘                       ▼
                                                              ┌──────────────────┐
                         ┌──────────────────────┐             │  Transparent     │
   Clinical note ──────► │  1. mask_pii()       │             │  decision rules  │──► JSON case record
                         │  2. NER + regex      │──► urgent   │  + disagreement  │    (REFER / ROUTINE,
                         │  3. urgent-term scan │    terms ──►│  flag            │     reasons, flags)
                         └──────────────────────┘             └──────────────────┘
```

**Decision rule:** `REFER` if `P(tumor) >= 0.986` **or** the masked note contains an urgent term; otherwise `ROUTINE`. If exactly one source is positive, the record also carries the flag `IMAGE/NOTE DISAGREE: send to a human reviewer`.

---

## Getting Started

### Prerequisites

| Requirement | Detail |
|---|---|
| Platform | Kaggle notebook (paths below are Kaggle paths) |
| Internet | **On** (EasyOCR downloads about 100 MB on first use) |
| Python libraries | `numpy`, `pandas`, `matplotlib`, `spacy`, `torch`, `torchvision`, `Pillow`, `scikit-learn`, `easyocr` |
| spaCy model | `en_core_web_sm` (preinstalled on Kaggle; downloaded automatically if missing) |

### Inputs to add (via **Add Input**)

| Input | Path |
|---|---|
| SMS Spam Collection | `/kaggle/input/datasets/uciml/sms-spam-collection-dataset/spam.csv` |
| Week 6 model checkpoint | `/kaggle/input/datasets/bilawalmakhdoom/week6-model-pt/week6_model.pt` |
| Brain-tumour MRI images (Part 6) | `/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset` |

Every path is first tried exactly and then searched for under `/kaggle/input`, so a slightly different folder name does not break the notebook. A path pointing to a **folder** is also accepted: the `first_existing` helper looks for the file inside it.

### Run

1. Create a Kaggle notebook and upload `week7-text-and-multimodel-ai.ipynb`.
2. Add the three inputs above and switch **Internet** on.
3. Choose **Run All**. The models are small and run on CPU just as well as on GPU.

---

## Detailed Findings

### Part 1: TF-IDF by hand

Three tiny messages (`d1` = "win free cash now", `d2` = "call me now", `d3` = "free call now") are vectorised with scikit-learn, then document `d1` is recomputed by hand and checked with an `assert`.

| Word | df | idf = ln(4 / (1 + df)) + 1 |
|---|---|---|
| win | 1 | 1.6931 |
| free | 2 | 1.2877 |
| cash | 1 | 1.6931 |
| call | 2 | 1.2877 |
| me | 1 | 1.6931 |
| now | 3 | 1.0000 |

- Hand calculation and library agree (`hand == library for d1: OK`).
- **cosine(d2, d3) = 0.544.** The shared word "call" contributes 0.340 but "now" only 0.204, because "now" appears in every message. This is why IDF matters for search and duplicate detection.

### Part 2: SMS spam classification

**Data:** SMS Spam Collection, 5,572 messages, 13.4% spam. An "always ham" classifier already scores 86.6% accuracy, so the notebook reports **macro-F1** and spam precision/recall instead.

**Model:** `TfidfVectorizer` → `LogisticRegression`, 80/20 stratified split (`random_state=42`), `class_weight='balanced'`, `sublinear_tf=True`, `min_df=2`.

| Model | Features | Macro-F1 | Spam precision | Spam recall | Errors (of 1,115) |
|---|---|---|---|---|---|
| Word 1-2 grams | 12,128 | 0.974 | 0.979 | 0.933 | 13 |
| Character `char_wb` 2-5 grams | 36,449 | 0.982 | 0.986 | 0.953 | 9 |

**Confusion matrices (test split)**

Word model:

| | Pred. ham | Pred. spam |
|---|---|---|
| **True ham (966)** | 963 | 3 |
| **True spam (149)** | 10 | 139 |

Character model:

| | Pred. ham | Pred. spam |
|---|---|---|
| **True ham (966)** | 964 | 2 |
| **True spam (149)** | 7 | 142 |

**What the word model learned**

| Direction | Top n-grams | Verdict |
|---|---|---|
| Spam-like | call, txt, free, text, reply, uk, to, www, mobile, claim, stop, your | Mostly sensible (calls to action, short-code replies); "to" and "your" are function-word shortcuts |
| Ham-like | my, me, ok, that, it, but, ll, gt, lt, can, what, da | Personal chat words; `gt`/`lt` are HTML-escape artefacts of the dataset |

**Robustness to spelling tricks** (threshold 0.5; the last three messages are my own)

| Message | Expected | Word p(spam) | Char p(spam) |
|---|---|---|---|
| W1N a FR33 pr1ze!! txt CLAIM now | spam | 0.99 ✅ | 0.85 ✅ |
| parso milenge, 6 baje | ham | 0.19 ✅ | 0.15 ✅ |
| C0ngrats!! U w0n Rs 40000 pr1ze. Cl@im by txt 7421 now | spam | 0.81 ✅ | 0.81 ✅ |
| Shabash! Aap ne 75000 ka inaam jeeta hai, foran 6200 par SMS karein | spam | 0.36 ❌ | 0.52 ✅ (barely) |
| Bhai class parso dopahar 2 baje hai, slides share kar dena | ham | 0.13 ✅ | 0.13 ✅ |

**Error analysis of the 13 word-model errors**

| Category | Count | Share |
|---|---|---|
| Model error | 8 | 61.5% |
| Genuinely ambiguous | 4 | 30.8% |
| Label error | 1 | 7.7% |

**Takeaway:** the character model is ahead by only 4 messages on one split, which is too small to call a real difference on clean English SMS. The case for characters rests on **robustness** to misspellings, not on the score gap. Neither model was trained on Roman Urdu, so those scores are untrained behaviour, not evidence of support.

### Part 3: Named entity recognition

NER is treated as token classification (BIO tags) and **scored at entity level**: an entity is correct only if both its exact text and its label match the gold annotation.

**Provided Pakistani sentences (6 sentences, 16 gold entities)**

| Pipeline | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| spaCy `en_core_web_sm` out of the box | 0.533 | 0.500 | 0.516 | 8 | 7 | 8 |
| + 7 entity-ruler patterns (before `ner`) | 0.875 | 0.875 | **0.875** | 14 | 2 | 2 |

**My domain: brain-tumour MRI reports (10 sentences, 38 gold entities)**

Sentences 1-5 are used to write the rules; sentences 6-10 are **held out** and never used for any rule.

| Set | Pipeline | Precision | Recall | F1 |
|---|---|---|---|---|
| 1-5 (rule-writing) | spaCy alone | 0.667 | 0.500 | 0.571 |
| 1-5 (rule-writing) | + domain rules | 1.000 | 1.000 | 1.000 |
| 6-10 (**held out**) | spaCy alone | 0.556 | 0.556 | 0.556 |
| 6-10 (**held out**) | + domain rules | 0.789 | 0.833 | **0.811** |

The same rules score 1.000 on the sentences they were written from but 0.811 on new ones: a visible **optimism gap of 0.189**, which is exactly why the held-out set exists.

**What the rules fixed on the held-out set:** money amounts (`PKR` / `Rs.`), `dd/mm/yyyy` dates and one hospital suffix pattern. **Still wrong:** two person names and `Metro MRI Centre` (the suffix pattern requires capitalised words, and "MRI" is all capitals).

**Remedy by error type**

| Error type | Right tool |
|---|---|
| Closed vocabularies (hospitals, cities) | Rules / gazetteer |
| Rigid formats (dates, amounts) | Regex |
| Open-class person names, org-versus-person cases | A model trained on annotated domain data |

### Part 4: Regex fields with negative tests

Some fields are rigid formats rather than named entities. A regex is only trustworthy if it also **rejects** look-alikes, so negative tests come first.

| Field | Pattern | Origin |
|---|---|---|
| CNIC | `\b\d{5}-\d{7}-\d\b` | lab |
| Mobile | `(?:\+92[- ]?\|\b0)3\d{2}[- ]?\d{7}\b` | lab |
| Amount | `\b(?:PKR\|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?)` | lab (fixed with `\b`) |
| Date | `\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b` | lab |
| E-mail | `\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b` | lab |
| Invoice no. | `\b(?:Invoice\s+)?No\.?\s*(\d{3,})\b` | lab |
| Scan file | `\b(?:Tr\|Te)-(?:gl\|me\|no\|pi)_\d+\.(?:jpe?g\|png)\b` | domain |
| Tumour size | `\b\d+(?:\.\d+)?\s?(?:mm\|cm)\b` | domain |
| MRN | `\bMRN[- ]?\d{6,8}\b` | domain |
| Age | `\baged?\s+(\d{1,2}\|1[01]\d)\b` | domain |
| MRI sequence | `(?-i:\b(?:T1\|T2\|FLAIR\|DWI)\b)` (case-sensitive) | domain |

**Test results:** 5/5 lab tests and 8/8 domain tests pass, of which 5 are negative tests.

| Negative test | Why it must return nothing |
|---|---|
| `BP 135 mmHg and glucose 7 mmol/L` | mmHg / mmol are not tumour sizes |
| `He has a flair for cooking` | the word "flair" is not the FLAIR sequence |
| `Tq-gl_3.jpg and Te-zz_8.jpg` | wrong prefix / wrong class code |
| `MRN 45 was seen in age 2027` | too few digits; a year is not an age |
| `Doctors 250 mg twice daily` | the "rs" in "Doctors" is not a currency |

**Bugs found and fixed**

| Pattern | Loose version | Result on `Doctors 250 mg; BP 135 mmHg` | Fix |
|---|---|---|---|
| Tumour size | `\d+\s?mm` | `['135 mm']` (read mmHg as a lesion) | leading and trailing `\b` |
| Amount | no `\b` before `Rs` | `['250']` (end of "Doctors") | leading `\b` |

### Part 5: Character error rate and OCR repair

**CER = (substitutions + deletions + insertions) / reference length**, computed with a Levenshtein implementation. `fix_ocr` repairs letters that look like digits **only inside numbers** (`O→0`, `l/I→1`, `N0.→No.`), so ordinary words are untouched.

| Reference | OCR output | CER before | CER after |
|---|---|---|---|
| `Invoice No. 6284` | `Invoice N0. 6284` | 0.062 | 0.000 |
| `Total: PKR 27,400` | `Total: PKR 27,4O0` | 0.059 | 0.000 |
| `Due 20/10/2026` | `Due 2O/10/2O26` | 0.143 | 0.000 |

**Synthetic invoice (EasyOCR):** raw CER = 0.000 and field-level accuracy over mobile, amount, date and invoice number = 4/4 = 1.00. This is expected for rendered text in one clean font; real phone photos (uneven light, perspective, folds, compression) will score worse.

> **Why CER alone can mislead:** `PKR 27,4O0` has a CER of only 0.059, yet the amount is wrong. One wrong character breaks the whole field, which is why field-level accuracy is reported as well.

### Part 6: Image + text case record

**Components**

| Component | Role |
|---|---|
| `mask_pii()` | Replaces CNIC, phone, e-mail and MRN with `[CNIC]`, `[PHONE]`, `[EMAIL]`, `[MRN]` **before** any extraction |
| `image_evidence()` | Week 6 ResNet-18 → softmax class probabilities (`no_tumor`, `tumor`) |
| `text_evidence()` | Entities (domain NER pipeline), regex fields, urgent symptom terms |
| `case_record()` | Combines everything into a JSON record with reasons and flags |

**Operating point:** `OP_THRESHOLD = 0.986`, the Week 6 threshold chosen for recall of at least 0.95 on validation. A probability such as 0.90 therefore does not trigger the image reason by itself.

**Main example** (`Te-gl_1.jpg` with a realistic clinical note)

| Evidence | Value | Fires? |
|---|---|---|
| Image: P(tumor) | 0.996 (≥ 0.986) | yes |
| Text: urgent terms | headache, seizure, vomit | yes |
| **Decision** | | **REFER** |
| Disagreement flag | both sources positive | none |

**Privacy check:** the JSON was searched for the full CNIC, phone, MRN and their digit runs: `PII fragments found in the JSON: none`.

**Disagreement cases**

| Case | P(tumor) | Urgent terms | Decision | Flag |
|---|---|---|---|---|
| A: healthy image + urgent note | 0.017 | headache, seizure, vomit | REFER | IMAGE/NOTE DISAGREE |
| B: tumour image + routine note | 0.996 | none | REFER | IMAGE/NOTE DISAGREE |

Because a missed tumour costs far more than an extra review, **any one positive source keeps the case on REFER**, and the flag tells a human reviewer that the evidence conflicts.

**Entity quality inside the record:** against a five-entity gold list the record scores precision 0.50, recall 1.00, F1 0.667. The wrong entities (`58` as DATE, `T1` as CARDINAL, `CNIC` as ORG) did not change the decision, but the `CNIC` / ORG entries are harmful as stored data, which argues for filtering placeholders out of the entity list.

---

## Key Formulas

| Name | Formula |
|---|---|
| Smooth IDF | `idf = ln((1 + N) / (1 + df)) + 1` |
| TF-IDF weight | `weight = tf * idf`, then divide by the vector length (L2 norm) |
| Cosine similarity | dot product of two length-1 vectors |
| Precision, recall, F1 | `P = TP/(TP+FP)`, `R = TP/(TP+FN)`, `F1 = 2PR/(P+R)` |
| Character error rate | `CER = (substitutions + deletions + insertions) / reference length` |
| Late fusion | `p = w * p_image + (1 - w) * p_text` |

---

## Privacy and Safety Design

- **Mask first, extract second.** Extraction copies text, so identifiers must be removed before NER and regex run. Otherwise `text_evidence` and every downstream log or JSON would carry them.
- **Automated leak check.** An `assert` fails the cell if any fragment of the CNIC, phone number or MRN appears in the record.
- **Transparent reasons.** Every decision lists which source triggered it, so a reviewer sees *why*, not only a probability.
- **Fail towards review.** Any single positive source keeps a case on REFER; disagreement is flagged for a human.
- **Decision support only.** Every record carries the disclaimer: *"Decision support only; a qualified human decides."*
- All names, amounts and dates used in the domain examples are invented.

---

## Limitations

| Limitation | Impact |
|---|---|
| Small evaluation sets (16 and 38 gold entities) | One entity moves F1 by 0.03 to 0.06 |
| Single spam train/test split | A 4-message difference is not evidence of a better model |
| No Roman Urdu training data | Roman Urdu scores are untrained behaviour |
| Synthetic OCR input | Real phone photos will give a higher CER |
| Hand-written rules | Unseen person names and acronym-containing organisations still fail |
| Held-out sentences share formats with rule-writing sentences | Shows format rules transfer within a style, not to unseen styles |
| Rule-based fusion | No labelled image-note pairs exist to learn a fusion weight |

---

## Next Steps

1. Annotate a few hundred real report sentences and fine-tune an NER model instead of extending the rules.
2. Allow all-capital words (such as "MRI") in the organisation suffix pattern and add a city gazetteer.
3. Measure OCR on five real scans with hand-typed reference text.
4. Collect labelled image-note pairs so the fusion weight `w` can be learned on validation data instead of using fixed rules.
5. Repeat the spam comparison with `lowercase=False` (or a capital-letter-share feature) and over several random splits.

---

## Notebook Structure

| Section | Contents |
|---|---|
| Formula sheet and setup | Imports, spaCy model, `show()` and `first_existing()` helpers |
| Part 1 | TF-IDF by hand |
| Part 2 | SMS spam: Tasks 2.1 to 2.5 plus written summary |
| Part 3 | NER: Tasks 3.1 to 3.4, written summary, F1 bar chart |
| Part 4 | Regex fields: Tasks 4.1 to 4.2 plus written summary |
| Part 5 | CER and OCR repair: Tasks 5.1 to 5.2 plus written summary |
| Part 6 | Image + text case record: Tasks 6.1 to 6.3, disagreement cases, written summary |
| Check Your Understanding | Six answers with working |
| Summary, limitations and next steps | Consolidated findings |

---

*Course project for Introduction to Applied AI. Decision-support prototype only; not for clinical use.*
