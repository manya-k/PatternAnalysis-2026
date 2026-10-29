# Siamese Network for ISIC Melanoma Triage
**Problem:** binary classification of dermoscopic images (benign vs. melanoma) with a confidence indication for  "refer to a specialist" option if unsure.
**Model:** Siamese metric-learning network (shared ResNet-18 encoder, contrastive loss, k-nearest-neighbour inference), compared against a conventional ResNet-18 classifier.
**Dataset:** [ISIC Melanoma Dataset, resized to 224x224 (Kaggle)](https://www.kaggle.com/datasets/roua91/isic-melanoma-dataset-resized-to-224x224) [7].

## Contents

1. [Overview](#1-overview)
2. [Feasibility Review](#2-feasibility-review)
3. [Repository Structure, Dependencies and Usage](#3-repository-structure-dependencies-and-usage)
4. [Data, Pre-processing and Splits](#4-data-pre-processing-and-splits)
5. [Method](#5-method)
6. [Results: Benchmarking Against the Baseline](#6-results-benchmarking-against-the-baseline)
7. [Resource Profiling](#7-resource-profiling)
8. [Failure Case Autopsy](#8-failure-case-autopsy)
9. [Recommendation to the Project Manager](#9-recommendation-to-the-project-manager)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [Artificial Intelligence Usage Disclosure](#11-artificial-intelligence-usage-disclosure)
12. [References](#12-references)

---

## 1. Overview

### The problem

Melanoma is the deadliest skin cancer, but it is curable when excised early. A triage clinician who reviews dermoscopic images has to decide quickly which lesions need urgent specialist attention, and benign atypical naevi can look very similar to early melanoma. An automated assistant could take routine cases off the clinician's desk, but only if it knows when it does not know: in screening, a single confidently wrong "benign" on a melanoma is far more damaging than a false alarm. This project investigates the **Confidence and Ambiguity Dilemma** of the task sheet: *does a Siamese network's distance-based confidence track diagnostic correctness better than a conventional CNN's softmax confidence, and can a confidence threshold safely automate the confident cases while referring the ambiguous ones to a human?*

### The algorithm

A **Siamese network** [1] passes two images through the *same* encoder (an ImageNet-pretrained ResNet-18 [2] with a small projection head) and maps each to a 128-dimensional, unit-length embedding. It is trained with **contrastive loss** [3] on image pairs: lesions of the same class are pulled together in the embedding space and lesions of different classes are pushed at least a margin apart. After training, every training image is embedded once to form a **reference bank**. A new image is embedded and compared with its **k nearest reference lesions**; their distance-weighted vote gives the melanoma probability, and the share of the vote for the predicted class is the confidence. Cases whose confidence is below a threshold chosen on validation data are **referred** to a specialist. The nearest reference images double as visual evidence for the clinician. The **baseline** is the same pretrained ResNet-18 trained directly as a two-class classifier with class-weighted cross-entropy and temperature scaling [4], so the comparison isolates metric learning versus direct classification rather than two different architectures.

### What the tool returns

For each image: a benign or melanoma decision, a confidence score, "auto" or "REFER", and (Siamese) the closest training lesions with their labels and distances. Green titles are correct predictions, red are errors.

![Siamese predictions on six random test images, each shown with its five nearest training images](images/predict_siamese.jpg)

*Figure 1. `python predict.py --model siamese`: six random test images (left column) and their nearest training images. The model's tuned settings (k = 25, class-balanced vote, refer if confidence < 0.764) are shown in the title.*

```mermaid
flowchart LR
    A[Dermoscopic image] --> B[Shared ResNet-18 encoder + projection head]
    B --> C[128-d unit-length embedding]
    C --> D[k = 25 nearest training embeddings]
    D --> E[Class-balanced, distance-weighted vote]
    E --> F{Confidence >= 0.764?}
    F -- yes --> G[Auto-decide benign or melanoma]
    F -- no --> H[Refer to specialist]
```

### Headline result

On the held-out test split (3,553 images, 814 melanomas), the Siamese model reaches **0.820 accuracy and 0.909 AUROC**, meeting the task's accuracy target, but the baseline is better on every discrimination and confidence metric (0.838 accuracy, 0.933 AUROC, ECE 0.023 vs 0.065, AURC 0.052 vs 0.062). The Siamese confidence is **not** a better indicator of correctness as 42% of its test predictions have confidence exactly 1.0, and these include confidently missed melanomas. The baseline's reject rule meets the safety target as melanoma recall among auto-decided cases 0.958 with 34% referred. On the other hand, the Siamese rule refers fewer cases (26%) but misses the target (0.921). Overall, Both models perform far worse on the realistic `IP` subset of real patients. The recommendation is not to advance the Siamese model as the triage engine.

---

## 2. Feasibility Review

**Project:** Siamese metric-learning classifier with a reject option for melanoma triage  
**Dataset:** [ISIC Melanoma Dataset, resized to 224x224 (Kaggle)](https://www.kaggle.com/datasets/roua91/isic-melanoma-dataset-resized-to-224x224)

### 2.1 User Need, Scope and Acceptance Criteria

**Intended user.** A dermatology triage clinician who reviews dermoscopic images and must decide which lesions need urgent specialist attention. The prototype is a decision-support tool.

**Prototype purpose.** Given a dermoscopic image, the system returns (a) a benign/melanoma decision, (b) a confidence score derived from embedding distance, and (c) the nearest reference images as visual evidence. Cases below a validated confidence threshold are referred to a human specialist instead of being auto-decided.

The central question is: *does the Siamese model's distance-based confidence track diagnostic correctness better than a conventional CNN's softmax confidence, and does that justify its extra cost?* A missed melanoma (false negative) is treated as far more costly than a false alarm.

**Scope.**

*In scope:*
- Binary classification (benign vs. melanoma) of single 224x224 dermoscopic images from the Kaggle ISIC dataset, using only the image and its label. Patient metadata such as age, sex and lesion site is not used as a model input.
- A Siamese network (shared ResNet-18 encoder, contrastive loss) and a conventional CNN baseline, trained and evaluated on the same patient-grouped train/validation/test split.
- Distance-based confidence, a reject (refer-to-specialist) rule chosen on validation data, and a calibration and uncertainty comparison against the baseline.
- Profiling of computational cost (parameters, peak GPU memory, inference latency), a failure-case autopsy of 3 to 5 errors, and a written deployment recommendation.

*Out of scope:*
- Clinical validation or regulatory approval. Results are a research prototype on retrospective public data, not evidence of real-world diagnostic safety.
- Multi-class lesion diagnosis (e.g. basal cell carcinoma or keratosis subtypes), lesion segmentation, and full-resolution images.
- Screening-prevalence estimates. The dataset's ~21.5% melanoma rate is artificially high, so precision and positive predictive value will not transfer to a real clinic (<2% malignant).
- Deployment engineering such as a user interface, integration with clinical records, or on-device inference.

**Acceptance criteria.** All are measured on a held-out test split that is only used once, after every choice has been fixed on validation data.

1. **Leakage-free data.** Train, validation and test splits share no `patient_id`. Exact duplicate images, found by file hash, are removed before splitting. Pairs are generated only within a split. The number of images with a true patient ID, a lesion-only ID, or no ID is reported, and the patient-level guarantee is claimed only for the true-patient subset. The split is saved to disk with a fixed seed so it can be reproduced.
2. **Discrimination.** The Siamese model reaches test accuracy of about **0.80**. It also reports per-class precision and recall, macro-F1, AUROC and PR-AUC, compared with the baseline on the same test images.
3. **Confidence.** For both models, report the AUROC of confidence for separating correct from incorrect predictions, the expected calibration error (ECE), and a risk-coverage curve. The baseline is temperature-scaled on validation data so the comparison is fair. The Siamese model passes this criterion if its confidence ranks its own errors at least as well as the calibrated baseline's.
4. **Safe reject rule.** A confidence threshold will be chosen on validation data only, such that **melanoma recall among auto-decided cases is ≥ 0.95**. On test, report the achieved melanoma recall and the **referral rate**, with a provisional target of ≤ 50% of cases referred.
5. **Resource constraint.** All training and evaluation run on a single laptop GPU (Apple M5 Pro, PyTorch MPS backend). Peak training GPU memory is **≤ 8 GiB** at batch size 32, measured by sampling `torch.mps.driver_allocated_memory()` after every training batch. A full training run takes **≤ 2 hours** per model. Per-image inference latency, including the k-nearest-neighbour search for the Siamese model, is **≤ 50 ms**. Parameter count, peak memory and latency are reported for both models, together with the hardware and batch size.

### 2.2 Model Choice and Course Concepts

**Dataset configuration.** According to the Kaggle data card, the dataset [7] merges ISIC 2018, 2019 and 2020 [6]. It contains 24,493 images resized to 224x224: 19,227 benign and 5,266 malignant, about 21.5% malignant after benign undersampling. Labels (`target`), image IDs (`isic_id`) and an optional `patient_id` are in `cleaned_metadata.csv`. The `patient_id` column holds three kinds of identifier (see 2.3): true ISIC 2020 patient IDs (`IP_*`), HAM10000 lesion IDs (`HAM_*`), which identify a lesion rather than a person, and placeholders (`dummy_*`) where no ID exists. This investigation will use a **70/15/15 train/validation/test split, stratified by class and grouped by `patient_id`** (`StratifiedGroupKFold`). Each `HAM_*` and `dummy_*` ID forms its own group. The split is also stratified by ID source (`IP` / `HAM` / `dummy`), because the three sources have very different melanoma rates. Augmentation will include flips, rotations, random resized crops and mild colour jitter, followed by ImageNet normalisation.

**Proposed model.**
- **Encoder:** an ImageNet-pretrained ResNet-18 with its classifier removed. A projection head (512 -> 256 -> 128, ReLU between the layers) produces an L2-normalised 128-dimensional embedding. 
- **Siamese training:** both images in a pair pass through the *same* encoder, which shares its weights. Training uses contrastive loss.
- **Pair generation:** new pairs are sampled every epoch, with equal numbers of same-class and different-class pairs. Malignant anchors are oversampled so the loss is not dominated by benign-benign pairs.
- **Inference:** the *training* split is embedded once to form a reference bank. A query is classified by distance-weighted k-nearest neighbours. Its confidence is the weighted vote fraction for the predicted class. Test images are never used as references.
- **Baseline:** the same pretrained ResNet-18 with a two-class softmax head, trained with class-weighted cross-entropy on the identical split and augmentation. Sharing the backbone isolates the effect of *metric learning versus direct classification* from the effect of the architecture. The baseline is temperature-scaled before the uncertainty comparison.

**Course concepts and literature.** The design draws on convolutional feature extraction and transfer learning [2], metric and contrastive learning [1, 3], handling class imbalance through sampling and class weights, calibration [4] and selective classification [5].

Siamese networks suit this problem because the decision becomes a *comparison against confirmed reference lesions*, which is close to how clinicians reason by example. Distance to those references gives a natural, inspectable signal of ambiguity. Whether that signal is actually better calibrated than softmax confidence is the hypothesis this project tests, not an assumption it makes.

### 2.3 Preliminary Feasibility Evidence

> **Status at the midpoint:** the data audit, the split, smoke tests of both models, peak-memory measurements and one full baseline training run had been completed. All results below were on the validation split; the test split had not been used.

| Check | Result |
|---|---|
| Local size / image format | 399 MB on disk; all 24,493 files are 224x224 JPEGs (6 to 38 KB each) |
| Images found vs metadata rows | 24,493 / 24,493; no missing files, no orphan files, no repeated `isic_id` |
| Class counts | benign 19,227, malignant 5,266 (21.5%); matches the data card |
| `IP_*`: true patient IDs (ISIC 2020) | 11,186 images (45.7%) from 1,919 patients (1 to 43 images each, median 4); 372 patients have both benign and malignant lesions |
| `HAM_*`: HAM10000 lesion IDs, not patient IDs | 2,513 images (10.3%), one image per ID, so grouping by them gives no patient-level protection |
| `dummy_*`: placeholders, no ID | 10,794 images (44.1%), using 10,514 distinct placeholders; 280 placeholders are shared by two images, and 130 of these pair a benign with a malignant image, so they are unlikely to be real patients |
| Exact duplicate images | 0; the 919 `ISIC_*_downsampled` files have no un-suffixed twin |
| Malignant rate by ID source | `dummy`: **37.7%** (4,071 / 10,794); `HAM`: **24.4%** (614 / 2,513); `IP`: **5.2%** (581 / 11,186) |
| Share of all melanomas by ID source | `dummy` 77.3%, `HAM` 11.7%, `IP` 11.0% |
| Melanomas in the older `isic_id` range (< `ISIC_0100000`) | **89.0%** (4,685 / 5,266), versus 45.0% of benign images |
| Patient-grouped split (seed 42, stratified by class x ID source) | train 17,253 (70.4%) / val 3,687 (15.1%) / test 3,553 (14.5%); malignant 21.2% / 21.5% / 22.9%; `IP`/`HAM`/`dummy` 46.0/10.2/43.9% (train), 46.0/10.1/43.9% (val), 43.9/10.9/45.3% (test). Exact balance is limited because `IP` patients (up to 43 images each) cannot be divided between splits. |
| Patient overlap after split | **0** `patient_id`s in more than one split |
| Baseline model | Pretrained ResNet-18 + `Linear(512->2)`: 11,177,538 trainable parameters |
| Siamese model | Same pretrained ResNet-18 + projection head (512->256->128, L2-normalised): 11,340,736 trainable parameters (+1.5%); output shapes, unit-length embeddings, weight sharing and contrastive-loss values checked on synthetic inputs |
| Baseline smoke test (1,000 train / 500 val images, 2 epochs, batch 32, Apple M5 Pro GPU via MPS) | runs end to end; val AUROC 0.873 -> 0.874, val loss 0.523 -> 0.437; about 40 s per epoch; total 1.3 min. Pipeline check only, not a performance result |
| Siamese smoke test (1,000 train pairs / 500 val images, 2 epochs, batch 32, k = 10, margin 1.0, same hardware) | runs end to end; contrastive loss 0.242 -> 0.226 (train), 0.226 -> 0.216 (val); kNN val AUROC 0.865 -> 0.852; about 47 s per epoch, including embedding the reference bank. Pipeline check only |
| Peak training GPU memory (batch 32, MPS, sampled after every batch) | baseline **1.19 GiB**; Siamese **2.20 GiB** (about 1.8x, since each training item is a pair of images). Both well under the 8 GiB limit |
| Baseline full training run (17,253 train / 3,687 val images, lr 1e-4, batch 32, patience 5) | best **val AUROC 0.927** at epoch 4 (val accuracy 0.851); early-stopped after 9 epochs; about 110 s per epoch; 17.1 min in total. Train loss kept falling (0.452 -> 0.318) while val loss rose after epoch 4, so early stopping on val AUROC is needed |

### 2.4 Risks, Budget and Fallback

**Key technical risks.**
1. **Patient leakage.** Only 11,186 images (45.7%, `IP_*`) carry a true patient ID. The other 13,307 have either a lesion ID (`HAM_*`) or no ID (`dummy_*`). The same patient, or the same lesion, may therefore appear in more than one split, or as re-encoded near-duplicates that an MD5 check cannot catch. *Mitigation:* use a `patient_id`-grouped split, so the `IP_*` subset is patient-disjoint and shared placeholders keep their images together. Run a perceptual-hash check for near-duplicates across splits. Claim the patient-level guarantee only for the `IP_*` subset, and state the remaining risk as a limitation.
2. **Source shortcut (confirmed by the audit).** 89% of melanomas lie in the older `isic_id` range. Melanoma rates differ sharply by ID source: 37.7% (`dummy`), 24.4% (`HAM`) and 5.2% (`IP`). The model could partly learn archive-specific appearance (camera, resolution, framing) instead of the pathology. *Mitigation:* stratify the split by class and ID source. Report every metric separately for the `IP`, `HAM` and `dummy` subsets. Check whether high-confidence errors cluster by source in the failure autopsy.
3. **Ineffective pair sampling.** Random pairs are mostly easy, so contrastive loss can plateau. *Mitigation:* oversample malignant anchors, select checkpoints by validation kNN AUROC rather than pair loss, and try triplet loss with batch-hard mining as an ablation.

**Computing budget.**
- Hardware: one Apple M5 Pro laptop GPU (MPS). Neither model needs the Rangpur cluster: the full baseline run took 17 minutes, and the Siamese model is expected to take roughly 3x as long per epoch.
- One smoke test for each model (done).
- About 3 full training runs per model, each ≤ 2 hours with early stopping (patience 3 to 5 epochs).
- One final evaluation on the test set.
- Total ≤ 12 hours of laptop GPU time.

**Fallback.**
- If the Siamese model stays clearly below 0.80 accuracy or AUROC: freeze the early ResNet stages, switch to triplet loss with hard-negative mining, or train a linear head on the frozen Siamese embeddings.
- If memory or runtime limits are exceeded: reduce the batch size or the number of pairs per epoch, use gradient accumulation, freeze the early backbone layers, or move training to the Rangpur cluster.

The split, the metrics and the reject-rule protocol stay fixed throughout, so any fallback can still answer the dilemma. A negative result, such as the Siamese confidence being *no better* than the calibrated baseline, is still a valid, evidence-based recommendation.

---

## 3. Repository Structure, Dependencies and Usage

### 3.1 Files

| File | Purpose |
|---|---|
| [`dataset.py`](dataset.py) | Loads the metadata, removes exact duplicates, builds and saves the patient-grouped split, checks patient overlap and cross-split near-duplicate, defines the augmentations, `LesionDataset` (single images) and `PairDataset` (pairs for contrastive training). |
| [`modules.py`](modules.py) | `BaselineClassifier`, `SiameseNet`, `ContrastiveLoss` and the shared ResNet-18 backbone. Running this alone performs a self-test. |
| [`train.py`](train.py) | Trains either model with early stopping, plots the curves, tunes the post-hoc settings on validation, tests once on the test split and saves every result. |
| [`metrics.py`](metrics.py) | Helper with measurement and plotting functions only: discrimination metrics, ECE, risk-coverage/AURC, single-image latency and the evaluation plots. |
| [`predict.py`](predict.py) | Example usage: loads the saved weights and tuned settings, classifies chosen or random test images, prints the results, draws the figures and runs the failure autopsy. |
| `images/` | Figures used in this README (copied from `outputs/`). |
| `data/`, `outputs/` | Dataset and generated files (weights, logs, JSON/CSV results, plots). Not committed (`.gitignore`). |

### 3.2 Dependencies

Tested with **Python 3.12.1** on macOS with an **Apple M5 Pro** GPU (PyTorch MPS backend). The code also runs on CUDA (e.g. Rangpur) or CPU. This is automatically chosen by `pick_device()` in `train.py`.

| Package | Version |
|---|---|
| torch | 2.8.0 |
| torchvision | 0.23.0 |
| numpy | 2.2.6 |
| pandas | 2.3.2 |
| scikit-learn | 1.7.1 |
| pillow | 11.2.1 |
| matplotlib | 3.10.6 |

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.3 Data

Download the [Kaggle dataset](https://www.kaggle.com/datasets/roua91/isic-melanoma-dataset-resized-to-224x224) and place it inside the `data/` folder so that the folder looks like this :

```
data/
  cleaned_metadata.csv
  combined_images/ISIC_0000000.jpg ...
```

### 3.4 Running the pipeline

```bash
python dataset.py  # creates data/splits.csv (seed 42) and prints the data audit
python modules.py  # model self-test: shapes, unit-length embeddings, weight sharing, loss values
python train.py --model baseline # train, tune on validation, test once, save to outputs/
python train.py --model siamese # same for the Siamese network; also draws the joint test plots
python predict.py --model siamese  # 6 random test images with their nearest training images
python predict.py --model baseline --images ISIC_0033837 ISIC_0066921
python predict.py --model siamese --autopsy  # 3 most confident missed melanomas + 2 false alarms
python predict.py --model siamese --show_full_test # saved results on the full test split
```

Useful extra options: `train.py --model siamese --skip_training` re-tunes and re-tests a saved checkpoint, `--test_only` re-tests it with its saved settings, and `--epochs`, `--batch_size`, `--patience`, `--margin`, `--seed` override the defaults.

**Outputs** (all in `outputs/`): `<model>_best.pt` (weights), `<model>_history.json` and `<model>_curves.png` (training), `<model>_tuned.json` (settings chosen on validation), `<model>_test_results.json` (all test metrics, overall and per ID source), `<model>_test_predictions.csv` (one row per test image), `siamese_k_tuning.png`, `roc_pr_curves.png`, `reliability_diagram.png`, `risk_coverage.png`, `siamese_bank.pt` (cached reference bank) and the `predict_*` / `autopsy_*` figures.

### 3.5 Reproducibility

- **Fixed split:** the split is created once with seed 42 and saved to `data/splits.csv`; every script reuses it. Deleting it and re-running `dataset.py` regenerated a byte-for-byte identical file.
- **Seeds:** `torch.manual_seed(0)` for training; pairs are drawn from a per-item seeded generator (`seed, epoch, index`), so they do not depend on the number of DataLoader workers; the random demo images use a fixed seed.
- **Verified:** for the final run, `outputs/` was emptied and the split, both trainings, tuning and testing were run from scratch. The test metrics matched the previous full run to three decimal places for both models. GPU kernels on MPS/CUDA are not guaranteed to be bit-for-bit deterministic, so small differences are possible on other hardware.
- **Checks on the inference path:** `predict.py` reproduces the test-step probabilities for the same images (maximum difference 3e-8), and the cached reference bank is invalidated automatically when the checkpoint changes.

## 4. Data, Pre-processing and Splits

### 4.1 Data audit (from `python dataset.py`)

| Check | Result |
|---|---|
| Images / metadata rows | 24,493 / 24,493 (19,227 benign, 5,266 melanoma) |
| Exact duplicates (MD5 of file bytes) | **0** removed |
| Patient IDs in more than one split | **0** |
| Near-duplicate images across splits (16x16 difference hash [10]) | **0** groups |

With the common 8x8 difference hash, 37 cross-split groups (101 images) matched. Visual inspection showed these were *different* lesions with the same coarse layout (a small central lesion on plain skin, or the black circle of a dermatoscope), and 15 groups even mixed benign and melanoma images. A 16x16 (256-bit) hash removes these false matches and finds no duplicates, so no near-duplicate leakage was detected.

### 4.2 Pre-processing

The Kaggle images are already resized to 224x224, the input size of ResNet-18, so no resizing is needed at evaluation time.

- **All splits:** conversion to a tensor and normalisation with the ImageNet mean and standard deviation, matching the statistics the pretrained backbone was trained with [2].
- **Training only (augmentation):** random resized crop (scale 0.8 to 1.0, aspect 0.9 to 1.1), horizontal and vertical flips, rotation up to 30 degrees, and mild colour jitter (brightness and contrast 0.2, saturation 0.1, hue 0.02). Dermoscopic lesions have no canonical orientation, so flips and rotations add label-preserving variety; geometric and mild colour augmentation are among the most effective augmentations for skin lesion analysis [11]. Colour jitter is kept mild because colour (blue-white veil, multiple browns) is diagnostically meaningful for melanoma.
- **No hair removal, cropping of the dermatoscope frame or colour constancy** were applied, so that the failure autopsy could test whether the models rely on such artefacts (Section 8).

### 4.3 Training, validation and test splits

| Split | Images | Melanoma rate | ID groups | IP / HAM / dummy |
|---|---|---|---|---|
| Train | 17,253 (70.4%) | 21.2% | 10,463 | 46.0 / 10.2 / 43.9% |
| Validation | 3,687 (15.1%) | 21.5% | 2,240 | 46.0 / 10.1 / 43.9% |
| Test | 3,553 (14.5%) | 22.9% (814) | 2,243 | 43.9 / 10.9 / 45.3% |

---

## 5. Method

### 5.1 Models

| | Baseline | Siamese |
|---|---|---|
| Backbone | ImageNet-pretrained ResNet-18, final layer removed (512 features) | same |
| Head | `Linear(512 -> 2)` | `Linear(512 -> 256) -> ReLU -> Linear(256 -> 128)`, L2-normalised |
| Training input | single images | image pairs, both through the same weights |
| Loss | cross-entropy with inverse-frequency class weights [0.63, 2.36] | contrastive loss, margin 1.0 |
| Prediction | softmax of logits / T | distance-weighted kNN vote over the training bank |
| Trainable parameters | 11,177,538 | 11,340,736 |

**Contrastive loss** [3] for a pair with embedding distance d and label s (1 = same class):

L = s * d^2 + (1 - s) * max(0, m - d)^2, with margin m = 1.0, averaged over the batch.

Embeddings have length 1, so distances lie in [0, 2] and the margin of 1.0 is meaningful across images.

**Pair sampling.** Each epoch draws 17,253 fresh pairs. Each pair first picks the anchor class with probability 0.5, then a same-class or different-class partner with probability 0.5, so the four pair types (melanoma-melanoma, melanoma-benign, benign-benign, benign-melanoma) each make up about 25% of an epoch. This oversamples melanoma anchors (21% of images, 50% of anchors) and balances the loss.

**Training.** AdamW, learning rate 1e-4, batch size 32, up to 20 epochs, early stopping when the validation AUROC has not improved for 5 epochs; the checkpoint with the best validation AUROC is kept. For the Siamese model, validation AUROC is computed by embedding the training bank and classifying validation images with a 5-nearest-neighbour vote, i.e. the way the model is used, rather than from the pair loss.

### 5.2 Confidence, calibration and the reject rule

- **Melanoma probability p.** Baseline: softmax of the logits divided by a temperature T. Siamese: among the k nearest training embeddings, each neighbour votes with weight 1/distance; p is the weighted share of melanoma votes.
- **Class-balanced vote.** The training bank is 79% benign, so in a plain vote an ambiguous lesion is surrounded mostly by benign neighbours simply because there are more of them. Optionally, each melanoma neighbour's vote is multiplied by n_benign / n_melanoma = 3.71, the kNN analogue of the baseline's class-weighted loss.
- **Confidence** is the probability of the predicted class, max(p, 1 - p), in [0.5, 1]; the decision threshold for the class is 0.5.
- **Temperature scaling** [4]: T is fitted on validation logits by minimising the negative log-likelihood (LBFGS on log T). It is applied to the baseline only; the Siamese vote has no logits, and with two classes a monotone rescaling cannot change the ranking of its confidences anyway.
- **Reject rule.** Auto-decide a case if its confidence is at least t, otherwise refer it. t is the *smallest* threshold for which melanoma recall among the auto-decided validation cases is at least 0.95, i.e. the one that automates the most cases while meeting the safety target. If no threshold reaches the target, every case is referred.

### 5.3 What is tuned on validation, and how

| Setting | Model | Candidates | Selection rule (validation only) | Chosen |
|---|---|---|---|---|
| Best epoch | both | every epoch | highest validation AUROC | baseline epoch 7, Siamese epoch 2 |
| Temperature T | baseline | continuous | minimum negative log-likelihood | 1.126 |
| k and vote | Siamese | k in {3, 5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200} x {plain, balanced} | lowest referral rate that meets the 95% recall target, ties broken by AURC | k = 25, balanced |
| Reject threshold t | both | every distinct confidence | smallest t with auto-decided melanoma recall ≥ 0.95 | baseline 0.773, Siamese 0.764 |

The k and vote are chosen by the operational goal (refer as few cases as possible while staying safe) rather than by AURC. AURC alone would always pick the plain vote, which can never meet the recall target (Figure 3).

![k tuning on validation](images/siamese_k_tuning.png)

*Figure 3. Siamese tuning on validation. Left: AURC (lower is better). Right: share of cases referred at the 95% recall target. The plain vote ranks errors better (lower AURC) but has to refer every case at every k; the balanced vote makes automation possible from k = 15 upwards.*

| Vote | k | Val AURC | Val AUROC | Val melanoma recall | Val referred |
|---|---|---|---|---|---|
| plain | 5 (training) | 0.057 | 0.895 | 0.704 | 100% |
| plain | 150 (best AURC) | 0.032 | 0.919 | 0.679 | 100% |
| balanced | 15 | 0.057 | 0.916 | 0.876 | 28.2% |
| **balanced** | **25 (chosen)** | **0.058** | **0.919** | **0.880** | **24.8%** |
| balanced | 100 | 0.057 | 0.919 | 0.880 | 32.2% |

The vote weight changes p monotonically, so it never changes AUROC; it moves cases across the 0.5 decision boundary, which raises melanoma recall from about 0.70 to 0.88 at the cost of more false alarms (hence the higher AURC).

### 5.4 Metrics

- **Discrimination:** accuracy, per-class precision and recall, macro-F1, AUROC, PR-AUC (average precision) and the confusion matrix, overall and per ID source.
- **Confidence:** AUROC of confidence for separating correct from wrong predictions; expected calibration error with 15 bins [4, 9]; risk-coverage curve and its area AURC [5]. In AURC, tied confidences are given the error rate of their whole tied group, because no threshold can split a tie; this matters for the Siamese model, which has many cases at exactly 1.0.
- **Reject rule:** referral rate, melanoma recall and accuracy among auto-decided cases, melanomas missed by automation, and system sensitivity if every referred case is diagnosed correctly by the specialist.
- **Resources:** trainable parameters, peak training GPU memory, training time and median single-image latency (batch size 1, after warm-up, including the kNN search over all 17,253 bank embeddings).

---

## 6. Results: Benchmarking Against the Baseline

All numbers are on the same held-out test split of **3,553 images (814 melanomas)**, with every setting frozen beforehand on validation.

### 6.1 Training curves

| ![Baseline training curves](images/baseline_curves.png) | ![Siamese training curves](images/siamese_curves.png) |
|---|---|
| *Baseline: weighted cross-entropy and validation AUROC. Best validation AUROC 0.928 at epoch 7; early-stopped after 12 epochs.* | *Siamese: contrastive loss and validation kNN AUROC (k = 5). Best 0.895 at epoch 2; early-stopped after 7 epochs.* |

In both models the training loss keeps falling while the validation loss stops improving after a few epochs, so early stopping on validation AUROC is needed. The Siamese validation AUROC plateaus early: after epoch 2 the encoder separates the training pairs better (pair loss 0.195 to 0.161) without improving the neighbourhoods of validation images.

### 6.2 Discrimination

| Metric (test) | Baseline | Siamese |
|---|---|---|
| Accuracy | **0.838** | 0.820 |
| AUROC | **0.933** | 0.909 |
| PR-AUC | **0.829** | 0.781 |
| Macro-F1 | **0.799** | 0.779 |
| Precision / recall: benign | 0.954 / **0.830** | 0.946 / 0.813 |
| Precision / recall: melanoma | **0.602** / **0.866** | 0.573 / 0.845 |
| Confusion matrix [[TN, FP], [FN, TP]] | [[2273, 466], [109, 705]] | [[2226, 513], [126, 688]] |

![ROC and precision-recall curves](images/roc_pr_curves.png)

*Figure 4. ROC and precision-recall curves on the test split. The baseline dominates at almost every operating point.*

### 6.3 Confidence and calibration

| Metric (test) | Baseline (T = 1.126) | Siamese (k = 25, balanced) |
|---|---|---|
| Confidence AUROC (correct vs wrong), higher is better | **0.810** | 0.803 |
| ECE, lower is better | **0.023** | 0.065 |
| AURC, lower is better | **0.052** | 0.062 |
| Test cases with confidence exactly 1.0 | 0% | **42.3%** |
| Errors among cases with confidence ≥ 0.99 | 1 of 402 | **21 of 1,512** |

| ![Reliability diagram](images/reliability_diagram.png) | ![Risk-coverage curves](images/risk_coverage.png) |
|---|---|
| *Figure 5. Reliability diagram. The baseline follows the diagonal; the Siamese vote is irregular and mostly overconfident.* | *Figure 6. Risk-coverage. The Siamese curve starts at about 1.4% error rather than 0: its most confident block of tied cases already contains errors.* |
---

## References

1] G. Koch, R. Zemel, and R. Salakhutdinov, "Siamese neural networks for one-shot image recognition," in *ICML Deep Learning Workshop*, 2015.  
[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. CVPR*, 2016.  
[3] R. Hadsell, S. Chopra, and Y. LeCun, "Dimensionality reduction by learning an invariant mapping," in *Proc. CVPR*, 2006.  
[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017.  
[5] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Proc. NeurIPS*, 2017.  
[6] V. Rotemberg *et al.*, "A patient-centric dataset of images and metadata for identifying melanomas using clinical context," *Scientific Data*, vol. 8, 2021.  
[7] Kaggle, "ISIC Melanoma Dataset (resized to 224x224)," roua91. CC BY-NC 4.0.
[8] H. Sulakhe, "siamesenetworks-pytorch: loss.py," GitHub repository. [Online]. Available: https://github.com/HarshSulakhe/siamesenetworks-pytorch/blob/master/loss.py (accessed Oct. 21, 2026). Basis of the contrastive loss implementation.
[9] "Expected calibration error explained," jev wiki, GitHub. [Online]. Available: https://github.com/feder-cr/jev/wiki/expected-calibration-error-explained
[10] N. Krawetz, "Kind of like that," *The Hacker Factor Blog*, 2013. Difference hash used for the near-duplicate check.
[11] F. Perez, C. Vasconcelos, S. Avila, and E. Valle, "Data augmentation for skin lesion analysis," in *ISIC Skin Image Analysis Workshop, MICCAI*, 2018.
## AI Declaration:
1. AI was used to add print statements to test and demonstrate that models are working as intentded before training and evaluation is completed. I did not know how I could check this with mock data so I turned to AI and it explained and help me add the prints at the end of `modules.py`
2. Ai was asked for suggestion on how to layout the plan i had in mind. It suggested me to add extra file `evaluate.py` and also recommended some  functions i should implement such as tyemperature scaling and aroc etc however, eventually my mental model did not align with the structure it recommendded so i refactored the entire codebase into how I had structured in my mind
3. Used AI's assistance to enhance the performance of KNN for begign as during autospy i found that the confidence is very low because of the imbalanced class labels. I provided the issue and the context of what i am building and AI gave me example code and I adapted that into mt training.
4. Inspiration for Autopsy and psuedocode for that was receieved from AI