# Siamese Classifier for ISIC Melanoma Triage

## Feasibility Review

**Project:** Siamese metric-learning classifier with a reject option for melanoma triage  
**Dataset:** [ISIC Melanoma Dataset, resized to 224x224 (Kaggle)](https://www.kaggle.com/datasets/roua91/isic-melanoma-dataset-resized-to-224x224)

### 1. User Need, Scope and Acceptance Criteria

**Intended user.** A dermatology triage clinician who reviews dermoscopic images and must decide which lesions need urgent specialist attention. The prototype is a decision-support tool.

**Prototype purpose.** Given a dermoscopic image, the system returns (a) a benign/melanoma decision, (b) a confidence score derived from embedding distance, and (c) the nearest reference images as visual evidence. Cases below a validated confidence threshold are referred to a human specialist instead of being auto-decided. 

The central question is: does the Siamese model's distance-based confidence track diagnostic correctness better than a conventional CNN's softmax confidence, and does that justify its extra cost?* A missed melanoma (false negative) is treated as far more costly than a false alarm.

**Scope.**

*In scope:*
- Binary classification (benign vs. melanoma) of single 224x224 dermoscopic images from the Kaggle ISIC dataset, using only the image and its label. Patient metadata such as age, sex and lesion site is not used as a model input.
- A Siamese network (shared ResNet-18 encoder, contrastive loss) and a conventional CNN baseline, trained and evaluated on the same patient-grouped train/validation/test split.
- Distance-based confidence, a reject (refer-to-specialist) rule chosen on validation data, and a calibration and uncertainty comparison against the baseline.
- Profiling of computational cost (parameters, peak GPU memory, inference latency), a failure-case autopsy of 3–5 errors, and a written deployment recommendation.

*Out of scope:*
- Clinical validation or regulatory approval. Results are a research prototype on retrospective public data, not evidence of real-world diagnostic safety.
- Multi-class lesion diagnosis (e.g. basal cell carcinoma or keratosis subtypes), lesion segmentation, and full-resolution images.
- Screening-prevalence estimates. The dataset's ~21.5% melanoma rate is artificially high, so precision and positive predictive value will not transfer to a real clinic (<2% malignant).
- Deployment engineering such as a user interface, integration with clinical records, or on-device inference.

**Acceptance criteria.** All are measured on a held-out test split that is only used once, after every choice has been fixed on validation data.

1. **Leakage-free data.** Train, validation and test splits share no `patient_id`. Exact duplicate images, found by file hash, are removed before splitting. Pairs are generated only within a split. The number of images with a true patient ID, a lesion-only ID, or no ID is reported, and the patient-level guarantee is claimed only for the true-patient subset. The split is saved to disk with a fixed seed so it can be reproduced.

2. **Discrimination:** The Siamese model reaches test accuracy of about **0.80**. It also reports per-class precision and recall, macro-F1, AUROC and PR-AUC, compared with the baseline on the same test images.

3. **Confidence:** For both models, report the AUROC of confidence for separating correct from incorrect predictions, the expected calibration error (ECE), and a risk–coverage curve. The baseline is temperature-scaled on validation data so the comparison is fair. The Siamese model passes this criterion if its confidence ranks its own errors at least as well as the calibrated baseline's.

4. **Safe reject rule.** a confidence threshold will be chosen on validation data only, such that **melanoma recall among auto-decided cases is ≥ 0.95**. On test, report the achieved melanoma recall and the **referral rate**, with a provisional target of ≤ 50% of cases referred.

5. **Resource constraint.** All training and evaluation run on a single laptop GPU (Apple M5 Pro, PyTorch MPS backend). Peak training GPU memory is **≤ 8 GiB** at batch size 32, measured by sampling `torch.mps.driver_allocated_memory()` after every training batch. A full training run takes **≤ 2 hours** per model. Per-image inference latency, including the k-nearest-neighbour search for the Siamese model, is **≤ 50 ms**. Parameter count, peak memory and latency are reported for both models, together with the hardware and batch size.

### 2. Model Choice and Course Concepts

**Dataset configuration.** According to the Kaggle data card, the dataset [7] merges ISIC 2018, 2019 and 2020 [6]. It contains 24,493 images resized to 224x224: 19,227 benign and 5,266 malignant, about 21.5% malignant after benign undersampling. Labels (`target`), image IDs (`isic_id`) and an optional `patient_id` are in `cleaned_metadata.csv`. The `patient_id` column holds three kinds of identifier (see section 3): true ISIC 2020 patient IDs (`IP_*`), HAM10000 lesion IDs (`HAM_*`), which identify a lesion rather than a person, and placeholders (`dummy_*`) where no ID exists. This investigation will use a **70/15/15 train/validation/test split, stratified by class and grouped by `patient_id`** (`StratifiedGroupKFold`). Each `HAM_*` and `dummy_*` ID forms its own group. The split is also stratified by ID source (`IP` / `HAM` / `dummy`), because the three sources have very different melanoma rates. Augmentation will include flips, rotations, random resized crops and mild colour jitter, followed by ImageNet normalisation. 

**Proposed model.**
- **Encoder:** an ImageNet-pretrained ResNet-18 with its classifier removed. A projection head (512 -> 256 -> 128, ReLU between the layers) produces an L2-normalised 128-dimensional embedding. 
- **Siamese training:** both images in a pair pass through the *same* encoder, which shares its weights. Training uses contrastive loss
- **Pair generation:** new pairs are sampled every epoch, with equal numbers of same-class and different-class pairs. Malignant anchors are oversampled so the loss is not dominated by benign–benign pairs.
- **Inference:** the *training* split is embedded once to form a reference bank. A query is classified by distance-weighted k-nearest neighbours. Its confidence is the weighted vote fraction for the predicted class. Test images are never used as references.
- **Baseline:** the same pretrained ResNet-18 with a two-class softmax head, trained with class-weighted cross-entropy on the identical split and augmentation. Sharing the backbone isolates the effect of *metric learning versus direct classification* from the effect of the architecture. The baseline is temperature-scale before the uncertainty comparison.

**Course concepts and literature.** The design draws on the following:
- convolutional feature extraction and transfer learning;
- metric and contrastive learning;
- handling class imbalance through sampling;


Siamese networks suit this problem because the decision becomes a *comparison against confirmed reference lesions*, which is close to how clinicians reason by example. Distance to those references gives a natural, inspectable signal of ambiguity. Whether that signal is actually better calibrated than softmax confidence is the hypothesis this project tests, not an assumption it makes.

### 3. Preliminary Feasibility Evidence

> **Status:** The data audit, the split, smoke tests of both models, peak-memory measurements and one full baseline training run have been completed. All results below are on the validation split; the test split has not been used.

| Check | Result |
|---|---|
| Local size / image format | 399 MB on disk; all 24,493 files are 224x224 JPEGs (6–38 KB each) |
| Images found vs metadata rows | 24,493 / 24,493; no missing files, no orphan files, no repeated `isic_id` |
| Class counts | benign 19,227, malignant 5,266 (21.5%); matches the data card |
| `IP_*`: true patient IDs (ISIC 2020) | 11,186 images (45.7%) from 1,919 patients (1–43 images each, median 4); 372 patients have both benign and malignant lesions |
| `HAM_*`: HAM10000 lesion IDs, not patient IDs | 2,513 images (10.3%), one image per ID, so grouping by them gives no patient-level protection |
| `dummy_*`: placeholders, no ID | 10,794 images (44.1%), using 10,514 distinct placeholders; 280 placeholders are shared by two images, and 130 of these pair a benign with a malignant image, so they are unlikely to be real patients |
| Exact duplicate images | 0; the 919 `ISIC_*_downsampled` files have no un-suffixed twin |
| Malignant rate by ID source | `dummy`: **37.7%** (4,071 / 10,794); `HAM`: **24.4%** (614 / 2,513); `IP`: **5.2%** (581 / 11,186) |
| Share of all melanomas by ID source | `dummy` 77.3%, `HAM` 11.7%, `IP` 11.0% |
| Melanomas in the older `isic_id` range (< `ISIC_0100000`) | **89.0%** (4,685 / 5,266), versus 45.0% of benign images |
| Patient-grouped split (seed 42, stratified by class × ID source) | train 17,253 (70.4%) / val 3,687 (15.1%) / test 3,553 (14.5%); malignant 21.2% / 21.5% / 22.9%; `IP`/`HAM`/`dummy` 46.0/10.2/43.9% (train), 46.0/10.1/43.9% (val), 43.9/10.9/45.3% (test). Exact balance is limited because `IP` patients (up to 43 images each) cannot be divided between splits. |
| Patient overlap after split | **0** `patient_id`s in more than one split |
| Baseline model | Pretrained ResNet-18 + `Linear(512→2)`: 11,177,538 trainable parameters |
| Siamese model | Same pretrained ResNet-18 + projection head (512→256→128, L2-normalised): 11,340,736 trainable parameters (+1.5%); output shapes, unit-length embeddings, weight sharing and contrastive-loss values checked on synthetic inputs |
| Baseline smoke test (1,000 train / 500 val images, 2 epochs, batch 32, Apple M5 Pro GPU via MPS) | runs end to end; val AUROC 0.873 → 0.874, val loss 0.523 → 0.437; about 40 s per epoch; total 1.3 min. Pipeline check only, not a performance result |
| Siamese smoke test (1,000 train pairs / 500 val images, 2 epochs, batch 32, k = 10, margin 1.0, same hardware) | runs end to end; contrastive loss 0.242 → 0.226 (train), 0.226 → 0.216 (val); kNN val AUROC 0.865 → 0.852; about 47 s per epoch, including embedding the reference bank. Pipeline check only |
| Peak training GPU memory (batch 32, MPS, sampled after every batch) | baseline **1.19 GiB**; Siamese **2.20 GiB** (≈1.8×, since each training item is a pair of images). Both well under the 8 GiB limit |
| Baseline full training run (17,253 train / 3,687 val images, lr 1e-4, batch 32, patience 5) | best **val AUROC 0.927** at epoch 4 (val accuracy 0.851); early-stopped after 9 epochs; about 110 s per epoch; 17.1 min in total. Train loss kept falling (0.452 → 0.318) while val loss rose after epoch 4, so early stopping on val AUROC is needed |

### 4. Risks, Budget and Fallback

**Key technical risks.**
1. **Patient leakage.** Only 11,186 images (45.7%, `IP_*`) carry a true patient ID. The other 13,307 have either a lesion ID (`HAM_*`) or no ID (`dummy_*`). The same patient, or the same lesion, may therefore appear in more than one split, or as re-encoded near-duplicates that an MD5 check cannot catch. *Mitigation:* use a `patient_id`-grouped split, so the `IP_*` subset is patient-disjoint and shared placeholders keep their images together. Run a perceptual-hash check for near-duplicates across splits. Claim the patient-level guarantee only for the `IP_*` subset, and state the remaining risk as a limitation.
2. **Source shortcut (confirmed by the audit).** 89% of melanomas lie in the older `isic_id` range. Melanoma rates differ sharply by ID source: 37.7% (`dummy`), 24.4% (`HAM`) and 5.2% (`IP`). The model could partly learn archive-specific appearance (camera, resolution, framing) instead of the pathology. *Mitigation:* stratify the split by class and ID source. Report every metric separately for the `IP`, `HAM` and `dummy` subsets. Check whether high-confidence errors cluster by source in the failure autopsy.
3. **Ineffective pair sampling.** Random pairs are mostly easy, so contrastive loss can plateau. *Mitigation:* oversample malignant anchors, select checkpoints by validation kNN AUROC rather than pair loss, and try triplet loss with batch-hard mining as an ablation.


**Computing budget.**
- Hardware: one Apple M5 Pro laptop GPU (MPS). Neither model needs the Rangpur cluster: the full baseline run took 17 minutes, and the Siamese model is expected to take roughly 3× as long per epoch.
- One smoke test for each model (done).
- About 3 full training runs per model, each ≤ 2 hours with early stopping (patience 3–5 epochs).
- One final evaluation on the test set.
- Total ≤ 12 hours of laptop GPU time.


**Fallback.**
- If the Siamese model stays clearly below 0.80 accuracy or AUROC: freeze the early ResNet stages, switch to triplet loss with hard-negative mining, or train a linear head on the frozen Siamese embeddings.
- If memory or runtime limits are exceeded: reduce the batch size or the number of pairs per epoch, use gradient accumulation, freeze the early backbone layers, or move training to the Rangpur cluster.

The split, the metrics and the reject-rule protocol stay fixed throughout, so any fallback can still answer the dilemma. A negative result, such as the Siamese confidence being *no better* than the calibrated baseline, is still a valid, evidence-based recommendation.

### References

[1] G. Koch, R. Zemel, and R. Salakhutdinov, "Siamese neural networks for one-shot image recognition," in *ICML Deep Learning Workshop*, 2015.  
[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. CVPR*, 2016.  
[3] R. Hadsell, S. Chopra, and Y. LeCun, "Dimensionality reduction by learning an invariant mapping," in *Proc. CVPR*, 2006.  
[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017.  
[5] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Proc. NeurIPS*, 2017.  
[6] V. Rotemberg *et al.*, "A patient-centric dataset of images and metadata for identifying melanomas using clinical context," *Scientific Data*, vol. 8, 2021.  
[7] Kaggle, "ISIC Melanoma Dataset (resized to 224x224)," roua91. CC BY-NC 4.0.
[8] H. Sulakhe, "siamesenetworks-pytorch: loss.py," GitHub repository. [Online]. Available: https://github.com/HarshSulakhe/siamesenetworks-pytorch/blob/master/loss.py (accessed Oct. 21, 2026).
[9] https://github.com/feder-cr/jev/wiki/expected-calibration-error-explained

## AI Declaration:
1. AI was used to add print statements to test and demonstrate that models are working as intentded before training and evaluation is completed. I did not know how I could check this with mock data so I turned to AI and it explained and help me add the prints at the end of `modules.py`
2. Ai was asked for suggestion on how to layout the plan i had in mind. It suggested me to add extra file `evaluate.py` and also recommended some  functions i should implement such as tyemperature scaling and aroc etc however, eventually my mental model did not align with the structure it recommendded so i refactored the entire codebase into how I had structured in my mind