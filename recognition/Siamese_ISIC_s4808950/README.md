# Siamese Classifier for ISIC Melanoma Triage

## Feasibility Review

**Project:** Siamese metric-learning classifier with a reject option for melanoma triage  
**Dataset:** [ISIC Melanoma Dataset, resized to 224x224 (Kaggle)](https://www.kaggle.com/datasets/roua91/isic-melanoma-dataset-resized-to-224x224)

### 1. User Need, Scope and Acceptance Criteria

**Intended user.** A dermatology triage clinician who reviews dermoscopic images and must decide which lesions need urgent specialist attention. The prototype is a decision-support tool.

**Prototype purpose.** Given a dermoscopic image, the system returns (a) a benign/melanoma decision, (b) a confidence score derived from embedding distance, and (c) the nearest reference images as visual evidence. Cases below a validated confidence threshold are referred to a human specialist instead of being auto-decided. 

The central question is: *does the Siamese model's distance-based confidence track diagnostic correctness better than a conventional CNN's softmax confidence, and does that justify its extra cost?* A missed melanoma (false negative) is treated as far more costly than a false alarm.

**Scope.**

*In scope:*
- Binary classification (benign vs. melanoma) of single 224x224 dermoscopic images from the Kaggle ISIC dataset, using only the image and its label. Patient metadata such as age, sex and lesion site is not used as a model input.
- A Siamese network (shared ResNet-18 encoder, contrastive loss) and a conventional CNN baseline, trained and evaluated on the same patient-grouped train/validation/test split.
- Distance-based confidence, a reject (refer-to-specialist) rule chosen on validation data, and a calibration and uncertainty comparison against the baseline.
- Profiling of computational cost (parameters, peak VRAM, inference latency), a failure-case autopsy of 3–5 errors, and a written deployment recommendation.

*Out of scope:*
- Clinical validation or regulatory approval. Results are a research prototype on retrospective public data, not evidence of real-world diagnostic safety.
- Multi-class lesion diagnosis (e.g. basal cell carcinoma or keratosis subtypes), lesion segmentation, and full-resolution images.
- Screening-prevalence estimates. The dataset's ~21.5% melanoma rate is artificially high, so precision and positive predictive value will not transfer to a real clinic (<2% malignant).
- Deployment engineering such as a user interface, integration with clinical records, or on-device inference.

**Acceptance criteria.** All are measured on a held-out test split that is only used once, after every choice has been fixed on validation data.

1. **Leakage-free data.** Train, validation and test splits share no `patient_id`. Exact duplicate images, found by file hash, are removed before splitting. Pairs are generated only *within* a split. Images without a `patient_id` are counted and reported. The split is saved to disk with a fixed seed so it can be reproduced.

2. **Discrimination:** The Siamese model reaches test accuracy of about **0.80** according to the task sheet. It also reports per-class precision and recall, macro-F1, AUROC and PR-AUC, compared with the baseline on the same test images.

3. **Confidence:** For both models, report the AUROC of confidence for separating correct from incorrect predictions, the expected calibration error (ECE), and a risk–coverage curve. The baseline is temperature-scaled on validation data so the comparison is fair. The Siamese model passes this criterion if its confidence ranks its own errors at least as well as the calibrated baseline's.

4. **Safe reject rule.** a confidence threshold will be chosen on validation data only, such that **melanoma recall among auto-decided cases is ≥ 0.95**. On test, report the achieved melanoma recall and the **referral rate**, with a provisional target of ≤ 50% of cases referred.

5. **Resource constraint.** Peak training GPU memory is **≤ 8 GiB** at the chosen batch size with mixed precision. A full training run takes **≤ 4 GPU-hours** on Rangpur. Report per-image inference latency, including the k-nearest-neighbour search, together with the hardware and batch size. The provisional target is ≤ 50 ms per image.

### 2. Model Choice and Course Concepts

**Dataset configuration.** According to the Kaggle data card, the dataset [7] merges ISIC 2018, 2019 and 2020 [6]. It contains 24,493 images resized to 224x224: 19,227 benign and 5,266 malignant, about 21.5% malignant after benign undersampling. Labels (`target`), image IDs (`isic_id`) and an optional `patient_id` are in `cleaned_metadata.csv`. This investigation will use a **70/15/15 train/validation/test split, stratified by class and grouped by patient** (`StratifiedGroupKFold`). Each placeholder (`dummy_*`) ID is treated as its own group, and the split is also stratified by ID type (real or placeholder), because the audit shows the two subsets have very different melanoma rates. Augmentation will include flips, rotations, random resized crops and mild colour jitter, followed by ImageNet normalisation. 

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

> **Status:** The smoke-test rows are still to be run. Values marked *(estimate)* have not been measured yet.

| Check | Result |
|---|---|
| Local size / image format | 399 MB on disk; all 24,493 files are 224x224 JPEGs (6–38 KB each) |
| Images found vs metadata rows | 24,493 / 24,493; no missing files, no orphan files, no repeated `isic_id` |
| Class counts | benign 19,227, malignant 5,266 (21.5%); matches the data card |
| Images with a real `patient_id` | 13,699 images from 4,432 patients (1–43 images each, median 1); 372 patients have both benign and malignant lesions |
| Images with a placeholder `patient_id` (`dummy_*`) | 10,794 (44.1%), using 10,514 distinct placeholder IDs; 280 placeholders are shared by more than one image |
| Exact duplicate images | 0; the 919 `ISIC_*_downsampled` files have no un-suffixed twin |
| Malignant rate by `patient_id` type | placeholder ID: **37.7%** (4,071 / 10,794); real ID: **8.7%** (1,195 / 13,699) |
| Melanomas in the older `isic_id` range (< `ISIC_0100000`) | **89.0%** (4,685 / 5,266), versus 45.0% of benign images |
| Patient-grouped split (seed 42) | train 17,208 (70.3%) / val 3,649 (14.9%) / test 3,636 (14.8%); malignant 21.4% / 21.4% / 21.9%; placeholder IDs 43.9% / 44.6% / 44.5% |
| Patient overlap after split | **0** patients in more than one split |

### 4. Risks, Budget and Fallback

**Key technical risks.**
1. **Patient leakage.** 10,794 images (44%) have placeholder `dummy_*` IDs. The same lesion, or the same patient, may therefore appear in more than one split under different placeholders, or as re-encoded near-duplicates that an MD5 check cannot catch. *Mitigation:* use a patient-grouped split. Treat each placeholder ID as its own group, so the 280 shared placeholders keep their images together. Run a perceptual-hash check for near-duplicates across splits. Report the leakage guarantee only for the real-ID subset, and state the remaining risk as a limitation.
2. **Source shortcut (confirmed by the audit).** 89% of melanomas lie in the older `isic_id` range. Images with placeholder IDs are 37.7% malignant, compared with 8.7% for images with real IDs. The model could partly learn archive-specific appearance. *Mitigation:* stratify the split by both class and ID type. Report every metric separately for the real-ID (ISIC 2020-style) subset and the placeholder-ID subset. Check whether high-confidence errors cluster by source in the failure autopsy.
3. **Ineffective pair sampling.** Random pairs are mostly easy, so contrastive loss can plateau. *Mitigation:* oversample malignant anchors, select checkpoints by validation kNN AUROC rather than pair loss, and try triplet loss with batch-hard mining as an ablation.


**Computing budget.**
- One smoke test for each model.
- About 3 full training runs per model, each ≤ 4 GPU-hours with early stopping (patience of 5 epochs).
- One final evaluation on the test set.
- Total ≤ 25 GPU-hours on Rangpur.


**Fallback.**
- If the Siamese model stays clearly below 0.80 accuracy or AUROC: freeze the early ResNet stages, switch to triplet loss with hard-negative mining, or train a linear head on the frozen Siamese embeddings.
- If memory or runtime limits are exceeded: reduce the batch size, use gradient accumulation and AMP, or freeze the backbone.

The split, the metrics and the reject-rule protocol stay fixed throughout, so any fallback can still answer the dilemma. A negative result, such as the Siamese confidence being *no better* than the calibrated baseline, is still a valid, evidence-based recommendation.

### References

[1] G. Koch, R. Zemel, and R. Salakhutdinov, "Siamese neural networks for one-shot image recognition," in *ICML Deep Learning Workshop*, 2015.  
[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. CVPR*, 2016.  
[3] R. Hadsell, S. Chopra, and Y. LeCun, "Dimensionality reduction by learning an invariant mapping," in *Proc. CVPR*, 2006.  
[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017.  
[5] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Proc. NeurIPS*, 2017.  
[6] V. Rotemberg *et al.*, "A patient-centric dataset of images and metadata for identifying melanomas using clinical context," *Scientific Data*, vol. 8, 2021.  
[7] Kaggle, "ISIC Melanoma Dataset (resized to 224x224)," roua91. CC BY-NC 4.0.
