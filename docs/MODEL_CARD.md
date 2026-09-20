# Model card: age-group estimation

This service estimates **age groups** from face images. Read this whole page before using it for anything that affects a person.

| | |
|---|---|
| **Pipeline** | YuNet face detector, then a Levi & Hassner GoogleNet classifier on each face crop (several crop scales and a mirrored view, aggregated) |
| **Age classifier** | `age_googlenet.onnx`: Levi & Hassner, *Age and Gender Classification Using Convolutional Neural Networks* (CVPR Workshops 2015), trained on the **Adience** benchmark, converted to ONNX by the [ONNX Model Zoo](https://github.com/onnx/models/tree/4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177/validated/vision/body_analysis/age_gender) |
| **Output** | One of 8 uneven age groups: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100, plus the probability per group, agreement between views, and an `uncertain` flag. **Never an exact age.** |
| **Face detector** | YuNet (MIT), vendored, see `models/YUNET_README.md` |

## License status of the age weights (important)

* The **ONNX Model Zoo repository is Apache-2.0**, but that licenses the repository, not necessarily the weights' provenance.
* The **original authors' page states only** "Copyright 2015, Gil Levi and Tal Hassner" and asks that you *cite their paper*. **No open-source license is stated**, and I found no explicit grant covering redistribution or commercial use.
* The weights were **trained on Adience**, which has its own terms (research benchmark; Flickr-sourced images).
* Therefore this repository **does not commit the weights**. `scripts/fetch_models.py` downloads them from the Model Zoo at a pinned commit and verifies a pinned SHA-256 (which matches the Model Zoo's own Git LFS record).
  The Docker image *does* contain them (fetched at build time), so the release workflow **does not publish images unless you set the repository variable `PUBLISH_IMAGE=true`** after your own review.
* If you need clear commercial rights, obtain permission from the authors or train/obtain a model with an explicit license, then replace the file and its pinned checksum. **This is not legal advice.**

Please cite: G. Levi and T. Hassner, "Age and Gender Classification Using Convolutional Neural Networks," IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG) at CVPR 2015.

## Reported accuracy (not reproduced here)

The authors report on Adience (as quoted by later papers; I could not open the original PDF from this environment): **about 49.5% exact-group accuracy with a single crop and 50.7% with over-sampling; about 84.6% / 84.7% within one group ("1-off")**.
That means the model is **wrong roughly one time in two** even in the lab, on a benchmark with 8 coarse groups. I have **not** evaluated it on labelled data; there is no age-labelled dataset in this repository, so no accuracy claim is made for this service.

## What I did observe

* **Overconfident probabilities.** The network's softmax saturates: on a clean portrait it reports 1.00 for `25-32`, and after a mild Gaussian blur it reports `38-43` with 0.99. Single-view "confidence" therefore **cannot** flag unreliable estimates.
* **Agreement between views is the uncertainty signal.** Classifying five views (crop scales 0.9 to 1.15x and mirrored) and averaging gave confidence 0.60 and 0.48 (agreement 0.6 and 0.4) on the blurred versions of the same face; the strongest blur was flagged `uncertain` at the default 0.6 threshold. This is a heuristic backed by four perturbed copies of one image, **not a calibrated probability**; a proper calibration needs labelled data.
* The prediction was stable across crop margins from 1.0 to 1.8 (same group), so the margin is not a fragile hyperparameter.
* Cost: about 43 ms per view in the Linux container on one ONNX thread (about 11 ms natively on an Apple M4 Pro), so 5 views is roughly 5x a single view.

## Known limitations

* Groups are **uneven and gapped** (4-6 then 8-12; nothing covers 3, 7, 13-14, 21-24, 33-37, 44-47, 54-59). The classifier still must pick one, so ages inside a gap are forced into a neighbouring group.
* Trained on unconstrained web photos of a limited demographic mix (Adience). Performance is expected to vary with **age, gender, skin tone, ethnicity, make-up, eyewear, expression, pose, image quality and lighting**. **Fairness has not been audited** here and the source paper does not report demographic breakdowns.
* Children and elderly faces, heavy make-up, filters/retouching, black-and-white photos, profile views and low resolution are all likely failure cases. Faces smaller than about 40 px are unreliable.
* It estimates *apparent* age from appearance, not chronological age.

## Intended and prohibited use

* **Intended:** aggregate analytics where errors average out and no individual is affected (for example rough audience composition in a consented study), demos, and education about CV pipelines.
* **Do not use** for age verification or gating (alcohol, tobacco, gambling, adult content, ID checks), hiring or employment, credit, insurance, law enforcement, immigration, safeguarding decisions, or **any decision with legal or similarly significant effect on a person**. At roughly 50% group accuracy it is unfit for those purposes even before considering fairness and legality.
* Do not run it on people without their knowledge or in contexts where notice and consent are required.

## Privacy

Face images and inferred age are personal data (and inferring attributes from faces can itself be regulated, e.g. GDPR, the EU AI Act's rules on biometric categorisation, BIPA). You need a lawful basis, notice and retention limits.
The service minimises retention: images for async jobs are deleted after processing, results contain geometry and probabilities but no pixels, and everything expires by TTL. It never writes images to logs, and it does not identify or re-identify people.
