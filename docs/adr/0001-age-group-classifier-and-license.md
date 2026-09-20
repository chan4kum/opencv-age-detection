# ADR 0001: Levi & Hassner age-group classifier; weights fetched, not committed

**Status:** accepted

**Context.** We need an age model that is small, runs on CPU with ONNX Runtime, and is honest about what it can do. Candidate sources were
the ONNX Model Zoo's Levi & Hassner GoogleNet (Adience, 8 groups), VGG-16 models trained on IMDB-WIKI (513 MB, research data), and
FairFace-based models (would require PyTorch conversion and add race prediction, which we do not want).
Age estimation models are almost universally trained on datasets with research-only or unclear terms.

**Decision.** Use the Levi & Hassner GoogleNet age classifier (24 MB) from the ONNX Model Zoo at a pinned commit, verified by SHA-256.
Do not commit the weights to git and do not publish images containing them by default.

**Why.** The original authors state only "Copyright 2015 Gil Levi and Tal Hassner, please cite our paper": no explicit license, and Adience has its
own terms. The Model Zoo repository is Apache-2.0 but that does not settle the weights' provenance. Fetching by pinned checksum keeps the
repository clean and reproducible while leaving the redistribution decision with the operator (`PUBLISH_IMAGE` gate in the release workflow).

**Consequences.**
- (+) Small and fast; a clear, honest license statement; reproducible builds; supply-chain checksum pinning.
- (-) The build and CI need network access to GitHub; tests fetch the model on first run.
- (-) Only 8 coarse groups and roughly 50% exact accuracy in the original paper. This limits legitimate uses (see the model card).
- (-) If you need clear commercial rights you must replace the model.
