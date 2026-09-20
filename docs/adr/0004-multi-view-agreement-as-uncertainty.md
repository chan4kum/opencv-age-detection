# ADR 0004: Multi-view agreement as the uncertainty signal

**Status:** accepted

**Context.** The network's softmax is saturated: clean crops give 1.00 for one group, and a mildly blurred crop gave a *different* group with 0.99.
Max-probability thresholds therefore cannot flag unreliable estimates, and calibrating (temperature scaling) needs labelled data we do not have.

**Decision.** Classify up to five views of each face (crop-margin multipliers 1.0, 1.0 mirrored, 0.9, 1.15, 1.15 mirrored), average the probabilities, and report
the mean top probability as `confidence`, the fraction of views that agree as `agreement`, and `uncertain = confidence < AD_AGE_MIN_CONFIDENCE` (default 0.6).
This is the same over-sampling idea the original paper used to gain accuracy (49.5% to 50.7%).

**Consequences.**
- (+) A usable, explainable uncertainty signal; measured on a blurred copy of one face it reduced confidence to 0.60 and 0.48 and flagged the strongest blur.
- (+) Small accuracy gain per the paper's over-sampling result (not reproduced here).
- (-) Latency and cost scale with the number of views (5 views is about 5x). `AD_AGE_VIEWS=1` restores speed but removes the signal.
- (-) It is a heuristic backed by perturbations of one image, not a calibrated probability. Real calibration requires labelled data.
