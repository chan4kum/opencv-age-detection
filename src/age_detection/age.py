"""Age-group classification on face crops (Levi & Hassner GoogleNet, Adience, via the ONNX Model Zoo).

The network classifies a face into one of eight *age groups*, not an exact age: the groups are uneven
and the model is far less accurate than the group boundaries suggest (see docs/MODEL_CARD.md).
The service therefore reports the group, its probability distribution and an ``uncertain`` flag,
and never fabricates a point estimate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

AGE_GROUPS: Final = ("0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100")
GROUP_RANGES: Final = ((0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100))
INPUT_SIZE: Final = 224
# Test-time-augmentation views as (crop-margin multiplier, mirrored). The first ``views`` entries are used.
# The network's softmax is saturated (it reports ~1.00 even when wrong), so agreement *between views* is the
# only usable uncertainty signal here. See docs/MODEL_CARD.md.
VIEWS: Final = ((1.0, False), (1.0, True), (0.9, False), (1.15, False), (1.15, True))
MEAN_BGR: Final = np.array([104.0, 117.0, 123.0], dtype=np.float32)


class ModelIntegrityError(RuntimeError):
    """The model file is missing or does not match the pinned checksum."""


@dataclass(frozen=True, slots=True)
class AgeEstimate:
    group: str
    age_low: int
    age_high: int
    confidence: float  # mean probability of the top group across views (about the share of agreeing views)
    agreement: float  # fraction of views whose own top group equals the aggregated top group
    views: int
    uncertain: bool  # confidence below the configured threshold
    probabilities: tuple[float, ...]  # mean over views, one per AGE_GROUPS entry, sums to 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def square_crop(
    image_bgr: NDArray[np.uint8], box_xywh: tuple[float, float, float, float], margin: float
) -> NDArray[np.uint8]:
    """Crop a square of side ``max(w, h) * margin`` centred on the face box, padding by edge replication
    where it extends beyond the image, then resize to the network input size."""
    x, y, w, h = box_xywh
    side = max(w, h) * margin
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, y0 = round(cx - side / 2.0), round(cy - side / 2.0)
    s = max(2, round(side))
    ih, iw = image_bgr.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + s - iw), max(0, y0 + s - ih)
    if pad_l or pad_t or pad_r or pad_b:
        image_bgr = cast(
            "NDArray[np.uint8]", cv2.copyMakeBorder(image_bgr, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        )
        x0, y0 = x0 + pad_l, y0 + pad_t
    crop = image_bgr[y0 : y0 + s, x0 : x0 + s]
    interp = cv2.INTER_AREA if s > INPUT_SIZE else cv2.INTER_LINEAR
    return cast("NDArray[np.uint8]", cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=interp))


def preprocess(crop_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
    """BGR, raw 0..255 minus the training mean, NCHW float32 (as specified by the ONNX Model Zoo)."""
    x = crop_bgr.astype(np.float32) - MEAN_BGR
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    e = np.exp(x - x.max())
    return cast("NDArray[np.float32]", (e / e.sum()).astype(np.float32))


class AgeClassifier:
    """Thread-safe (``InferenceSession.run`` is re-entrant) age-group classifier."""

    def __init__(
        self,
        model_path: Path,
        *,
        expected_sha256: str | None,
        min_confidence: float = 0.6,
        crop_margin: float = 1.3,
        views: int = 5,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
        providers: list[str] | None = None,
    ) -> None:
        if not model_path.is_file():
            raise ModelIntegrityError(
                f"age model not found: {model_path} (run `python scripts/fetch_models.py` or `make models`)"
            )
        if expected_sha256 is not None:
            actual = sha256_file(model_path)
            if actual != expected_sha256:
                raise ModelIntegrityError(
                    f"age model checksum mismatch for {model_path}: expected {expected_sha256}, got {actual}"
                )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = inter_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=providers or ["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self.min_confidence = min_confidence
        if not 1 <= views <= len(VIEWS):
            raise ValueError(f"views must be between 1 and {len(VIEWS)}")
        self.crop_margin = crop_margin
        self.views = views
        self.providers = self._session.get_providers()

    def warmup(self) -> None:
        self.classify_crop(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))

    def _probs(self, crop_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        raw = self._session.run(None, {self._input: preprocess(crop_bgr)})[0][0].astype(np.float32)
        return raw if abs(float(raw.sum()) - 1.0) < 1e-3 and float(raw.min()) >= 0.0 else softmax(raw)

    def classify_crop(self, crop_bgr: NDArray[np.uint8]) -> AgeEstimate:
        """Single-view classification of an already-cropped 224x224 face."""
        return self._aggregate([self._probs(crop_bgr)])

    def classify_face(self, image_bgr: NDArray[np.uint8], box_xywh: tuple[float, float, float, float]) -> AgeEstimate:
        """Classify a detected face over several crop scales / mirrored views and aggregate."""
        outs = []
        for scale, mirror in VIEWS[: self.views]:
            crop = square_crop(image_bgr, box_xywh, self.crop_margin * scale)
            outs.append(self._probs(cast("NDArray[np.uint8]", cv2.flip(crop, 1)) if mirror else crop))
        return self._aggregate(outs)

    def _aggregate(self, per_view: list[NDArray[np.float32]]) -> AgeEstimate:
        mean = np.mean(per_view, axis=0)
        mean = mean / mean.sum()
        top = int(mean.argmax())
        agree = sum(1 for p in per_view if int(p.argmax()) == top) / len(per_view)
        conf = float(mean[top])
        low, high = GROUP_RANGES[top]
        return AgeEstimate(
            group=AGE_GROUPS[top],
            age_low=low,
            age_high=high,
            confidence=conf,
            agreement=agree,
            views=len(per_view),
            uncertain=conf < self.min_confidence,
            probabilities=tuple(float(p) for p in mean),
        )
