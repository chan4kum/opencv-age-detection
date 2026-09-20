from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from age_detection.age import (
    AGE_GROUPS,
    GROUP_RANGES,
    INPUT_SIZE,
    VIEWS,
    AgeClassifier,
    ModelIntegrityError,
    preprocess,
    softmax,
    square_crop,
)
from age_detection.config import DEFAULT_AGE_MODEL_SHA256
from age_detection.detector import YuNetDetector
from tests.conftest import AGE_MODEL, DATA, MODEL, make_settings


@pytest.fixture(scope="module")
def classifier() -> AgeClassifier:
    return AgeClassifier(AGE_MODEL, expected_sha256=DEFAULT_AGE_MODEL_SHA256)


@pytest.fixture(scope="module")
def face_and_image() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    det = YuNetDetector(MODEL, expected_sha256=make_settings().model_sha256, max_side=1024)
    img = cv2.imread(str(DATA / "astronaut.jpg"))
    f = det.detect(img)[0]
    return img, (f.x, f.y, f.width, f.height)


def test_groups_and_ranges_are_consistent() -> None:
    assert len(AGE_GROUPS) == len(GROUP_RANGES) == 8
    for label, (lo, hi) in zip(AGE_GROUPS, GROUP_RANGES, strict=True):
        assert label == f"{lo}-{hi}" and lo <= hi
    assert [r[0] for r in GROUP_RANGES] == sorted(r[0] for r in GROUP_RANGES)


class TestSquareCrop:
    def test_output_is_always_input_size(self) -> None:
        img = np.zeros((300, 400, 3), np.uint8)
        for box in [(100, 100, 50, 80), (0, 0, 40, 40), (350, 250, 60, 60), (-20, -20, 60, 60), (10, 10, 500, 500)]:
            assert square_crop(img, box, 1.3).shape == (INPUT_SIZE, INPUT_SIZE, 3)

    def test_crop_is_centred_on_the_face(self) -> None:
        img = np.zeros((400, 400, 3), np.uint8)
        img[180:220, 180:220] = 255  # 40x40 white square centred at (200, 200)
        crop = square_crop(img, (180, 180, 40, 40), 2.0)
        ys, xs = np.where(crop[:, :, 0] == 255)
        assert abs(xs.mean() - INPUT_SIZE / 2) < 4 and abs(ys.mean() - INPUT_SIZE / 2) < 4

    def test_out_of_image_area_is_edge_replicated_not_black(self) -> None:
        img = np.full((100, 100, 3), 200, np.uint8)
        crop = square_crop(img, (0, 0, 40, 40), 2.0)  # extends past the top-left corner
        assert crop.min() == 200

    def test_larger_margin_shows_more_context(self) -> None:
        img = np.zeros((400, 400, 3), np.uint8)
        img[190:210, 190:210] = 255
        tight = (square_crop(img, (190, 190, 20, 20), 1.0)[:, :, 0] == 255).mean()
        loose = (square_crop(img, (190, 190, 20, 20), 2.0)[:, :, 0] == 255).mean()
        assert tight > loose


def test_preprocess_layout_and_mean_subtraction() -> None:
    crop = np.full((INPUT_SIZE, INPUT_SIZE, 3), (104, 117, 123), np.uint8)  # exactly the mean (BGR)
    x = preprocess(crop)
    assert x.shape == (1, 3, INPUT_SIZE, INPUT_SIZE) and x.dtype == np.float32 and float(np.abs(x).max()) == 0.0
    ramp = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
    ramp[:, :, 0] = 200  # blue channel only
    assert float(preprocess(ramp)[0, 0, 0, 0]) == pytest.approx(200 - 104)


def test_softmax_is_a_distribution_and_stable() -> None:
    p = softmax(np.array([1000.0, 1001.0, 999.0], np.float32))
    assert p.sum() == pytest.approx(1.0) and np.isfinite(p).all() and p.argmax() == 1


class TestClassifier:
    def test_real_face_returns_a_valid_estimate(
        self, classifier: AgeClassifier, face_and_image: tuple[np.ndarray, tuple[float, float, float, float]]
    ) -> None:
        img, box = face_and_image
        e = classifier.classify_face(img, box)
        assert (
            e.group in AGE_GROUPS
            and e.views == 5
            and (e.age_low, e.age_high) == GROUP_RANGES[AGE_GROUPS.index(e.group)]
        )
        assert sum(e.probabilities) == pytest.approx(1.0, abs=1e-4) and 0.0 <= e.agreement <= 1.0
        assert e.confidence == pytest.approx(max(e.probabilities))

    def test_is_deterministic(
        self, classifier: AgeClassifier, face_and_image: tuple[np.ndarray, tuple[float, float, float, float]]
    ) -> None:
        img, box = face_and_image
        assert classifier.classify_face(img, box) == classifier.classify_face(img, box)

    def test_single_view_matches_classify_crop(
        self, face_and_image: tuple[np.ndarray, tuple[float, float, float, float]]
    ) -> None:
        img, box = face_and_image
        one = AgeClassifier(AGE_MODEL, expected_sha256=DEFAULT_AGE_MODEL_SHA256, views=1)
        assert one.classify_face(img, box) == one.classify_crop(square_crop(img, box, one.crop_margin))

    def test_blur_lowers_view_agreement_or_confidence(
        self, classifier: AgeClassifier, face_and_image: tuple[np.ndarray, tuple[float, float, float, float]]
    ) -> None:
        """The network's softmax is saturated (~1.0 even when wrong), so multi-view agreement is the uncertainty signal."""
        img, box = face_and_image
        sharp = classifier.classify_face(img, box)
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        soft = classifier.classify_face(blurred, box)
        assert (soft.agreement, soft.confidence) < (sharp.agreement, sharp.confidence) or soft.group != sharp.group

    def test_uncertain_flag_follows_the_threshold(
        self, face_and_image: tuple[np.ndarray, tuple[float, float, float, float]]
    ) -> None:
        img, box = face_and_image
        never = AgeClassifier(AGE_MODEL, expected_sha256=DEFAULT_AGE_MODEL_SHA256, min_confidence=0.0)
        always = AgeClassifier(AGE_MODEL, expected_sha256=DEFAULT_AGE_MODEL_SHA256, min_confidence=1.0)
        assert never.classify_face(img, box).uncertain is False
        assert always.classify_face(cv2.GaussianBlur(img, (11, 11), 0), box).uncertain is True

    def test_aggregation_math(self, classifier: AgeClassifier) -> None:
        a = np.array([0, 0, 0, 0, 1, 0, 0, 0], np.float32)
        b = np.array([0, 0, 0, 0, 0, 1, 0, 0], np.float32)
        est = classifier._aggregate([a, a, a, b, b])
        assert est.group == "25-32" and est.confidence == pytest.approx(0.6) and est.agreement == pytest.approx(0.6)
        assert est.uncertain is False  # 0.6 >= default threshold 0.6
        split = classifier._aggregate([a, b])
        assert split.confidence == pytest.approx(0.5) and split.uncertain is True

    @pytest.mark.parametrize("views", [0, len(VIEWS) + 1])
    def test_views_out_of_range_rejected(self, views: int) -> None:
        with pytest.raises(ValueError, match="views must be between"):
            AgeClassifier(AGE_MODEL, expected_sha256=DEFAULT_AGE_MODEL_SHA256, views=views)


class TestModelIntegrity:
    def test_missing_model_gives_actionable_error(self, tmp_path: Path) -> None:
        with pytest.raises(ModelIntegrityError, match="fetch_models"):
            AgeClassifier(tmp_path / "nope.onnx", expected_sha256=None)

    def test_tampered_model_is_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "m.onnx"
        bad.write_bytes(AGE_MODEL.read_bytes() + b"\x00")
        with pytest.raises(ModelIntegrityError, match="checksum mismatch"):
            AgeClassifier(bad, expected_sha256=DEFAULT_AGE_MODEL_SHA256)

    def test_pinned_checksum_matches_the_fetched_file(self) -> None:
        from age_detection.age import sha256_file

        assert sha256_file(AGE_MODEL) == DEFAULT_AGE_MODEL_SHA256
