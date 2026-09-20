from __future__ import annotations

import json
from pathlib import Path

import pytest

from age_detection.cli import main
from age_detection.config import hash_api_key
from tests.conftest import AGE_MODEL, DATA, MODEL


def test_estimate_prints_json_and_writes_annotated_image(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.jpg"
    assert main(["estimate", str(DATA / "two_faces.jpg"), "-o", str(out), "--views", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["faces"]) == 2 and out.stat().st_size > 1000
    assert payload["faces"][0]["age"]["views"] == 1 and payload["faces"][0]["age"]["group"]


def test_estimate_defaults_to_the_configured_views(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["estimate", str(DATA / "astronaut.jpg")]) == 0
    assert json.loads(capsys.readouterr().out)["faces"][0]["age"]["views"] == 5


def test_estimate_no_face_prints_empty_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["estimate", str(DATA / "noise.png")]) == 0
    assert json.loads(capsys.readouterr().out)["faces"] == []


def test_estimate_missing_file_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["estimate", "/no/such/file.jpg"]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_estimate_invalid_image_exit_code(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"garbage")
    assert main(["estimate", str(bad)]) == 2


def test_estimate_unwritable_output_fails_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["estimate", str(DATA / "astronaut.jpg"), "-o", str(tmp_path / "nodir" / "x.jpg")])
    assert rc == 2 and "could not write" in capsys.readouterr().err


def test_keygen_hash_matches_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["keygen"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert hash_api_key(lines[0].split()[-1]) == lines[1].split()[-1]


def test_verify_models_ok_and_reports_each_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify-models"]) == 0
    out = capsys.readouterr().out
    assert "face detector" in out and "age classifier" in out and out.count("OK") == 2


def test_verify_models_missing_age_model_points_at_the_fetch_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AD_AGE_MODEL_PATH", str(tmp_path / "missing.onnx"))
    assert main(["verify-models"]) == 2
    assert "make models" in capsys.readouterr().err


def test_tampered_age_model_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "age.onnx"
    bad.write_bytes(AGE_MODEL.read_bytes() + b"\x00")
    monkeypatch.setenv("AD_AGE_MODEL_PATH", str(bad))
    assert main(["estimate", str(DATA / "astronaut.jpg")]) == 3
    assert "checksum mismatch" in capsys.readouterr().err
    assert MODEL.is_file()


class TestInitStorage:
    def test_creates_bucket_and_lifecycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import boto3
        from moto import mock_aws

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "t")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "t")
        monkeypatch.setenv("AD_S3_BUCKET", "cli-bucket")
        with mock_aws():
            assert main(["init-storage", "--expire-days", "3"]) == 0
            rules = boto3.client("s3", region_name="us-east-1").get_bucket_lifecycle_configuration(Bucket="cli-bucket")
            assert rules["Rules"][0]["Expiration"]["Days"] == 3

    def test_unreachable_storage_exits_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AD_S3_ENDPOINT_URL", "http://127.0.0.1:1")
        monkeypatch.setenv("AD_S3_ACCESS_KEY_ID", "a")
        monkeypatch.setenv("AD_S3_SECRET_ACCESS_KEY", "b")
        assert main(["init-storage"]) == 1


class TestWebcam:
    """Fake camera: no real device and no macOS camera-permission prompt."""

    def test_camera_unavailable(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        import cv2

        class Closed:
            def __init__(self, *_: object) -> None: ...
            def isOpened(self) -> bool:
                return False

            def release(self) -> None: ...

        monkeypatch.setattr(cv2, "VideoCapture", Closed)
        assert main(["webcam"]) == 2
        assert "cannot open camera" in capsys.readouterr().err

    def test_stream_end_is_clean_exit_and_labels_faces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cv2

        frames = [cv2.imread(str(DATA / "astronaut.jpg"))]
        labels: list[str] = []

        class Cam:
            def __init__(self, *_: object) -> None: ...
            def isOpened(self) -> bool:
                return True

            def read(self) -> tuple[bool, object]:
                return (True, frames.pop()) if frames else (False, None)

            def release(self) -> None: ...

        monkeypatch.setattr(cv2, "VideoCapture", Cam)
        monkeypatch.setattr(cv2, "putText", lambda img, text, *a, **k: labels.append(text))
        monkeypatch.setattr(cv2, "imshow", lambda *_: None)
        monkeypatch.setattr(cv2, "waitKey", lambda *_: 0)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        assert main(["webcam"]) == 0 and len(labels) == 1

    def test_headless_opencv_gives_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import cv2
        import numpy as np

        class Cam:
            def __init__(self, *_: object) -> None: ...
            def isOpened(self) -> bool:
                return True

            def read(self) -> tuple[bool, object]:
                return True, np.zeros((64, 64, 3), np.uint8)

            def release(self) -> None: ...

        def no_gui(*_: object) -> None:
            raise cv2.error("The function is not implemented (imshow)")

        monkeypatch.setattr(cv2, "VideoCapture", Cam)
        monkeypatch.setattr(cv2, "imshow", no_gui)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        assert main(["webcam"]) == 2
        assert "opencv-python-headless" in capsys.readouterr().err
