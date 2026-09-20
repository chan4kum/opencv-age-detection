"""Environment-driven configuration (12-factor). Every setting is prefixed with ``AD_``."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# SHA-256 of the pinned upstream model (opencv_zoo face_detection_yunet_2026may.onnx, MIT).
DEFAULT_MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"

# SHA-256 of the pinned age classifier (ONNX Model Zoo age_googlenet.onnx; see scripts/fetch_models.py).
DEFAULT_AGE_MODEL_SHA256 = "fa2a3228e425056aa2b080b3afd3cf607327c86616e952602ed67b5fc16ab356"

ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP", "BMP")


class Settings(BaseSettings):
    """Validated runtime configuration. Invalid values fail fast at start-up."""

    model_config = SettingsConfigDict(env_prefix="AD_", extra="ignore", frozen=True)

    # --- general -----------------------------------------------------------------
    environment: Literal["dev", "test", "prod"] = "dev"
    service_name: str = "age-detection"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = True
    docs_enabled: bool = True  # serve /docs, /redoc and /openapi.json

    # --- model / inference -------------------------------------------------------
    model_path: Path = Path("models/face_detection_yunet_2026may.onnx")
    model_sha256: str = DEFAULT_MODEL_SHA256  # the YuNet face detector
    age_model_path: Path = Path("models/age_googlenet.onnx")
    age_model_sha256: str = DEFAULT_AGE_MODEL_SHA256
    age_min_confidence: float = Field(0.6, ge=0.0, le=1.0)  # below this a face is flagged `uncertain`
    age_crop_margin: float = Field(1.3, ge=1.0, le=3.0)
    age_views: int = Field(5, ge=1, le=5)  # test-time augmentation views; 1 = fastest, least reliable uncertainty
    score_threshold: float = Field(0.7, ge=0.0, le=1.0)
    nms_threshold: float = Field(0.3, ge=0.0, le=1.0)
    max_faces: int = Field(20, ge=1, le=200)  # each face costs up to ~57 ms with the default 5 views
    # Longest image side fed to the network. Larger images are down-scaled (aspect ratio kept).
    inference_max_side: int = Field(960, ge=64, le=4096)
    ort_intra_op_threads: int = Field(1, ge=0, le=64)  # 0 = ONNX Runtime default
    ort_inter_op_threads: int = Field(1, ge=0, le=64)

    # --- request limits (defence against oversized / decompression-bomb inputs) ----
    max_upload_bytes: int = Field(10 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(40_000_000, ge=1024)
    max_concurrent_inference: int = Field(4, ge=1, le=256)
    inference_queue_timeout_s: float = Field(2.0, gt=0)

    # --- security ----------------------------------------------------------------
    # Comma-separated SHA-256 hex digests of accepted API keys (never store raw keys).
    api_key_hashes: Annotated[frozenset[str], NoDecode] = frozenset()
    auth_disabled: bool = False
    cors_allow_origins: Annotated[tuple[str, ...], NoDecode] = ()

    # --- async pipeline (NATS JetStream + S3-compatible object storage) ----------
    async_enabled: bool = False
    nats_url: str = "nats://localhost:4222"
    nats_provision: bool = True  # create stream/KV/consumer if missing
    nats_replicas: int = Field(1, ge=1, le=5)  # use 3 against a 3-node NATS cluster
    nats_stream: str = "AD_JOBS"
    nats_subject: str = "ad.jobs.estimate"
    nats_consumer: str = "ad-workers"
    nats_kv_bucket: str = "ad_jobs"
    job_ttl_s: int = Field(3600, ge=60)
    job_max_deliver: int = Field(4, ge=1, le=20)
    job_ack_wait_s: int = Field(30, ge=5)
    job_retry_backoff_s: float = Field(2.0, ge=0)
    worker_concurrency: int = Field(4, ge=1, le=256)

    s3_bucket: str = "age-detection"
    s3_endpoint_url: AnyUrl | None = None  # unset = AWS S3; set for MinIO/SeaweedFS/Ceph/etc.
    s3_region: str = "us-east-1"
    s3_access_key_id: SecretStr | None = None  # unset = default credential chain (IRSA, env, ...)
    s3_secret_access_key: SecretStr | None = None
    s3_force_path_style: bool = True
    s3_server_side_encryption: Literal["AES256", "aws:kms"] | None = None
    delete_input_after_processing: bool = True  # privacy by default: do not retain uploaded images

    @field_validator("api_key_hashes", mode="before")
    @classmethod
    def _split_hashes(cls, v: object) -> object:
        if isinstance(v, str):
            return frozenset(h.strip().lower() for h in v.split(",") if h.strip())
        return v

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return tuple(o.strip() for o in v.split(",") if o.strip())
        return v

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        for h in self.api_key_hashes:
            if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("AD_API_KEY_HASHES must contain 64-char lowercase SHA-256 hex digests")
        if self.environment == "prod" and not self.api_key_hashes and not self.auth_disabled:
            raise ValueError(
                "AD_ENVIRONMENT=prod requires AD_API_KEY_HASHES (or an explicit AD_AUTH_DISABLED=true "
                "when authentication is enforced upstream, e.g. by a service mesh or gateway)"
            )
        if (self.s3_access_key_id is None) != (self.s3_secret_access_key is None):
            raise ValueError("AD_S3_ACCESS_KEY_ID and AD_S3_SECRET_ACCESS_KEY must be set together")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def hash_api_key(key: str) -> str:
    """SHA-256 hex digest used to store API keys."""
    return hashlib.sha256(key.encode()).hexdigest()
