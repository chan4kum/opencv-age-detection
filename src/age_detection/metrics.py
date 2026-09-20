"""Prometheus metrics. One registry per process (scale out with pods, not worker processes)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

HTTP_REQUESTS = Counter("ad_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_DURATION = Histogram(
    "ad_http_request_duration_seconds", "HTTP request latency", ["method", "route"], buckets=_LATENCY_BUCKETS
)
HTTP_IN_FLIGHT = Gauge("ad_http_requests_in_flight", "HTTP requests currently being served")

INFERENCE_DURATION = Histogram(
    "ad_inference_duration_seconds",
    "Model inference latency (pre-process + ONNX Runtime + post-process)",
    ["source"],
    buckets=_LATENCY_BUCKETS,
)
FACES_PER_IMAGE = Histogram(
    "ad_faces_per_image", "Number of faces detected per image", buckets=(0, 1, 2, 3, 5, 8, 13, 21, 50, 100, 500)
)
INFERENCE_REJECTED = Counter(
    "ad_inference_rejected_total", "Requests shed because inference capacity was exhausted", ["reason"]
)
AGE_PREDICTIONS = Counter("ad_age_predictions_total", "Age-group predictions", ["group", "uncertain"])
IMAGES_REJECTED = Counter("ad_images_rejected_total", "Inputs rejected before inference", ["reason"])

JOBS_SUBMITTED = Counter("ad_jobs_submitted_total", "Async jobs accepted")
JOBS_PROCESSED = Counter("ad_jobs_processed_total", "Async jobs finished by workers", ["outcome"])
JOB_E2E_DURATION = Histogram(
    "ad_job_end_to_end_seconds",
    "Time from job submission to completion",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300),
)
WORKER_IN_FLIGHT = Gauge("ad_worker_jobs_in_flight", "Jobs currently being processed by this worker")

BUILD_INFO = Gauge("ad_build_info", "Build information", ["version", "model_sha256", "provider"])
