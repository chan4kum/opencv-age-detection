# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/), versioning: [SemVer](https://semver.org/).

## [Unreleased]

## [1.0.0]

### Changed
- Rebuilt from a Caffe/OpenCV webcam script into a production-grade service.

### Added
- YuNet face detection (ONNX Runtime) plus a Levi & Hassner age-group classifier with checksum-pinned models.
- Multi-view aggregation with `confidence`, `agreement` and an `uncertain` flag, because the network's softmax is saturated.
- `POST /v1/estimate`, `/v1/estimate/annotated`, async `/v1/jobs` on NATS JetStream + S3, health/readiness/metrics, RFC 9457 errors.
- API-key auth (hashed), request limits, load shedding, security headers.
- Prometheus metrics (including age-group and uncertain-share), OpenTelemetry tracing across API and worker, JSON logs, Grafana dashboard, alerts.
- Docker image (non-root, read-only rootfs), Compose stack, Helm chart, CI/CD, documentation, model card with license status and limitations.
- `scripts/fetch_models.py`: checksum-verified download of the age weights (not stored in git).
- Release workflow publishes images only when `PUBLISH_IMAGE=true`, because of the weights' license status.
