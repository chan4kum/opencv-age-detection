"""Generate the Grafana dashboard JSON (kept as code so panels stay reviewable)."""

import json
import sys

DS = {"type": "prometheus", "uid": "prometheus"}


def panel(
    pid: int,
    title: str,
    exprs: list[tuple[str, str]],
    x: int,
    y: int,
    unit: str = "short",
    w: int = 12,
    h: int = 8,
    desc: str = "",
) -> dict:
    return {
        "id": pid,
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 8}}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
        "targets": [
            {"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": legend}
            for i, (e, legend) in enumerate(exprs)
        ],
    }


def q(p: float, metric: str, by: str = "", extra: str = "") -> str:
    return f"histogram_quantile({p}, sum by (le{by}) (rate({metric}_bucket{extra}[$__rate_interval])))"


panels = [
    panel(
        1,
        "Request rate by route",
        [
            (
                'sum by (route) (rate(ad_http_requests_total{route!~"/healthz|/readyz|/metrics"}[$__rate_interval]))',
                "{{route}}",
            )
        ],
        0,
        0,
        "reqps",
    ),
    panel(
        2,
        "Error ratio (5xx)",
        [
            (
                'sum(rate(ad_http_requests_total{status=~"5.."}[$__rate_interval])) / clamp_min(sum(rate(ad_http_requests_total[$__rate_interval])), 1e-9)',
                "5xx ratio",
            )
        ],
        12,
        0,
        "percentunit",
    ),
    panel(
        3,
        "Latency: /v1/estimate",
        [
            (q(0.5, "ad_http_request_duration_seconds", extra='{route="/v1/estimate"}'), "p50"),
            (q(0.95, "ad_http_request_duration_seconds", extra='{route="/v1/estimate"}'), "p95"),
            (q(0.99, "ad_http_request_duration_seconds", extra='{route="/v1/estimate"}'), "p99"),
        ],
        0,
        8,
        "s",
    ),
    panel(
        4,
        "Model inference latency",
        [
            (q(0.5, "ad_inference_duration_seconds", ",source"), "p50 {{source}}"),
            (q(0.95, "ad_inference_duration_seconds", ",source"), "p95 {{source}}"),
        ],
        12,
        8,
        "s",
    ),
    panel(
        5,
        "In-flight HTTP requests / worker jobs",
        [
            ("sum(ad_http_requests_in_flight)", "http in flight"),
            ("sum(ad_worker_jobs_in_flight)", "worker jobs in flight"),
        ],
        0,
        16,
    ),
    panel(
        6,
        "Load shedding & rejected inputs",
        [
            ("sum by (reason) (rate(ad_inference_rejected_total[$__rate_interval]))", "shed: {{reason}}"),
            ("sum by (reason) (rate(ad_images_rejected_total[$__rate_interval]))", "rejected: {{reason}}"),
        ],
        12,
        16,
        "ops",
    ),
    panel(
        7,
        "Async jobs by outcome",
        [
            ("sum by (outcome) (rate(ad_jobs_processed_total[$__rate_interval]))", "{{outcome}}"),
            ("sum(rate(ad_jobs_submitted_total[$__rate_interval]))", "submitted"),
        ],
        0,
        24,
        "ops",
    ),
    panel(
        8,
        "Job end-to-end latency (submit to done)",
        [(q(0.5, "ad_job_end_to_end_seconds"), "p50"), (q(0.95, "ad_job_end_to_end_seconds"), "p95")],
        12,
        24,
        "s",
    ),
    panel(
        9,
        "Queue depth (pending jobs)",
        [
            ('max(nats_consumer_num_pending{consumer_name="ad-workers"})', "pending"),
            ('max(nats_consumer_num_ack_pending{consumer_name="ad-workers"})', "ack pending"),
        ],
        0,
        32,
        desc="Requires the NATS Prometheus exporter.",
    ),
    panel(
        10,
        "Faces per image (p50 / p95)",
        [(q(0.5, "ad_faces_per_image"), "p50"), (q(0.95, "ad_faces_per_image"), "p95")],
        12,
        32,
    ),
    panel(
        11,
        "Age-group predictions (rate)",
        [("sum by (group) (rate(ad_age_predictions_total[$__rate_interval]))", "{{group}}")],
        0,
        40,
        "ops",
    ),
    panel(
        12,
        "Share of uncertain estimates",
        [
            (
                'sum(rate(ad_age_predictions_total{uncertain="true"}[$__rate_interval])) / clamp_min(sum(rate(ad_age_predictions_total[$__rate_interval])), 1e-9)',
                "uncertain share",
            )
        ],
        12,
        40,
        "percentunit",
        desc="Estimates whose views disagree (mean probability below the threshold). A rising share usually means worse input quality.",
    ),
]
dash = {
    "uid": "age-detection",
    "title": "Age Detection Service",
    "schemaVersion": 39,
    "version": 1,
    "editable": True,
    "refresh": "10s",
    "time": {"from": "now-1h", "to": "now"},
    "tags": ["age-detection"],
    "panels": panels,
}
json.dump(dash, sys.stdout, indent=2)
print()
