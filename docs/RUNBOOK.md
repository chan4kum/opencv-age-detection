# Runbook

Each alert in `deploy/prometheus/alerts.yml` links to a section here. Metric names are `ad_*` (see ARCHITECTURE.md).

## Quick triage

```bash
kubectl get pods -l app.kubernetes.io/name=age-detection
kubectl logs deploy/<release>-age-detection-api --tail=100 | jq -c 'select(.level!="info")'
kubectl logs deploy/<release>-age-detection-worker --tail=100 | jq -c .
curl -s localhost:8000/readyz            # which dependency is failing
```

Every response carries `X-Request-ID`; logs and traces are keyed by it (`request_id`, `trace_id`).

## High error rate
*Alert `AgeDetectionHighErrorRate`: more than 2% of requests are 5xx for 10 min.*

1. Split by status/route: `sum by (route,status) (rate(ad_http_requests_total{status=~"5.."}[5m]))`.
2. `503` with `overloaded` = capacity (see Load shedding). `503` with `dependency-unavailable` = NATS or S3 (see below).
3. `500` = bug: find the `unhandled_exception` log line with the request ID; open a bug with the trace.
4. Recently deployed? `helm rollback <release>`.

## High latency
*Alert `AgeDetectionHighLatency`: p95 of `/v1/estimate` above 500 ms for 10 min.*

1. Compare `ad_inference_duration_seconds` to `ad_http_request_duration_seconds`. If inference is fast but requests slow, the time is
   queueing (see Load shedding) or large uploads.
2. Very large images cost more (decode + down-scale). Check `image.width/height` span attributes.
3. CPU throttling? The chart sets no CPU limit on purpose; check node contention.

## Load shedding
*Alert `AgeDetectionLoadShedding`: requests are answered 503 `overloaded` (`ad_inference_rejected_total`).*

* Scale out: raise API `maxReplicas`, or lower the HPA CPU target.
* Or raise `config.maxConcurrentInference` together with the CPU request (one slot is roughly one core).
* Clients should honour `Retry-After` and use the async API for bursts.

## Jobs failing
*Alert `AgeDetectionJobsFailing`: `ad_jobs_processed_total{outcome="failed_exhausted"|"poison"}` is increasing.*

`failed_permanent` is normal (bad user input). `failed_exhausted` means an infrastructure fault persisted through all retries:

1. Worker logs: `job_retry` / `job_retries_exhausted` include the error.
2. Object storage reachable? `kubectl exec` a worker and run `python -c "from age_detection.storage import ObjectStore; ..."`, or check `/readyz` on the API.
3. NATS KV writable? `nats kv status ad_jobs` (nats-box).
4. `poison` messages: something other than this API published to the subject. Inspect with `nats stream view AD_JOBS`.

## Queue backlog
*Alert `AgeDetectionQueueBacklog`: more than 500 pending messages for 10 min.*

1. Are workers running and ready? `kubectl get pods -l app.kubernetes.io/component=worker`.
2. Enable KEDA (`worker.keda.enabled=true`) or raise `worker.autoscaling.maxReplicas`.
3. `nats consumer info AD_JOBS fd-workers`: high `Ack Pending` with no progress means stuck jobs; they are redelivered after `job_ack_wait_s`.
4. Do NOT purge the stream unless you accept losing those jobs (their inputs stay in the bucket until lifecycle expiry).

## Many uncertain estimates
*Alert `AgeDetectionManyUncertainEstimates`: more than half of estimates are flagged `uncertain` for 30 minutes.*

`uncertain` means the five views of a face disagree (mean probability below `config.ageMinConfidence`); it is the service's only quality signal because the network's own probabilities are saturated.

1. Did the input distribution change? Blurry, tiny (under about 40 px), heavily compressed or dark faces raise the share. Compare `ad_faces_per_image` and image sizes.
2. Did the model file change? `age-detection verify-models` must report both checksums OK.
3. Did `config.ageViews` drop to 1? With one view the uncertain share collapses to near zero and the alert can never fire (the signal is gone, not the problem).
4. This is a data-quality symptom, not an outage. Do not "fix" it by lowering the threshold; consumers rely on `uncertain` to know when not to trust a group.

## Instance down
Check `kubectl describe pod` (OOMKilled? raise the memory limit; probes failing?) and the startup log line `started`.
If the pod exits at start with `model checksum mismatch`, the image is corrupted or the model was replaced: rebuild from a clean checkout.
If startup hangs, the API waits for NATS/JetStream at start; check the NATS pods first.

## Common operations

| Task | How |
|---|---|
| Rotate an API key | `age-detection keygen`; add the new digest to `auth.apiKeyHashes`, roll clients, remove the old digest |
| Change thresholds | `config.scoreThreshold` etc. in values, `helm upgrade` (pods roll via config checksum) |
| Roll back | `helm rollback <release> <revision>` |
| Drain a worker | `kubectl delete pod` is safe: SIGTERM stops pulling, finishes in-flight jobs; anything unfinished is redelivered |
| Verify a model file | `age-detection verify-model --model path.onnx` |
