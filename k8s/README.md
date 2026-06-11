# APEX Kubernetes Deployment

This directory contains the Helm chart for deploying APEX to any Kubernetes cluster.

## Chart Structure

```
k8s/helm/apex/
├── Chart.yaml               # Chart metadata (name, version, appVersion)
├── values.yaml              # Default values — override for your environment
└── templates/
    ├── deployment.yaml      # Main app deployment with startup/liveness/readiness probes
    ├── service.yaml         # ClusterIP service (port 8000)
    ├── ingress.yaml         # Optional ingress (disabled by default)
    ├── hpa.yaml             # HorizontalPodAutoscaler (2–10 replicas @ 70% CPU)
    ├── pdb.yaml             # PodDisruptionBudget (minAvailable: 1)
    ├── configmap.yaml       # Non-secret environment variables
    ├── servicemonitor.yaml  # Prometheus ServiceMonitor (disabled by default)
    └── NOTES.txt            # Post-install instructions
```

## Prerequisites

- Kubernetes 1.25+
- Helm 3.10+
- Container registry access (default: `ghcr.io/pavanbadempet/apex-backend`)
- A Kubernetes Secret named `apex-secrets` with sensitive keys (see below)

## Quick Start

### 1. Create the secrets

```bash
kubectl create secret generic apex-secrets \
  --from-literal=jwt-secret-key="$(openssl rand -hex 32)" \
  --from-literal=tmdb-api-key="your-tmdb-key" \
  --from-literal=admin-token="$(openssl rand -hex 16)"
```

### 2. Install the chart

```bash
# Tier 2 (ONNX CPU — default)
helm install apex ./k8s/helm/apex \
  --set secretRefs.jwtSecretKey=jwt-secret-key \
  --set secretRefs.tmdbApiKey=tmdb-api-key \
  --set secretRefs.adminToken=admin-token

# Tier 3 (FAISS only — minimal resources)
helm install apex ./k8s/helm/apex \
  --set servingTier=tier3 \
  --set servingProfile=lite \
  --set resources.requests.memory=512Mi \
  --set resources.limits.memory=1Gi
```

### 3. Verify deployment

```bash
kubectl get pods -l app.kubernetes.io/name=apex
kubectl port-forward svc/apex-apex 8000:8000
curl http://localhost:8000/health
```

## Tier Configuration

| Tier | `servingTier` | `servingProfile` | Memory | Use Case |
|---|---|---|---|---|
| **Tier 1** | `tier1` | `full` | ≥16Gi | GPU server, full 6-model ensemble + online learning |
| **Tier 2** | `tier2` | `full` | 4–8Gi | CPU server, ONNX quantized inference (default) |
| **Tier 3** | `tier3` | `lite` | 0.5–2Gi | Low-memory, FAISS + TF-IDF only |

### Upgrading to Tier 1 (GPU)

```bash
helm upgrade apex ./k8s/helm/apex \
  --set servingTier=tier1 \
  --set resources.requests.memory=16Gi \
  --set resources.limits.memory=24Gi \
  --set replicaCount=1
```

Add GPU resource requests if your cluster has GPU nodes:
```yaml
# values-gpu.yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

## Scaling

The chart ships with a HorizontalPodAutoscaler enabled by default:

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

Scale-up is aggressive (0s stabilization window, 2 pods/minute).
Scale-down is conservative (5-minute stabilization window, 1 pod/2 minutes).
This prevents recommendation latency spikes during traffic bursts.

## Probes

The deployment uses three probes tuned for APEX's warmup characteristics:

| Probe | Path | Purpose |
|---|---|---|
| **Startup** | `/health` | Allows up to ~130s for model artifact loading before liveness kicks in |
| **Liveness** | `/health` | Restarts containers that become unresponsive (PyTorch deadlock, OOM) |
| **Readiness** | `/v1/platform/readiness` | Removes pods from service until artifacts are loaded and validated |

The startup probe is critical — without it, the liveness probe would kill the container
during the 30–60s model warmup on cold start.

## Observability

### Prometheus metrics

APEX exposes metrics at `/metrics`. Enable the ServiceMonitor (requires prometheus-operator):

```bash
helm upgrade apex ./k8s/helm/apex \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.namespace=monitoring \
  --set serviceMonitor.labels.release=prometheus
```

### SLO endpoint

```bash
kubectl port-forward svc/apex-apex 8000:8000
curl http://localhost:8000/v1/platform/slo | jq .
```

Returns real-time p50/p95/p99 latency, error rates, and online learning coordinator status.

## Security

The deployment template enforces:
- `runAsNonRoot: true` (user 1000)
- `allowPrivilegeEscalation: false`
- `capabilities: drop: [ALL]`

All sensitive environment variables are sourced from Kubernetes Secrets, never
stored in the ConfigMap or values.yaml.

## Values Reference

See [`values.yaml`](helm/apex/values.yaml) for all configurable values with inline documentation.

Key values:

| Value | Default | Description |
|---|---|---|
| `replicaCount` | `2` | Pod replicas |
| `servingTier` | `tier2` | ML serving tier |
| `dpEpsilon` | `1.0` | Differential privacy budget |
| `autoscaling.enabled` | `true` | Enable HPA |
| `podDisruptionBudget.enabled` | `true` | Enable PDB |
| `serviceMonitor.enabled` | `false` | Enable Prometheus scraping |
| `terminationGracePeriodSeconds` | `60` | Graceful shutdown window |
