# APEX — Kubernetes Deployment

This guide covers deploying APEX to any Kubernetes cluster using the bundled Helm chart.

---

## Prerequisites

- `kubectl >= 1.28`
- `helm >= 3.x` (tested with 3.16)
- A running Kubernetes cluster (minikube, kind, EKS, GKE, AKS, etc.)

---

## Quick Install

```bash
helm install apex ./k8s/helm/apex \
  --namespace apex \
  --create-namespace
```

This deploys APEX with default values (Tier 2 — ONNX CPU, 1 replica, no ingress).

---

## Configure Serving Tier

Override the serving tier at install time using `--set`:

```bash
# Tier 2: ONNX CPU (default)
helm install apex ./k8s/helm/apex --set servingTier=tier2

# Tier 1: Full ensemble with GPU
helm install apex ./k8s/helm/apex --set servingTier=tier1

# Tier 3: Lite (FAISS + TF-IDF only)
helm install apex ./k8s/helm/apex --set servingTier=tier3
```

| Tier | Hardware Condition | Active Models | Typical Latency |
|---|---|---|---|
| **Tier 1** | GPU present + RAM ≥ 16 GB | Full 6-model ensemble + RL + Active Inference | 50–200 ms |
| **Tier 2** | No GPU + RAM ≥ 8 GB | ONNX-quantized ensemble | 200–800 ms |
| **Tier 3** | RAM < 8 GB | FAISS + TF-IDF only | 800–2000 ms |

---

## Set Secrets

### Option 1 — Via `--set` (simple, non-production)

```bash
helm install apex ./k8s/helm/apex \
  --set secretRefs.jwtSecretKey=jwt-secret-key
```

### Option 2 — Via a Kubernetes Secret (recommended for production)

First, create the secret in the cluster:

```bash
kubectl create secret generic apex-secrets \
  --from-literal=jwt-secret-key=<your-jwt-secret> \
  --from-literal=tmdb-api-key=<your-tmdb-key> \
  --namespace apex
```

Then reference the secret keys during install:

```bash
helm install apex ./k8s/helm/apex \
  --set secretRefs.jwtSecretKey=jwt-secret-key \
  --set secretRefs.tmdbApiKey=tmdb-api-key \
  --namespace apex
```

The deployment mounts each referenced key as an environment variable via `secretKeyRef`. Leaving a `secretRefs.*` value empty (`""`) skips mounting that secret.

---

## Upgrade

```bash
helm upgrade apex ./k8s/helm/apex --namespace apex
```

Pass additional `--set` flags to change values at upgrade time (e.g., `--set replicaCount=3`).

---

## Uninstall

```bash
helm uninstall apex --namespace apex
```

This removes all Kubernetes resources created by the chart. The namespace is not deleted automatically — remove it manually if no longer needed:

```bash
kubectl delete namespace apex
```

---

## Verify

Check that pods are running and the service is available:

```bash
kubectl get pods -n apex
kubectl get svc -n apex
kubectl logs -n apex -l app.kubernetes.io/name=apex --tail=50
```

Once the service is up, forward the port and hit the health endpoint:

```bash
kubectl port-forward svc/apex 8000:8000 -n apex
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "movie_count": 10000,
  "serving_tier": "tier2",
  "app_version": "2.0.0"
}
```

---

## Enable Ingress

```bash
helm upgrade apex ./k8s/helm/apex \
  --set ingress.enabled=true \
  --set ingress.host=api.yourdomain.com
```

To add TLS, pass a values file:

```yaml
# myvalues.yaml
ingress:
  enabled: true
  host: api.yourdomain.com
  tls:
    - secretName: apex-tls
      hosts:
        - api.yourdomain.com
```

```bash
helm upgrade apex ./k8s/helm/apex -f myvalues.yaml --namespace apex
```

---

## Notes

- The HPA requires `metrics-server` to be installed in the cluster. Install it with:
  ```bash
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
  ```
- The `autoscaling/v2` API requires Kubernetes >= 1.23
- Resource limits default to 2 CPU / 4Gi RAM; increase for Tier 1 GPU workloads
- The chart does not create a `PersistentVolumeClaim` — APEX uses an in-memory FAISS index loaded at startup. For artifact persistence across restarts, mount a PVC at `/app/models/` and pre-populate it with serving artifacts
