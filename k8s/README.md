# APEX Recommendation API — Kubernetes Deployment Guide

This directory contains Helm charts and Kubernetes manifests for deploying the APEX Recommendation System in production clusters.

## Prerequisites

- `kubectl >= 1.28`
- `helm >= 3.x`
- Kubernetes cluster (EKS, GKE, AKS, or local kind/minikube)

## Quick Install

```bash
helm install apex ./k8s/helm/apex --namespace apex --create-namespace
```

## Configure Serving Tier

APEX dynamically adapts its neural recommendation engine to available hardware:

- **Tier 1 (GPU / TensorRT / High-RAM)**: `--set servingTier=tier1 --set servingProfile=full`
- **Tier 2 (ONNX Runtime / C++ Inference)**: `--set servingTier=tier2 --set servingProfile=full`
- **Tier 3 (Pure PyTorch Lite - Default)**: `--set servingTier=tier3 --set servingProfile=lite`

```bash
helm upgrade --install apex ./k8s/helm/apex \
  --namespace apex \
  --set servingTier=tier3 \
  --set servingProfile=lite
```

## Secret Management

Pass reference names of pre-created Kubernetes secrets:

```bash
helm upgrade --install apex ./k8s/helm/apex \
  --namespace apex \
  --set secretRefs.jwtSecretKey=apex-secrets \
  --set secretRefs.tmdbApiKey=apex-secrets \
  --set secretRefs.adminToken=apex-secrets
```

## Verification

```bash
# Check pod status
kubectl get pods -n apex

# Tail container logs
kubectl logs -n apex -l app.kubernetes.io/name=apex -f

# Port-forward and check health
kubectl port-forward svc/apex-apex 8000:8000 -n apex
curl http://localhost:8000/health
```

## Uninstall

```bash
helm uninstall apex --namespace apex
```
