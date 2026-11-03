# NVIDIA NIM & NeMo Microservices Architecture Guide

## 1. Overview of NVIDIA Inference Microservices (NIM)
NVIDIA NIM (NVIDIA Inference Microservice) is a set of accelerated containers designed to deploy AI models with standard enterprise APIs. NIM packages model weights, the inference engine (TensorRT-LLM or vLLM), and an OpenAI-compliant HTTP API into a self-contained container image.

## 2. Container Ports and Default Health Endpoints
When launching a standard NIM container (e.g., `meta/llama-3.1-70b-instruct`), the microservice exposes the following endpoints:

| Port / Endpoint Path | Protocol | Purpose / Description |
| :--- | :--- | :--- |
| **Port 8000** | HTTP | Primary service port for all API interactions |
| **`/v1/chat/completions`** | HTTP POST | OpenAI-compatible chat completion inference endpoint |
| **`/v1/completions`** | HTTP POST | Raw text completion endpoint |
| **`/v1/models`** | HTTP GET | Lists active deployed model identifiers |
| **`/v1/health/ready`** | HTTP GET | Kubernetes readiness probe (returns 200 when model is loaded) |
| **`/v1/health/live`** | HTTP GET | Kubernetes liveness probe (returns 200 while process is running) |
| **`/v1/metrics`** | HTTP GET | Prometheus-formatted GPU telemetry, TTFT, and latency metrics |

## 3. Engine Backend Configuration: TRT-LLM vs. vLLM
NIM automatically selects or permits manual override of the execution engine:
- `ENGINE_BACKEND=trt-llm`: Optimal throughput and latency for NVIDIA GPUs using TensorRT-LLM and compiled engine plans (`.plan`).
- `ENGINE_BACKEND=vllm`: Dynamic fallback engine supporting broad Hugging Face architectures without ahead-of-time compilation.
- `KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.90`: Allocates 90% of available GPU VRAM for KV-cache paging.
