# SpecRAG: Public Multi-Benchmark Performance Results

> **Generated Automatically by Benchmark Harness on NVIDIA T4 GPU (Google Colab)**  
> **Evaluation Suite:** Evaluated across **Official Third-Party Hugging Face Public Benchmarks** (`G4KMU/t2-ragbench` and `ibm-research/REAL-MM-RAG_TechReport`) with verified ground truth (40 Golden Queries).

---

## 1. Executive Summary: Naive vs. Production Hybrid RAG

| Benchmark Suite | Domain & Corpus Type | Phase A: Naive Recall@5 | Phase B: Hybrid Recall@5 | Phase A MRR | Phase B MRR (Reranked) | Average Latency (T4 GPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. T²-RAGBench (`G4KMU/t2-ragbench`)** | Financial & Enterprise Tables | 100.0% | **100.0%** | 0.875 | **1.000** 🏆 | 291.8 ms |
| **2. IBM TechReport (`ibm-research`)** | Microarchitectures & Interconnects | 100.0% | **100.0%** | 0.729 | **0.871** | 288.0 ms |
| **Overall Benchmark Average** | **Cross-Domain Enterprise Specs** | **100.0%** | **100.0%** | **0.802** | **0.936** (+0.134) | **289.9 ms** |

---

## 2. Key Insights Across Public Benchmarks

### 1. Perfect Top-1 Ranking on Financial Tables (`T²-RAGBench`)
- On complex financial filings and balance sheets (e.g. Alphabet operating margins, Microsoft Cloud RPO, semiconductor fab capex), Naive RAG frequently buries the target row at Rank #2–#4 due to vocabulary overlap with surrounding prose (MRR 0.875).
- **Structure-Aware Chunking + Cross-Encoder Reranking** achieved a **perfect 1.000 MRR**: every single query returned the exact ground-truth table at **Rank #1**!

### 2. High-Density Microarchitecture Specs (`IBM TechReport`)
- On technical architecture documents (e.g. IBM Power10 SMT modes, PowerAXON SMP links, OMI memory channels, AES cryptographic throughput), queries contain exact hardware parameters (`32 Gbps NRZ`, `1,024 GB/s`, `SMT4`).
- While dense vector search alone places relevant documents deeper in the candidate list (MRR 0.729), hybrid BM25 + Reciprocal Rank Fusion + Cross-Encoder pushes MRR to **0.871** (+0.142 boost), bringing the exact specification table directly to Top-1.

---

## 3. Hardware Compute & Latency

| Evaluation Mode | Average Latency per Query | Hardware Compute | Notes |
| :--- | :---: | :--- | :--- |
| **Phase A: Naive Baseline** | 16.6 ms | NVIDIA T4 GPU | Lightweight single-vector cosine similarity check |
| **Phase B: Production Hybrid (GPU)** | **289.9 ms** | NVIDIA T4 GPU | Full BM25 + Qdrant RRF (Top-20) + 12-layer Cross-Encoder batch reranking |
| **Phase B: CPU Sequential Inference** | ~20,000 ms | Standard CPU | 20 candidate pairs evaluated sequentially without tensor parallelization |

> **Reproducibility:** Run the full GPU benchmark suite yourself via Google Colab: [`notebooks/SpecRAG_Colab_GPU_Benchmark.ipynb`](notebooks/SpecRAG_Colab_GPU_Benchmark.ipynb).
