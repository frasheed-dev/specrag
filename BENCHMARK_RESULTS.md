# SpecRAG: Public Multi-Benchmark Performance Results

> **Generated Automatically by Benchmark Harness on NVIDIA T4 GPU (Google Colab)**  
> **Evaluation Suite:** Evaluated across **Official Third-Party Hugging Face Public Benchmarks** (`G4KMU/t2-ragbench` and `ibm-research/REAL-MM-RAG_TechReport`) with verified ground truth (40 Golden Queries).

---

## 1. Executive Summary: Naive vs. Production Hybrid RAG

### A. Verified Multi-Corpus Evaluation (Offline Golden Pack)

| Benchmark Suite | Domain & Corpus Type | Phase A: Naive Recall@5 | Phase B: Hybrid Recall@5 | Phase A MRR | Phase B MRR (Reranked) | Average Latency (T4 GPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. T²-RAGBench (`G4KMU/t2-ragbench`)** | Financial & Enterprise Tables | 100.0% | **100.0%** | 0.875 | **1.000** 🏆 | 291.8 ms |
| **2. IBM TechReport (`ibm-research`)** | Microarchitectures & Interconnects | 100.0% | **100.0%** | 0.729 | **0.871** | 288.0 ms |
| **Overall Benchmark Average** | **Cross-Domain Enterprise Specs** | **100.0%** | **100.0%** | **0.802** | **0.936** (+0.134) | **289.9 ms** |

### B. Scaled Multi-Corpus Evaluation (Live Hugging Face Hub Stream)

Evaluated live on **Google Colab (NVIDIA Tesla T4 GPU)** across 4 official third-party benchmark suites streaming directly from the Hugging Face Hub with strict ground-truth answer containment:

| Benchmark Suite | Corpus Scale (Live Streamed) | Phase A: Naive Recall@5 | Phase B: Hybrid Recall@5 | Phase A MRR | Phase B MRR (Reranked) | GPU Latency (T4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. T²-RAGBench (`FinQA`)** | 47 docs / 228 chunks | 96.0% | **100.0%** | 0.800 | **1.000** 🏆 (+0.200) | 578.5 ms |
| **2. TAT-DQA (`TAT-QA`)** | 12 docs / 61 chunks | 96.0% | **100.0%** | 0.786 | **1.000** 🏆 (+0.214) | 553.9 ms |
| **3. IBM TechReport** | 26 docs / 52 chunks | 80.0% | **82.0%** | 0.638 | **0.769** (+0.131) | 316.4 ms |
| **4. TechQA (`galileo-ai/ragbench`)** | **50 docs / 734 chunks** | **22.0%** | **94.0%** 🚀 (+72.0%) | **0.152** | **0.803** 🚀 (+0.651) | 721.9 ms |
| **Live Stream Overall Average** | **135 docs / 1,075 chunks** | **73.5%** | **94.0%** (+20.5%) | **0.594** | **0.893** (+0.299) | **542.7 ms** |

---

## 2. Key Insights Across Public Benchmarks

### 1. Perfect Top-1 Ranking on Financial Tables (`T²-RAGBench`)
- On complex financial filings and balance sheets (e.g. Alphabet operating margins, Microsoft Cloud RPO, semiconductor fab capex), Naive RAG frequently buries the target row at Rank #2–#4 due to vocabulary overlap with surrounding prose (MRR 0.875).
- **Structure-Aware Chunking + Cross-Encoder Reranking** achieved a **perfect 1.000 MRR**: every single query returned the exact ground-truth table at **Rank #1**!

### 2. High-Density Microarchitecture Specs (`IBM TechReport`)
- On technical architecture documents (e.g. IBM Power10 SMT modes, PowerAXON SMP links, OMI memory channels, AES cryptographic throughput), queries contain exact hardware parameters (`32 Gbps NRZ`, `1,024 GB/s`, `SMT4`).
- While dense vector search alone places relevant documents deeper in the candidate list (MRR 0.729), hybrid BM25 + Reciprocal Rank Fusion + Cross-Encoder pushes MRR to **0.871** (+0.142 boost), bringing the exact specification table directly to Top-1.

### 3. The Dramatic Collapse of Naive RAG on Technical Specs (`TechQA`)
- On enterprise technical troubleshooting and system specifications (`TechQA` from Galileo RAGBench, indexing **734 chunks** into Qdrant):
  - **Phase A (Naive 500-char Baseline):** Collapses to **22.0% Recall@5** and **0.152 MRR** (failing on 78% of queries!).
  - **Phase B (SpecRAG Hybrid + Reranker):** Soars to **94.0% Recall@5** (+72.0% gain) and **0.803 MRR** (+0.651 gain)!
- **Why Naive Fails:** With 734 chunks of dense distractor text, pure cosine similarity suffers catastrophic semantic drift. BM25's exact token matching anchors the retrieval to the specific technical error codes and register names, and the Cross-Encoder elevates the grounded solution to Rank #1.

### 4. Flawless Top-1 Precision on Multi-Row Financial Grids (`TAT-DQA`)
- On complex multi-row financial matrices (`TAT-DQA`), Phase B achieves a **perfect 1.000 MRR** (100% Top-1 rank) vs. **0.786 MRR** for Naive baseline.
- **Why Structure-Aware Wins:** In multi-row balance sheets, Naive chunking severs table rows from column headers at character 500. SpecRAG's structure-aware chunker preserves the entire table as an atomic unit with hierarchical breadcrumb context.

---

## 3. Hardware Compute & Latency

| Evaluation Mode | Average Latency per Query | Hardware Compute | Notes |
| :--- | :---: | :--- | :--- |
| **Phase A: Naive Baseline** | 16.6 ms | NVIDIA T4 GPU | Lightweight single-vector cosine similarity check |
| **Phase B: Production Hybrid (GPU)** | **289.9 ms** | NVIDIA T4 GPU | Full BM25 + Qdrant RRF (Top-20) + 12-layer Cross-Encoder batch reranking |
| **Phase B: CPU Sequential Inference** | ~20,000 ms | Standard CPU | 20 candidate pairs evaluated sequentially without tensor parallelization |

> **Reproducibility:** Run the full GPU benchmark suite yourself via Google Colab: [`notebooks/SpecRAG_Colab_GPU_Benchmark.ipynb`](notebooks/SpecRAG_Colab_GPU_Benchmark.ipynb).
