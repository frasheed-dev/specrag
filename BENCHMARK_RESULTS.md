# SpecRAG: Public Multi-Benchmark Performance Results

> **Evaluation Environment:** Evaluated on NVIDIA Tesla T4 GPU (Google Colab) across official third-party Hugging Face public benchmarks with verified ground-truth answer containment.

---

## 1. Executive Summary: Naive vs. Production Hybrid Architecture

SpecRAG is evaluated across two configurations:
1. **Curated Offline Verification Pack (40 Golden Queries):** Deterministic regression suite across 44 chunks from `G4KMU/t2-ragbench` and `ibm-research/REAL-MM-RAG_TechReport`.
2. **Scaled Live Hub Stream (200 Queries, 1,075 Chunks):** Streaming evaluation across 4 benchmark suites with 734 distractor chunks.

### A. Scaled Multi-Corpus Evaluation (Live Hugging Face Hub Stream)

- **Total Queries Evaluated:** 200 (50 per dataset)
- **Total Corpus Indexed:** 135 documents / 1,075 chunks
- **Baseline (Phase A):** 500-character fixed-window chunking (~100–125 tokens, 50-character overlap) + `BAAI/bge-small-en-v1.5` dense cosine similarity.
- **Production Hybrid (Phase B):** Structure-aware table preservation + hierarchical breadcrumbs + Sparse BM25Okapi + Qdrant Dense + Reciprocal Rank Fusion (Top-20) + `BAAI/bge-reranker-base` Cross-Encoder.

| Benchmark Suite | Corpus Scale | Phase A: Naive Recall@5 | Phase B: Hybrid Recall@5 | Phase A MRR | Phase B MRR (Reranked) | GPU Latency (T4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. T²-RAGBench (`FinQA`)** | 47 docs / 228 chunks | 96.0% | **100.0%** | 0.800 | **1.000** (+0.200) | 578.5 ms |
| **2. TAT-DQA (`TAT-QA`)** | 12 docs / 61 chunks | 96.0% | **100.0%** | 0.786 | **1.000** (+0.214) | 553.9 ms |
| **3. IBM TechReport** | 26 docs / 52 chunks | 80.0% | **82.0%** | 0.638 | **0.769** (+0.131) | 316.4 ms |
| **4. TechQA (`galileo-ai/ragbench`)** | 50 docs / 734 chunks | 22.0% | **94.0%** (+72.0%) | 0.152 | **0.803** (+0.651) | 721.9 ms |
| **Live Multi-Corpus Average** | **135 docs / 1,075 chunks** | **73.5%** | **94.0%** (+20.5%) | **0.594** | **0.893** (+0.299) | **542.7 ms** |

### B. Curated Offline Verification Pack

- **Total Queries Evaluated:** 40 (20 financial, 20 microarchitecture)
- **Total Corpus Indexed:** 2 documents / 44 chunks

| Benchmark Suite | Domain | Phase A Recall@5 | Phase B Recall@5 | Phase A MRR | Phase B MRR | GPU Latency (T4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **T²-RAGBench (`FinQA`)** | SEC Financial Tables | 100.0% | **100.0%** | 0.875 | **1.000** (+0.125) | 291.8 ms |
| **IBM TechReport** | Processor Buses & Interconnects | 100.0% | **100.0%** | 0.729 | **0.871** (+0.142) | 288.0 ms |
| **Offline Pack Average** | Technical & Financial Documents | **100.0%** | **100.0%** | **0.802** | **0.936** (+0.134) | **289.9 ms** |

---

## 2. Technical Analysis Across Benchmark Domains

### 1. Tabular Precision on Financial Statements (`T²-RAGBench` and `TAT-DQA`)
- On financial balance sheets and disclosure notes, naive 500-character chunking splits multi-column tables midway. When numerical entries lose column headers, dense vector similarity returns topical prose instead of the target cell.
- In both financial suites, the hybrid pipeline achieves **100.0% Recall@5** and **1.000 MRR**. The structure-aware chunker preserves entire tables as atomic units with table captions and column headers intact, while the cross-encoder ranks the exact table at position 1 for every query.

### 2. High-Density Hardware Parameters (`IBM TechReport`)
- Processor architecture specifications involve exact technical identifiers (`32 Gbps NRZ`, `1,024 GB/s`, `PowerAXON`, `OMI`).
- Pure dense retrieval places target chunks lower in the candidate list due to token out-of-vocabulary effects on rare alphanumeric strings (MRR 0.638). BM25 lexical matching indexes these exact terms, raising MRR to **0.769** (+0.131) in live streaming and **0.871** (+0.142) in the offline pack.

### 3. Resistance to Distractor Noise at Scale (`TechQA`)
- The `TechQA` benchmark indexes 734 chunks of enterprise technical troubleshooting text into Qdrant.
- **Naive baseline failure:** Pure dense retrieval collapses to **22.0% Recall@5** and **0.152 MRR**. In 78% of queries, dense cosine similarity selects distractor chunks that share semantic vocabulary but omit the factual answer.
- **Hybrid resilience:** SpecRAG achieves **94.0% Recall@5** (+72.0% absolute gain) and **0.803 MRR** (+0.651 gain). BM25 anchors candidate selection on exact error codes and subsystem identifiers, while joint cross-attention filters out semantically similar distractors.

---

## 3. Latency Profile: Live Stream vs. Offline Pack vs. CPU

Batch inference latency scales with candidate pool size and compute availability:

| Evaluation Environment | Indexed Corpus Scale | Queries Evaluated | Hardware Platform | Phase A Latency | Phase B Latency | Latency Characteristics |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Offline Verification Pack** | 44 chunks | 40 | NVIDIA Tesla T4 GPU | 16.6 ms | **289.9 ms** | Low candidate competition; minimal Qdrant filtering overhead |
| **Live IBM TechReport** | 52 chunks | 50 | NVIDIA Tesla T4 GPU | 15.6 ms | **316.4 ms** | Moderate corpus; short document segments |
| **Live TAT-DQA** | 61 chunks | 50 | NVIDIA Tesla T4 GPU | 14.9 ms | **553.9 ms** | Multi-row table text increases cross-encoder sequence length |
| **Live T²-RAGBench** | 228 chunks | 50 | NVIDIA Tesla T4 GPU | 15.6 ms | **578.5 ms** | Longer document contexts with multi-column financial disclosures |
| **Live TechQA** | 734 chunks | 50 | NVIDIA Tesla T4 GPU | 19.7 ms | **721.9 ms** | High distractor density; cross-encoder evaluates larger sequence payloads |
| **Live Stream Overall Average** | 1,075 chunks | 200 | NVIDIA Tesla T4 GPU | 16.5 ms | **542.7 ms** | Production interactive SLA (<600 ms) |
| **CPU Sequential Reference** | Any corpus | Per Query | x86-64 CPU (Unbatched) | ~25 ms | ~15,000–20,000 ms | Unbatched 12-layer transformer attention without tensor parallelization |

---

## 4. Ablation Analysis: Component-by-Component Contribution

To quantify the marginal value of each architectural component, retrieval performance was measured cumulatively across the benchmark suites:

| Pipeline Configuration | Recall@5 | MRR | Latency (T4 GPU) | Primary Failure Mode Addressed |
| :--- | :---: | :---: | :---: | :--- |
| **1. Naive Baseline**<br>*(500-char fixed-window + Dense Bi-Encoder)* | 73.5% | 0.594 | ~16 ms | Baseline: fails on severed tables and distractor noise. |
| **2. + Structure-Aware Chunking**<br>*(Atomic tables + Header breadcrumbs)* | 82.0% | 0.680 | ~18 ms | Fixes table severing; guarantees column headers remain bound to data rows. |
| **3. + BM25 Lexical Search & RRF**<br>*(Dual-engine hybrid retrieval)* | 88.5% | 0.745 | ~35 ms | Fixes alphanumeric blindness; recovers exact SKUs, register offsets, and error codes. |
| **4. + Cross-Encoder Reranking**<br>*(Full SpecRAG pipeline with `bge-reranker-base`)* | **94.0%** | **0.893** | **542.7 ms** | Eliminates distractor crowding; promotes exact answer chunk to Rank #1 via joint attention. |

---

## 5. System Limitations & Production Trade-offs

1. **Compute and Latency Budget:**
   - Cross-encoder reranking adds ~500 ms on a GPU. While acceptable for enterprise search and interactive chat (<1.5 s target), applications requiring sub-50 ms response times require latency-optimized alternatives such as ColBERT (late interaction) or model distillation (e.g., FlashRank).
   - CPU-only deployments cannot support real-time cross-encoder inference without quantization (e.g., ONNX Runtime or INT8 quantization).

2. **In-Memory BM25 Scaling:**
   - The current implementation stores BM25Okapi inverted indices in process memory. This design works efficiently for corpora up to hundreds of thousands of chunks. Corpora exceeding tens of millions of documents require external search backends such as Elasticsearch, OpenSearch, or Qdrant sparse vectors.

3. **Complex Table Geometries:**
   - Structure-aware chunking handles rectangular Markdown tables cleanly. However, PDFs containing non-standard layouts—such as borderless multi-line headers, diagonally split cells, or embedded rasterized charts—require multimodal layout parsers (e.g., DocLayout-YOLO or ColPali) to preserve fidelity before text extraction.

---

## 6. Reproducibility

Reproduce all benchmarks on a Tesla T4 GPU with zero local installation:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frasheed-dev/specrag/blob/main/notebooks/SpecRAG_Colab_GPU_Benchmark.ipynb)
