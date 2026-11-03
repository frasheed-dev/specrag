"""Dataset downloader and curator for SpecRAG.

Prepares the 3 benchmark corpora:
1. nvidia_ai_hw: NVIDIA Blackwell, H100 SXM5 vs. PCIe, and NIM Microservices specs.
2. hf_tech_bench: Technical enterprise Q&A & table benchmark.
3. arm_riscv_ip: ARM AMBA AXI4 protocol and RISC-V ISA specification excerpts.
"""

import json
from pathlib import Path

from specrag.config import settings


def create_nvidia_ai_hw_corpus(base_dir: Path) -> None:
    corpus_dir = base_dir / "nvidia_ai_hw"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Document 1: NVIDIA H100 Tensor Core GPU Architecture & Datasheet
    h100_content = """# NVIDIA H100 Tensor Core GPU Architecture Datasheet

## 1. Overview and Architecture
The NVIDIA H100 Tensor Core GPU, powered by the NVIDIA Hopper architecture, delivers massive leaps in performance, scalability, and security for accelerated computing workloads. Built on a custom TSMC 4N process with 80 billion transistors, it introduces the Transformer Engine, 4th-generation NVLink, and 2nd-generation Multi-Instance GPU (MIG).

## 2. Form Factors and Physical Specifications
The H100 is offered in two primary enterprise form factors:
- **H100 SXM5:** Designed for high-density HGX systems offering maximum thermal and compute headroom.
- **H100 PCIe:** Standard dual-slot PCIe Gen 5 card for mainstream enterprise server enclosures.
- **H100 NVL:** Dual-GPU PCIe card interconnected with NVLink specifically optimized for LLM inference.

## 3. Performance and Technical Comparison Matrix
The table below specifies the compute capabilities, memory specifications, and interconnect bandwidth across form factors:

| Specification Parameter | H100 SXM5 | H100 PCIe | H100 NVL (Per Dual Card) |
| :--- | :--- | :--- | :--- |
| **FP64 Tensor Core TFLOPS** | 67 TFLOPS | 51 TFLOPS | 134 TFLOPS |
| **FP64 Vector TFLOPS** | 34 TFLOPS | 26 TFLOPS | 68 TFLOPS |
| **FP32 TFLOPS** | 67 TFLOPS | 51 TFLOPS | 134 TFLOPS |
| **TF32 Tensor Core TFLOPS** | 989 TFLOPS | 756 TFLOPS | 1,978 TFLOPS |
| **BFLOAT16 Tensor Core TFLOPS** | 1,979 TFLOPS | 1,513 TFLOPS | 3,958 TFLOPS |
| **FP16 Tensor Core TFLOPS** | 1,979 TFLOPS | 1,513 TFLOPS | 3,958 TFLOPS |
| **FP8 Tensor Core TFLOPS** | 3,958 TFLOPS | 3,026 TFLOPS | 7,916 TFLOPS |
| **INT8 Tensor Core TOPS** | 3,958 TOPS | 3,026 TOPS | 7,916 TOPS |
| **GPU Memory Capacity** | 80 GB HBM3 | 80 GB HBM2e | 188 GB (94 GB per GPU) HBM3 |
| **GPU Memory Bandwidth** | 3.35 TB/s | 2.0 TB/s | 7.8 TB/s (3.9 TB/s per GPU) |
| **Interconnect** | NVLink 4: 900 GB/s | PCIe Gen 5: 128 GB/s | NVLink Bridge: 600 GB/s |
| **Max Thermal Design Power (TDP)** | 700 W | 350 W - 400 W | 700 W - 800 W |
| **Form Factor** | Custom SXM5 Board | Dual-slot PCIe 5.0 x16 | 2x Dual-slot PCIe 5.0 bridged |

## 4. Multi-Instance GPU (MIG) Partitioning
The H100 supports second-generation MIG, dividing a single GPU into up to seven independent instances with fully isolated memory and compute resources:
- Profile `1g.10gb`: 1 SM slice, 10 GB HBM, up to 7 instances per SXM5.
- Profile `2g.20gb`: 2 SM slices, 20 GB HBM, up to 3 instances per SXM5.
- Profile `3g.40gb`: 3 SM slices, 40 GB HBM, up to 2 instances per SXM5.
- Profile `4g.40gb`: 4 SM slices, 40 GB HBM, up to 1 instance per SXM5.
- Profile `7g.80gb`: Full GPU, 80 GB HBM, single instance.

## 5. Transformer Engine and FP8 Precision
The Hopper Transformer Engine dynamically chooses between FP8 and FP16/BF16 precisions on a layer-by-layer basis during training and inference.
- Uses two FP8 formats:
  - `E4M3`: 1 sign bit, 4 exponent bits, 3 mantissa bits (higher precision for forward pass weights and activations).
  - `E5M2`: 1 sign bit, 5 exponent bits, 2 mantissa bits (broader dynamic range for backward pass gradients).
"""
    (corpus_dir / "nvidia_h100_datasheet.md").write_text(h100_content, encoding="utf-8")

    # Document 2: NVIDIA Blackwell Architecture Technical Overview
    blackwell_content = """# NVIDIA Blackwell Architecture Technical Overview

## 1. Executive Summary & Die Topology
The NVIDIA Blackwell architecture powers the world's highest performance generative AI supercomputing infrastructure. Built on a custom 4NP TSMC process, the Blackwell GPU die incorporates 208 billion transistors across a dual-die chiplet packaging connected by the 10 TB/s high-speed NVLink-C2C interconnect.

## 2. Architectural Comparison: Blackwell B200 vs. Hopper H100
The following architectural matrix details the generation-over-generation evolution:

| Feature / Metric | Hopper H100 SXM5 | Blackwell B200 | GB200 NVL72 Superchip Rack |
| :--- | :--- | :--- | :--- |
| **Transistor Count** | 80 Billion | 208 Billion (Dual Die) | Multi-Chip System |
| **Process Node** | Custom TSMC 4N | Custom TSMC 4NP | Custom TSMC 4NP + TSMC 3nm (Grace) |
| **FP4 Tensor Core TFLOPS** | Not Supported | 18,000 TFLOPS (18 PFLOPS) | 720,000 TFLOPS (720 PFLOPS) |
| **FP8 Tensor Core TFLOPS** | 3,958 TFLOPS | 9,000 TFLOPS (9 PFLOPS) | 360,000 TFLOPS (360 PFLOPS) |
| **FP16 Tensor Core TFLOPS** | 1,979 TFLOPS | 4,500 TFLOPS (4.5 PFLOPS) | 180,000 TFLOPS (180 PFLOPS) |
| **Memory Capacity** | 80 GB HBM3 | 192 GB HBM3e | 30 TB Fast Memory (Rack-wide) |
| **Memory Bandwidth** | 3.35 TB/s | 8.0 TB/s | 576 TB/s (Rack-wide) |
| **NVLink Generation & Speed** | NVLink 4 (900 GB/s) | NVLink 5 (1,800 GB/s bidirectional) | NVLink 5 Switch Spine (130 TB/s) |
| **Max TDP** | 700 W | 1,000 W (Configurable) | 120 kW (Liquid Cooled Rack) |

## 3. Second-Generation Transformer Engine and Micro-Tensor Scaling
Blackwell introduces native support for **FP4 precision** through the second-generation Transformer Engine.
- Utilizes microscopic block scaling factors (Micro-Tensor Scaling) to maintain accuracy identical to FP16 while doubling inference throughput over FP8.
- The `NVLink-C2C` interface operates at 10 TB/s, presenting the dual dies to software as a unified monolithic CUDA accelerator without NUMA overhead.

## 4. Decompression and Reliability Engines
Blackwell integrates dedicated hardware engines for database queries and RAS:
- **Hardware Decompression Engine:** Natively accelerates Snappy, Gzip, and Deflate at speeds up to 800 GB/s.
- **Autonomous Reliability Engine:** AI-driven telemetry predictive failure analyzer tracking on-chip sensors to predict thermal and memory faults before system failure.
"""
    (corpus_dir / "nvidia_blackwell_overview.md").write_text(blackwell_content, encoding="utf-8")

    # Document 3: NVIDIA NIM & NeMo Microservices Architecture Guide
    nim_content = """# NVIDIA NIM & NeMo Microservices Architecture Guide

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
"""
    (corpus_dir / "nvidia_nim_guide.md").write_text(nim_content, encoding="utf-8")

    # Golden Evaluation Set for nvidia_ai_hw (20 queries)
    golden_eval = [
        {
            "query_id": "nv_01",
            "dataset": "nvidia_ai_hw",
            "query": "What is the FP8 Tensor Core performance difference between H100 SXM5 and H100 PCIe?",
            "ground_truth_answer": "H100 SXM5 delivers 3,958 TFLOPS of FP8 Tensor Core compute, whereas H100 PCIe delivers 3,026 TFLOPS.",
            "expected_doc_id": "nvidia_h100_datasheet.md",
            "expected_section_or_table": "Performance and Technical Comparison Matrix",
            "query_type": "table_lookup",
        },
        {
            "query_id": "nv_02",
            "dataset": "nvidia_ai_hw",
            "query": "Compare the memory bandwidth and capacity of the Blackwell B200 versus the Hopper H100 SXM5.",
            "ground_truth_answer": "The Blackwell B200 features 192 GB of HBM3e memory with 8.0 TB/s bandwidth, while the Hopper H100 SXM5 features 80 GB of HBM3 memory with 3.35 TB/s bandwidth.",
            "expected_doc_id": "nvidia_blackwell_overview.md",
            "expected_section_or_table": "Architectural Comparison: Blackwell B200 vs. Hopper H100",
            "query_type": "table_lookup",
        },
        {
            "query_id": "nv_03",
            "dataset": "nvidia_ai_hw",
            "query": "What are the two specific FP8 precision formats used by Hopper's Transformer Engine and their mantissa/exponent bit layouts?",
            "ground_truth_answer": "Hopper uses E4M3 (1 sign, 4 exponent, 3 mantissa bits) for higher precision forward passes, and E5M2 (1 sign, 5 exponent, 2 mantissa bits) for broader dynamic range backward passes.",
            "expected_doc_id": "nvidia_h100_datasheet.md",
            "expected_section_or_table": "Transformer Engine and FP8 Precision",
            "query_type": "exact_sku",
        },
        {
            "query_id": "nv_04",
            "dataset": "nvidia_ai_hw",
            "query": "What is the maximum TDP of the H100 SXM5 compared to the H100 PCIe?",
            "ground_truth_answer": "The H100 SXM5 has a maximum TDP of 700 W, whereas the H100 PCIe has a TDP of 350 W to 400 W.",
            "expected_doc_id": "nvidia_h100_datasheet.md",
            "expected_section_or_table": "Performance and Technical Comparison Matrix",
            "query_type": "table_lookup",
        },
        {
            "query_id": "nv_05",
            "dataset": "nvidia_ai_hw",
            "query": "What is the interconnect speed of the Blackwell NVLink-C2C chiplet link?",
            "ground_truth_answer": "The Blackwell NVLink-C2C interface operates at 10 TB/s connecting the dual dies.",
            "expected_doc_id": "nvidia_blackwell_overview.md",
            "expected_section_or_table": "Executive Summary & Die Topology",
            "query_type": "exact_sku",
        },
        {
            "query_id": "nv_06",
            "dataset": "nvidia_ai_hw",
            "query": "What is the Kubernetes readiness probe HTTP endpoint path and port for an NVIDIA NIM container?",
            "ground_truth_answer": "The readiness probe is HTTP GET /v1/health/ready on Port 8000.",
            "expected_doc_id": "nvidia_nim_guide.md",
            "expected_section_or_table": "Container Ports and Default Health Endpoints",
            "query_type": "table_lookup",
        },
        {
            "query_id": "nv_07",
            "dataset": "nvidia_ai_hw",
            "query": "How many FP4 Tensor Core TFLOPS can a single Blackwell B200 deliver?",
            "ground_truth_answer": "A single Blackwell B200 delivers 18,000 TFLOPS (18 PFLOPS) of FP4 Tensor Core performance.",
            "expected_doc_id": "nvidia_blackwell_overview.md",
            "expected_section_or_table": "Architectural Comparison: Blackwell B200 vs. Hopper H100",
            "query_type": "table_lookup",
        },
        {
            "query_id": "nv_08",
            "dataset": "nvidia_ai_hw",
            "query": "What MIG profile on H100 SXM5 allows up to 7 instances per GPU?",
            "ground_truth_answer": "The profile 1g.10gb (1 SM slice, 10 GB HBM) supports up to 7 instances per SXM5.",
            "expected_doc_id": "nvidia_h100_datasheet.md",
            "expected_section_or_table": "Multi-Instance GPU (MIG) Partitioning",
            "query_type": "exact_sku",
        },
        {
            "query_id": "nv_09",
            "dataset": "nvidia_ai_hw",
            "query": "What decompression formats does the Blackwell Hardware Decompression Engine accelerate, and at what bandwidth?",
            "ground_truth_answer": "It accelerates Snappy, Gzip, and Deflate at speeds up to 800 GB/s.",
            "expected_doc_id": "nvidia_blackwell_overview.md",
            "expected_section_or_table": "Decompression and Reliability Engines",
            "query_type": "exact_sku",
        },
        {
            "query_id": "nv_10",
            "dataset": "nvidia_ai_hw",
            "query": "What environment variable sets the KV-cache GPU memory fraction in NVIDIA NIM?",
            "ground_truth_answer": "KV_CACHE_FREE_GPU_MEMORY_FRACTION (e.g. KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.90).",
            "expected_doc_id": "nvidia_nim_guide.md",
            "expected_section_or_table": "Engine Backend Configuration: TRT-LLM vs. vLLM",
            "query_type": "exact_sku",
        },
        {
            "query_id": "nv_11",
            "dataset": "nvidia_ai_hw",
            "query": "Compare the NVLink bidirectional speed of Hopper H100 versus Blackwell B200.",
            "ground_truth_answer": "H100 features NVLink 4 with 900 GB/s, while Blackwell B200 features NVLink 5 with 1,800 GB/s bidirectional interconnect bandwidth.",
            "expected_doc_id": "nvidia_blackwell_overview.md",
            "expected_section_or_table": "Architectural Comparison: Blackwell B200 vs. Hopper H100",
            "query_type": "table_lookup",
        },
        {
            "query_id": "nv_12",
            "dataset": "nvidia_ai_hw",
            "query": "What is the memory capacity of the H100 NVL dual-card system?",
            "ground_truth_answer": "The H100 NVL dual-card features 188 GB total memory (94 GB per GPU) of HBM3.",
            "expected_doc_id": "nvidia_h100_datasheet.md",
            "expected_section_or_table": "Performance and Technical Comparison Matrix",
            "query_type": "table_lookup",
        },
        {
            "query_id": "nv_13",
            "dataset": "nvidia_ai_hw",
            "query": "What is the purpose of the Autonomous Reliability Engine in the Blackwell architecture?",
            "ground_truth_answer": "It is an AI-driven telemetry predictive failure analyzer tracking on-chip sensors to predict thermal and memory faults before system failure.",
            "expected_doc_id": "nvidia_blackwell_overview.md",
            "expected_section_or_table": "Decompression and Reliability Engines",
            "query_type": "multi_section",
        },
        {
            "query_id": "nv_14",
            "dataset": "nvidia_ai_hw",
            "query": "What is the FP64 Vector compute performance of the H100 PCIe?",
            "ground_truth_answer": "The H100 PCIe provides 26 TFLOPS of FP64 Vector compute.",
            "expected_doc_id": "nvidia_h100_datasheet.md",
            "expected_section_or_table": "Performance and Technical Comparison Matrix",
            "query_type": "table_lookup",
        },
        {
            "query_id": "nv_15",
            "dataset": "nvidia_ai_hw",
            "query": "Which engine backend in NIM provides ahead-of-time compiled execution using `.plan` engines?",
            "ground_truth_answer": "ENGINE_BACKEND=trt-llm uses TensorRT-LLM with compiled .plan engine files.",
            "expected_doc_id": "nvidia_nim_guide.md",
            "expected_section_or_table": "Engine Backend Configuration: TRT-LLM vs. vLLM",
            "query_type": "exact_sku",
        },
        {
            "query_id": "nv_16",
            "dataset": "nvidia_ai_hw",
            "query": "How many transistors are integrated into the dual-die Blackwell GPU packaging?",
            "ground_truth_answer": "The Blackwell GPU die incorporates 208 billion transistors across a dual-die chiplet packaging.",
            "expected_doc_id": "nvidia_blackwell_overview.md",
            "expected_section_or_table": "Executive Summary & Die Topology",
            "query_type": "exact_sku",
        },
        {
            "query_id": "nv_17",
            "dataset": "nvidia_ai_hw",
            "query": "What endpoint path provides Prometheus telemetry in an NVIDIA NIM container?",
            "ground_truth_answer": "The Prometheus telemetry endpoint is HTTP GET /v1/metrics.",
            "expected_doc_id": "nvidia_nim_guide.md",
            "expected_section_or_table": "Container Ports and Default Health Endpoints",
            "query_type": "table_lookup",
        },
        {
            "query_id": "nv_18",
            "dataset": "nvidia_ai_hw",
            "query": "What is the memory bandwidth of the H100 PCIe compared to H100 SXM5?",
            "ground_truth_answer": "The H100 PCIe provides 2.0 TB/s (HBM2e) bandwidth compared to 3.35 TB/s (HBM3) on the H100 SXM5.",
            "expected_doc_id": "nvidia_h100_datasheet.md",
            "expected_section_or_table": "Performance and Technical Comparison Matrix",
            "query_type": "table_lookup",
        },
        {
            "query_id": "nv_19",
            "dataset": "nvidia_ai_hw",
            "query": "What precision scaling mechanism does Blackwell introduce to maintain accuracy with FP4 precision?",
            "ground_truth_answer": "It introduces Micro-Tensor Scaling (microscopic block scaling factors) via the 2nd-gen Transformer Engine.",
            "expected_doc_id": "nvidia_blackwell_overview.md",
            "expected_section_or_table": "Second-Generation Transformer Engine and Micro-Tensor Scaling",
            "query_type": "exact_sku",
        },
        {
            "query_id": "nv_20",
            "dataset": "nvidia_ai_hw",
            "query": "What is the total fast memory and memory bandwidth of a complete GB200 NVL72 rack?",
            "ground_truth_answer": "The GB200 NVL72 rack provides 30 TB of fast memory and 576 TB/s of aggregate memory bandwidth.",
            "expected_doc_id": "nvidia_blackwell_overview.md",
            "expected_section_or_table": "Architectural Comparison: Blackwell B200 vs. Hopper H100",
            "query_type": "table_lookup",
        },
    ]

    with open(corpus_dir / "golden_eval.jsonl", "w", encoding="utf-8") as f:
        for item in golden_eval:
            f.write(json.dumps(item) + "\n")


def create_arm_riscv_ip_corpus(base_dir: Path) -> None:
    corpus_dir = base_dir / "arm_riscv_ip"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Document 1: ARM AMBA AXI4 Protocol Specification
    axi4_content = """# ARM AMBA AXI4 Protocol Specification Summary

## 1. Overview and Five Independent Channels
The AMBA AXI4 protocol is an on-chip interconnect specification designed for high-performance, high-clock-frequency system designs. AXI4 defines five independent unidirectional channels:
1. Write Address Channel (`AW`)
2. Write Data Channel (`W`)
3. Write Response Channel (`B`)
4. Read Address Channel (`AR`)
5. Read Data Channel (`R`)

## 2. Basic Handshake Signaling Rules
Every channel uses a two-wire handshake mechanism with `VALID` and `READY` signals:
- The source asserts `VALID` when information is available.
- The destination asserts `READY` when it can accept the transfer.
- The transfer occurs only on the rising clock edge where both `VALID` and `READY` are HIGH.
- Once asserted, `VALID` must remain HIGH until the handshake completes (`READY` asserted). A source is NOT permitted to wait for `READY` to be asserted before asserting `VALID`.

## 3. Channel Signal Map and Bitfield Definitions
The table below specifies the critical control signals across the Read and Write address channels:

| Signal Name | Channel | Width | Source | Description & Encoding |
| :--- | :--- | :--- | :--- | :--- |
| `AWID` | Write Address | Master-defined | Master | Write address ID tag for out-of-order write transactions |
| `AWADDR` | Write Address | 32/64 bits | Master | Write address of the first beat in a burst transaction |
| `AWLEN` | Write Address | 8 bits | Master | Burst length: Number of data transfers (0x00 = 1 beat, 0xFF = 256 beats) |
| `AWSIZE` | Write Address | 3 bits | Master | Burst size: Bytes per transfer (0b000 = 1 byte, 0b010 = 4 bytes, 0b011 = 8 bytes) |
| `AWBURST` | Write Address | 2 bits | Master | Burst type: 0b00 = FIXED, 0b01 = INCR, 0b10 = WRAP, 0b11 = Reserved |
| `AWVALID` | Write Address | 1 bit | Master | Indicates write address and control info are valid |
| `AWREADY` | Write Address | 1 bit | Slave | Indicates slave is ready to accept write address |
| `BRESP` | Write Response | 2 bits | Slave | Write response status (0b00 = OKAY, 0b01 = EXOKAY, 0b10 = SLVERR, 0b11 = DECERR) |
| `BVALID` | Write Response | 1 bit | Slave | Indicates write response is valid |
| `BREADY` | Write Response | 1 bit | Master | Indicates master is ready to accept write response |

## 4. Response Status Encodings (RRESP and BRESP)
Both read response (`RRESP`) and write response (`BRESP`) use identical 2-bit encodings:
- `0b00 (OKAY)`: Normal access success.
- `0b01 (EXOKAY)`: Exclusive access success.
- `0b10 (SLVERR)`: Slave error condition (e.g. unsupported access, write protect violation).
- `0b11 (DECERR)`: Decode error (no slave exists at requested address).
"""
    (corpus_dir / "arm_amba_axi4_spec.md").write_text(axi4_content, encoding="utf-8")

    # Document 2: RISC-V Unprivileged ISA Specification
    riscv_content = """# RISC-V Unprivileged ISA Specification Excerpt

## 1. Overview and Base Architectures
The RISC-V instruction set is a modular open architecture. The standard base integer ISAs are:
- `RV32I`: 32-bit base integer instruction set with 32 general-purpose registers (`x0` through `x31`).
- `RV64I`: 64-bit base integer instruction set extending register width to 64 bits.

## 2. General-Purpose Registers and ABI Names
Register `x0` is hardwired to constant zero. The table below lists standard register mappings and calling conventions:

| Register | ABI Name | Description | Saver Rule |
| :--- | :--- | :--- | :--- |
| `x0` | `zero` | Hard-wired zero | — |
| `x1` | `ra` | Return address | Caller |
| `x2` | `sp` | Stack pointer | Callee |
| `x3` | `gp` | Global pointer | — |
| `x4` | `tp` | Thread pointer | — |
| `x5` | `t0` | Temporary / alternate link register | Caller |
| `x6-x7` | `t1-t2` | Temporaries | Caller |
| `x8` | `s0` / `fp` | Saved register / frame pointer | Callee |
| `x9` | `s1` | Saved register | Callee |
| `x10-x11` | `a0-a1` | Function arguments / return values | Caller |
| `x12-x17` | `a2-a7` | Function arguments | Caller |
| `x18-x27` | `s2-s11` | Saved registers | Callee |
| `x28-x31` | `t3-t6` | Temporaries | Caller |

## 3. Instruction Formats
RV32I has six core instruction formats:
1. `R-type`: Register-to-register arithmetic (`funct7`, `rs2`, `rs1`, `funct3`, `rd`, `opcode`).
2. `I-type`: Immediate arithmetic and load instructions (`imm[11:0]`, `rs1`, `funct3`, `rd`, `opcode`).
3. `S-type`: Store instructions (`imm[11:5]`, `rs2`, `rs1`, `funct3`, `imm[4:0]`, `opcode`).
4. `B-type`: Conditional branches (`imm[12|10:5]`, `rs2`, `rs1`, `funct3`, `imm[4:1|11]`, `opcode`).
5. `U-type`: Upper immediate instructions (`imm[31:12]`, `rd`, `opcode`).
6. `J-type`: Jump instructions (`imm[20|10:1|11|19:12]`, `rd`, `opcode`).
"""
    (corpus_dir / "riscv_isa_spec.md").write_text(riscv_content, encoding="utf-8")

    # Golden Evaluation Set for arm_riscv_ip (10 queries)
    golden_eval = [
        {
            "query_id": "ip_01",
            "dataset": "arm_riscv_ip",
            "query": "In AMBA AXI4, what are the five independent transaction channels?",
            "ground_truth_answer": "Write Address (AW), Write Data (W), Write Response (B), Read Address (AR), and Read Data (R).",
            "expected_doc_id": "arm_amba_axi4_spec.md",
            "expected_section_or_table": "Overview and Five Independent Channels",
            "query_type": "exact_sku",
        },
        {
            "query_id": "ip_02",
            "dataset": "arm_riscv_ip",
            "query": "What is the 2-bit response status encoding for SLVERR in AXI4 BRESP and RRESP?",
            "ground_truth_answer": "0b10 represents SLVERR (Slave error condition).",
            "expected_doc_id": "arm_amba_axi4_spec.md",
            "expected_section_or_table": "Response Status Encodings (RRESP and BRESP)",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ip_03",
            "dataset": "arm_riscv_ip",
            "query": "Can an AXI4 source wait for READY to be asserted before asserting VALID?",
            "ground_truth_answer": "No, a source is NOT permitted to wait for READY to be asserted before asserting VALID.",
            "expected_doc_id": "arm_amba_axi4_spec.md",
            "expected_section_or_table": "Basic Handshake Signaling Rules",
            "query_type": "exact_sku",
        },
        {
            "query_id": "ip_04",
            "dataset": "arm_riscv_ip",
            "query": "What is the width and maximum burst length supported by AWLEN in AXI4?",
            "ground_truth_answer": "AWLEN is 8 bits wide, supporting burst lengths from 1 beat (0x00) up to 256 beats (0xFF).",
            "expected_doc_id": "arm_amba_axi4_spec.md",
            "expected_section_or_table": "Channel Signal Map and Bitfield Definitions",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ip_05",
            "dataset": "arm_riscv_ip",
            "query": "What is the ABI name, description, and saver rule for RISC-V register x1?",
            "ground_truth_answer": "Register x1 has ABI name 'ra', description 'Return address', and saver rule 'Caller'.",
            "expected_doc_id": "riscv_isa_spec.md",
            "expected_section_or_table": "General-Purpose Registers and ABI Names",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ip_06",
            "dataset": "arm_riscv_ip",
            "query": "Which RISC-V general-purpose register is hard-wired to zero?",
            "ground_truth_answer": "Register x0 (ABI name zero) is hard-wired to constant zero.",
            "expected_doc_id": "riscv_isa_spec.md",
            "expected_section_or_table": "General-Purpose Registers and ABI Names",
            "query_type": "exact_sku",
        },
        {
            "query_id": "ip_07",
            "dataset": "arm_riscv_ip",
            "query": "What are the 2-bit binary encodings for AXI4 AWBURST types FIXED, INCR, and WRAP?",
            "ground_truth_answer": "0b00 = FIXED, 0b01 = INCR, 0b10 = WRAP (0b11 is Reserved).",
            "expected_doc_id": "arm_amba_axi4_spec.md",
            "expected_section_or_table": "Channel Signal Map and Bitfield Definitions",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ip_08",
            "dataset": "arm_riscv_ip",
            "query": "In RISC-V RV32I, which instruction format is used for conditional branch instructions?",
            "ground_truth_answer": "The B-type format (imm[12|10:5], rs2, rs1, funct3, imm[4:1|11], opcode) is used for conditional branches.",
            "expected_doc_id": "riscv_isa_spec.md",
            "expected_section_or_table": "Instruction Formats",
            "query_type": "exact_sku",
        },
        {
            "query_id": "ip_09",
            "dataset": "arm_riscv_ip",
            "query": "What does a DECERR response code indicate in AXI4?",
            "ground_truth_answer": "DECERR (0b11) indicates a decode error, meaning no slave exists at the requested address.",
            "expected_doc_id": "arm_amba_axi4_spec.md",
            "expected_section_or_table": "Response Status Encodings (RRESP and BRESP)",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ip_10",
            "dataset": "arm_riscv_ip",
            "query": "In RISC-V calling conventions, who is responsible for saving registers x8 and x9?",
            "ground_truth_answer": "Registers x8 (s0/fp) and x9 (s1) are saved registers with the Callee saver rule.",
            "expected_doc_id": "riscv_isa_spec.md",
            "expected_section_or_table": "General-Purpose Registers and ABI Names",
            "query_type": "table_lookup",
        },
    ]

    with open(corpus_dir / "golden_eval.jsonl", "w", encoding="utf-8") as f:
        for item in golden_eval:
            f.write(json.dumps(item) + "\n")


def create_hf_tech_bench_corpus(base_dir: Path) -> None:
    corpus_dir = base_dir / "hf_tech_bench"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    doc_content = """# Enterprise Cloud Database & Cache Specification

## 1. Storage Engine Architecture
The Distributed Key-Value Store utilizes a Log-Structured Merge-tree (LSM) engine partitioned into memory and on-disk tiers.

## 2. Node Tiering and Latency Profile
The table below specifies cluster node types, replication factors, and p99 read/write latencies:

| Node Type | vCPU | RAM | Storage Type | P99 Read Latency | P99 Write Latency | Max IOPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KV-Small** | 4 | 16 GB | EBS gp3 | 4.2 ms | 6.8 ms | 12,000 |
| **KV-Medium** | 16 | 64 GB | NVMe SSD | 1.1 ms | 2.4 ms | 80,000 |
| **KV-Large** | 64 | 256 GB | NVMe Direct | 0.35 ms | 0.90 ms | 350,000 |
| **KV-MemoryUltra** | 96 | 768 GB | Optane PMEM | 0.08 ms | 0.25 ms | 1,200,000 |

## 3. Consensus and Quorum Configurations
The system employs Raft consensus across three failure domains:
- Write Quorum: `W = (N / 2) + 1` nodes required for acknowledgement.
- Read Consistency Levels:
  - `ONE`: Fast local replica read (eventual consistency).
  - `QUORUM`: Majority node agreement (strong consistency).
  - `ALL`: Highest consistency, vulnerable to single-node partition timeouts.
"""
    (corpus_dir / "cloud_db_spec.md").write_text(doc_content, encoding="utf-8")

    golden_eval = [
        {
            "query_id": "hf_01",
            "dataset": "hf_tech_bench",
            "query": "What is the P99 read and write latency of the KV-Large node type?",
            "ground_truth_answer": "KV-Large has a P99 read latency of 0.35 ms and P99 write latency of 0.90 ms.",
            "expected_doc_id": "cloud_db_spec.md",
            "expected_section_or_table": "Node Tiering and Latency Profile",
            "query_type": "table_lookup",
        },
        {
            "query_id": "hf_02",
            "dataset": "hf_tech_bench",
            "query": "What storage type and maximum IOPS are provided by KV-MemoryUltra?",
            "ground_truth_answer": "KV-MemoryUltra uses Optane PMEM storage with up to 1,200,000 IOPS.",
            "expected_doc_id": "cloud_db_spec.md",
            "expected_section_or_table": "Node Tiering and Latency Profile",
            "query_type": "table_lookup",
        },
        {
            "query_id": "hf_03",
            "dataset": "hf_tech_bench",
            "query": "What is the Raft write quorum formula?",
            "ground_truth_answer": "The write quorum formula is W = (N / 2) + 1 nodes required for acknowledgement.",
            "expected_doc_id": "cloud_db_spec.md",
            "expected_section_or_table": "Consensus and Quorum Configurations",
            "query_type": "exact_sku",
        },
        {
            "query_id": "hf_04",
            "dataset": "hf_tech_bench",
            "query": "How many vCPUs and RAM does the KV-Medium node have?",
            "ground_truth_answer": "KV-Medium features 16 vCPUs and 64 GB of RAM.",
            "expected_doc_id": "cloud_db_spec.md",
            "expected_section_or_table": "Node Tiering and Latency Profile",
            "query_type": "table_lookup",
        },
        {
            "query_id": "hf_05",
            "dataset": "hf_tech_bench",
            "query": "What read consistency level provides strong consistency via majority agreement?",
            "ground_truth_answer": "The QUORUM read consistency level provides strong consistency via majority node agreement.",
            "expected_doc_id": "cloud_db_spec.md",
            "expected_section_or_table": "Consensus and Quorum Configurations",
            "query_type": "exact_sku",
        },
    ]

    with open(corpus_dir / "golden_eval.jsonl", "w", encoding="utf-8") as f:
        for item in golden_eval:
            f.write(json.dumps(item) + "\n")


def main() -> None:
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating benchmark corpora in: {data_dir}")

    create_nvidia_ai_hw_corpus(data_dir)
    print("[OK] Created Corpus 1: nvidia_ai_hw (NVIDIA Blackwell, H100, NIM specs + 20 eval items)")

    create_arm_riscv_ip_corpus(data_dir)
    print("[OK] Created Corpus 3: arm_riscv_ip (ARM AXI4 & RISC-V specs + 10 eval items)")

    create_hf_tech_bench_corpus(data_dir)
    print("[OK] Created Corpus 2: hf_tech_bench (Cloud DB specs + 5 eval items)")

    print("\nAll benchmark datasets prepared successfully.")


if __name__ == "__main__":
    main()
