# NVIDIA Blackwell Architecture Technical Overview

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
