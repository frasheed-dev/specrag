# NVIDIA H100 Tensor Core GPU Architecture Datasheet

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
