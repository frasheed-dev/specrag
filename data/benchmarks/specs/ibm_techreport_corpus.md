# IBM Research: Enterprise Architecture & Power Systems Specifications

## Module 1: IBM Power10 Processor Microarchitecture
The IBM Power10 processor is fabricated on Samsung's 7nm EUV process technology.

### Table 1. Power10 Microarchitectural Parameters
| Parameter | SMT8 Core Mode | SMT4 Core Mode | Native Matrix Engine (MMA) |
|---|---|---|---|
| Cores per Socket | Up to 15 | Up to 30 | 4 MMA engines per SMT8 core |
| L2 Cache | 2 MB per core | 1 MB per sub-core | Private 8-way associative |
| L3 Cache | 120 MB shared | 120 MB shared | Infini-hybrid eDRAM |
| Peak Memory Bandwidth | 409 GB/s per socket | 409 GB/s per socket | OMI (Open Memory Interface) |

## Module 2: Enterprise High-Speed Interconnects & PCIe
Power10 integrates native PCIe Gen5 controllers and PowerAXON symmetric multiprocessing links.

### Table 2. PowerAXON and PCIe Subsystem Bandwidth
| Interface Type | Lanes per Socket | Raw Signaling Rate | Aggregate Bidirectional Bandwidth |
|---|---|---|---|
| PowerAXON (SMP) | 16 links | 32 Gbps NRZ | 1 TB/s |
| PCIe Gen5 Host | 64 lanes | 32 GT/s | 256 GB/s |
| OMI Memory Channels | 16 channels | 32 Gbps | 1,024 GB/s peak |
| NVLink-C2C Bridge Mode | 8 links | 25 Gbps | 400 GB/s |

## Module 3: Security & Cryptographic Accelerators
Every Power10 core includes dedicated cryptographic pipelines for quantum-safe security.

### Table 3. Core Cryptographic Throughput Specifications
| Algorithm | Key Length | Hardware Engines per Core | Max Throughput per Socket |
|---|---|---|---|
| AES-GCM | 256-bit | 4 | 240 Gbps |
| SHA-3 / SHAKE | 512-bit | 2 | 160 Gbps |
| Dilithium (PQC) | Level 3 | Microcode / Vector | 85,000 verifications/sec |
| Kyber-768 (KEM) | Level 3 | Microcode / Vector | 120,000 encapsulations/sec |
