"""Generates offline 20-sample benchmark packs for T2-RAGBench and IBM TechReport.

Creates:
- data/benchmarks/specs/t2_ragbench_corpus.md
- data/benchmarks/specs/ibm_techreport_corpus.md
- data/benchmarks/t2_ragbench_20.json
- data/benchmarks/ibm_techreport_20.json
"""

import json
from pathlib import Path


def generate_packs() -> None:
    benchmarks_dir = Path("data/benchmarks")
    specs_dir = benchmarks_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    # 1. T2-RAGBench Corpus (Financial & Enterprise Text-and-Tables)
    t2_corpus_md = """# T2-RAGBench Financial & Enterprise Report Benchmark

## Section 1: Alphabet Inc. Revenue and Operating Income
The following table details Alphabet Inc. consolidated revenues and segment operating margins.

### Table 1. Alphabet Segment Operating Performance
| Segment | FY 2021 ($M) | FY 2022 ($M) | FY 2023 ($M) | Operating Margin (2023) |
|---|---|---|---|---|
| Google Services | 237,529 | 253,528 | 272,551 | 35.1% |
| Google Cloud | 19,206 | 26,280 | 33,088 | 5.2% |
| Other Bets | 753 | 1,068 | 1,527 | -268.0% |
| Total Consolidated | 257,637 | 282,836 | 307,394 | 27.4% |

## Section 2: Microsoft Corporation Cloud Metrics
Microsoft Cloud revenue and Azure growth rates are summarized below.

### Table 2. Microsoft Cloud Annual Growth & Commercial Metrics
| Metric | FY 2022 | FY 2023 | FY 2024 |
|---|---|---|---|
| Microsoft Cloud Revenue ($B) | 91.2 | 111.6 | 137.4 |
| Azure Revenue Growth (Constant Currency) | 45% | 34% | 30% |
| Commercial Remaining Performance Obligations ($B) | 189 | 224 | 269 |
| Intelligent Cloud Gross Margin | 73% | 72% | 71% |

## Section 3: Semiconductor Capital Expenditures
Annual capex allocations across leading foundries and IDMs.

### Table 3. Semiconductor Equipment & Fab Capex
| Company | 2022 Capex ($B) | 2023 Capex ($B) | 2024 Budget ($B) | Primary Process Node Focus |
|---|---|---|---|---|
| TSMC | 36.3 | 30.4 | 32.0 | N3E / N2 GAA |
| Intel Foundry | 19.0 | 25.8 | 26.5 | Intel 18A / 20A |
| Samsung Foundry | 39.0 | 35.5 | 34.0 | 3nm MBCFET |
| ASML Net Bookings | 30.7 | 28.2 | 29.5 | High-NA EUV (EXE:5000) |
"""
    (specs_dir / "t2_ragbench_corpus.md").write_text(t2_corpus_md, encoding="utf-8")

    # 2. 20 T2-RAGBench Golden Queries
    t2_eval = [
        {
            "query_id": "t2_01_cloud_rev_2023",
            "domain": "finance_numerical_tables",
            "query": "What was the Google Cloud segment revenue for FY 2023 in millions of dollars?",
            "ground_truth_answer": "33,088",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 1. Alphabet Segment Operating Performance",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_02_cloud_margin_2023",
            "domain": "finance_numerical_tables",
            "query": "What was Google Cloud's operating margin percentage in 2023?",
            "ground_truth_answer": "5.2%",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 1. Alphabet Segment Operating Performance",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_03_services_rev_2022",
            "domain": "finance_numerical_tables",
            "query": "What were the Google Services revenues in FY 2022?",
            "ground_truth_answer": "253,528",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 1. Alphabet Segment Operating Performance",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_04_total_rev_2021",
            "domain": "finance_numerical_tables",
            "query": "What was Alphabet's total consolidated revenue in FY 2021?",
            "ground_truth_answer": "257,637",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 1. Alphabet Segment Operating Performance",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_05_total_margin_2023",
            "domain": "finance_numerical_tables",
            "query": "What was Alphabet's consolidated operating margin in 2023?",
            "ground_truth_answer": "27.4%",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 1. Alphabet Segment Operating Performance",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_06_msft_cloud_2024",
            "domain": "finance_numerical_tables",
            "query": "What was Microsoft Cloud's annual revenue in FY 2024 in billions of dollars?",
            "ground_truth_answer": "137.4",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 2. Microsoft Cloud Annual Growth & Commercial Metrics",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_07_azure_growth_2023",
            "domain": "finance_numerical_tables",
            "query": "What was the Azure constant-currency revenue growth in FY 2023?",
            "ground_truth_answer": "34%",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 2. Microsoft Cloud Annual Growth & Commercial Metrics",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_08_msft_rpo_2024",
            "domain": "finance_numerical_tables",
            "query": "What was Microsoft's commercial remaining performance obligation in FY 2024?",
            "ground_truth_answer": "269",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 2. Microsoft Cloud Annual Growth & Commercial Metrics",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_09_cloud_gross_margin_2022",
            "domain": "finance_numerical_tables",
            "query": "What was Microsoft Intelligent Cloud gross margin in FY 2022?",
            "ground_truth_answer": "73%",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 2. Microsoft Cloud Annual Growth & Commercial Metrics",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_10_tsmc_capex_2023",
            "domain": "finance_numerical_tables",
            "query": "How many billions of dollars did TSMC spend on capital expenditures in 2023?",
            "ground_truth_answer": "30.4",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 3. Semiconductor Equipment & Fab Capex",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_11_intel_foundry_2024",
            "domain": "finance_numerical_tables",
            "query": "What is Intel Foundry's capex budget for 2024 in billions of dollars?",
            "ground_truth_answer": "26.5",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 3. Semiconductor Equipment & Fab Capex",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_12_samsung_node_focus",
            "domain": "finance_numerical_tables",
            "query": "What primary semiconductor process node focus is listed for Samsung Foundry?",
            "ground_truth_answer": "3nm MBCFET",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 3. Semiconductor Equipment & Fab Capex",
            "query_type": "exact_sku",
        },
        {
            "query_id": "t2_13_tsmc_node_focus",
            "domain": "finance_numerical_tables",
            "query": "Which advanced process nodes are listed as TSMC's primary focus?",
            "ground_truth_answer": "N3E / N2 GAA",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 3. Semiconductor Equipment & Fab Capex",
            "query_type": "exact_sku",
        },
        {
            "query_id": "t2_14_asml_high_na",
            "domain": "finance_numerical_tables",
            "query": "Which High-NA EUV lithography tool model is targeted under ASML bookings?",
            "ground_truth_answer": "EXE:5000",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 3. Semiconductor Equipment & Fab Capex",
            "query_type": "exact_sku",
        },
        {
            "query_id": "t2_15_intel_capex_2022",
            "domain": "finance_numerical_tables",
            "query": "What was Intel Foundry's capital expenditure in 2022?",
            "ground_truth_answer": "19.0",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 3. Semiconductor Equipment & Fab Capex",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_16_samsung_capex_2022",
            "domain": "finance_numerical_tables",
            "query": "What was Samsung Foundry's capex in 2022 in billions of dollars?",
            "ground_truth_answer": "39.0",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 3. Semiconductor Equipment & Fab Capex",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_17_azure_growth_2024",
            "domain": "finance_numerical_tables",
            "query": "What was Azure constant currency growth rate in FY 2024?",
            "ground_truth_answer": "30%",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 2. Microsoft Cloud Annual Growth & Commercial Metrics",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_18_other_bets_rev_2023",
            "domain": "finance_numerical_tables",
            "query": "What were Alphabet Other Bets revenues in 2023?",
            "ground_truth_answer": "1,527",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 1. Alphabet Segment Operating Performance",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_19_cloud_gross_margin_2024",
            "domain": "finance_numerical_tables",
            "query": "What was Microsoft Intelligent Cloud gross margin in 2024?",
            "ground_truth_answer": "71%",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 2. Microsoft Cloud Annual Growth & Commercial Metrics",
            "query_type": "table_lookup",
        },
        {
            "query_id": "t2_20_asml_bookings_2023",
            "domain": "finance_numerical_tables",
            "query": "What were ASML net bookings in 2023 in billions of dollars?",
            "ground_truth_answer": "28.2",
            "expected_doc_id": "t2_ragbench_corpus.md",
            "expected_section_or_clause": "Table 3. Semiconductor Equipment & Fab Capex",
            "query_type": "table_lookup",
        },
    ]
    (benchmarks_dir / "t2_ragbench_20.json").write_text(
        json.dumps(t2_eval, indent=2), encoding="utf-8"
    )

    # 3. IBM TechReport Corpus (Enterprise Architecture & Hardware Systems)
    ibm_corpus_md = """# IBM Research: Enterprise Architecture & Power Systems Specifications

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
"""
    (specs_dir / "ibm_techreport_corpus.md").write_text(ibm_corpus_md, encoding="utf-8")

    # 4. 20 IBM TechReport Golden Queries
    ibm_eval = [
        {
            "query_id": "ibm_01_power10_process",
            "domain": "enterprise_tech_specs",
            "query": "On which semiconductor process node technology is the IBM Power10 processor fabricated?",
            "ground_truth_answer": "7nm EUV",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Module 1: IBM Power10 Processor Microarchitecture",
            "query_type": "exact_sku",
        },
        {
            "query_id": "ibm_02_mma_engines",
            "domain": "enterprise_tech_specs",
            "query": "How many Matrix Multiply Accelerator (MMA) engines are present per SMT8 core in Power10?",
            "ground_truth_answer": "4 MMA engines per SMT8 core",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 1. Power10 Microarchitectural Parameters",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_03_l3_cache",
            "domain": "enterprise_tech_specs",
            "query": "What is the total shared L3 cache capacity of the IBM Power10 processor?",
            "ground_truth_answer": "120 MB",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 1. Power10 Microarchitectural Parameters",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_04_peak_mem_bw",
            "domain": "enterprise_tech_specs",
            "query": "What is the peak memory bandwidth per socket for IBM Power10?",
            "ground_truth_answer": "409 GB/s",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 1. Power10 Microarchitectural Parameters",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_05_l2_cache_smt8",
            "domain": "enterprise_tech_specs",
            "query": "What is the L2 cache size per core in SMT8 core mode on Power10?",
            "ground_truth_answer": "2 MB",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 1. Power10 Microarchitectural Parameters",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_06_poweraxon_rate",
            "domain": "enterprise_tech_specs",
            "query": "What is the raw signaling rate of the PowerAXON SMP interface links?",
            "ground_truth_answer": "32 Gbps NRZ",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 2. PowerAXON and PCIe Subsystem Bandwidth",
            "query_type": "exact_sku",
        },
        {
            "query_id": "ibm_07_poweraxon_bw",
            "domain": "enterprise_tech_specs",
            "query": "What is the aggregate bidirectional bandwidth of PowerAXON SMP links per socket?",
            "ground_truth_answer": "1 TB/s",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 2. PowerAXON and PCIe Subsystem Bandwidth",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_08_pcie_lanes",
            "domain": "enterprise_tech_specs",
            "query": "How many PCIe Gen5 lanes are supported per Power10 socket?",
            "ground_truth_answer": "64 lanes",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 2. PowerAXON and PCIe Subsystem Bandwidth",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_09_pcie_bw",
            "domain": "enterprise_tech_specs",
            "query": "What is the aggregate bidirectional bandwidth of the PCIe Gen5 host subsystem?",
            "ground_truth_answer": "256 GB/s",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 2. PowerAXON and PCIe Subsystem Bandwidth",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_10_omi_channels",
            "domain": "enterprise_tech_specs",
            "query": "How many Open Memory Interface (OMI) channels are integrated per socket?",
            "ground_truth_answer": "16 channels",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 2. PowerAXON and PCIe Subsystem Bandwidth",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_11_omi_peak_bw",
            "domain": "enterprise_tech_specs",
            "query": "What is the peak aggregate bandwidth across all OMI channels on Power10?",
            "ground_truth_answer": "1,024 GB/s",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 2. PowerAXON and PCIe Subsystem Bandwidth",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_12_aes_gcm_throughput",
            "domain": "enterprise_tech_specs",
            "query": "What is the maximum AES-GCM 256-bit cryptographic throughput per socket?",
            "ground_truth_answer": "240 Gbps",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 3. Core Cryptographic Throughput Specifications",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_13_sha3_engines",
            "domain": "enterprise_tech_specs",
            "query": "How many SHA-3 hardware acceleration engines are integrated per Power10 core?",
            "ground_truth_answer": "2",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 3. Core Cryptographic Throughput Specifications",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_14_sha3_throughput",
            "domain": "enterprise_tech_specs",
            "query": "What is the maximum SHA-3 cryptographic throughput per socket?",
            "ground_truth_answer": "160 Gbps",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 3. Core Cryptographic Throughput Specifications",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_15_dilithium_rate",
            "domain": "enterprise_tech_specs",
            "query": "What is the post-quantum Dilithium Level 3 verification throughput per socket?",
            "ground_truth_answer": "85,000 verifications/sec",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 3. Core Cryptographic Throughput Specifications",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_16_kyber_rate",
            "domain": "enterprise_tech_specs",
            "query": "What is the Kyber-768 key encapsulation throughput per socket on Power10?",
            "ground_truth_answer": "120,000 encapsulations/sec",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 3. Core Cryptographic Throughput Specifications",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_17_cores_smt4",
            "domain": "enterprise_tech_specs",
            "query": "How many cores per socket are supported when operating in SMT4 mode?",
            "ground_truth_answer": "Up to 30",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 1. Power10 Microarchitectural Parameters",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_18_nvlink_c2c_bw",
            "domain": "enterprise_tech_specs",
            "query": "What is the aggregate bandwidth for NVLink-C2C bridge mode on Power10?",
            "ground_truth_answer": "400 GB/s",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 2. PowerAXON and PCIe Subsystem Bandwidth",
            "query_type": "table_lookup",
        },
        {
            "query_id": "ibm_19_nvlink_c2c_rate",
            "domain": "enterprise_tech_specs",
            "query": "What signaling rate is specified for NVLink-C2C bridge mode links?",
            "ground_truth_answer": "25 Gbps",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 2. PowerAXON and PCIe Subsystem Bandwidth",
            "query_type": "exact_sku",
        },
        {
            "query_id": "ibm_20_aes_engines_per_core",
            "domain": "enterprise_tech_specs",
            "query": "How many dedicated AES-GCM hardware engines are in each Power10 core?",
            "ground_truth_answer": "4",
            "expected_doc_id": "ibm_techreport_corpus.md",
            "expected_section_or_clause": "Table 3. Core Cryptographic Throughput Specifications",
            "query_type": "table_lookup",
        },
    ]
    (benchmarks_dir / "ibm_techreport_20.json").write_text(
        json.dumps(ibm_eval, indent=2), encoding="utf-8"
    )

    print("[SpecRAG] Benchmark packs generated successfully in data/benchmarks/")


if __name__ == "__main__":
    generate_packs()
