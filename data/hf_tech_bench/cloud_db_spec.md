# Enterprise Cloud Database & Cache Specification

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
