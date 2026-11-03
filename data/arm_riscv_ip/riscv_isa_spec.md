# RISC-V Unprivileged ISA Specification Excerpt

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
