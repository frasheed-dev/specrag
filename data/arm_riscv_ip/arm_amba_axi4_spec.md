# ARM AMBA AXI4 Protocol Specification Summary

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
